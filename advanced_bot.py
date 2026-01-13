import certifi
import telebot
from telebot.types import Message
from pymongo import MongoClient
import google.genai as genai
from google.api_core.exceptions import ResourceExhausted
import fitz  # PyMuPDF
import os
import json
import re
import time
import sys
import math
import traceback
import requests
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running"

# ——— कन्फिगरेसन (Render Environment Variables बाट पढ्ने) ———
BOT_TOKEN = os.getenv("BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Simplified to a single key
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))
BACKUP_CHANNEL_ID = int(os.getenv("BACKUP_CHANNEL_ID", 0))

DOWNLOAD_PATH = "temp_pdfs"
if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH)

# ——— INITIALIZATION (Render-optimized) ———
bot = telebot.TeleBot(BOT_TOKEN, threaded=True) # Threaded mode for performance
client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    socketTimeoutMS=20000,
    connectTimeoutMS=20000,
    tlsCAFile=certifi.where()
)
db = client['TelegramBotDB']
pdf_collection = db['PDF_Store']
notes_collection = db['Notes']
counters_collection = db['Counters']
history_collection = db['Chat_History']

# Configure Gemini with the single API key
if GEMINI_API_KEY:
    from google.genai import Client
    genai_client = Client(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY environment variable not set.")
    genai_client = None

MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro"
]

# अपडेटेड मॉडल लिस्ट (2026 के अनुसार)
FREE_TIER_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite", 
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",  # लेगेसी फेलबैक
]

# क्वोटा ट्रैकिंग के लिए
failed_models = set()

# ——— CORE HELPER FUNCTIONS ———

def log_exception(e):
    """विस्तृत त्रुटि लग गर्नका लागि।"""
    print(f"An exception occurred: {e}")
    traceback.print_exc(file=sys.stdout)

def clean_json(raw_text):
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    return match.group(0) if match else raw_text

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    # Handle potential zero division
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def get_embedding(text, task_type="RETRIEVAL_DOCUMENT"):
    try:
        res = genai_client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config={
                "task_type": task_type,
            }
        )
        return res.embeddings[0].values
    except ResourceExhausted:
        return "QUOTA_EXCEEDED"
    except Exception as e:
        log_exception(e)
        return None

def get_next_serial_number(sequence_name):
    sequence_doc = counters_collection.find_one_and_update(
        {'_id': sequence_name}, {'$inc': {'sequence_value': 1}},
        return_document=True, upsert=True
    )
    return sequence_doc['sequence_value']

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            # यूनिकोड और लेआउट मोड का उपयोग
            text += page.get_text("text", sort=True, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE)
        doc.close()
        
        # यदि टेक्स्ट कम है तो OCR का उपयोग
        if len(text.strip()) < 100:
            return extract_vision_text(file_path)
        return text
    except Exception as e:
        print(f"PDF टेक्स्ट निकालने में त्रुटि: {e}")
        log_exception(e)
        return extract_vision_text(file_path)  # फेलबैक के रूप में OCR

def extract_vision_text(file_path):
    img_path = f"temp_scan_{os.path.basename(file_path)}.png"
    try:
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        pix.save(img_path)
        doc.close()

        uploaded_file = genai_client.files.upload(img_path)
        response = genai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                "Extract all text from this document page:",
                uploaded_file
            ]
        )
        genai_client.files.delete(name=uploaded_file.name)
        return response.text
    except ResourceExhausted as e:
        print(f"Vision OCR quota error: {e}")
        log_exception(e)
        return "QUOTA_EXCEEDED_VISION"
    except Exception as e:
        print(f"Vision OCR असफल भयो: {e}")
        log_exception(e)
        return None
    finally:
        # FIX: Ensure temp image is always deleted
        if os.path.exists(img_path):
            os.remove(img_path)

def send_long_message(chat_id, text, reply_to_message_id=None, parse_mode="Markdown"):
    if not text: return
    if len(text) <= 4000:
        bot.send_message(chat_id, text, reply_to_message_id=reply_to_message_id, parse_mode=parse_mode)
    else:
        parts = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for part in parts:
            bot.send_message(chat_id, part, parse_mode="Markdown") # reply_to only for first part maybe
            time.sleep(1)

def get_chat_history(user_id):
    history = history_collection.find({"user_id": user_id}).sort("_id", -1).limit(10)
    formatted_history = []
    for msg in reversed(list(history)):
        formatted_history.append({"role": "user", "parts": [msg['user_msg']]})
        formatted_history.append({"role": "model", "parts": [msg['bot_res']]})
    return formatted_history
def save_chat_history(user_id, user_msg, bot_res):
    history_collection.insert_one({"user_id": user_id, "user_msg": user_msg, "bot_res": bot_res})

def fallback_to_alternative_api(prompt):
    """अन्य फ्री एपीआई का उपयोग (OpenAI, HuggingFace, आदि)"""
    import requests
    
    try:
        # HuggingFace Inference API (फ्री)
        
        # वैकल्पिक 1: HuggingFace Zephyr
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            hf_response = requests.post(
                "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
                headers={"Authorization": f"Bearer {hf_token}"},
                json={"inputs": prompt}
            )
            
            if hf_response.status_code == 200:
                generated_text = hf_response.json()[0]['generated_text']
                # Clean up prompt from response if present
                if generated_text.startswith(prompt):
                    return generated_text[len(prompt):].strip()
                return generated_text
                
        # वैकल्पिक 2: OpenRouter (फ्री मॉडल)
        openrouter_key = os.getenv('OPENROUTER_KEY')
        if openrouter_key:
            openrouter_response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemini-2.0-flash-lite:free", # Using a free model on OpenRouter
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if openrouter_response.status_code == 200:
                return openrouter_response.json()['choices'][0]['message']['content']
                
    except Exception as e:
        print(f"फेलबैक API त्रुटि: {e}")
        log_exception(e)
    
    return "❌ सभी AI सेवाएं अस्थाई रूप से अनुपलब्ध हैं। कृपया कुछ समय बाद प्रयास करें।"

def call_gemini_smart_improved(prompt, history=None):
    """क्वोटा मैनेजमेंट और फेलबैक के साथ अपडेटेड फंक्शन"""
    if not GEMINI_API_KEY:
        return "सेवा उपलब्ध नहीं है।"
    
    # सिस्टम इंस्ट्रक्शन जोड़ें (नेपाली फोंट के लिए)
    system_instruction = """
    आप एक मल्टीलिंगुअल असिस्टेंट हैं। 
    नेपाली, हिंदी और अंग्रेजी सभी भाषाओं को समझें और प्रोसेस करें।
    संख्याओं और विशेष करैक्टर्स को सही से हैंडल करें।
    """
    
    # Prepare contents with system instruction and history
    contents = []
    if system_instruction:
        contents.append({"role": "user", "parts": [{"text": system_instruction}]})
        contents.append({"role": "model", "parts": [{"text": "ठीक है, मैं तैयार हूँ।"}]})
    
    if history:
        contents.extend(history)
    
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    for model_name in FREE_TIER_MODELS: # Use the new FREE_TIER_MODELS list
        if model_name in failed_models:
            continue
            
        try:
            print(f"ट्राइंग मॉडल: {model_name}")
            response = genai_client.models.generate_content(
                model=f"models/{model_name}",
                contents=contents
            )
            if response and response.text:
                return response.text
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"मॉडल {model_name} फेल्ड: {error_msg}")
            
            # क्वोटा एरर की पहचान
            if "quota" in error_msg or "429" in error_msg or "resource exhausted" in error_msg:
                print(f"क्वोटा समाप्त: {model_name}")
                failed_models.add(model_name)
                continue  # अगले मॉडल की कोशिश करें
            elif "not found" in error_msg or "invalid" in error_msg:
                print(f"मॉडल नहीं मिला: {model_name}")
                failed_models.add(model_name)
                continue
            else:
                # अन्य एरर - थोड़ी देर रुक कर कोशिश करें
                time.sleep(2)
                log_exception(e)
                continue
    
    # सभी मॉडल फेल होने पर
    return fallback_to_alternative_api(prompt)


# ——— BOT MESSAGE HANDLERS ———

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "नमस्ते! म तपाईंको निजी ज्ञान आधार (Knowledge Base) बोट हुँ।")

@bot.message_handler(content_types=['document'])
def handle_pdf_universal(message):
    if message.document.mime_type != 'application/pdf': return
    if message.document.file_size > 20 * 1024 * 1024: return bot.reply_to(message, "यो फाइल धेरै ठूलो छ (20MB+)।")
    if pdf_collection.find_one({"file_id": message.document.file_id}):
        try: bot.delete_message(message.chat.id, message.message_id)
        except: pass
        return bot.send_message(message.chat.id, f"यो PDF पहिले नै बचत गरिएको छ।")

    status_msg = bot.send_message(message.chat.id, f"⏳ '{message.document.file_name}' प्रशोधन गर्दै...")
    file_path = None
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        file_path = os.path.join(DOWNLOAD_PATH, message.document.file_name)
        with open(file_path, 'wb') as new_file: new_file.write(downloaded_file)

        text = extract_text_from_pdf(file_path)
        pdf_type = "digital"
        if not text:
            bot.edit_message_text(f"डिजिटल पाठ फेला परेन, Vision OCR प्रयास गर्दै...", status_msg.chat.id, status_msg.message_id)
            text = extract_vision_text(file_path)
            if text == "QUOTA_EXCEEDED_VISION":
                return bot.edit_message_text("❌ AI Quota Error: The daily free limit for processing scanned documents has been reached. Please try again tomorrow.", status_msg.chat.id, status_msg.message_id)
            pdf_type = "scanned"
        
        if not text: return bot.edit_message_text("❌ माफ गर्नुहोस्, यो PDF बाट कुनै पाठ निकाल्न सकिएन।", status_msg.chat.id, status_msg.message_id)

        summary_prompt = f"यो सामग्रीलाई खोज अनुक्रमणिकाको लागि २ वाक्यमा सारांश गर्नुहोस्: {text[:2000]}"
        summary = call_gemini_smart_improved(summary_prompt)
        vector = get_embedding(summary, task_type="RETRIEVAL_DOCUMENT")
        if vector == "QUOTA_EXCEEDED":
            return bot.edit_message_text("❌ AI Quota Error: The daily free limit for processing new documents has been reached. Please try again tomorrow.", status_msg.chat.id, status_msg.message_id)
        if not vector: return bot.edit_message_text("❌ AI Error: Vector generation failed. Try again.", status_msg.chat.id, status_msg.message_id)
        
        serial_no = get_next_serial_number('pdf_id')
        backup_msg = bot.forward_message(BACKUP_CHANNEL_ID, message.chat.id, message.message_id)
        
        pdf_collection.insert_one({
            "serial_number": serial_no, "file_name": message.document.file_name, "file_id": message.document.file_id,
            "summary": summary, "embedding": vector, "full_text": text, "type": pdf_type, "backup_msg_id": backup_msg.message_id,
            "uploader_id": message.from_user.id
        })

        try: bot.delete_message(message.chat.id, message.message_id)
        except: pass
        bot.edit_message_text(f"✅ PDF #{serial_no} ({pdf_type}) '{message.document.file_name}' सफलतापूर्वक प्रशोधन र सुरक्षित गरियो।", status_msg.chat.id, status_msg.message_id)

    except Exception as e:
        log_exception(e)
        bot.edit_message_text(f"माफ गर्नुहोस्, फाइल प्रशोधन गर्दा त्रुटि आयो: {e}", status_msg.chat.id, status_msg.message_id)
    finally:
        if file_path and os.path.exists(file_path): os.remove(file_path)

@bot.message_handler(commands=['get'])
def retrieve_pdf(message):
    if message.from_user.id != ADMIN_ID:
        return bot.reply_to(message, "❌ सुरक्षा कारणले गर्दा PDF फाइल डाउनलोड गर्न अनुमति छैन। तपाईं यसको बारेमा AI सँग सोध्न सक्नुहुन्छ।")
    if message.chat.type != 'private':
        try: bot.send_message(message.from_user.id, "🛡️ सुरक्षाका लागि, म तपाईंलाई यो फाइल Private Message (PM) मा पठाउँदैछु।")
        except: return bot.reply_to(message, "Please start a chat with me privately first so I can PM you.")
        return bot.reply_to(message, "🛡️ म तपाईंलाई यो फाइल PM मा पठाउँदैछु।")

    try:
        args = message.text.split()
        if len(args) < 2: return bot.reply_to(message, "नम्बर दिनुहोस्। Ex: /get 1")
        index_no = int(args[1])
        res = pdf_collection.find_one({"serial_number": index_no})
        if res: bot.send_document(ADMIN_ID, res['file_id'], caption=f"📄 Admin Copy: {res['file_name']}")
        else: bot.reply_to(message, "फाइल भेटिएन।")
    except Exception as e: 
        log_exception(e)
        bot.reply_to(message, f"त्रुटि भयो: {e}")

@bot.message_handler(commands=['ask_file'])
def ask_from_file(message):
    query = message.text.replace('/ask_file', '').strip()
    if not query:
        return bot.reply_to(message, "कृपया फाइलको बारेमा केही सोध्नुहोस्। उदाहरण: `/ask_file यो PDF के बारेमा छ?`")
    
    status_msg = bot.reply_to(message, "🔍 फाइलों में खोज रहा हूं...")
    
    try:
        # Step 1: Generate embedding for the query
        vector = get_embedding(query, task_type="RETRIEVAL_QUERY")
        if vector == "QUOTA_EXCEEDED":
            return bot.edit_message_text("❌ AI Quota Error: The daily free limit for asking questions has been reached. Please try again tomorrow.", status_msg.chat.id, status_msg.message_id)
        if not vector:
            return bot.edit_message_text(
                "❌ AI त्रुटि: प्रश्न का वेक्टर बनाने में असफल।",
                status_msg.chat.id, 
                status_msg.message_id
            )
        
        # Manual Similarity Search (Option 1)
        all_pdfs = list(pdf_collection.find({}, {"serial_number": 1, "file_name": 1, "summary": 1, "embedding": 1, "full_text": 1, "_id": 0}))
        
        if not all_pdfs:
            bot.edit_message_text(
                "📭 कुनै पनि PDF फाइल भेटिएन। AI बाट सामान्य जवाफ लिँदै छु...",
                status_msg.chat.id, 
                status_msg.message_id
            )
            general_prompt = f"User asked: {query}\n\nPlease provide a helpful answer to this question based on your general knowledge."
            ai_response = call_gemini_smart_improved(general_prompt)
            
            bot.delete_message(message.chat.id, status_msg.message_id)
            send_long_message(
                message.chat.id, 
                f"📘 **AI का सामान्य जवाब:**\n\n"
                f"{ai_response}\n\n"
                f"_💡 नोट: यो जवाफ मेरो सामान्य जानकारीमा आधारित छ, कुनै विशेष फाइलबाट होइन।_",
                reply_to_message_id=message.message_id,
                parse_mode="Markdown"
            )
            return

        best_doc = None
        best_score = -1

        for doc in all_pdfs:
            if "embedding" in doc and doc["embedding"]:
                score = cosine_similarity(vector, doc["embedding"])
                if score > best_score:
                    best_score = score
                    best_doc = doc
        
        # Step 2: अगर कोई PDF नहीं मिला या स्कोर कम है
                if not best_doc or best_score < 0.50: # 0.50 is the similarity threshold
                    bot.edit_message_text(
                        "📭 फाइलमा जानकारी भेटिएन, AI बाट सामान्य जवाफ लिँदै छु...",
                        status_msg.chat.id,
                        status_msg.message_id
                    )
                    
                    general_prompt = f"User asked: {query}\n\nPlease provide a helpful answer to this question based on your general knowledge."
                    ai_response = call_gemini_smart_improved(general_prompt)
                    
                    bot.delete_message(message.chat.id, status_msg.message_id)
                    send_long_message(
                        message.chat.id,
                        f"📘 **AI का सामान्य जवाब:**\n\n"
                        f"{ai_response}\n\n"
                        f"_💡 नोट: यो जवाफ मेरो सामान्य जानकारीमा आधारित छ, कुनै विशेष फाइलबाट होइन।_",
                        reply_to_message_id=message.message_id,
                        parse_mode="Markdown"
                    )
                    return
        
        # Step 3: PDF मिला है - सबसे relevant PDF का उपयोग करें
        context = best_doc['full_text'] if 'full_text' in best_doc else best_doc['summary']
        context = context[:3000] # Limit context to prevent Gemini overload
        
        bot.edit_message_text(
            f"📄 **{best_doc['file_name']}** में खोज रहा हूं...",
            status_msg.chat.id, 
            status_msg.message_id
        )
        
        # Enhanced prompt with page finding logic
        enhanced_prompt = f"""
        PDF Context (Relevant section from {best_doc['file_name']}):
        {context}
        
        User Question: {query}
        
        Instructions:
        1. Answer based ONLY on the given PDF context above
        2. If information is found, mention that it's from the PDF and indicate the serial number of the PDF.
        3. If possible, estimate which page this information might be on (e.g., "beginning," "middle," or "end" of the document, or "page X" if an exact number can be inferred from context, though exact page numbers are not available).
        4. If information is NOT in the context, say clearly "यह जानकारी PDF में नहीं मिली।"
        5. Answer in a natural, helpful way. Ensure all responses are primarily in Nepali if possible.
        
        Answer:
        """
        
        ai_response = call_gemini_smart_improved(enhanced_prompt)
        
        # Step 5: Format the response
        pdf_info = f"PDF #{best_doc['serial_number']} ({best_doc['file_name']})"
        
        # Check if AI found the answer in PDF
        not_found_phrases = ["not found", "नहीं मिला", "जानकारी नहीं है", "not in the pdf"] # Added Nepali phrase
        if any(phrase in ai_response.lower() for phrase in not_found_phrases):
            # Fallback to general AI answer
            bot.edit_message_text(
                "📭 PDF में जानकारी नहीं मिली, AI से सामान्य जवाब ले रहा हूं...",
                status_msg.chat.id, 
                status_msg.message_id
            )
            
            general_prompt = f"User asked: {query}\n\nPlease provide a helpful answer based on your general knowledge. Answer in Nepali."
            ai_response = call_gemini_smart_improved(general_prompt)
            
            bot.delete_message(message.chat.id, status_msg.message_id)
            send_long_message(
                message.chat.id,
                f"📘 **AI का सामान्य जवाब:**\n\n"
                f"{ai_response}\n\n"
                f"_💡 नोट: यो जवाफ मेरो सामान्य जानकारीमा आधारित छ, कुनै विशेष फाइलबाट होइन।_",
                reply_to_message_id=message.message_id,
                parse_mode="Markdown"
            )
        else:
            # Found in PDF - show with PDF info
            bot.delete_message(message.chat.id, status_msg.message_id)
            
            # Try to extract page number from AI response
            # Modified regex to be more flexible and capture page number hints
            page_match = re.search(r'(पेज\s*\d+|beginning|middle|end)', ai_response.lower())
            page_info = ""
            if page_match:
                page_info = f" ({page_match.group(0)})" # Use the captured group directly
            
            send_long_message(
                message.chat.id,
                f"📄 **{pdf_info}{page_info} के आधार पर:**\n\n"
                f"{ai_response}\n\n"
                f"_✅ जानकारी PDF से ली गई है_",
                reply_to_message_id=message.message_id,
                parse_mode="Markdown"
            )
            
    except Exception as e:
        log_exception(e)
        bot.edit_message_text(
            f"❌ त्रुटि: {str(e)[:100]}",
            status_msg.chat.id, 
            status_msg.message_id
        )

def ask_general_ai(message, query, status_msg=None):
    """सामान्य AI से जवाब लें"""
    if status_msg:
                    bot.edit_message_text(
                        "🤖 AI बाट सामान्य ज्ञान लिँदै छु...",
                        status_msg.chat.id,
                        status_msg.message_id
                    )
    else:
        status_msg = bot.reply_to(message, "🤖 AI बाट सामान्य ज्ञान लिँदै छु...")
    
    prompt = f"""
    User Question: {query}
    
    Instructions:
    1. Provide a helpful, accurate answer
    2. If you're not sure, say so
    3. Be concise but informative
    4. Answer in Hindi/English as appropriate. Prefer Nepali if context allows.
    
    Answer:
    """
    
    ai_response = call_gemini_smart_improved(prompt)
    
    bot.delete_message(message.chat.id, status_msg.message_id)
    send_long_message(
        message.chat.id,
        f"🤖 **AI को सामान्य जवाफ:**\n\n"
        f"{ai_response}\n\n"
        f"_💡 नोट: यो जवाफ मेरो सामान्य जानकारीमा आधारित छ, कुनै विशेष फाइलबाट होइन।_",
        reply_to_message_id=message.message_id,
        parse_mode="Markdown"
    )

@bot.message_handler(commands=['ask_ai'])
def ask_ai_command(message):
    query = message.text.replace('/ask_ai', '').strip()
    if not query:
        return bot.reply_to(message, "कृपया AI बाट केही सोध्नुहोस्।")
    ask_general_ai(message, query)


@bot.message_handler(commands=['ask_smart'])
def ask_smart(message):
    """स्मार्टली डिसाइड करें - PDF में है या AI से पूछें"""
    query = message.text.replace('/ask_smart', '').strip()
    
    if not query:
        return bot.reply_to(message, "प्रश्न दर्ज करें।")
    
    # First, check if this is a PDF-related question
    pdf_keywords = ['pdf', 'file', 'document', 'फाइल', 'दस्तावेज', 'पीडीएफ']
    
    is_pdf_question = any(keyword in query.lower() for keyword in pdf_keywords)
    
    if is_pdf_question:
        # Use /ask_file logic
        ask_from_file(message) # Note: this calls the modified ask_from_file
    else:
        # Direct AI chat
        handle_chat_improved(message) # This will be the new improved handler

@bot.message_handler(commands=['help'])
def help_command(message):
    help_text = """
    🤖 **कमांड्स:**
    
    `/ask_file [प्रश्न]` - सिर्फ PDFs में खोजे (मैनुअल वेक्टर खोज)
    `/ask_ai [प्रश्न]` - सिर्फ AI से पूछे (सामान्य ज्ञान)
    `/ask_smart [प्रश्न]` - पहले PDF में खोजेगा अगर कीवर्ड मिलते हैं, वरना AI से पूछेगा
    `/quiz [PDF नंबर]` - PDF पर आधारित क्विज बनाएगा
    `/start` - बॉट का परिचय
    
    **उदाहरण:**
    `/ask_file इस PDF में क्या है?`
    `/ask_ai भारत की राजधानी क्या है?`
    `/ask_smart machine learning क्या है?`
    """
    bot.reply_to(message, help_text, parse_mode="Markdown")

# Modify handle_chat to incorporate the ask_smart logic
@bot.message_handler(func=lambda message: not message.text.startswith('/'))
def handle_chat(message):
    if message.chat.type == 'private' or (message.reply_to_message and message.reply_to_message.from_user.id == bot.get_me().id):
        # Incorporate ask_smart logic
        query = message.text.strip()
        pdf_keywords = ['pdf', 'file', 'document', 'फाइल', 'दस्तावेज', 'पीडीएफ']
        is_pdf_question = any(keyword in query.lower() for keyword in pdf_keywords)
        
        if is_pdf_question:
            ask_from_file(message)
        else:
            # Original handle_chat logic for general AI
            history = get_chat_history(message.from_user.id)
            bot.send_chat_action(message.chat.id, 'typing')
            res = call_gemini_smart_improved(message.text, history)
            save_chat_history(message.from_user.id, message.text, res)
            send_long_message(message.chat.id, res, reply_to_message_id=message.message_id)

# ——— BOT START (Render Safe) ———
def run_bot():
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)

if __name__ == "__main__":
    print("Bot started...")
    threading.Thread(target=run_bot).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))