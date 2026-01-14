import certifi
import telebot
from telebot.types import Message
from pymongo import MongoClient
import google.generativeai as genai
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
from PIL import Image

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

# API Setup
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY missing")

# Working model list as of Jan 2026, supporting vision
WORKING_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.0-flash-lite",
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

def get_embedding(text, task_type="retrieval_document"):
    try:
        # Use the new API for embedding
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type=task_type
        )
        return result['embedding']
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

def extract_vision_text(file_path):
    """
    यो फसनले PDF को पहिलो पेजलाई फोटोमा बदल्छ र Gemini Vision लाई पठाउँछ।
    यसले नेपाली फन्ट र गणितीय फर्मुलाहरू (Math) एकदम सही निकाल्छ।
    """
    img_path = f"{file_path}_temp.png"
    uploaded_file = None # Use a different name to avoid confusion
    try:
        # 1. PDF लाई फोटोमा बदल्ने (Zoom गरेर)
        doc = fitz.open(file_path)
        mat = fitz.Matrix(2, 2)
        pix = doc[0].get_pixmap(matrix=mat)
        doc.close()
        pix.save(img_path)

        # 2. Gemini मा फोटो अपलोड गर्ने र ACTIVE हुन पर्खिने
        print("Uploading image to Gemini for OCR...")
        uploaded_file = genai.upload_file(path=img_path, display_name=os.path.basename(img_path))
        
        print(f"File uploaded: {uploaded_file.name}, State: {uploaded_file.state.name}")
        while uploaded_file.state.name == "PROCESSING":
            print("Waiting for file to be processed...")
            time.sleep(4) # Increased sleep time
            uploaded_file = genai.get_file(name=uploaded_file.name)
            print(f"File state: {uploaded_file.state.name}")
            
        if uploaded_file.state.name != "ACTIVE":
            print(f"Error: Uploaded file is not active. State: {uploaded_file.state.name}")
            return None

        # 3. फोटोबाट टेक्स्ट माग्ने (Fallback logic সহ)
        prompt_parts = ["Extract all text from this document page exactly as it is. Preserve Nepali text and Math formulas.", uploaded_file]
        
        for model_name in WORKING_MODELS:
            try:
                print(f"Trying vision model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt_parts)
                # If we get a response, return it immediately
                return response.text
            except Exception as e:
                error_msg = str(e).lower()
                print(f"Vision model {model_name} failed: {error_msg}")
                # If model is not found, or quota is hit, or it's an invalid argument for this model, try the next one.
                if any(err in error_msg for err in ["404", "not found", "quota", "invalid argument"]):
                    continue
                else:
                    log_exception(e)
                    continue # Try next model even for other errors
        
        # If all models failed
        print("All vision models failed to extract text.")
        return None
        
    except Exception as e:
        print(f"Vision Error: {e}")
        log_exception(e)
        return None
        
    finally:
        # टेम्पोररी फोटो र अपलोड गरिएको फाइल डिलिट गर्ने
        if uploaded_file:
            print(f"Deleting uploaded file: {uploaded_file.name}")
            genai.delete_file(name=uploaded_file.name)
        if os.path.exists(img_path):
            os.remove(img_path)

def smart_pdf_extract(file_path):
    """
    यो 'Smart' फसन हो। पहिले यसले सामान्य तरिकाले टेक्स्ट निकाल्न खोज्छ।
    यदि टेक्स्ट बुझिएन वा एकदम कम आयो (जस्तै स्क्यान गरेको फाइल),
    तब यसले माथिको 'extract_vision_text' प्रयोग गर्छ।
    """
    try:
        # सामान्य तरिका (छिटो हुन्छ)
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # यदि टेक्स्ट ५० अक्षर भन्दा कम छ वा खाली छ भने -> Vision प्रयोग गर्ने
        if len(text.strip()) < 50:
            print("Low quality text detected, switching to Vision OCR...")
            vision_text = extract_vision_text(file_path)
            if vision_text:
                return vision_text, "Vision OCR (Image)"
            else:
                print("Vision OCR also failed to extract text.")
                return text, "Vision OCR Failed"  # Return original text but with a failure status
        
        return text, "Digital Text"
    except Exception as e:
        print(f"Standard text extraction failed: {e}. Falling back to Vision OCR.")
        vision_text = extract_vision_text(file_path)
        if vision_text:
            return vision_text, "Fallback OCR"
        else:
            return None, "Extraction Failed"

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
    आप एक मल्टीलिंगुअल असिस्टेंट हैं जो नेपाली, हिंदी और अंग्रेजी सभी भाषाओं को समझते और प्रोसेस करते हैं।
    आपका प्राथमिक लक्ष्य नेपाली में जवाब देना है। आप कभी-कभी अंग्रेजी या हिंदी शब्दों का उपयोग कर सकते हैं, लेकिन सुनिश्चित करें कि मुख्य भाषा नेपाली ही हो।
    संख्याओं और विशेष करैक्टर्स को सही से हैंडल करें।
    """
    
    # Prepare contents with system instruction and history
    contents = []
    if system_instruction:
        contents.append({"role": "user", "parts": [{"text": system_instruction}]})
        contents.append({"role": "model", "parts": [{"text": "ठीक है, मैं तैयार हूँ।"}]}
    
    if history:
        contents.extend(history)
    
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    for model_name in WORKING_MODELS: # Use the new WORKING_MODELS list
        if model_name in failed_models:
            continue
            
        try:
            print(f"ट्राइंग मॉडल: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(contents)
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
def handle_pdf(message):
    if message.document.mime_type != 'application/pdf':
        return bot.reply_to(message, "कृपया PDF फाइल मात्र पठाउनुहोस्।")

    status_msg = bot.send_message(message.chat.id, "📥 फाइल डाउनलोड र स्क्यान गर्दै...")
    
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_path = os.path.join(DOWNLOAD_PATH, message.document.file_name)
    
    with open(file_path, 'wb') as f:
        f.write(downloaded_file)

    try:
        # १. टेक्स्ट निकाल्ने
        text, method = smart_pdf_extract(file_path)
        
        if method in ["Vision OCR Failed", "Extraction Failed"] or not text or not text.strip():
            return bot.edit_message_text("❌ माफ गर्नुहोस्, यो PDF खाली छ वा पढ्न सकिएन।", message.chat.id, status_msg.message_id)

        # २. डिबग म्यासेज
        debug_msg = f"🔍 **DEBUG: Extracted Content ({method})**\n\n```\n{text[:800]}...\n```"
        bot.send_message(message.chat.id, debug_msg, parse_mode="Markdown")

        # ३. सारांश र Embedding
        summary_prompt = f"Summarize this in 3 sentences: {text[:4000]}"
        summary = call_gemini_smart_improved(summary_prompt)

        emb_result = genai.embed_content(
            model="models/text-embedding-004",
            content=summary,
            task_type="retrieval_document"
        )
        vector = emb_result['embedding']

        # ४. DB मा सेभ (FIXED KEYS)
        serial = get_next_serial_number("pdf_id")
        pdf_collection.insert_one({
            "serial": serial,                     # Key: serial
            "file_name": message.document.file_name, # Key: file_name
            "file_id": message.document.file_id,     # NEW: file_id (for /get command)
            "text": text,
            "summary": summary,
            "embedding": vector,
            "uploader": message.from_user.id,
            "timestamp": time.time()
        })

        bot.edit_message_text(
            f"✅ **PDF #{serial} सुरक्षित भयो!**\n\n📝 **सारांश:**\n{summary}", 
            message.chat.id, status_msg.message_id, parse_mode="Markdown"
        )

    except Exception as e:
        log_exception(e)
        bot.edit_message_text(f"❌ त्रुटि आयो: {str(e)}", message.chat.id, status_msg.message_id)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@bot.message_handler(commands=['get'])
def retrieve_pdf(message):
    if message.from_user.id != ADMIN_ID:
        return bot.reply_to(message, "❌ Admin Only Command.")

    try:
        args = message.text.split()
        if len(args) < 2: return bot.reply_to(message, "नम्बर दिनुहोस्। Ex: /get 1")
        
        index_no = int(args[1])
        
        # FIX: 'serial' वा 'serial_number' दुबै चेक गर्ने
        res = pdf_collection.find_one({"$or": [{"serial": index_no}, {"serial_number": index_no}]})
        
        if res:
            # FIX: file_id प्रयोग गर्ने
            file_id = res.get('file_id')
            file_name = res.get('file_name', res.get('filename', 'document.pdf'))
            
            if file_id:
                bot.send_document(ADMIN_ID, file_id, caption=f"📄 Admin Copy: {file_name}")
            else:
                bot.reply_to(message, "❌ यो फाइलको ID डाटाबेसमा छैन (सायद पुरानो भर्सनबाट अपलोड भएको हो)।")
        else:
            bot.reply_to(message, "❌ फाइल भेटिएन।")
    except Exception as e: 
        log_exception(e)
        bot.reply_to(message, f"त्रुटि भयो: {e}")

@bot.message_handler(commands=['ask_file'])
def ask_from_file(message):
    query = message.text.replace('/ask_file', '').strip()
    if not query:
        return bot.reply_to(message, "कृपया फाइलको बारेमा केही सोध्नुहोस्।")
    
    status_msg = bot.reply_to(message, "🔍 फाइलहरूमा खोज्दै छु...")
    
    try:
        vector = get_embedding(query, task_type="RETRIEVAL_QUERY")
        
        # १. डाटाबेसबाट फाइलहरू तान्ने
        all_pdfs = list(pdf_collection.find({}, {"serial": 1, "file_name": 1, "embedding": 1, "text": 1, "_id": 0}))
        
        if not all_pdfs:
            return ask_general_ai(message, query, status_msg)

        best_doc = None
        best_score = -1

        # २. Similarity Search गर्ने
        for doc in all_pdfs:
            if "embedding" in doc and doc["embedding"]:
                score = cosine_similarity(vector, doc["embedding"])
                if score > best_score:
                    best_score = score
                    best_doc = doc
        
        # ३. स्कोर चेक गर्ने (लुप भन्दा बाहिर)
        # यदि स्कोर ४०% भन्दा कम छ भने सामान्य AI लाई सोध्ने
        if not best_doc or best_score < 0.40: 
            return ask_general_ai(message, query, status_msg)
        
        # ४. डेटा निकाल्दा .get() प्रयोग गर्ने (यसले KeyError रोक्छ)
        context = best_doc.get('text', '')[:3500]
        f_name = best_doc.get('file_name', 'Unknown File')
        f_serial = best_doc.get('serial', 'N/A')

        bot.edit_message_text(f"📄 **{f_name}** मा जवाफ भेटियो, प्रोसेस गर्दैछु...", status_msg.chat.id, status_msg.message_id)
        
        prompt = f"Context from PDF (File: {f_name}):\n{context}\n\nQuestion: {query}\n\nAnswer in Nepali strictly based on the context provided."
        ai_response = call_gemini_smart_improved(prompt)
        
        bot.delete_message(message.chat.id, status_msg.message_id)
        send_long_message(
            message.chat.id,
            f"📄 **PDF #{f_serial} ({f_name}) को आधारमा:**\n\n{ai_response}",
            reply_to_message_id=message.message_id
        )
            
    except Exception as e:
        log_exception(e)
        bot.edit_message_text(f"❌ त्रुटि: {str(e)}", status_msg.chat.id, status_msg.message_id)

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

# ——— BOT & SERVER RUNNER ———

def run_polling():
    """Runs the bot's polling loop in a resilient way."""
    while True:
        try:
            print("🤖 Bot Polling Started...")
            bot.infinity_polling(timeout=20, long_polling_timeout=20, skip_pending=True)
        except Exception as e:
            print(f"💥 Polling Crash: {e}")
            log_exception(e)
            print("Restarting polling in 5 seconds...")
            time.sleep(5)

# Start the bot polling in a background thread.
# This runs when the module is imported by Gunicorn.
print("✅ Starting bot polling in a background thread...")
threading.Thread(target=run_polling, daemon=True).start()

# The if __name__ block is now only for local development.
# Gunicorn will not run this, but it will run the code above.
if __name__ == "__main__":
    # When running locally, Flask's dev server is used.
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Flask dev server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
