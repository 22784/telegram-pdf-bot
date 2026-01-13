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

from flask import Flask
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
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text if len(text.strip()) >= 100 else None
    except Exception as e:
        print(f"PDF पाठ निकाल्दा त्रुटि: {e}")
        log_exception(e)
        return None

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

def call_gemini_smart(prompt, history=None):
    if not genai_client:
        return "❌ Gemini API key missing."

    contents = []
    if history:
        contents.extend(history)
    contents.append({"role": "user", "parts": [prompt]})

    for model_name in MODELS:
        try:
            response = genai_client.models.generate_content(
                model=f"models/{model_name}",
                contents=contents
            )
            if response and response.text:
                return response.text
        except ResourceExhausted:
            return "❌ AI Quota exhausted. Try tomorrow."
        except Exception as e:
            print(f"⚠️ Model failed: {model_name} → {e}")
            time.sleep(1)
            continue
    return "❌ All Gemini models failed."


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
        summary = call_gemini_smart(summary_prompt)
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
    if not query: return bot.reply_to(message, "कृपया फाइलको बारेमा केही सोध्नुहोस्।")
    status_msg = bot.reply_to(message, "🔍 फाइलहरूमा खोज्दै...")
    try:
        vector = get_embedding(query, task_type="RETRIEVAL_QUERY")
        if vector == "QUOTA_EXCEEDED":
            return bot.edit_message_text("❌ AI Quota Error: The daily free limit for asking questions has been reached. Please try again tomorrow.", status_msg.chat.id, status_msg.message_id)
        if not vector:
            return bot.edit_message_text("❌ AI Error: तपाईंको प्रश्नको लागि भेक्टर बनाउन सकिएन। कृपया आफ्नो API कुञ्जीहरू जाँच गर्नुहोस्।", status_msg.chat.id, status_msg.message_id)
        
        # Manual Similarity Search (Option 1 from user)
        all_pdfs = list(pdf_collection.find({}, {"summary": 1, "embedding": 1, "full_text": 1, "_id": 0}))
        
        if not all_pdfs:
            return bot.edit_message_text("❌ कुनै पनि PDF हरू भेटिएनन्। कृपया पहिले PDF अपलोड गर्नुहोस्।", message.chat.id, status_msg.message_id)

        best_doc = None
        best_score = -1

        for doc in all_pdfs:
            # Ensure the document has an embedding
            if "embedding" in doc and doc["embedding"]:
                score = cosine_similarity(vector, doc["embedding"])
                if score > best_score:
                    best_score = score
                    best_doc = doc
        
        if not best_doc or best_score < 0.65:
            return bot.edit_message_text(
                "❌ इस सवाल से रिलेटेड कोई strong content नहीं मिला।",
                message.chat.id, status_msg.message_id
            )

        context = best_doc['full_text'] if 'full_text' in best_doc else best_doc['summary']
        # Apply full text limit to prevent Gemini overload
        context = context[:3000]
        prompt = f"Context from PDF: {context}\n\nUser Question: {query}\n\nAnswer based on context only:"
        
        bot.edit_message_text("✍️ सान्दर्भिक जानकारी भेटियो, जवाफ तयार पार्दै...", status_msg.chat.id, status_msg.message_id)
        ai_response = call_gemini_smart(prompt)

        bot.delete_message(message.chat.id, status_msg.message_id)
        send_long_message(message.chat.id, f"📄 **फाइलको आधारमा जवाफ:**\n\n{ai_response}", reply_to_message_id=message.message_id, parse_mode="Markdown")

    except Exception as e:
        log_exception(e)
        bot.edit_message_text(f"माफ गर्नुहोस्, सोध्दा त्रुटि आयो: {e}", status_msg.chat.id, status_msg.message_id)

@bot.message_handler(commands=['quiz'])
def generate_pdf_quiz(message):
    args = message.text.split()
    if len(args) < 2: return bot.reply_to(message, "कृपया PDF नम्बर दिनुहोस्। उदाहरण: `/quiz 1`")
    try:
        pdf_id = int(args[1])
        res = pdf_collection.find_one({"serial_number": pdf_id})
        if not res: return bot.reply_to(message, "यो नम्बरको फाइल भेटिएन।")

        status_msg = bot.reply_to(message, f"⏳ {res['file_name']} बाट क्विज तयार गर्दैछु...")
        prompt = f"Create 1 MCQ quiz in JSON based on this: {res['summary']}. Return only JSON."
        ai_res = call_gemini_smart(prompt)
        data = json.loads(clean_json(ai_res))
        bot.send_poll(message.chat.id, question=data['question'][:255], options=[o[:100] for o in data['options']], correct_option_id=data['correct_option_id'], type='quiz', explanation=data.get('explanation', '')[:200])
        bot.delete_message(message.chat.id, status_msg.message_id)
    except Exception as e: 
        log_exception(e)
        bot.edit_message_text(f"त्रुटि: {e}", message.chat.id, status_msg.message_id if 'status_msg' in locals() else message.message_id)

@bot.message_handler(func=lambda message: not message.text.startswith('/'))
def handle_chat(message):
    if message.chat.type == 'private' or (message.reply_to_message and message.reply_to_message.from_user.id == bot.get_me().id):
        history = get_chat_history(message.from_user.id)
        bot.send_chat_action(message.chat.id, 'typing')
        res = call_gemini_smart(message.text, history)
        save_chat_history(message.from_user.id, message.text, res)
        # FIX: Use the correct variable 'res' instead of 'bot_response'
        send_long_message(message.chat.id, res, reply_to_message_id=message.message_id)

# ——— BOT START (Render Safe) ———
def run_bot():
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)

if __name__ == "__main__":
    print("Bot started...")
    threading.Thread(target=run_bot).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
