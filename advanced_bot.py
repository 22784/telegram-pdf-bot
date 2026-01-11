import telebot
from telebot.types import Message
from pymongo import MongoClient
import google.generativeai as genai
import fitz  # PyMuPDF
import os
import json
import re
import time
from flask import Flask
from threading import Thread

# ——— कन्फिगरेसन (Render Environment Variables बाट पढ्ने) ———
BOT_TOKEN = os.getenv("BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
# একাধিক API कुञ्जीहरूलाई अल्पविराम (comma) द्वारा छुट्याएर एउटै लाइनमा राख्नुहोस्
API_KEYS = [key.strip() for key in os.getenv("API_KEYS", "").split(',') if key.strip()]
ADMIN_ID = int(os.getenv("ADMIN_ID", 0))
BACKUP_CHANNEL_ID = int(os.getenv("BACKUP_CHANNEL_ID", 0))

DOWNLOAD_PATH = "temp_pdfs"
if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH)

# ——— INITIALIZATION ———
bot = telebot.TeleBot(BOT_TOKEN)
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client['TelegramBotDB']
pdf_collection = db['PDF_Store']
notes_collection = db['Notes']
counters_collection = db['Counters']
history_collection = db['Chat_History']

# genai.configure अब call_gemini_smart भित्र ह्यान्डल गरिन्छ
MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]

# ——— RENDER KEEP-ALIVE SERVER ———
app = Flask('')
@app.route('/')
def home():
    return "I am alive!"

def run_server():
  app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run_server)
    t.start()

# ——— CORE HELPER FUNCTIONS ———
def clean_json(raw_text):
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    return match.group(0) if match else raw_text

def get_embedding(text):
    try:
        return genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")['embedding']
    except Exception as e:
        print(f"इम्बेडिङ बनाउँदा त्रुटि: {e}")
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
        return None

def extract_vision_text(file_path):
    img_path = f"temp_scan_{os.path.basename(file_path)}.png"
    try:
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        pix.save(img_path)
        doc.close()
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        img_file = genai.upload_file(img_path)
        response = model.generate_content(["Extract all text from this document page:", img_file])
        return response.text
    except Exception as e:
        print(f"Vision OCR असफल भयो: {e}")
        return None
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

def send_long_message(chat_id, text, reply_to_message_id=None, parse_mode="Markdown"):
    """टेलीग्रामको क्यारेक्टर लिमिटलाई ह्यान्डल गर्दै लामो सन्देशहरू पठाउँछ।"""
    if not text:
        return
    if len(text) <= 4000:
        bot.send_message(chat_id, text, reply_to_message_id=reply_to_message_id, parse_mode=parse_mode)
    else:
        parts = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for part in parts:
            bot.send_message(chat_id, part, reply_to_message_id=reply_to_message_id, parse_mode=parse_mode)
            time.sleep(1) # Spam filtering से बचने के लिए

def get_chat_history(user_id):
    history = history_collection.find({"user_id": user_id}).sort("_id", -1).limit(10)
    formatted_history = []
    for msg in reversed(list(history)):
        formatted_history.append({"role": "user", "parts": [msg['user_msg']]})
        formatted_history.append({"role": "model", "parts": [msg['bot_res']]})
    return formatted_history
def save_chat_history(user_id, user_msg, bot_res):
    history_collection.insert_one({"user_id": user_id, "user_msg": user_msg, "bot_res": bot_res})
def call_gemini_smart(prompt, history=[]):
    """AI लाई कल गर्छ, फलब्याक तर्कका साथ (कुञ्जी र मोडल दुवैमा)।"""
    if not API_KEYS:
        print("API कुञ्जीहरू कन्फिगर गरिएका छैनन्!")
        return "SERVICE_ERROR: API कुञ्जीहरू कन्फिगर गरिएका छैनन्।"

    for key in API_KEYS: # प्रत्येक कुञ्जी प्रयोग गरी हेर्नुहोस्
        genai.configure(api_key=key) 
        for model_name in MODELS: # त्यसपछि प्रत्येक मोडल प्रयोग गरी हेर्नुहोस्
            try:
                model = genai.GenerativeModel(model_name)
                if history:
                    chat = model.start_chat(history=history)
                    response = chat.send_message(prompt)
                else:
                    response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"कुञ्जी असफल भयो: {key[:5]}... मोडल: {model_name} त्रुटि: {e}. अर्को प्रयास गर्दैछु...")
                time.sleep(1) # Rate limit बाट बच्न थोरै पर्खनुहोस्
                continue # अर्को मोडल वा कुञ्जी प्रयास गर्नुहोस्
    return "भाई, अहिले सबै AI कुञ्जीहरू र मोडलहरू व्यस्त छन्। केही बेर पछि प्रयास गर्नुहोस्।"


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
            pdf_type = "scanned"
        
        if not text: return bot.edit_message_text("❌ माफ गर्नुहोस्, यो PDF बाट कुनै पाठ निकाल्न सकिएन।", status_msg.chat.id, status_msg.message_id)

        summary_prompt = f"यो सामग्रीलाई खोज अनुक्रमणिकाको लागि २ वाक्यमा सारांश गर्नुहोस्: {text[:2000]}"
        summary = call_gemini_smart(summary_prompt)
        vector = get_embedding(summary)
        if not vector: return bot.edit_message_text("❌ AI Error: Vector generation failed. Try again.", status_msg.chat.id, status_msg.message_id)
        
        serial_no = get_next_serial_number('pdf_id')
        backup_msg = bot.forward_message(BACKUP_CHANNEL_ID, message.chat.id, message.message_id)
        
        pdf_collection.insert_one({
            "serial_number": serial_no, "file_name": message.document.file_name, "file_id": message.document.file_id,
            "summary": summary, "embedding": vector, "type": pdf_type, "backup_msg_id": backup_msg.message_id,
            "uploader_id": message.from_user.id
        })

        try: bot.delete_message(message.chat.id, message.message_id)
        except: pass
        bot.edit_message_text(f"✅ PDF #{serial_no} ({pdf_type}) '{message.document.file_name}' सफलतापूर्वक प्रशोधन र सुरक्षित गरियो।", status_msg.chat.id, status_msg.message_id)

    except Exception as e:
        bot.edit_message_text(f"माफ गर्नुहोस्, फाइल प्रशोधन गर्दा त्रुटि आयो: {e}", status_msg.chat.id, status_msg.message_id)
        print(f"Universal handler error: {e}")
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
    except Exception as e: bot.reply_to(message, f"त्रुटि भयो: {e}")

@bot.message_handler(commands=['ask_file'])
def ask_from_file(message):
    query = message.text.replace('/ask_file', '').strip()
    if not query: return bot.reply_to(message, "कृपया फाइलको बारेमा केही सोध्नुहोस्।")
    status_msg = bot.reply_to(message, "🔍 फाइलहरूमा खोज्दै...")
    try:
        vector = get_embedding(query)
        results = list(pdf_collection.aggregate([{"$vectorSearch": {"index": "vector_index", "path": "embedding", "queryVector": vector, "numCandidates": 10, "limit": 1}}]))
        if not results: return bot.edit_message_text("❌ सम्बन्धित जानकारी भेटिएन।", message.chat.id, status_msg.message_id)
        
        context = results[0]['summary']
        prompt = f"Context from PDF: {context}\n\nUser Question: {query}\n\nAnswer based on context only:"
        ai_response = call_gemini_smart(prompt)
        bot.delete_message(message.chat.id, status_msg.message_id) # Delete status message
        send_long_message(message.chat.id, f"📄 **फाइलको आधारमा जवाफ:**\n\n{ai_response}", reply_to_message_id=message.message_id, parse_mode="Markdown")
    except Exception as e: bot.edit_message_text(f"त्रुटि: {e}", message.chat.id, status_msg.message_id)

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
    except Exception as e: bot.edit_message_text(f"त्रुटि: {e}", message.chat.id, status_msg.message_id if 'status_msg' in locals() else message.message_id)

@bot.message_handler(func=lambda message: not message.text.startswith('/'))
def handle_chat(message):
    if message.chat.type == 'private' or (message.reply_to_message and message.reply_to_message.from_user.id == bot.get_me().id):
        history = get_chat_history(message.from_user.id)
        bot.send_chat_action(message.chat.id, 'typing')
        res = call_gemini_smart(message.text, history)
        save_chat_history(message.from_user.id, message.text, res)
        send_long_message(message.chat.id, bot_response, reply_to_message_id=message.message_id)

# --- BOT START ---
if __name__ == "__main__":
    keep_alive()
    print("बोट लाइभ भयो...")
    bot.infinity_polling(skip_pending=True)
