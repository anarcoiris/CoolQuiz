# telegram_quiz_bot_sqlite.py
"""
Bot Telegram - Cuestionario con SQLite (sencillo pero funcional)
Dependencias: python-telegram-bot, sqlite3 (builtin)
Uso:
  export TELEGRAM_TOKEN="..."    (o pon TOKEN en el script)
  python telegram_quiz_bot_sqlite.py
"""
import logging
import sqlite3
import json
import random
import os
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler

# CONFIG
TOKEN = os.environ.get("TELEGRAM_TOKEN") or "TU_TOKEN_AQUI"
QUESTIONS_FILE = "questions_for_bot.json"  # generar desde tu .txt: ver nota abajo
DB_FILE = "telegram_quiz.db"

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# DB helpers
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        chat_id INTEGER PRIMARY KEY,
        username TEXT,
        started_at TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        timestamp TEXT,
        question_idx INTEGER,
        pregunta TEXT,
        eleccion TEXT,
        correctas TEXT,
        ok INTEGER
    )""")
    conn.commit()
    return conn

conn = init_db()

# Load questions JSON (convert your .txt to JSON using the Streamlit export or a simple script)
if not os.path.exists(QUESTIONS_FILE):
    print("Genera questions_for_bot.json a partir de tu .txt. (ver instrucciones en el script de Streamlit).")
    questions = []
else:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

# Simple in-memory user state (for demo). For production, store in DB or redis.
user_state = {}

# Conversation states
ASKING = range(1)

def start(update: Update, context: CallbackContext):
    cid = update.effective_chat.id
    username = update.effective_user.username or ""
    conn.execute("INSERT OR IGNORE INTO users (chat_id, username, started_at) VALUES (?, ?, datetime('now'))", (cid, username))
    conn.commit()
    update.message.reply_text("¡Hola! Bienvenido al bot de examen. Envía /quiz para empezar.")

def quiz(update: Update, context: CallbackContext):
    cid = update.effective_chat.id
    if not questions:
        update.message.reply_text("No hay preguntas cargadas en el servidor.")
        return
    q = random.choice(questions)
    user_state[cid] = {"current": q}
    # If multiple corrects, allow multiple selections: we'll instruct user to send comma-separated answers.
    if len(q.get("correctas", [])) > 1:
        text = f"{q['pregunta']}\n\nOpciones:\n"
        for i,opt in enumerate(q['opciones']):
            text += f"{i+1}) {opt}\n"
        text += "\nResponde con los números separados por comas (ej: 1,3)"
        update.message.reply_text(text)
    else:
        # single choice: send keyboard
        keyboard = [[o] for o in q['opciones']]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        update.message.reply_text(q['pregunta'], reply_markup=reply_markup)
    return ASKING

def answer(update: Update, context: CallbackContext):
    cid = update.effective_chat.id
    text = update.message.text.strip()
    state = user_state.get(cid)
    if not state:
        update.message.reply_text("Usa /quiz para pedir una pregunta.")
        return ConversationHandler.END
    q = state["current"]
    correctas = q.get("correctas", [])
    # parse response
    if len(correctas) > 1:
        # expect numbers
        try:
            nums = [int(x.strip())-1 for x in text.split(",") if x.strip()]
            elec = [q["opciones"][i] for i in nums if 0 <= i < len(q["opciones"])]
        except Exception:
            update.message.reply_text("Formato inválido. Envía números separados por comas.")
            return ASKING
        ok = set(elec) == set(correctas)
    else:
        elec = text
        ok = (elec == correctas[0]) if correctas else None

    # save to DB
    conn.execute("INSERT INTO results (chat_id,timestamp,question_idx,pregunta,eleccion,correctas,ok) VALUES (?, datetime('now'), ?, ?, ?, ?, ?)",
                 (cid, questions.index(q), q['pregunta'], json.dumps(elec, ensure_ascii=False), json.dumps(correctas, ensure_ascii=False), 1 if ok else 0))
    conn.commit()

    if ok:
        update.message.reply_text("✅ Correcto!", reply_markup=ReplyKeyboardRemove())
    else:
        update.message.reply_text("❌ Incorrecto." + (f" Correctas: {', '.join(correctas)}" if correctas else ""), reply_markup=ReplyKeyboardRemove())

    # ask if wants another
    update.message.reply_text("¿Otra pregunta? Envía /quiz o /score para ver tu puntuación.")
    return ConversationHandler.END

def score(update: Update, context: CallbackContext):
    cid = update.effective_chat.id
    c = conn.cursor()
    c.execute("SELECT COUNT(*), SUM(ok) FROM results WHERE chat_id=?", (cid,))
    total, corrects = c.fetchone()
    if total is None:
        total = 0
        corrects = 0
    update.message.reply_text(f"Has contestado {total} preguntas. Correctas: {corrects or 0}")

def main():
    updater = Updater(TOKEN)
    dp = updater.dispatcher
    conv = ConversationHandler(
        entry_points=[CommandHandler('quiz', quiz)],
        states={ASKING: [MessageHandler(Filters.text & ~Filters.command, answer)]},
        fallbacks=[]
    )
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(conv)
    dp.add_handler(CommandHandler('score', score))
    dp.add_handler(MessageHandler(Filters.command, lambda u,c: u.message.reply_text("Comando desconocido.")))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
