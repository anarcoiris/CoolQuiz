import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import csv
import sqlite3
import random
import hashlib
from datetime import datetime, timezone
try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

random.seed(42)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PROFILES_FILE = os.path.join(BASE_DIR, "profiles.json")
RESULTS_DIR = os.path.join(BASE_DIR, "user_results")
QUESTIONS_FOR_BOT = os.path.join(BASE_DIR, "questions_for_bot.json")
DB_FILE = os.path.join(BASE_DIR, "quiz_app.db")
os.makedirs(RESULTS_DIR, exist_ok=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def parse_questions(text):
    text = text.replace("\r\n", "\n")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    out = []
    for b in blocks:
        lines = [l.rstrip() for l in b.splitlines() if l.strip() != ""]
        if len(lines) < 2:
            continue
        if (lines[0].endswith("?") or lines[0].endswith("¿")) and len(lines) >= 2:
            tema = "General"
            pregunta = lines[0]
            opts_raw = lines[1:]
        else:
            tema = lines[0]
            pregunta = lines[1] if len(lines) > 1 else ""
            opts_raw = lines[2:] if len(lines) > 2 else []
            if not opts_raw:
                tema = "General"
                pregunta = lines[0]
                opts_raw = lines[1:]
        opts = []
        corrects = []
        for o in opts_raw:
            s = o.strip()
            if not s:
                continue
            if s.startswith("*"):
                val = s[1:].strip()
                opts.append(val)
                corrects.append(val)
            else:
                opts.append(s)
        out.append({"tema": tema, "pregunta": pregunta, "opciones": opts, "correctas": corrects})
    return out

def ensure_db(path=DB_FILE):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        timestamp TEXT,
        question_idx INTEGER,
        pregunta TEXT,
        eleccion TEXT,
        correctas TEXT,
        ok INTEGER
    )""")
    conn.commit()
    return conn

conn = ensure_db()

def save_result_db(r):
    try:
        c = conn.cursor()
        c.execute("""INSERT INTO results (username,timestamp,question_idx,pregunta,eleccion,correctas,ok)
            VALUES (?, ?, ?, ?, ?, ?, ?)""", (
            r.get("username"),
            r.get("timestamp"),
            r.get("question_idx"),
            r.get("pregunta"),
            json.dumps(r.get("eleccion"), ensure_ascii=False),
            json.dumps(r.get("correctas"), ensure_ascii=False),
            1 if r.get("ok") else 0 if r.get("ok") is not None else None
        ))
        conn.commit()
    except Exception:
        pass

st.set_page_config(page_title="Quiz Suite — Advanced", layout="wide")
if "auth" not in st.session_state:
    st.session_state["auth"] = {"logged_in": False, "username": None}
if "ui_mode" not in st.session_state:
    st.session_state["ui_mode"] = "main"
users = load_json(USERS_FILE)
profiles = load_json(PROFILES_FILE)

def register_user(username, password):
    if username in users:
        return False, "exists"
    users[username] = {"password": sha256(password), "created_at": now_iso()}
    save_json(USERS_FILE, users)
    profiles.setdefault(username, {"preferences": {}, "progress": {}})
    save_json(PROFILES_FILE, profiles)
    return True, "ok"

def login_user(username, password):
    if username not in users:
        return False, "no_user"
    if users[username]["password"] != sha256(password):
        return False, "bad_pass"
    st.session_state["auth"]["logged_in"] = True
    st.session_state["auth"]["username"] = username
    return True, "ok"

def logout_user():
    st.session_state["auth"] = {"logged_in": False, "username": None}
    st.session_state["ui_mode"] = "main"

def user_dir(username):
    d = os.path.join(RESULTS_DIR, username)
    os.makedirs(d, exist_ok=True)
    return d

def save_user_results(username, results):
    d = user_dir(username)
    fname = os.path.join(d, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_json(fname, results)
    return fname

def export_questions_json(qs):
    save_json(QUESTIONS_FOR_BOT, qs)
    return QUESTIONS_FOR_BOT

st.title("Quiz Suite — Advanced Demo")
col1, col2 = st.columns([2,1])
with col2:
    if st.session_state["auth"]["logged_in"]:
        st.write("Usuario:", st.session_state["auth"]["username"])
        if st.button("Cerrar sesión"):
            logout_user()
            st.experimental_rerun()
    else:
        st.write("No autenticado")
with col1:
    st.write("Interfaz avanzada, determinista y conservadora")

if not st.session_state["auth"]["logged_in"]:
    st.sidebar.header("Acceso")
    action = st.sidebar.radio("Selecciona", ["Iniciar sesión", "Registrar"])
    uname = st.sidebar.text_input("Usuario", key="login_user")
    pwd = st.sidebar.text_input("Contraseña", type="password", key="login_pass")
    if action == "Registrar":
        if st.sidebar.button("Registrar"):
            ok, msg = register_user(uname, pwd)
            if ok:
                st.sidebar.success("Registrado. Inicia sesión.")
            else:
                st.sidebar.error("Usuario ya existe")
    else:
        if st.sidebar.button("Entrar"):
            ok, msg = login_user(uname, pwd)
            if ok:
                st.sidebar.success("Autenticado")
                st.experimental_rerun()
            else:
                st.sidebar.error("Usuario o contraseña incorrectos")
    st.stop()

username = st.session_state["auth"]["username"]
user_profile = profiles.get(username, {"preferences": {}, "progress": {}})
st.sidebar.header("Usuario")
st.sidebar.write(username)
st.sidebar.markdown("Preferencias")
pref_pool = st.sidebar.number_input("Pool por defecto (0=todas)", min_value=0, value=user_profile["preferences"].get("pool_default", 0))
pref_shuffle = st.sidebar.checkbox("Barajar preguntas por usuario", value=user_profile["preferences"].get("shuffle", True))
pref_auto_check = st.sidebar.checkbox("Comprobar al elegir", value=user_profile["preferences"].get("auto_check", True))
if st.sidebar.button("Guardar preferencias"):
    profiles.setdefault(username, {})["preferences"] = {"pool_default": pref_pool, "shuffle": pref_shuffle, "auto_check": pref_auto_check}
    save_json(PROFILES_FILE, profiles)
    st.sidebar.success("Preferencias guardadas")

st.sidebar.markdown("---")
if st.sidebar.button("Generar Dockerfile + requirements"):
    dockerfile = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_quiz_complete.py", "--server.port=8501", "--server.address=0.0.0.0"]"""
    reqs = "streamlit\npandas\nnumpy\nscikit-learn\npillow\npydeck\nopenai"
    bdf = dockerfile.encode("utf-8")
    st.download_button("Descargar Dockerfile", data=bdf, file_name="Dockerfile", mime="text/plain")
    st.download_button("Descargar requirements.txt", data=reqs, file_name="requirements.txt", mime="text/plain")

uploaded = st.file_uploader("Sube .txt de preguntas o deja vacío para ejemplo", type=["txt"])
if uploaded:
    content = uploaded.read().decode("utf-8")
    questions_all = parse_questions(content)
else:
    sample_text = """Vulnerabilidad
¿Qué es una vulnerabilidad?
Una amenaza externa
*Debilidad en un sistema que puede ser explotada
Una copia de seguridad
Un firewall

Amenaza
¿Qué es una amenaza en ciberseguridad?
Un dato cifrado
*Circunstancia que puede afectar negativamente a un activo
Un error corregido
Un antivirus

Riesgo
¿Cómo se define el riesgo en ciberseguridad?
*Probabilidad de que una amenaza explote una vulnerabilidad causando impacto
Cantidad de usuarios conectados
Un antivirus no actualizado
Nivel de permisos en Windows
"""
    questions_all = parse_questions(sample_text)

seed = sha256(username)[:8]
seed_int = int(seed, 16) % (2**31)
random.seed(seed_int)

if pref_shuffle:
    random.shuffle(questions_all)

pool_size = pref_pool if pref_pool and pref_pool <= len(questions_all) else len(questions_all)
questions = questions_all[:pool_size]

st.sidebar.markdown("---")
mode = st.sidebar.selectbox("Modo", ["Práctica libre", "Story Mode (curso)", "Prompt Playground", "Administración"])
st.sidebar.markdown(f"Preguntas cargadas: {len(questions_all)} | Usando: {len(questions)}")

if mode == "Práctica libre":
    st.header("Modo: Práctica libre")
    if "answers" not in st.session_state:
        st.session_state["answers"] = {i: None for i in range(len(questions))}
    for idx, q in enumerate(questions):
        st.subheader(f"{idx+1}. {q.get('pregunta')}")
        opts = q.get("opciones", [])
        corrects = q.get("correctas", [])
        if len(corrects) > 1:
            selected = []
            for i,opt in enumerate(opts):
                key = f"{username}_q_{idx}_opt_{i}"
                if key not in st.session_state:
                    st.session_state[key] = False
                st.session_state[key] = st.checkbox(opt, key=key)
                if st.session_state[key]:
                    selected.append(opt)
            st.session_state["answers"][idx] = selected
            if pref_auto_check and selected:
                ok = set(selected) == set(corrects)
                if ok:
                    st.success("✅ Correcto")
                else:
                    st.error("❌ Incorrecto")
        else:
            placeholder = "-- Selecciona --"
            options_for_select = [placeholder] + opts
            key = f"{username}_q_{idx}_sel"
            prev = st.session_state["answers"].get(idx)
            default = prev if prev in opts else placeholder
            choice = st.selectbox("", options_for_select, index=options_for_select.index(default), key=key)
            if choice == placeholder:
                st.session_state["answers"][idx] = None
            else:
                st.session_state["answers"][idx] = choice
            if pref_auto_check and st.session_state["answers"][idx] is not None:
                elec = st.session_state["answers"][idx]
                ok = (elec == corrects[0]) if corrects else None
                if ok is True:
                    st.success("✅ Correcto")
                elif ok is False:
                    st.error("❌ Incorrecto")
        if st.button(f"Comprobar y guardar pregunta {idx+1}", key=f"save_{idx}"):
            elec = st.session_state["answers"].get(idx)
            corrects_local = q.get("correctas", [])
            if len(corrects_local) > 1:
                ok = set(elec or []) == set(corrects_local)
            else:
                ok = (elec == corrects_local[0]) if corrects_local and elec is not None else None
            r = {"username": username, "timestamp": now_iso(), "question_idx": idx, "pregunta": q.get("pregunta"), "eleccion": elec, "correctas": corrects_local, "ok": ok}
            save_result_db(r)
            dpath = save_user_results(username, [r])
            st.success("Guardado")
    if st.button("Exportar preguntas para bot (questions_for_bot.json)"):
        pth = export_questions_json(questions)
        with open(pth, "rb") as f:
            st.download_button("Descargar JSON para bot", data=f.read(), file_name="questions_for_bot.json", mime="application/json")
elif mode == "Story Mode (curso)":
    st.header("Modo: Story Mode")
    prog = profiles.setdefault(username, {}).get("progress", {})
    if "story_idx" not in prog:
        prog["story_idx"] = 0
    idx = prog["story_idx"]
    if idx >= len(questions):
        st.success("Has completado el curso. Reinicia para volver a empezar.")
        if st.button("Reiniciar curso"):
            profiles[username]["progress"]["story_idx"] = 0
            save_json(PROFILES_FILE, profiles)
            st.experimental_rerun()
    else:
        q = questions[idx]
        st.subheader(f"Lección {idx+1}: {q.get('tema')}")
        st.write(q.get("pregunta"))
        opts = q.get("opciones", [])
        corrects = q.get("correctas", [])
        if len(corrects) > 1:
            chosen = []
            for i,opt in enumerate(opts):
                key = f"{username}_story_{idx}_opt_{i}"
                if key not in st.session_state:
                    st.session_state[key] = False
                st.session_state[key] = st.checkbox(opt, key=key)
                if st.session_state[key]:
                    chosen.append(opt)
            if st.button("Comprobar lección"):
                ok = set(chosen) == set(corrects)
                r = {"username": username, "timestamp": now_iso(), "question_idx": idx, "pregunta": q.get("pregunta"), "eleccion": chosen, "correctas": corrects, "ok": ok}
                save_result_db(r)
                save_user_results(username, [r])
                if ok:
                    st.success("✅ Lección correcta. Avanzando...")
                    profiles[username].setdefault("progress", {})["story_idx"] = idx + 1
                    save_json(PROFILES_FILE, profiles)
                    st.experimental_rerun()
                else:
                    st.error("❌ Revisa la respuesta e inténtalo de nuevo.")
        else:
            placeholder = "-- Selecciona --"
            options_for_select = [placeholder] + opts
            key = f"{username}_story_{idx}_sel"
            prev = st.session_state.get(key, placeholder)
            choice = st.selectbox("Selecciona", options_for_select, index=options_for_select.index(prev), key=key)
            if st.button("Comprobar lección"):
                if choice == placeholder:
                    st.warning("Selecciona una opción")
                else:
                    ok = (choice == corrects[0]) if corrects else None
                    r = {"username": username, "timestamp": now_iso(), "question_idx": idx, "pregunta": q.get("pregunta"), "eleccion": choice, "correctas": corrects, "ok": ok}
                    save_result_db(r)
                    save_user_results(username, [r])
                    if ok:
                        st.success("✅ Lección correcta. Avanzando...")
                        profiles[username].setdefault("progress", {})["story_idx"] = idx + 1
                        save_json(PROFILES_FILE, profiles)
                        st.experimental_rerun()
                    else:
                        st.error("❌ Incorrecto. Vuelve a intentarlo.")
elif mode == "Prompt Playground":
    st.header("Prompt Playground")
    openai_key_local = st.text_input("OpenAI API Key (opcional)", type="password", key="openai_local")
    prompt = st.text_area("Prompt / pregunta", value="Explica brevemente por qué la respuesta correcta es la marcada para:\n\n" + (questions[0]["pregunta"] if questions else ""), height=200)
    model = st.selectbox("Modelo", ["gpt-4o-mini","gpt-4o","gpt-4"], index=0)
    if st.button("Enviar a OpenAI"):
        key_to_use = openai_key_local if openai_key_local else None
        if not key_to_use:
            st.error("Proporciona una OpenAI API key en el campo superior para usar esta función.")
        else:
            try:
                openai.api_key = key_to_use
                resp = openai.ChatCompletion.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=400)
                out = resp["choices"][0]["message"]["content"]
                st.text_area("Respuesta", value=out, height=300)
            except Exception as e:
                st.error(f"Error llamando a OpenAI: {e}")
elif mode == "Administración":
    st.header("Administración")
    df = pd.read_sql_query("SELECT username, timestamp, question_idx, pregunta, eleccion, correctas, ok FROM results ORDER BY timestamp DESC LIMIT 500", conn)
    st.dataframe(df)
    if st.button("Vaciar resultados (solo demo)"):
        try:
            c = conn.cursor()
            c.execute("DELETE FROM results")
            conn.commit()
            st.success("Resultados eliminados")
        except Exception:
            st.error("No se pudo eliminar")
st.markdown("---")
st.sidebar.markdown("Quick links")
if st.sidebar.button("Descargar todos mis resultados (CSV)"):
    d = user_dir(username)
    rows = []
    for f in os.listdir(d):
        if f.endswith(".json"):
            j = load_json(os.path.join(d, f))
            if isinstance(j, list):
                rows.extend(j)
    if rows:
        si = io.StringIO()
        w = csv.writer(si)
        w.writerow(["timestamp","question_idx","pregunta","eleccion","correctas","ok"])
        for r in rows:
            w.writerow([r.get("timestamp"), r.get("question_idx"), r.get("pregunta"), json.dumps(r.get("eleccion"), ensure_ascii=False), json.dumps(r.get("correctas"), ensure_ascii=False), r.get("ok")])
        st.download_button("Descargar CSV", data=si.getvalue().encode("utf-8"), file_name=f"{username}_results_all.csv", mime="text/csv")
    else:
        st.warning("No hay resultados guardados")
