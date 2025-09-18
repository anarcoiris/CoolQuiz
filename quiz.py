# streamlit_quiz_complete.py
"""
Streamlit Cuestionario — Versión completa y revisada
Funciones incluidas:
- Parseo de .txt con formato (tema opcional, pregunta, opciones; * marca correctas)
- Single-choice: selectbox con placeholder "-- Selecciona --" (no preselección)
- Multi-choice: checkboxes (cuando hay >1 respuestas correctas)
- Pool de preguntas, barajar preguntas y opciones
- Comprobación inmediata (opcional) o manual
- Guardado automático por comprobación y export CSV/JSON
- Export JSON para bot de Telegram (questions_for_bot.json)
- Guardado opcional en SQLite
- Integración opcional con OpenAI (clave en sidebar)
- Perfil: guardar/cargar preferencias
- Manejo seguro de session_state y de Streamlit API
"""
import streamlit as st
import random
import json
import os
import io
import csv
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any

# -------------------- Config --------------------
PROFILES_FILE = "quiz_profiles.json"
RESULTS_DIR = "quiz_results"
QUESTIONS_FOR_BOT = "questions_for_bot.json"
DB_FILE = "quiz_data.db"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------- Helpers: time / IO --------------------
def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def save_json_atomic(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def results_to_csv_bytes(results: List[Dict[str,Any]]) -> bytes:
    si = io.StringIO()
    w = csv.writer(si)
    w.writerow(["timestamp","profile","question_idx","tema","pregunta","eleccion","correctas","ok"])
    for r in results:
        w.writerow([
            r.get("timestamp"),
            r.get("profile"),
            r.get("question_idx"),
            r.get("tema"),
            r.get("pregunta"),
            json.dumps(r.get("eleccion"), ensure_ascii=False),
            json.dumps(r.get("correctas"), ensure_ascii=False),
            r.get("ok"),
        ])
    return si.getvalue().encode("utf-8")

# -------------------- Parse questions --------------------
def parse_questions(file_content: str) -> List[Dict[str,Any]]:
    """
    Parse blocks separated by blank lines.
    Block forms:
      Tema
      Pregunta
      Opcion
      *Opcion correcta
    or:
      Pregunta?
      Opcion
      *Opcion correcta
    Returns list of dicts: {tema, pregunta, opciones:list[str], correctas:list[str]}
    """
    text = file_content.replace("\r\n", "\n")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    out = []
    for b in blocks:
        lines = [l.rstrip() for l in b.splitlines() if l.strip() != ""]
        if len(lines) < 2:
            continue
        # Detect if first line is question (ends with ?/¿) or topic
        if (lines[0].endswith("?") or lines[0].endswith("¿")) and len(lines) >= 2:
            tema = "General"
            pregunta = lines[0]
            opts_raw = lines[1:]
        else:
            # treat first line as topic
            tema = lines[0]
            pregunta = lines[1] if len(lines) > 1 else ""
            opts_raw = lines[2:] if len(lines) > 2 else []
            if not opts_raw:
                # fallback: maybe there's no topic, first line was question
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
        out.append({
            "tema": tema,
            "pregunta": pregunta,
            "opciones": opts,
            "correctas": corrects
        })
    return out

# -------------------- Profiles --------------------
def load_profiles() -> Dict[str,Any]:
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_profiles(profiles: Dict[str,Any]):
    save_json_atomic(PROFILES_FILE, profiles)

# -------------------- SQLite backup --------------------
def get_sqlite_conn(dbfile=DB_FILE):
    try:
        conn = sqlite3.connect(dbfile, check_same_thread=False)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            profile TEXT,
            question_idx INTEGER,
            tema TEXT,
            pregunta TEXT,
            eleccion TEXT,
            correctas TEXT,
            ok INTEGER
        )""")
        conn.commit()
        return conn
    except Exception:
        return None

def save_result_to_db(conn, r: Dict[str,Any]):
    if conn is None:
        return
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO results (timestamp,profile,question_idx,tema,pregunta,eleccion,correctas,ok)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["timestamp"],
            r["profile"],
            r["question_idx"],
            r["tema"],
            r["pregunta"],
            json.dumps(r["eleccion"], ensure_ascii=False),
            json.dumps(r["correctas"], ensure_ascii=False),
            1 if r["ok"] else 0 if r["ok"] is not None else None
        ))
        conn.commit()
    except Exception:
        pass

# -------------------- Session state defaults --------------------
def ensure_session_defaults():
    defaults = {
        "quiz_questions": None,
        "answers": {},
        "results": [],
        "started_at": None,
        "quiz_generated": False,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# -------------------- App --------------------
st.set_page_config(page_title="Cuestionario completo", layout="wide")
ensure_session_defaults()

st.title("📘 Cuestionario — versión completa y revisada")
st.markdown("Sube un `.txt` (usa `*` delante de la(s) opción(es) correcta(s)).")

# Sidebar: profiles and options
st.sidebar.header("Configuración")
profiles = load_profiles()
profile_list = ["--nuevo--"] + list(profiles.keys())
sel_profile = st.sidebar.selectbox("Perfil", profile_list)

if sel_profile == "--nuevo--":
    profile_name = st.sidebar.text_input("Nombre de perfil", value=st.session_state.get("temp_profile_name",""))
    st.session_state["temp_profile_name"] = profile_name
else:
    profile_name = sel_profile

pool_default = st.sidebar.number_input("Tamaño pool (0 = todas)", min_value=0, value=0)
shuffle_q = st.sidebar.checkbox("Barajar preguntas", value=True)
shuffle_opts = st.sidebar.checkbox("Barajar opciones", value=True)
auto_check = st.sidebar.checkbox("Comprobar al elegir (feedback inmediato)", value=True)
auto_save_on_check = st.sidebar.checkbox("Guardar automáticamente al comprobar", value=True)
use_sqlite = st.sidebar.checkbox("Guardar copia en SQLite", value=False)
openai_key = st.sidebar.text_input("OpenAI API key (opcional)", type="password")

if st.sidebar.button("Guardar perfil"):
    if not profile_name:
        st.sidebar.error("Introduce un nombre de perfil.")
    else:
        profiles[profile_name] = {
            "pool_default": pool_default,
            "shuffle_q": shuffle_q,
            "shuffle_opts": shuffle_opts,
            "auto_check": auto_check,
            "auto_save_on_check": auto_save_on_check,
            "use_sqlite": use_sqlite
        }
        save_profiles(profiles)
        st.sidebar.success(f"Perfil '{profile_name}' guardado.")

if sel_profile != "--nuevo--" and st.sidebar.button("Cargar perfil"):
    p = profiles.get(sel_profile, {})
    if p:
        # Apply by updating session (widgets cannot be programmatically changed easily)
        st.session_state["loaded_profile"] = p
        st.sidebar.info("Perfil cargado: revisa las opciones en la sidebar.")
    else:
        st.sidebar.error("Perfil no encontrado.")

# File upload
uploaded = st.file_uploader("Carga tu archivo .txt", type=["txt"])
if not uploaded:
    st.info("Sube un archivo .txt con las preguntas para continuar.")
    st.stop()

content = uploaded.read().decode("utf-8")
questions_all = parse_questions(content)
if not questions_all:
    st.error("No se detectaron preguntas. Revisa el formato.")
    st.stop()

st.markdown(f"**Preguntas detectadas:** {len(questions_all)}")
n_total = len(questions_all)
pool_size = st.number_input("Pool a usar (0 = todas)", min_value=0, max_value=n_total, value=pool_default)

# Generate quiz
if st.button("Generar cuestionario"):
    qs = [q.copy() for q in questions_all]
    if shuffle_q:
        random.shuffle(qs)
    if pool_size and pool_size > 0:
        qs = qs[:pool_size]
    # Shuffle options per question
    for q in qs:
        if not q.get("opciones"):
            q["opciones"] = []
        if shuffle_opts:
            random.shuffle(q["opciones"])
    st.session_state["quiz_questions"] = qs
    st.session_state["answers"] = {i: None for i in range(len(qs))}
    st.session_state["results"] = []
    st.session_state["started_at"] = now_iso_utc()
    st.session_state["quiz_generated"] = True
    # optional rerun if available
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass

if not st.session_state["quiz_generated"] or not st.session_state["quiz_questions"]:
    st.info("Pulsa 'Generar cuestionario' para iniciar.")
    st.stop()

qs = st.session_state["quiz_questions"]
sqlite_conn = get_sqlite_conn(DB_FILE) if use_sqlite else None

# Layout
col_main, col_side = st.columns([3,1])
with col_side:
    st.markdown("### Estado")
    st.write("Iniciado:", st.session_state.get("started_at"))
    st.write(f"Preguntas en sesión: {len(qs)}")
    st.write(f"Perfil: {profile_name or 'anon'}")
    if st.button("Reiniciar (vaciar sesión)"):
        for k in ["quiz_questions","answers","results","started_at","quiz_generated"]:
            if k in st.session_state:
                del st.session_state[k]
        try:
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                st.stop()
        except Exception:
            st.stop()

    st.markdown("---")
    if st.button("Exportar preguntas -> questions_for_bot.json"):
        try:
            save_json_atomic(QUESTIONS_FOR_BOT, qs)
            st.success(f"Exportado {QUESTIONS_FOR_BOT}")
        except Exception as e:
            st.error(f"No se pudo exportar JSON: {e}")

    if st.session_state["results"]:
        bts = results_to_csv_bytes(st.session_state["results"])
        st.download_button("Descargar resultados (CSV)", data=bts, file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

with col_main:
    st.markdown("### Preguntas")
    for idx, q in enumerate(qs):
        st.markdown(f"**{idx+1}. [{q.get('tema')}] {q.get('pregunta')}**")
        corrects = q.get("correctas", [])
        opts = q.get("opciones", [])

        # ensure answer slot
        if idx not in st.session_state["answers"]:
            st.session_state["answers"][idx] = None

        # MULTIPLE corrects -> checkboxes
        if len(corrects) > 1:
            chosen = []
            for i, opt in enumerate(opts):
                key = f"q_{idx}_opt_{i}"
                if key not in st.session_state:
                    st.session_state[key] = False
                st.session_state[key] = st.checkbox(opt, key=key)
                if st.session_state[key]:
                    chosen.append(opt)
            st.session_state["answers"][idx] = chosen
            # auto-check only if some selected
            if auto_check and chosen:
                ok = set(chosen) == set(corrects)
                if ok:
                    st.success("✅ Correcto")
                else:
                    st.error("❌ Incorrecto")
                    if corrects:
                        st.info("Correctas: " + ", ".join(corrects))
        else:
            # SINGLE choice -> selectbox with placeholder
            placeholder = "-- Selecciona --"
            options_for_select = [placeholder] + opts
            key = f"q_{idx}_select"
            prev = st.session_state["answers"].get(idx)
            default = prev if prev in opts else placeholder
            try:
                choice = st.selectbox("", options_for_select, index=options_for_select.index(default), key=key)
            except Exception:
                # fallback safe: index 0
                choice = st.selectbox("", options_for_select, index=0, key=key)
            if choice == placeholder:
                st.session_state["answers"][idx] = None
            else:
                st.session_state["answers"][idx] = choice
            # auto-check only if user selected a real option
            if auto_check and st.session_state["answers"][idx] is not None:
                elec = st.session_state["answers"][idx]
                ok = (elec == corrects[0]) if corrects else None
                if ok is True:
                    st.success("✅ Correcto")
                elif ok is False:
                    st.error("❌ Incorrecto")
                    if corrects:
                        st.info("Correcta: " + ", ".join(corrects))
                    else:
                        st.info("No hay respuesta marcada en el fichero.")

        # Button to check & save this question
        if st.button(f"Comprobar y guardar pregunta {idx+1}", key=f"check_{idx}"):
            elec = st.session_state["answers"].get(idx)
            if len(corrects) > 1:
                ok = set(elec or []) == set(corrects)
            else:
                ok = (elec == corrects[0]) if corrects and elec is not None else None
            r = {
                "timestamp": now_iso_utc(),
                "profile": profile_name or "anon",
                "question_idx": idx,
                "tema": q.get("tema"),
                "pregunta": q.get("pregunta"),
                "eleccion": elec,
                "correctas": corrects,
                "ok": ok
            }
            st.session_state["results"].append(r)
            # auto-save JSON file
            if auto_save_on_check:
                try:
                    fname = os.path.join(RESULTS_DIR, f"results_{(profile_name or 'anon')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    save_json_atomic(fname, st.session_state["results"])
                    st.info(f"Guardado local: {fname}")
                except Exception:
                    st.warning("No se pudo guardar resultados en disco.")
            # sqlite backup
            if sqlite_conn is not None:
                save_result_to_db(sqlite_conn, r)
            if r["ok"]:
                st.success("✅ Correcto (guardado)")
            else:
                st.error("❌ Incorrecto (guardado)")
                if corrects:
                    st.info("Correctas: " + ", ".join(corrects))

# -------------------- Bottom controls --------------------
st.markdown("---")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("Comprobar todas las respuestas"):
        total = len(qs)
        aciertos = 0
        batch = []
        for idx, q in enumerate(qs):
            elec = st.session_state["answers"].get(idx)
            corrects = q.get("correctas", [])
            if len(corrects) > 1:
                ok = set(elec or []) == set(corrects)
            else:
                ok = (elec == corrects[0]) if corrects and elec is not None else None
            if ok:
                aciertos += 1
            r = {
                "timestamp": now_iso_utc(),
                "profile": profile_name or "anon",
                "question_idx": idx,
                "tema": q.get("tema"),
                "pregunta": q.get("pregunta"),
                "eleccion": elec,
                "correctas": corrects,
                "ok": ok
            }
            batch.append(r)
            st.session_state["results"].append(r)
            if sqlite_conn is not None:
                save_result_to_db(sqlite_conn, r)
        st.success(f"Resultado: {aciertos} / {total} correctas")
        if auto_save_on_check:
            try:
                fname = os.path.join(RESULTS_DIR, f"results_batch_{(profile_name or 'anon')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                save_json_atomic(fname, st.session_state["results"])
                st.info(f"Guardado lote: {fname}")
            except Exception:
                st.warning("No se pudo guardar el lote en disco.")

with c2:
    if st.session_state["results"]:
        bts = results_to_csv_bytes(st.session_state["results"])
        st.download_button("Descargar resultados (CSV)", data=bts, file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    else:
        st.write("No hay resultados guardados todavía.")

with c3:
    st.markdown("### ChatGPT — explicación (opcional)")
    qtext = st.text_input("Pega la pregunta para pedir explicación (opcional)", key="gpt_q_text")
    if st.button("Pedir explicación a OpenAI"):
        if not openai_key:
            st.error("Introduce tu OpenAI API key en la sidebar.")
        elif not qtext:
            st.error("Pega la pregunta para la que quieres explicación.")
        else:
            try:
                import openai
                openai.api_key = openai_key
                # Use ChatCompletion if available, else fallback for older sdk versions
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":f"Explica por qué la respuesta es correcta:\n\n{qtext}"}],
                        max_tokens=400
                    )
                    ans = completion["choices"][0]["message"]["content"]
                except Exception:
                    # fallback minimal
                    ans = openai.Completion.create(
                        model="gpt-4o-mini",
                        prompt=f"Explica por qué la respuesta es correcta:\n\n{qtext}",
                        max_tokens=400
                    )["choices"][0]["text"]
                st.markdown("**Explicación GPT:**")
                st.write(ans)
            except Exception as e:
                st.error(f"No se pudo llamar a OpenAI: {e}")

st.markdown("---")
st.caption("Listo — esta versión evita preselecciones, maneja multi-respuesta, guarda resultados y puede exportar JSON para el bot. Si quieres, genero también el Dockerfile o el script que convierta tu .txt en questions_for_bot.json y subo ese JSON aquí mismo.")
