# streamlit_showcase.py
"""
Streamlit Showcase — despliegue interactivo y avanzado
Ejecutar: streamlit run streamlit_showcase.py

Instalar dependencias mínimas:
pip install streamlit pandas numpy altair plotly scikit-learn pillow pydeck
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io, os, json, time
from datetime import datetime, timezone

# Optional imports guard (pydeck)
try:
    import pydeck as pdk
    _HAS_PYDECK = True
except Exception:
    _HAS_PYDECK = False

# ---------------- UI: Page & style ----------------
st.set_page_config(page_title="Streamlit Showcase — Avanzado", layout="wide", initial_sidebar_state="expanded")
# Hero
st.markdown("<style> .big-title {font-size:32px; font-weight:700;} .muted {color:#6c757d;} </style>", unsafe_allow_html=True)
st.markdown('<div class="big-title">🚀 Streamlit Showcase — Lo que puedes construir hoy</div>', unsafe_allow_html=True)
st.caption("Interactividad, visualizaciones y deploy — todo en un solo archivo. Experimenta los bloques y prueba cada sección.")

# Sidebar controls (global)
st.sidebar.header("Ajustes globales")
THEME = st.sidebar.selectbox("Tema visual (solo demostración)", ["Auto", "Claro", "Oscuro"])
if THEME != "Auto":
    st.sidebar.info("Cambia el tema en Settings → Theme (Streamlit) para ver el efecto real.")
SAMPLE_SIZE = st.sidebar.slider("Tamaño ejemplo (para demos de ML / mapas)", 100, 5000, 800, step=100)

st.sidebar.markdown("---")
st.sidebar.markdown("📦 Dependencias:\n`pandas numpy altair plotly scikit-learn pillow pydeck`")

# session_state defaults
if "log" not in st.session_state:
    st.session_state["log"] = []
if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None

def log(msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    st.session_state["log"].append(f"{ts} — {msg}")

# ---------- Section 1: Quick Data Explorer ----------
st.header("1) Data Explorer — subir, explorar y editar")
with st.expander("Abrir Data Explorer (uploader + editor + filtros)", expanded=True):
    uploaded = st.file_uploader("Sube un CSV/TSV o deja que cargue un dataset de ejemplo", type=["csv","tsv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_csv(uploaded, sep="\t")
            st.session_state["uploaded_df"] = df
            st.success(f"Cargado {uploaded.name} — {df.shape[0]} filas × {df.shape[1]} columnas")
            log(f"CSV cargado: {uploaded.name}")
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
    else:
        # sample dataset auto-generado
        if st.button("Cargar dataset de ejemplo (blobs)"):
            X, y = make_blobs(n_samples=SAMPLE_SIZE, centers=5, n_features=5, random_state=42)
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["label"] = y
            st.session_state["uploaded_df"] = df
            st.success(f"Generado ejemplo — {df.shape[0]} filas")
            log("Dataset de ejemplo generado")

    df = st.session_state["uploaded_df"]
    if df is None:
        st.info("Sube un archivo o genera el dataset de ejemplo para ver herramientas de edición y filtros.")
    else:
        st.markdown("### Vista rápida")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("### Editor rápido (edita la tabla y descarga)")
        try:
            # new Streamlit has st.data_editor; fallback to st.experimental_data_editor if present
            if hasattr(st, "data_editor"):
                edited = st.data_editor(df, num_rows="dynamic")
            else:
                edited = st.experimental_data_editor(df, num_rows="dynamic")
        except Exception:
            # fallback: show dataframe only
            st.warning("Editor interactivo no soportado en esta versión de Streamlit — mostrando dataframe sólo.")
            edited = df

        # filter example
        st.markdown("### Filter: selecciona columna y rango")
        col_filter = st.selectbox("Columna numérica para filtrar (si existe)", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])] + [None])
        if col_filter:
            lo, hi = st.slider("Rango", float(df[col_filter].min()), float(df[col_filter].max()), (float(df[col_filter].quantile(0.05)), float(df[col_filter].quantile(0.95))))
            filtered = edited[(edited[col_filter] >= lo) & (edited[col_filter] <= hi)]
            st.write(f"Filtradas {filtered.shape[0]} filas")
            st.dataframe(filtered.head(200), use_container_width=True)
        else:
            filtered = edited

        # export edited
        buf = io.BytesIO()
        filtered.to_csv(buf, index=False)
        st.download_button("Descargar CSV (filtrado/editar)", data=buf.getvalue(), file_name="filtered_export.csv", mime="text/csv")

# ---------- Section 2: Visual Analytics ----------
st.header("2) Visual Analytics — charts reactivas")
with st.expander("Abrir Visual Analytics", expanded=False):
    st.write("Ejemplo: gráfico Altair con selección (brush) que controla histograma y tabla.")
    df = st.session_state["uploaded_df"]
    if df is None or df.shape[0] < 2:
        st.info("Carga o genera un dataset para ver visualizaciones interactivas.")
    else:
        # pick two numeric columns for scatter
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            st.warning("Necesitamos al menos 2 columnas numéricas en el dataset.")
        else:
            xcol = st.selectbox("Eje X", numeric_cols, index=0)
            ycol = st.selectbox("Eje Y", numeric_cols, index=1)
            color_col = st.selectbox("Color (opcional)", [None] + list(df.columns), index=0)

            base = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X(xcol, scale=alt.Scale(zero=False)),
                y=alt.Y(ycol, scale=alt.Scale(zero=False)),
                tooltip=list(df.columns)[:6]
            ).interactive()

            brush = alt.selection(type='interval')
            points = base.add_selection(brush).properties(width=700, height=400)
            st.altair_chart(points, use_container_width=True)

            selected = df
            sel = df
            # we can create linked views: histogram conditioned on brush selection
            if brush:
                # create histogram of xcol with selection highlight
                hist = alt.Chart(df).mark_bar().encode(
                    x=alt.X(f"{xcol}:Q", bin=alt.Bin(maxbins=30)),
                    y='count()',
                    color=alt.condition(brush, alt.value("steelblue"), alt.value("lightgray"))
                ).transform_filter(brush).properties(width=700)
                st.altair_chart(hist, use_container_width=True)

            # Also an interactive Plotly chart
            st.markdown("**Plotly — 3D scatter (si hay >2 columnas numéricas)**")
            if len(numeric_cols) >= 3:
                zcol = numeric_cols[2]
                fig = px.scatter_3d(df, x=xcol, y=ycol, z=zcol, color=color_col if color_col in df.columns else None,
                                    hover_data=list(df.columns)[:6])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Se requieren 3 columnas numéricas para ver scatter 3D.")

# ---------- Section 3: Geospatial ----------
st.header("3) Geospatial — mapas y capas")
with st.expander("Abrir Geospatial (mapa rápido + pydeck)", expanded=False):
    # create fake geo points if dataset numeric
    df = st.session_state["uploaded_df"]
    if df is None:
        st.info("Carga datos o genera dataset de ejemplo para ver mapas.")
    else:
        st.write("Generamos coordenadas aleatorias centradas en Madrid para demostrar capas.")
        center_lat, center_lon = 40.4168, -3.7038  # Madrid
        n = min(df.shape[0], SAMPLE_SIZE)
        rng = np.random.RandomState(42)
        lats = center_lat + 0.05 * (rng.rand(n) - 0.5)
        lons = center_lon + 0.05 * (rng.rand(n) - 0.5)
        gdf = pd.DataFrame({"lat": lats, "lon": lons})
        st.map(gdf)  # simple map

        if _HAS_PYDECK:
            st.markdown("**pydeck layer example**")
            layer = pdk.Layer(
                "HexagonLayer",
                data=gdf,
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=50,
                pickable=True,
                elevation_range=[0, 1000],
                extruded=True,
            )
            deck = pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=40),
                layers=[layer],
            )
            st.pydeck_chart(deck)
        else:
            st.info("pydeck no instalado — instala `pydeck` para ver capas 3D.")

# ---------- Section 4: Model Playground (ML en vivo) ----------
st.header("4) Model Playground — clustering y PCA en vivo")
with st.expander("Abrir Model Playground", expanded=False):
    st.write("Ajusta número de clusters y escala, observa resultados en tiempo real.")
    df = st.session_state["uploaded_df"]
    if df is None:
        st.info("Carga o genera un dataset para probar modelos.")
    else:
        # choose numeric subset
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            st.warning("Necesitamos al menos 2 columnas numéricas para ML demo.")
        else:
            cols = st.multiselect("Columnas para modelo (selecciona 2-6)", numeric_cols, default=numeric_cols[:3])
            if 2 <= len(cols) <= 6:
                X = df[cols].dropna().to_numpy()
                n_clusters = st.slider("Número de clusters (KMeans)", 2, 12, 4)
                scale = st.slider("Ruido / escala (aleatorizar datos)", 0.0, 1.0, 0.0, step=0.05)
                if scale > 0:
                    X = X + np.random.normal(scale=scale * np.std(X), size=X.shape)
                # PCA to 2D for visualization
                pca = PCA(n_components=2)
                X2 = pca.fit_transform(X)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X2)
                labels = kmeans.labels_
                df_vis = pd.DataFrame(X2, columns=["PC1", "PC2"])
                df_vis["cluster"] = labels.astype(str)
                fig = px.scatter(df_vis, x="PC1", y="PC2", color="cluster", title=f"PCA + KMeans (k={n_clusters})", width=800, height=500)
                st.plotly_chart(fig, use_container_width=True)
                # Show cluster centers and basic metrics
                centers = kmeans.cluster_centers_
                st.write("Centros (PCA space):")
                st.dataframe(pd.DataFrame(centers, columns=["PC1", "PC2"]))
                # export model labels
                if st.button("Exportar etiquetas como CSV"):
                    out = df.assign(_label=labels)[list(df.columns) + ["_label"]]
                    buf = io.BytesIO()
                    out.to_csv(buf, index=False)
                    st.download_button("Descargar CSV con etiquetas", data=buf.getvalue(), file_name="labels_export.csv", mime="text/csv")
            else:
                st.info("Selecciona entre 2 y 6 columnas numéricas para la demo.")

# ---------- Section 5: Media Studio (images + audio) ----------
st.header("5) Media Studio — edición de imágenes y audio dinámico")
with st.expander("Abrir Media Studio", expanded=False):
    st.write("Sube una imagen y prueba filtros en tiempo real.")
    uploaded_img = st.file_uploader("Sube una imagen (png/jpg)", type=["png","jpg","jpeg"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Original", use_column_width=True)
        # image controls
        st.markdown("Filtros:")
        blur = st.slider("Blur (radius)", 0.0, 10.0, 0.0)
        contrast = st.slider("Contraste", 0.5, 2.0, 1.0)
        grayscale = st.checkbox("Convertir a escala de grises")
        flip = st.checkbox("Flip horizontal")
        rotate = st.slider("Rotación grados", -180, 180, 0)
        # apply filters
        img2 = img.copy()
        if blur > 0:
            img2 = img2.filter(ImageFilter.GaussianBlur(radius=blur))
        if grayscale:
            img2 = ImageOps.grayscale(img2).convert("RGB")
        if contrast != 1.0:
            img2 = ImageEnhance.Contrast(img2).enhance(contrast)
        if flip:
            img2 = ImageOps.mirror(img2)
        if rotate != 0:
            img2 = img2.rotate(rotate, expand=True)
        st.image(img2, caption="Transformada", use_column_width=True)
        # download
        buf = io.BytesIO()
        img2.save(buf, format="PNG")
        st.download_button("Descargar imagen transformada", data=buf.getvalue(), file_name="transformed.png", mime="image/png")
    else:
        st.info("Sube una imagen para editarla aquí.")

    st.markdown("### Audio demo: generar tono simple y escuchar")
    freq = st.slider("Frecuencia (Hz)", 220, 880, 440)
    duration = st.slider("Duración (s)", 0.5, 5.0, 1.5)
    samplerate = 22050
    t = np.linspace(0, duration, int(samplerate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    # convert to 16-bit PCM
    audio = (tone * 32767).astype(np.int16)
    buf = io.BytesIO()
    # WAV header using scipy or write simple wave (we'll use wave)
    import wave
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    st.audio(buf.getvalue(), format="audio/wav")

# ---------- Section 6: Real-time & Interactivity ----------
st.header("6) Real-time, estado y notificaciones")
with st.expander("Abrir Real-time demo", expanded=False):
    st.write("Progress bar, logs en sesión y notificaciones.")
    if st.button("Empezar proceso simulado (5 pasos)"):
        st.info("Proceso iniciado — mira el panel 'Logs' abajo.")
        log("Proceso simulado iniciado")
        progress = st.progress(0)
        for i in range(5):
            time.sleep(0.35)
            progress.progress(int((i+1)/5*100))
            log(f"Paso {i+1} completado")
        st.success("Proceso finalizado")
        log("Proceso finalizado")
    st.markdown("#### Logs de la sesión")
    st.text_area("Logs", value="\n".join(st.session_state["log"][-50:]), height=200)

# ---------- Section 7: Embeds / Custom HTML/JS ----------
st.header("7) Embeds: incrusta HTML/JS y componentes personalizados")
with st.expander("Abrir Embeds (HTML/JS)", expanded=False):
    st.write("Puedes incrustar pequeños widgets HTML/JS con `st.components.v1.html`.")
    sample_html = """
    <div style="font-family:Arial, sans-serif; padding:10px; border-radius:8px; border:1px solid #eee;">
      <h3 style="margin:0 0 8px 0">Mini contador (HTML/JS)</h3>
      <div id="count" style="font-size:22px;">0</div>
      <button onclick="document.getElementById('count').innerText = parseInt(document.getElementById('count').innerText) + 1">+1</button>
      <button onclick="document.getElementById('count').innerText = 0">reset</button>
    </div>
    """
    st.components.v1.html(sample_html, height=150)

    st.write("También puedes incrustar visualizaciones D3, iframes con dashboards externos o tus React components empaquetados.")

# ---------- Section 8: Patterns & Tips ----------
st.header("8) Buenas prácticas y patterns")
with st.expander("Abrir buenas prácticas", expanded=False):
    st.markdown("""
    - Usa `st.cache_data` / `st.cache_resource` para operaciones costosas (carga de datos, modelos).
    - Mantén la lógica de estado en `st.session_state` para flows multi-página.
    - Divide UI en columnas y `expander` para no saturar la pantalla.
    - Para apps complejas: separa en módulos y añade tests.
    - Para producción: dockeriza, usa `gunicorn` si usas multiples procesos, y monitoriza con logs+prometheus.
    """)
    st.markdown("**Ejemplo rápido de cache:**")
    st.code("""
    @st.cache_data
    def load_big_data(path):
        return pd.read_parquet(path)
    """)

# ---------- Section 9: Export & Deploy ----------
st.header("9) Export & Deploy")
with st.expander("Instrucciones de deploy", expanded=False):
    st.markdown("""
    **Opciones rápidas:**
    1. Streamlit Community Cloud — push a GitHub y conecta. (rápido, gratis para demos)
    2. Render / Fly / Heroku — contenedor Docker o `gunicorn` + `streamlit run`.
    3. Docker local — ejemplo abajo.

    **Dockerfile mínimo:**
    ```dockerfile
    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    EXPOSE 8501
    CMD ["streamlit", "run", "streamlit_showcase.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ```
    """)
    st.markdown("**Generar requirements.txt básico**")
    st.code("pandas\nnumpy\nstreamlit\naltair\nplotly\nscikit-learn\npillow\npydeck")

# ---------- Footer: CTA ----------
st.markdown("---")
st.markdown("### ¿Te ha gustado la demo? 🎉")
col_a, col_b = st.columns([2,1])
with col_a:
    st.write("Puedo convertir cualquiera de estas secciones en una app independiente, añadir autenticación, conectar a DBs (Postgres), añadir CI/CD y empaquetar con Docker + nginx. Dime qué quieres y lo preparo.")
with col_b:
    if st.button("Mostrar resumen rápido (export)"):
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "has_uploaded": st.session_state["uploaded_df"] is not None,
            "rows": int(st.session_state["uploaded_df"].shape[0]) if st.session_state["uploaded_df"] is not None else 0
        }
        st.json(summary)
