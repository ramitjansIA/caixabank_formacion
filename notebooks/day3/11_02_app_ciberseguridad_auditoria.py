# app_ciberseguridad_auditoria.py
# ------------------------------------------------------------
# App Streamlit autocontenida para demo de ciberseguridad bancaria.
# No usa CSV externo: genera datos sintéticos y entrena modelos en local.
# Módulos:
#   1) Clasificación tabular de riesgo de incidente
#   2) Análisis de texto de alerta / correo sospechoso
#   3) Análisis opcional de imagen con Hugging Face
# ------------------------------------------------------------

import re
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CyberAudit AI - Demo bancaria",
    page_icon="🛡️",
    layout="wide"
)


# ------------------------------------------------------------
# 1. DATOS SINTÉTICOS TABULARES
# ------------------------------------------------------------

@st.cache_data
def generar_datos_ciberseguridad(n=2500, random_state=42):
    rng = np.random.default_rng(random_state)

    canal = rng.choice(["web", "mobile", "api", "oficina", "atm"], size=n, p=[0.35, 0.30, 0.18, 0.10, 0.07])
    pais_riesgo = rng.choice(["bajo", "medio", "alto"], size=n, p=[0.70, 0.22, 0.08])
    tipo_evento = rng.choice(
        ["login", "transferencia", "cambio_password", "alta_beneficiario", "descarga_datos", "intento_api"],
        size=n,
        p=[0.32, 0.24, 0.13, 0.12, 0.09, 0.10]
    )
    dispositivo_nuevo = rng.binomial(1, 0.18, size=n)
    hora = rng.integers(0, 24, size=n)
    intentos_fallidos = rng.poisson(1.2, size=n)
    importe = rng.gamma(shape=2.2, scale=900, size=n)
    importe = np.where(tipo_evento == "transferencia", importe, importe * 0.08)
    antiguedad_cliente_meses = rng.integers(1, 180, size=n)
    num_ips_24h = rng.poisson(1.5, size=n) + 1
    score_reputacion_ip = rng.normal(72, 18, size=n).clip(0, 100)
    distancia_km_ultimo_login = rng.gamma(2.0, 600, size=n).clip(0, 9000)
    cambios_perfil_7d = rng.poisson(0.4, size=n)

    # Regla latente de riesgo: combina fraude + ciberseguridad.
    riesgo = (
        0.9 * dispositivo_nuevo
        + 0.8 * (pais_riesgo == "alto").astype(int)
        + 0.45 * (pais_riesgo == "medio").astype(int)
        + 0.9 * (tipo_evento == "alta_beneficiario").astype(int)
        + 0.75 * (tipo_evento == "intento_api").astype(int)
        + 0.65 * (tipo_evento == "descarga_datos").astype(int)
        + 0.7 * (hora <= 5).astype(int)
        + 0.28 * intentos_fallidos
        + 0.00035 * importe
        + 0.25 * num_ips_24h
        - 0.018 * score_reputacion_ip
        + 0.00018 * distancia_km_ultimo_login
        + 0.32 * cambios_perfil_7d
        - 0.004 * antiguedad_cliente_meses
        + rng.normal(0, 0.9, size=n)
    )

    # Umbral para generar una clase minoritaria moderada.
    prob = 1 / (1 + np.exp(-(riesgo - 2.0)))
    incidente = rng.binomial(1, prob)

    df = pd.DataFrame({
        "canal": canal,
        "pais_riesgo": pais_riesgo,
        "tipo_evento": tipo_evento,
        "dispositivo_nuevo": dispositivo_nuevo,
        "hora": hora,
        "intentos_fallidos": intentos_fallidos,
        "importe": importe.round(2),
        "antiguedad_cliente_meses": antiguedad_cliente_meses,
        "num_ips_24h": num_ips_24h,
        "score_reputacion_ip": score_reputacion_ip.round(2),
        "distancia_km_ultimo_login": distancia_km_ultimo_login.round(2),
        "cambios_perfil_7d": cambios_perfil_7d,
        "incidente_alto_riesgo": incidente
    })
    return df


@st.cache_resource
def entrenar_modelos():
    df = generar_datos_ciberseguridad()
    X = df.drop(columns=["incidente_alto_riesgo"])
    y = df["incidente_alto_riesgo"]

    cat_cols = ["canal", "pais_riesgo", "tipo_evento"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    modelos = {
        "Regresión logística / GLM": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=9,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    resultados = {}
    for nombre, modelo in modelos.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", modelo)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            y_prob = None
            auc = np.nan
        resultados[nombre] = {
            "pipeline": pipe,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "auc": auc
        }

    return df, resultados, cat_cols, num_cols


# ------------------------------------------------------------
# 2. MOTOR SIMPLE DE TEXTO PARA DEMO
# ------------------------------------------------------------

PALABRAS_PHISHING = [
    "urgente", "bloqueada", "verifique", "contraseña", "password", "credenciales",
    "pin", "otp", "token", "enlace", "link", "caduca", "suspendida", "reactivar"
]
PALABRAS_FRAUDE = [
    "beneficiario nuevo", "importe elevado", "transferencia", "cripto", "mule",
    "cuenta puente", "país de riesgo", "operación inusual", "retirada", "cashout"
]
PALABRAS_CIBER = [
    "ransomware", "malware", "exfiltración", "exfiltracion", "api", "bruteforce",
    "fuerza bruta", "credenciales", "ip sospechosa", "tor", "vpn", "sesión anómala"
]


def analizar_texto_alerta(texto):
    texto_norm = texto.lower()
    score_phishing = sum(1 for p in PALABRAS_PHISHING if p in texto_norm)
    score_fraude = sum(1 for p in PALABRAS_FRAUDE if p in texto_norm)
    score_ciber = sum(1 for p in PALABRAS_CIBER if p in texto_norm)

    scores = {
        "phishing / ingeniería social": score_phishing,
        "fraude transaccional": score_fraude,
        "ciberseguridad técnica": score_ciber
    }

    categoria = max(scores, key=scores.get)
    total = sum(scores.values())

    if total == 0:
        nivel = "Bajo"
        categoria = "sin patrón fuerte"
        recomendacion = "Revisión ordinaria. No se observan términos críticos en esta demo."
    elif total <= 2:
        nivel = "Medio"
        recomendacion = "Revisar contexto, usuario afectado, canal y trazabilidad del evento."
    else:
        nivel = "Alto"
        recomendacion = "Escalar a equipo de fraude/ciberseguridad, preservar evidencias y validar controles afectados."

    return scores, categoria, nivel, recomendacion


# ------------------------------------------------------------
# 3. MÓDULO OPCIONAL HUGGING FACE PARA IMAGEN
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cargar_detector_imagen():
    from transformers import pipeline
    return pipeline("object-detection", model="hustvl/yolos-tiny")


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.title("🛡️ CyberAudit AI — Demo integral para auditores bancarios")
st.caption("Clasificación tabular, análisis de texto y análisis visual. Datos sintéticos. Sin CSV externo.")

with st.sidebar:
    st.header("Módulos")
    pagina = st.radio(
        "Selecciona una demo",
        [
            "1. Riesgo tabular de incidente",
            "2. Análisis de texto: phishing/fraude/ciber",
            "3. Imagen: detección visual con Hugging Face",
            "4. Dataset sintético y métricas"
        ]
    )

    st.markdown("---")
    st.info("App pensada para formación: no sustituye un sistema real de scoring ni una investigación forense.")


df, resultados, cat_cols, num_cols = entrenar_modelos()


if pagina == "1. Riesgo tabular de incidente":
    st.header("1. Clasificación tabular de riesgo de incidente")
    st.write(
        "Este módulo simula un caso de auditoría donde un registro operativo se clasifica como "
        "incidente de alto riesgo o evento ordinario."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        canal = st.selectbox("Canal", ["web", "mobile", "api", "oficina", "atm"])
        pais_riesgo = st.selectbox("Riesgo del país", ["bajo", "medio", "alto"])
        tipo_evento = st.selectbox(
            "Tipo de evento",
            ["login", "transferencia", "cambio_password", "alta_beneficiario", "descarga_datos", "intento_api"]
        )
        dispositivo_nuevo = st.selectbox("¿Dispositivo nuevo?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

    with col2:
        hora = st.slider("Hora del evento", 0, 23, 2)
        intentos_fallidos = st.number_input("Intentos fallidos", min_value=0, max_value=50, value=4)
        importe = st.number_input("Importe asociado (€)", min_value=0.0, max_value=500000.0, value=25000.0, step=500.0)
        antiguedad_cliente_meses = st.number_input("Antigüedad cliente (meses)", min_value=1, max_value=240, value=8)

    with col3:
        num_ips_24h = st.number_input("Nº IPs en 24h", min_value=1, max_value=50, value=5)
        score_reputacion_ip = st.slider("Score reputación IP", 0, 100, 35)
        distancia_km_ultimo_login = st.number_input("Distancia desde último login (km)", min_value=0.0, max_value=10000.0, value=3200.0)
        cambios_perfil_7d = st.number_input("Cambios de perfil en 7 días", min_value=0, max_value=20, value=2)

    modelo_nombre = st.selectbox("Modelo", list(resultados.keys()), index=2)
    pipe = resultados[modelo_nombre]["pipeline"]

    registro = pd.DataFrame([{
        "canal": canal,
        "pais_riesgo": pais_riesgo,
        "tipo_evento": tipo_evento,
        "dispositivo_nuevo": dispositivo_nuevo,
        "hora": hora,
        "intentos_fallidos": intentos_fallidos,
        "importe": importe,
        "antiguedad_cliente_meses": antiguedad_cliente_meses,
        "num_ips_24h": num_ips_24h,
        "score_reputacion_ip": score_reputacion_ip,
        "distancia_km_ultimo_login": distancia_km_ultimo_login,
        "cambios_perfil_7d": cambios_perfil_7d
    }])

    pred = int(pipe.predict(registro)[0])
    prob = float(pipe.predict_proba(registro)[0, 1]) if hasattr(pipe.named_steps["model"], "predict_proba") else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicción", "Alto riesgo" if pred == 1 else "Riesgo ordinario")
    c2.metric("Probabilidad estimada", f"{prob:.1%}")
    c3.metric("Modelo usado", modelo_nombre)

    if prob >= 0.70:
        st.error("Decisión sugerida: escalar inmediatamente y bloquear/revisar la operación.")
    elif prob >= 0.40:
        st.warning("Decisión sugerida: revisión manual y contraste con evidencias adicionales.")
    else:
        st.success("Decisión sugerida: monitorización ordinaria.")

    with st.expander("Ver registro enviado al modelo"):
        st.dataframe(registro, use_container_width=True)


elif pagina == "2. Análisis de texto: phishing/fraude/ciber":
    st.header("2. Análisis de texto de alerta, correo o comentario de auditoría")
    ejemplo = (
        "URGENTE: su cuenta será suspendida. Verifique su contraseña y OTP en este enlace. "
        "Además, se detecta transferencia de importe elevado a beneficiario nuevo."
    )
    texto = st.text_area("Texto a analizar", value=ejemplo, height=170)

    scores, categoria, nivel, recomendacion = analizar_texto_alerta(texto)

    c1, c2, c3 = st.columns(3)
    c1.metric("Nivel de alerta", nivel)
    c2.metric("Categoría dominante", categoria)
    c3.metric("Indicadores detectados", sum(scores.values()))

    st.subheader("Scores por familia de riesgo")
    st.bar_chart(pd.Series(scores))

    if nivel == "Alto":
        st.error(recomendacion)
    elif nivel == "Medio":
        st.warning(recomendacion)
    else:
        st.success(recomendacion)

    st.markdown("### Lectura para auditores")
    st.write(
        "Este módulo no pretende ser un clasificador NLP definitivo. Sirve para explicar cómo un modelo de texto "
        "puede transformar descripciones no estructuradas en señales de riesgo: phishing, fraude transaccional "
        "o ciberseguridad técnica. En producción se podría sustituir por BERT, RoBERTa o un modelo ajustado "
        "con históricos internos etiquetados."
    )


elif pagina == "3. Imagen: detección visual con Hugging Face":
    st.header("3. Análisis visual con Hugging Face")
    st.write(
        "Carga una imagen/captura para ejecutar un detector visual generalista. Para entornos bancarios reales, "
        "habría que ajustar el modelo con clases propias: documento manipulado, QR sospechoso, pantalla de login falsa, etc."
    )

    archivo = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
    umbral = st.slider("Umbral de confianza", 0.01, 0.90, 0.20, 0.01)

    if archivo is not None:
        from PIL import Image, ImageDraw
        image = Image.open(archivo).convert("RGB")
        st.image(image, caption="Imagen cargada", use_container_width=True)

        if st.button("Ejecutar detección visual"):
            try:
                detector = cargar_detector_imagen()
                detecciones = detector(image, threshold=umbral)
                st.write("Detecciones:")
                st.dataframe(pd.DataFrame(detecciones), use_container_width=True)

                img_draw = image.copy()
                draw = ImageDraw.Draw(img_draw)
                for det in detecciones:
                    box = det["box"]
                    label = det["label"]
                    score = det["score"]
                    draw.rectangle(
                        [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                        outline="red",
                        width=3
                    )
                    draw.text((box["xmin"], max(0, box["ymin"] - 15)), f"{label} {score:.2f}", fill="red")
                st.image(img_draw, caption="Resultado con bounding boxes", use_container_width=True)

                if len(detecciones) == 0:
                    st.info(
                        "No se detectaron objetos. Es normal si la imagen contiene pantallas sintéticas, textos o conceptos "
                        "bancarios que el modelo generalista no conoce."
                    )
            except Exception as e:
                st.error("No se pudo cargar o ejecutar el modelo de Hugging Face.")
                st.code(str(e))
    else:
        st.info("Sube una imagen para activar este módulo.")


else:
    st.header("4. Dataset sintético y métricas de modelos")
    st.write("Distribución de la variable objetivo generada sintéticamente:")
    st.dataframe(df["incidente_alto_riesgo"].value_counts().rename(index={0: "ordinario", 1: "alto riesgo"}), use_container_width=True)

    st.subheader("Muestra de datos")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Comparativa de métricas")
    filas = []
    for nombre, info in resultados.items():
        rep = info["classification_report"]
        filas.append({
            "modelo": nombre,
            "AUC": info["auc"],
            "precision_clase_riesgo": rep["1"]["precision"],
            "recall_clase_riesgo": rep["1"]["recall"],
            "f1_clase_riesgo": rep["1"]["f1-score"],
            "accuracy": rep["accuracy"]
        })
    metricas = pd.DataFrame(filas)
    st.dataframe(metricas, use_container_width=True)

    st.markdown("### Nota metodológica")
    st.write(
        "Los modelos usan `class_weight='balanced'` para que el aprendizaje no ignore la clase de alto riesgo. "
        "En un caso real se añadiría validación temporal, calibración de probabilidades, explainability con SHAP, "
        "trazabilidad de variables y revisión de sesgos operativos."
    )

