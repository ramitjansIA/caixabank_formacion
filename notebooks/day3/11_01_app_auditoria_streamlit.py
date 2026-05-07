# app_auditoria_streamlit.py
# ------------------------------------------------------------
# Aplicación Streamlit: predicción de riesgo de auditoría bancaria
# ------------------------------------------------------------
# Uso:
#   streamlit run app_auditoria_streamlit.py
#
# Requiere que el CSV auditoria_bancaria_desbalanceada.csv esté en la
# misma carpeta que este archivo .py, o indicar su ruta en CSV_PATH.

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False
    SMOTE = None
    ImbPipeline = Pipeline


CSV_PATH = "../../data/auditoria_bancaria_desbalanceada.csv"
MODEL_PATH = "modelo_riesgo_auditoria.joblib"
TARGET = "riesgo_auditoria"
ID_COL = "id_operacion"


st.set_page_config(
    page_title="Demo auditoría bancaria - Predicción de riesgo",
    page_icon="🏦",
    layout="wide",
)


@st.cache_data
def cargar_datos(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encuentra el archivo {path}. Colócalo en la misma carpeta que este script."
        )
    return pd.read_csv(path)


def construir_pipeline(df: pd.DataFrame, usar_smote: bool = True):
    X = df.drop(columns=[TARGET])
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    y = df[TARGET].astype(int)

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    if usar_smote and SMOTE_AVAILABLE:
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                ("model", model),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    return pipeline, X, y, numeric_features, categorical_features


@st.cache_resource
def entrenar_o_cargar_modelo(df: pd.DataFrame, usar_smote: bool = True):
    pipeline, X, y, numeric_features, categorical_features = construir_pipeline(df, usar_smote=usar_smote)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    artifact = {
        "pipeline": pipeline,
        "features": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": metrics,
        "usar_smote": usar_smote and SMOTE_AVAILABLE,
    }

    joblib.dump(artifact, MODEL_PATH)
    return artifact


def clasificar_riesgo(probabilidad: float) -> tuple[str, str]:
    if probabilidad >= 0.70:
        return "RIESGO ALTO", "🔴"
    elif probabilidad >= 0.40:
        return "RIESGO MEDIO", "🟠"
    else:
        return "RIESGO BAJO", "🟢"


def crear_formulario(df: pd.DataFrame, features: list[str]):
    st.sidebar.header("Formulario de operación")
    st.sidebar.caption("Introduce los datos del registro a evaluar.")

    valores = {}

    for col in features:
        serie = df[col]

        if pd.api.types.is_numeric_dtype(serie):
            min_val = float(np.nanmin(serie))
            max_val = float(np.nanmax(serie))
            med_val = float(np.nanmedian(serie))

            if pd.api.types.is_integer_dtype(serie):
                valores[col] = st.sidebar.number_input(
                    col,
                    min_value=int(np.floor(min_val)),
                    max_value=int(np.ceil(max_val * 2 if max_val > 0 else max_val + 1)),
                    value=int(round(med_val)),
                    step=1,
                )
            else:
                valores[col] = st.sidebar.number_input(
                    col,
                    min_value=float(min_val),
                    max_value=float(max_val * 2 if max_val > 0 else max_val + 1),
                    value=float(med_val),
                    step=1.0,
                    format="%.2f",
                )
        else:
            opciones = sorted(serie.dropna().astype(str).unique().tolist())
            valores[col] = st.sidebar.selectbox(col, opciones)

    return pd.DataFrame([valores])


try:
    df = cargar_datos(CSV_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

st.title("🏦 Predicción de riesgo de auditoría bancaria")
st.markdown(
    """
    Esta aplicación entrena un modelo supervisado con la base de datos adjunta y permite introducir
    una nueva operación mediante formulario para obtener una predicción de **riesgo de auditoría**.

    Variable objetivo: `riesgo_auditoria`  
    - `0`: operación sin riesgo detectado  
    - `1`: operación con riesgo potencial para revisión/auditoría
    """
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Registros", f"{len(df):,}".replace(",", "."))
with col_b:
    st.metric("Variables", df.shape[1])
with col_c:
    tasa = df[TARGET].mean() * 100
    st.metric("Tasa clase minoritaria", f"{tasa:.2f}%")

with st.expander("Ver distribución de la variable objetivo"):
    st.dataframe(df[TARGET].value_counts().rename_axis("riesgo_auditoria").reset_index(name="n"))

usar_smote = st.sidebar.checkbox(
    "Usar SMOTE en entrenamiento",
    value=True,
    help="SMOTE genera ejemplos sintéticos de la clase minoritaria solo en entrenamiento.",
)

if usar_smote and not SMOTE_AVAILABLE:
    st.sidebar.warning("imblearn no está instalado. Se entrenará sin SMOTE.")

artifact = entrenar_o_cargar_modelo(df, usar_smote=usar_smote)
pipeline = artifact["pipeline"]
features = artifact["features"]
metrics = artifact["metrics"]

nuevo_registro = crear_formulario(df, features)

st.subheader("Registro introducido")
st.dataframe(nuevo_registro, use_container_width=True)

if st.button("Predecir riesgo de auditoría", type="primary"):
    prob = float(pipeline.predict_proba(nuevo_registro)[:, 1][0])
    pred = int(prob >= 0.5)
    etiqueta, icono = clasificar_riesgo(prob)

    st.subheader("Resultado de la predicción")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicción binaria", pred)
    with c2:
        st.metric("Probabilidad de riesgo", f"{prob:.2%}")
    with c3:
        st.metric("Nivel interpretativo", f"{icono} {etiqueta}")

    if pred == 1:
        st.error("El modelo recomienda revisión manual/auditoría de la operación.")
    else:
        st.success("El modelo no detecta riesgo elevado según el patrón aprendido.")

    st.progress(min(prob, 1.0))

st.subheader("Rendimiento del modelo en test")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
with m2:
    st.metric("Precision", f"{metrics['precision']:.3f}")
with m3:
    st.metric("Recall", f"{metrics['recall']:.3f}")
with m4:
    st.metric("F1", f"{metrics['f1']:.3f}")
with m5:
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

with st.expander("Matriz de confusión"):
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Real 0", "Real 1"],
        columns=["Predicho 0", "Predicho 1"],
    )
    st.dataframe(cm_df)

with st.expander("Informe de clasificación"):
    st.text(metrics["classification_report"])

with st.expander("Notas metodológicas para auditoría"):
    st.markdown(
        f"""
        - El modelo utilizado es un **Random Forest** dentro de un pipeline de preprocesamiento.
        - Las variables numéricas se imputan con la mediana y se escalan.
        - Las variables categóricas se imputan con la moda y se codifican mediante One-Hot Encoding.
        - SMOTE activo: **{artifact['usar_smote']}**.
        - La probabilidad devuelta debe interpretarse como una **señal de priorización**, no como una decisión automática definitiva.
        - En un contexto bancario real, esta predicción debería combinarse con reglas de negocio, trazabilidad, explicabilidad y revisión humana.
        """
    )
