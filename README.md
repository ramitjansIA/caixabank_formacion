<p align="center">
  <img src="https://www.upc.edu/comunicacio/ca/identitat/descarrega-arxius-grafics/fitxers-marca-principal/upc-positiu-p3005.png" height="60"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://www.hipotecalowcost.com/wp-content/uploads/2019/08/Logo-CaixaBank.png" height="60"/>
</p>

<h1 align="center">Model Risk & Data Science Training</h1>
<h3 align="center">CaixaBank · Advanced Analytics Program</h3>

---

## 📌 Descripción

Este repositorio contiene el material completo del **Curso de Auditoría de Modelos en Python**, diseñado para la formación de perfiles analíticos en entornos financieros.

El curso combina:

* fundamentos teóricos sólidos
* desarrollo práctico paso a paso
* enfoque aplicado a casos reales de riesgo y modelización

El objetivo principal es capacitar al participante para **entender, construir, evaluar y auditar modelos predictivos**, con especial foco en su uso dentro del sector bancario.

---

## 🎯 Objetivos del curso

Al finalizar el curso, el participante será capaz de:

* Comprender el flujo completo de un problema de machine learning supervisado
* Implementar modelos clásicos y avanzados en Python
* Evaluar modelos con métricas adecuadas al contexto de negocio
* Interpretar resultados y justificar decisiones
* Detectar errores metodológicos (data leakage, sesgos, mala validación)
* Analizar modelos desde una perspectiva de auditoría

---

## 🧱 Estructura del curso

### 📅 Día 1 · Fundamentos del Machine Learning

* Introducción a modelos supervisados
* Regresión logística (GLM)
* Árboles de decisión
* Random Forest

---

### 📅 Día 2 · Modelos avanzados e interpretabilidad

* Recapitulación y corrección de ejercicios
* Boosting: AdaBoost, XGBoost, CatBoost
* Support Vector Machine (SVM)
* Evaluación de modelos
* Interpretabilidad
* Redes neuronales (ANN)

---

### 📅 Día 3 · Deep Learning

* Recapitulación y corrección de ejercicios
* Redes neuronales convolucionales (CNN)
* Transfer Learning
* End-to-end: Aplicación del modelaje
  
---

## 📂 Contenido del repositorio

```bash
├── data/                # Input datasets (CSV format)
├── notebooks/
│ ├── day1/              # Foundational models
│ ├── day2/              # Advanced models and evaluation
│ └── day3/              # Deep Learning
├── assets/              # Visual resources (logos, figures)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Requisitos

Este curso está desarrollado en Python 3.10+.

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

Principales librerías utilizadas:

* pandas
* numpy
* scikit-learn
* matplotlib / seaborn
* xgboost
* catboost
* shap
* tensorflow / keras

---

## ▶️ Uso

1. Clonar el repositorio:

```bash
git clone <repo_url>
cd <repo_name>
```

2. Abrir el proyecto en Visual Studio Code:

<p align="center">
  <img src="https://images-eds-ssl.xboxlive.com/image?url=4rt9.lXDC4H_93laV1_eHHFT949fUipzkiFOBH3fAiZZUCdYojwUyX2aTonS1aIwMrx6NUIsHfUHSLzjGJFxxj7kCzMIlSC20SNjaJf9GmESvWFqgy6FNrwzWSIu2lzePyWSz8zg09RAX43OFexidzEE3_7l3auaKk4w9ktJdqg-&format=source" 
       width="15%"/>
</p>

```bash
code .
```

3. Configurar el entorno de Python:

* Asegúrate de tener instalada la extensión **Python** y **Jupyter** en VS Code
* Selecciona el intérprete de Python adecuado (`Ctrl + Shift + P → Python: Select Interpreter`)
* Instala las dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecutar los notebooks:

* Abre cualquier archivo `.ipynb` desde VS Code
* Ejecuta las celdas de forma secuencial utilizando el botón `Run` o `Shift + Enter`

5. Seguir el orden recomendado:

* Día 1 → fundamentos
* Día 2 → modelos avanzados
* Día 3 → deep learning

---

💡 **Recomendación:**
Trabajar siempre con un entorno virtual (`venv` o `conda`) para asegurar reproducibilidad y evitar conflictos de dependencias.


## 🧪 Metodología

El enfoque del curso es **hands-on**:

* Desarrollo paso a paso
* Código explícito (sin abstracciones innecesarias)
* Ejercicios prácticos en cada bloque
* Resolución guiada
* Comparación entre modelos

---

## 📊 Filosofía del curso

Más allá del rendimiento del modelo, este curso pone énfasis en:

* la **interpretabilidad**
* la **robustez metodológica**
* la **trazabilidad del pipeline**
* la **coherencia con negocio**

Porque en entornos financieros:

> *Un modelo no solo debe funcionar, debe poder explicarse.*

---

## ⚠️ Consideraciones importantes

* Los notebooks asumen la existencia de un dataset con variable objetivo `default`
* No se generan datos sintéticos en esta versión
* Es responsabilidad del usuario adaptar nombres de columnas si es necesario

---

## 👨‍🏫 Público objetivo

* Analistas de datos
* Perfiles de riesgo
* Auditores de modelos
* Data scientists junior/intermediate
* Profesionales del sector financiero

---

## 📬 Contacto

Para dudas, mejoras o sugerencias, se recomienda utilizar los issues del repositorio. También puedes enviar un correo a sergi.ramirez@upc.edu

---

## 📄 Licencia

Este material está destinado a uso formativo interno.
Para otros usos, consultar con los responsables del curso.

---

**CaixaBank · Data Science & AI**
