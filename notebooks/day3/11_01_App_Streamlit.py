import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load('modelo_riesgo_auditoria_bancaria.joblib')

st.title('Predicción de Riesgo de Operación Bancaria')

importe_operacion = st.number_input('Importe de la operación (€)', min_value=0.0, value=10000.0)
antiguedad_cliente = st.number_input('Antigüedad del cliente en meses', min_value=0, value=36)
num_operaciones_30d = st.number_input('Número de operaciones en últimos 30 días', min_value=0, value=15)
ratio_efectivo = st.slider('Ratio de efectivo', min_value=0.0, max_value=1.0, value=0.25)
pais_riesgo = st.selectbox('País de riesgo', [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
alertas_previas = st.number_input('Alertas previas', min_value=0, value=0)
documentacion_completa = st.selectbox('Documentación completa', [1, 0], format_func=lambda x: 'Completa' if x == 1 else 'Incompleta')
canal = st.selectbox('Canal', ['oficina', 'online', 'movil', 'cajero', 'telefonica'])
segmento_cliente = st.selectbox('Segmento', ['retail', 'empresa', 'patrimonial', 'autonomo'])
tipo_operacion = st.selectbox('Tipo de operación', ['transferencia', 'retirada_efectivo', 'ingreso_efectivo', 'pago_tarjeta', 'prestamo'])
umbral = st.slider('Umbral de decisión', min_value=0.05, max_value=0.95, value=0.50)

if st.button('Predecir'):
    nuevo_caso = pd.DataFrame([{
        'importe_operacion': importe_operacion,
        'antiguedad_cliente': antiguedad_cliente,
        'num_operaciones_30d': num_operaciones_30d,
        'ratio_efectivo': ratio_efectivo,
        'pais_riesgo': pais_riesgo,
        'alertas_previas': alertas_previas,
        'documentacion_completa': documentacion_completa,
        'canal': canal,
        'segmento_cliente': segmento_cliente,
        'tipo_operacion': tipo_operacion
    }])
    prob = modelo.predict_proba(nuevo_caso)[:, 1][0]
    clase = int(prob >= umbral)
    st.write(f'Probabilidad de riesgo alto: {prob:.3f}')
    if clase == 1:
        st.error('RIESGO ALTO: revisión prioritaria recomendada')
    else:
        st.success('RIESGO NORMAL/BAJO: monitorización ordinaria')