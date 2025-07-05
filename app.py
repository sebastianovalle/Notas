import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Cargar el modelo entrenado
model = joblib.load('mimillon.bin')

# Configurar los LabelEncoders (necesitamos los mismos que se usaron para entrenar)
# Para este ejemplo simple, los creamos manualmente basándonos en cómo se codificaron
encoders = {
    "Horas de Estudio": LabelEncoder(),
    "Asistencia": LabelEncoder(),
    "Resultado": LabelEncoder()
}

# Ajustar los encoders con los datos originales para que puedan decodificar
data_original = {
    "Horas de Estudio": ["Alta", "Baja", "Baja", "Alta", "Alta"],
    "Asistencia": ["Buena", "Buena", "Mala", "Mala", "Buena"],
    "Resultado": ["Sí", "No", "No", "Sí", "Sí"]
}
df_original = pd.DataFrame(data_original)

for col in df_original.columns:
    encoders[col].fit(df_original[col])


st.title("Predicción de Clase")
st.markdown('<p style="color:red;">Elaborado por: Sebastian Ovalle</p>', unsafe_allow_html=True)

st.write("Seleccione los valores de las variables de entrada:")

# Obtener entradas del usuario
horas_estudio = st.selectbox("Horas de Estudio:", ["Alta", "Baja"])
asistencia = st.selectbox("Asistencia:", ["Buena", "Mala"])

# Codificar las entradas del usuario
horas_estudio_encoded = encoders["Horas de Estudio"].transform([horas_estudio])[0]
asistencia_encoded = encoders["Asistencia"].transform([asistencia])[0]

# Preparar los datos para la predicción
input_data = pd.DataFrame([[horas_estudio_encoded, asistencia_encoded]],
                          columns=["Horas de Estudio", "Asistencia"])

# Realizar la predicción
prediction_encoded = model.predict(input_data)[0]

# Decodificar la predicción
prediction = encoders["Resultado"].inverse_transform([prediction_encoded])[0]

st.write("---")

# Mostrar el resultado con emojis
if prediction == "Sí":
    st.subheader("Felicitaciones Aprueba 😊")
else:
    st.subheader("No aprueba 😞")

# Generar requirements.txt
requirements = ["streamlit", "pandas", "sklearn", "joblib"]
with open("requirements.txt", "w") as f:
    for item in requirements:
        f.write(f"{item}\n")

