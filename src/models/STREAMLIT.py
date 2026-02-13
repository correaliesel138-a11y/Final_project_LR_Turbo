import streamlit as st
import pandas as pd
from pickle import load

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Calidad de Vino",
    page_icon="🍷",
    layout="wide"
)

# Cargar los modelos y datos guardados
@st.cache_resource
def load_models():
    knn_model = load(open("models/knn_wine_model.sav", "rb"))
    scaler = load(open("models/knn_wine_scaler.sav", "rb"))
    return knn_model, scaler

# Cargar modelos
knn_final, scaler = load_models()

# Título y descripción
st.title("🍷 Predictor de Calidad de Vino")
st.markdown("""
Esta aplicación predice la calidad de un vino tinto basándose en sus características químicas.
Ingresa los valores de las características del vino y obtén una predicción.
""")

# Crear dos columnas para mejor organización
col1, col2 = st.columns(2)

with col1:
    st.subheader("Características del Vino")
    
    fixed_acidity = st.number_input(
        "Acidez Fija (fixed acidity)", 
        min_value=0.0, 
        max_value=20.0, 
        value=8.0, 
        step=0.1
    )
    
    volatile_acidity = st.number_input(
        "Acidez Volátil (volatile acidity)", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.5, 
        step=0.01
    )
    
    citric_acid = st.number_input(
        "Ácido Cítrico (citric acid)", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.3, 
        step=0.01
    )
    
    residual_sugar = st.number_input(
        "Azúcar Residual (residual sugar)", 
        min_value=0.0, 
        max_value=20.0, 
        value=2.5, 
        step=0.1
    )
    
    chlorides = st.number_input(
        "Cloruros (chlorides)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.08, 
        step=0.001
    )
    
    free_sulfur_dioxide = st.number_input(
        "Dióxido de Azufre Libre (free sulfur dioxide)", 
        min_value=0.0, 
        max_value=100.0, 
        value=15.0, 
        step=1.0
    )

with col2:
    st.subheader("Características Adicionales")
    
    total_sulfur_dioxide = st.number_input(
        "Dióxido de Azufre Total (total sulfur dioxide)", 
        min_value=0.0, 
        max_value=300.0, 
        value=46.0, 
        step=1.0
    )
    
    density = st.number_input(
        "Densidad (density)", 
        min_value=0.99, 
        max_value=1.01, 
        value=0.9968, 
        step=0.0001
    )
    
    pH = st.number_input(
        "pH", 
        min_value=2.5, 
        max_value=4.5, 
        value=3.3, 
        step=0.01
    )
    
    sulphates = st.number_input(
        "Sulfatos (sulphates)", 
        min_value=0.0, 
        max_value=2.0, 
        value=0.65, 
        step=0.01
    )
    
    alcohol = st.number_input(
        "Alcohol (%)", 
        min_value=8.0, 
        max_value=15.0, 
        value=10.5, 
        step=0.1
    )

# Botón de predicción
st.markdown("---")
if st.button("🔮 Predecir Calidad del Vino", type="primary"):
    # Crear array con las características
    notes = [
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]
    
    # Escalar y predecir
    notes_scaled = scaler.transform([notes])
    prediction = knn_final.predict(notes_scaled)[0]
    
    # Mensajes de predicción
    messages = {
        0: "🍷 El vino es de **BAJA CALIDAD**",
        1: "🍷 El vino es de **CALIDAD MEDIA**",
        2: "🍷 El vino es de **ALTA CALIDAD**"
    }
    
    # Mostrar resultado
    st.markdown("### Resultado de la Predicción:")
    
    if prediction == 0:
        st.error(messages[prediction])
    elif prediction == 1:
        st.warning(messages[prediction])
    else:
        st.success(messages[prediction])
    
    # Mostrar características ingresadas en un dataframe
    with st.expander("Ver características ingresadas"):
        features_df = pd.DataFrame([notes], columns=[
            'Acidez Fija', 'Acidez Volátil', 'Ácido Cítrico', 
            'Azúcar Residual', 'Cloruros', 'SO2 Libre', 
            'SO2 Total', 'Densidad', 'pH', 'Sulfatos', 'Alcohol'
        ])
        st.dataframe(features_df)

# Información adicional en el sidebar
st.sidebar.title("ℹ️ Información")
st.sidebar.markdown("""
### Sobre el modelo
Este modelo utiliza el algoritmo K-Nearest Neighbors (KNN) 
para clasificar vinos tintos en tres categorías de calidad.

### Características
El modelo analiza 11 características químicas del vino:
- Acidez fija
- Acidez volátil
- Ácido cítrico
- Azúcar residual
- Cloruros
- Dióxido de azufre libre
- Dióxido de azufre total
- Densidad
- pH
- Sulfatos
- Alcohol
""")