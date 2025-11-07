# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# === CARGAR MODELO ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fashion_ecommerce_model.h5')

model = load_model()

# === CLASES DE FASHION MNIST ===
class_names = [
    'Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
    'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín'
]

# === PREPROCESAMIENTO DE IMAGEN ===
def preprocess_image(img):
    img = img.resize((96, 96))
    arr = np.array(img).astype('float32')
    
    # Convertir a RGB si es gris
    if len(arr.shape) == 2:
        arr = np.stack([arr]*3, axis=-1)
    # Quitar canal alfa si existe
    if arr.shape[2] == 4:
        arr = arr[:,:,:3]
    
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# === INTERFAZ WEB ===
st.set_page_config(page_title="Fashion E-commerce", page_icon="👗")
st.title("👗 Clasificador de Ropa para E-commerce")
st.write("Sube una foto de ropa y te diré qué es.")

uploaded_file = st.file_uploader("Sube una imagen...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    
    with st.spinner("Analizando..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        st.success(f"**Predicción:** {class_names[predicted_class]}")
        st.metric("Confianza", f"{confidence:.1%}")
        
        # Mostrar todas las probabilidades
        st.write("### Probabilidades por clase:")
        probs = {class_names[i]: f"{p:.1%}" for i, p in enumerate(prediction[0])}
        st.json(probs)

st.markdown("---")
st.caption("Desarrollado con ❤️ usando MobileNetV2 + Streamlit | [GitHub Repo](#)")
