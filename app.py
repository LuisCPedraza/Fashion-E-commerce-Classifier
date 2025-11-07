# app.py (VERSIÓN CORREGIDA CON PREDICCIONES + INVENTARIO + MEJOR LAYOUT)
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import sqlite3
from datetime import datetime
import io  # Para mostrar imágenes desde BLOB

# === BASE DE DATOS ===
DB_NAME = "inventario.db"

def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)  # Para Streamlit multi-thread
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS productos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT,
            categoria TEXT,
            confianza REAL,
            fecha TEXT,
            imagen BLOB
        )
    ''')
    conn.commit()
    return conn

def save_product(conn, nombre, categoria, confianza, imagen):
    c = conn.cursor()
    img_bytes = imagen.read()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO productos (nombre, categoria, confianza, fecha, imagen) VALUES (?, ?, ?, ?, ?)",
              (nombre, categoria, confianza, fecha, img_bytes))
    conn.commit()

def load_inventory(conn):
    c = conn.cursor()
    c.execute("SELECT id, nombre, categoria, confianza, fecha, imagen FROM productos ORDER BY fecha DESC")
    rows = c.fetchall()
    return rows

# === CARGAR MODELO ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fashion_ecommerce_model.h5')

model = load_model()

class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

# === PREPROCESAMIENTO ===
def preprocess_image(img):
    img = img.resize((96, 96))
    arr = np.array(img).astype('float32')
    if len(arr.shape) == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:,:,:3]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# === INICIALIZAR ===
conn = init_db()

# === INTERFAZ ===
st.title("👗 Clasificador de Ropa para E-commerce")
st.write("Sube una foto, clasifica y agrega al inventario.")

# Usar tabs para mejor distribución
tab1, tab2 = st.tabs(["📤 Subir Producto", "📋 Inventario"])

with tab1:
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])
    nombre = st.text_input("Nombre del producto", placeholder="Ej: Camiseta roja")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)

        with st.spinner("Clasificando..."):
            processed = preprocess_image(image)
            pred = model.predict(processed)
            cat_idx = np.argmax(pred[0])
            confianza = pred[0][cat_idx] * 100  # Porcentaje

            st.success(f"**Predicción: {class_names[cat_idx]}** ({confianza:.1f}%)")

            # Mostrar probabilidades detalladas
            st.subheader("Probabilidades por clase")
            probs = {class_names[i]: f"{p*100:.1f}%" for i, p in enumerate(pred[0])}
            st.json(probs)

            if st.button("Agregar al Inventario"):
                save_product(conn, nombre, class_names[cat_idx], confianza, uploaded_file)
                st.success("¡Producto agregado exitosamente!")

with tab2:
    inventory = load_inventory(conn)
    if inventory:
        st.subheader("Inventario Actual")
        for id_prod, nombre, cat, conf, fecha, img_blob in inventory:
            col1, col2 = st.columns([1, 3])
            with col1:
                # Mostrar imagen desde BLOB
                if img_blob:
                    img = Image.open(io.BytesIO(img_blob))
                    st.image(img, width=100)
            with col2:
                st.write(f"**{nombre}** - {cat} ({conf:.1f}%)")
                st.write(f"Fecha: {fecha}")
            st.divider()
    else:
        st.info("Aún no hay productos en el inventario.")

st.markdown("---")
st.caption("Desarrollado con ❤️ usando MobileNetV2 + Streamlit | [GitHub Repo](#)")
