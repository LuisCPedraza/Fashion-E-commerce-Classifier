# app.py (VERSIÓN CON INVENTARIO)
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import sqlite3
from datetime import datetime
import os

# === BASE DE DATOS ===
DB_NAME = "inventario.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS productos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT,
            categoria TEXT,
            fecha TEXT,
            imagen BLOB
        )
    ''')
    conn.commit()
    conn.close()

def save_product(nombre, categoria, imagen):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    img_bytes = imagen.read()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO productos (nombre, categoria, fecha, imagen) VALUES (?, ?, ?, ?)",
              (nombre, categoria, fecha, img_bytes))
    conn.commit()
    conn.close()

def load_inventory():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT nombre, categoria, fecha FROM productos ORDER BY fecha DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# === INICIALIZAR ===
init_db()
model = tf.keras.models.load_model('fashion_ecommerce_model.h5')
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

# === INTERFAZ ===
st.title("Clasificador de Ropa para E-commerce")
st.write("Sube una foto y agrega al inventario.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Subir Producto")
    uploaded_file = st.file_uploader("Imagen", type=['png', 'jpg', 'jpeg'])
    nombre = st.text_input("Nombre del producto")

    if uploaded_file and nombre:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", width=200)
        
        with st.spinner("Clasificando..."):
            processed = preprocess_image(image)
            pred = model.predict(processed)
            cat_idx = np.argmax(pred[0])
            confianza = pred[0][cat_idx]
            
            st.success(f"**{class_names[cat_idx]}** ({confianza:.1%})")
            
            if st.button("Agregar al Inventario"):
                save_product(nombre, class_names[cat_idx], uploaded_file)
                st.success("Producto agregado!")

with col2:
    st.subheader("Inventario Actual")
    inventory = load_inventory()
    if inventory:
        for nombre, cat, fecha in inventory:
            st.write(f"**{nombre}** - {cat} - {fecha}")
    else:
        st.info("Aún no hay productos.")

st.markdown("---")
st.caption("Desarrollado con ❤️ usando MobileNetV2 + Streamlit | [GitHub Repo](#)")
