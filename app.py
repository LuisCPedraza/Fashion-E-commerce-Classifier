# app.py (VERSIÓN FINAL 100% FUNCIONAL)
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import sqlite3
from datetime import datetime
import io
import os

# === CONFIGURACIÓN ===
DB_NAME = "inventario.db"
MODEL_PATH = "fashion_ecommerce_model.h5"

# === BASE DE DATOS ===
def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Crear tabla base
    c.execute('''
        CREATE TABLE IF NOT EXISTS productos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT,
            categoria TEXT,
            fecha TEXT,
            imagen BLOB
        )
    ''')
    
    # Agregar 'confianza' si no existe
    try:
        c.execute("ALTER TABLE productos ADD COLUMN confianza REAL")
    except sqlite3.OperationalError:
        pass  # Ya existe
    
    conn.commit()
    conn.close()

# === GUARDAR PRODUCTO ===
def save_product(nombre, categoria, confianza, imagen):
    conn = get_connection()
    c = conn.cursor()
    img_bytes = imagen.read()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("PRAGMA table_info(productos)")
    columns = [col[1] for col in c.fetchall()]
    
    if 'confianza' in columns:
        c.execute("INSERT INTO productos (nombre, categoria, confianza, fecha, imagen) VALUES (?, ?, ?, ?, ?)",
                  (nombre, categoria, confianza, fecha, img_bytes))
    else:
        c.execute("INSERT INTO productos (nombre, categoria, fecha, imagen) VALUES (?, ?, ?, ?)",
                  (nombre, categoria, fecha, img_bytes))
    
    conn.commit()
    conn.close()

# === CARGAR INVENTARIO ===
def load_inventory():
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("PRAGMA table_info(productos)")
    columns = [col[1] for col in c.fetchall()]
    has_confianza = 'confianza' in columns
    
    if has_confianza:
        c.execute("SELECT id, nombre, categoria, confianza, fecha, imagen FROM productos ORDER BY fecha DESC")
    else:
        c.execute("SELECT id, nombre, categoria, NULL, fecha, imagen FROM productos ORDER BY fecha DESC")
    
    rows = c.fetchall()
    conn.close()
    return rows, has_confianza

# === CARGAR MODELO ===
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modelo no encontrado: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

# === PREPROCESAMIENTO ===
def preprocess_image(img):
    img = img.resize((96, 96))
    arr = np.array(img).astype('float32')
    if len(arr.shape) == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# === INICIALIZAR ===
init_db()

# === INTERFAZ ===
st.set_page_config(page_title="Fashion E-commerce", page_icon="clothing", layout="wide")
st.title("Clasificador de Ropa para E-commerce")
st.markdown("**Sube una foto → Clasifica → Agrega al inventario**")

tab1, tab2 = st.tabs(["Subir Producto", "Inventario"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])
        nombre = st.text_input("Nombre del producto", placeholder="Ej: Camiseta negra talla M")
    
    with col2:
        if uploaded_file and nombre:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen subida", use_column_width=True)

            with st.spinner("Clasificando..."):
                processed = preprocess_image(image)
                pred = model.predict(processed, verbose=0)
                cat_idx = np.argmax(pred[0])
                confianza = pred[0][cat_idx]

                st.success(f"**Predicción:** {class_names[cat_idx]}")
                st.metric("Confianza", f"{confianza:.1%}")

                st.subheader("Probabilidades")
                probs = {class_names[i]: f"{p:.1%}" for i, p in enumerate(pred[0])}
                st.json(probs)

                if st.button("Agregar al Inventario", type="primary"):
                    save_product(nombre, class_names[cat_idx], confianza, uploaded_file)
                    st.success("¡Producto agregado!")
                    st.balloons()

with tab2:
    st.subheader("Inventario Actual")
    inventory_data = load_inventory()
    inventory = inventory_data[0]
    has_confianza = inventory_data[1]
    
    if inventory:
        for idx, (id_prod, nombre, cat, conf, fecha, img_blob) in enumerate(inventory):
            col1, col2 = st.columns([1, 4])
            with col1:
                if img_blob:
                    img = Image.open(io.BytesIO(img_blob))
                    st.image(img, width=120)
            with col2:
                st.write(f"**{nombre}**")
                st.write(f"**Categoría:** {cat}")
                # SOLUCIÓN: MOSTRAR CONFIANZA SOLO SI ES VÁLIDA
                if has_confianza and conf is not None and isinstance(conf, (int, float)):
                    st.write(f"**Confianza:** {conf * 100:.1f}%")
                else:
                    st.write("**Confianza:** No disponible")
                st.write(f"**Fecha:** {fecha}")
            if idx < len(inventory) - 1:
                st.divider()
    else:
        st.info("No hay productos en el inventario aún.")

# === FOOTER ===
st.markdown("---")
st.caption("Desarrollado con ❤️ usando MobileNetV2 + Streamlit | [GitHub Repo](#)")
st.caption("Desarrollado en **Zarzal, Valle del Cauca** | Noviembre 2025 | [GitHub](#)")
