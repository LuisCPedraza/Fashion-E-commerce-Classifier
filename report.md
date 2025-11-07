# Informe del Proyecto: Clasificador de Imágenes para E-commerce basado en Fashion MNIST

**Autores:** Nicolas González y Luis Carlos Pedraza  
**Fecha:** 6 de noviembre de 2025 

**Descripción general:** Este informe resume el desarrollo de un clasificador de imágenes de prendas de ropa utilizando el dataset Fashion MNIST. Comenzamos con un análisis descriptivo de los datos, implementamos una CNN básica, mejoramos con Transfer Learning usando MobileNetV2, y desplegamos una app web en Streamlit para clasificar imágenes reales. El objetivo es agilizar la gestión de inventario en e-commerce. El modelo final (CNN) alcanza ~92-93% de accuracy en test, mientras que MobileNetV2 logra ~88% en una versión optimizada para RAM limitada. Se incluyen resultados detallados de ejecuciones en Google Colab, gráficos y tablas generados.

## Notebook en Google Colab
**Enlace al código completo ejecutado en Google Colab del entrenamiento CNN:**
https://colab.research.google.com/drive/1Szvz4rff_lv59lcs34KHpzJJz2l3_ujF?usp=sharing

## 1. Introducción
El proyecto utiliza Machine Learning para clasificar imágenes de ropa en 10 categorías: 
- Camiseta/Top
- Pantalón
- Jersey
- Vestido
- Abrigo
- Sandalia
- Camisa
- Zapatilla de deporte
- Bolsa
- Botín.  
  
Tecnologías: TensorFlow/Keras, MobileNetV2, tf.data para eficiencia, Streamlit para la app.  
Dataset: 60,000 imágenes train (28x28 grises), 10,000 test. No hay valores null en los datos.

## 2. Análisis del Dataset
### 2.1 Características
- **Train:** 60,000 imágenes, cada una 28x28 píxeles (784 píxeles en escala de grises, 0-255). Forma: (60000, 785) con 'label' y píxeles.
- **Test:** 10,000 imágenes. Forma: (10000, 785) con 'label' y píxeles.
- **Categorías:** Distribución equilibrada (~6,000 por clase en train, ~1,000 en test).
- Verificación: No hay valores null en train ni test.

### 2.2 Ejemplo de Datos (Head/Tail de Train)
Ejemplo de las primeras 4 y últimas 3 filas del dataset train:

| Index | label | pixel1 | pixel2 | ... | pixel783 | pixel784 | labelName          |
|-------|-------|--------|--------|-----|----------|----------|---------------------|
| 0     | 2     | 0      | 0      | ... | 0        | 0        | Jersey             |
| 1     | 9     | 0      | 0      | ... | 0        | 0        | Botín              |
| 2     | 6     | 0      | 0      | ... | 0        | 0        | Camisa             |
| 3     | 0     | 0      | 0      | ... | 0        | 0        | Camiseta / Top     |
| 59997 | 8     | 0      | 0      | ... | 0        | 0        | Bolsa              |
| 59998 | 8     | 0      | 0      | ... | 0        | 0        | Bolsa              |
| 59999 | 7     | 0      | 0      | ... | 0        | 0        | Zapatilla de deporte |

### 2.3 Tabla de Categorías
| Índice | Categoría            |
|--------|----------------------|
| 0      | Camiseta / Top      |
| 1      | Pantalón            |
| 2      | Jersey              |
| 3      | Vestido             |
| 4      | Abrigo              |
| 5      | Sandalia            |
| 6      | Camisa              |
| 7      | Zapatilla de deporte|
| 8      | Bolsa               |
| 9      | Botín               |

### 2.4 Distribución por Clases (Gráficas en Colab)
- Pie chart para train: Cada clase ~10% (6,000 ítems). Gráfico muestra distribución equilibrada con etiquetas como "Jersey 10.00% (6000)", "Vestido 10.00% (6000)", etc.
- Pie chart para test: Cada clase ~10% (1,000 ítems). Similar al train, equilibrado.
- Distribución después del split (70/30): Train ~10% por clase (e.g., Sandalia 10.04% (4217)), Val ~10% por clase (e.g., Sandalia 9.77% (1759)).

### 2.5 Visualización de Imágenes
- Muestras: Imagen de botín (label 9, train): Matriz de píxeles con valores altos en contornos (e.g., 227, 197, 186).
- Imagen de vestido (label 3, test): Matriz de píxeles con patrones de vestido (e.g., 193, 174, 204).

## 3. Preprocesamiento de Datos
- Normalización: Píxeles / 255.0.
- Reshape: (N, 28, 28) → (N, 96, 96, 3) para MobileNetV2 (resize, gris a RGB, preprocess_input).
- One-hot encoding para etiquetas (e.g., label 0 → [1,0,0,...]).
- Split: 70% train, 30% val usando tf.data.Dataset para evitar OOM.
- Eficiencia: tf.map_fn y prefetch para no agotar RAM.

Tabla de formas después de preprocesamiento:
| Dataset     | Forma X              | Forma Y      |
|-------------|----------------------|--------------|
| Train full  | (60000, 96, 96, 3)  | (60000, 10) |
| Test        | (10000, 96, 96, 3)  | (10000, 10) |
| Train split | (42000, 96, 96, 3)  | (42000, 10) |
| Val split   | (18000, 96, 96, 3)  | (18000, 10) |

## 4. Modelo Inicial (CNN Básica)
- Capas: Conv2D (32,64,128 filtros), LeakyReLU, MaxPooling2D, Dropout (0.3-0.5), Flatten, Dense (128,10).
- Compilación: Adam, categorical_crossentropy, accuracy.
- Entrenamiento: 50 épocas, batch 70. Warning: alpha depreciado en LeakyReLU (usar negative_slope).
- Resultados: Accuracy test 0.9208, loss 0.2558.
- Gráficas: Curvas de loss (baja de ~0.8 a ~0.15 en train, val estable ~0.22), accuracy (sube a ~0.94 en train, ~0.93 val).

Summary del Modelo CNN:
| Layer (type)            | Output Shape      | Param #    |
|-------------------------|-------------------|------------|
| conv2d_6 (Conv2D)       | (None, 28, 28, 32)| 320        |
| leaky_re_lu (LeakyReLU) | (None, 28, 28, 32)| 0          |
| max_pooling2d_4         | (None, 14, 14, 32)| 0          |
| dropout_6               | (None, 14, 14, 32)| 0          |
| conv2d_7                | (None, 14, 14, 64)| 18,496     |
| leaky_re_lu_1           | (None, 14, 14, 64)| 0          |
| max_pooling2d_5         | (None, 7, 7, 64)  | 0          |
| dropout_7               | (None, 7, 7, 64)  | 0          |
| conv2d_8                | (None, 5, 5, 128) | 73,856     |
| leaky_re_lu_2           | (None, 5, 5, 128) | 0          |
| flatten_2               | (None, 3200)      | 0          |
| dense_4                 | (None, 128)       | 409,728    |
| leaky_re_lu_3           | (None, 128)       | 0          |
| dropout_8               | (None, 128)       | 0          |
| dense_5                 | (None, 10)        | 1,290      |

Total params: 503,690 (1.92 MB)

Tabla de Resultados Iniciales:
| Métrica | Train | Val | Test |
|---------|-------|-----|------|
| Accuracy | 0.94 | 0.93 | 0.92 |
| Loss    | 0.15 | 0.22 | 0.26 |

Matriz de Confusión: Diagonal alta (e.g., Camiseta 892 correctas), errores como Jersey confundido con Camisa (54).

## 5. Modelo Mejorado (MobileNetV2 + Transfer Learning)
- Input: 96x96x3.
- Base: MobileNetV2 pre-entrenado, congelado.
- Top: GlobalAveragePooling2D, Dropout 0.3, Dense 10.
- Compilación: Adam (lr=0.001), categorical_crossentropy.
- Entrenamiento: 7 épocas (RAM limitada), batch 32, tf.data.Dataset.
- Resultados: Val accuracy ~0.90, Test accuracy ~0.8831.

Summary del Modelo MobileNetV2:
| Layer (type)                | Output Shape        | Param #     |
|-----------------------------|---------------------|-------------|
| input_layer_1 (InputLayer)  | (None, 96, 96, 3)  | 0           |
| mobilenetv2_1.00_96         | (None, 3, 3, 1280) | 2,257,984   |
| global_average_pooling2d    | (None, 1280)       | 0           |
| dropout                     | (None, 1280)       | 0           |
| dense                       | (None, 10)         | 12,810      |

Total params: 2,270,794 (8.66 MB)  
Trainable params: 12,810 (50.04 KB)  
Non-trainable params: 2,257,984 (8.61 MB)

Tabla de Métricas por Época (MobileNetV2, resumida):
| Época | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1     | 0.7676    | 0.6637     | 0.8814  | 0.3208   |
| 2     | 0.8677    | 0.3631     | 0.8836  | 0.3168   |
| 3     | 0.8742    | 0.3472     | 0.8936  | 0.2942   |
| 4     | 0.8812    | 0.3326     | 0.8968  | 0.2856   |
| 5     | 0.8839    | 0.3251     | 0.8894  | 0.3197   |
| 6     | 0.8854    | 0.3229     | 0.8992  | 0.2818   |
| 7     | 0.8862    | 0.3150     | 0.8958  | 0.2933   |

- Predicciones: Correctas ~9208/10000 (92.08%).
- Matriz de Confusión: Alta precisión diagonal, errores menores (e.g., 792 erróneas totales).

## 6. App Web (Streamlit)
- Frontend: Formulario para subir imagen + nombre del producto.
- Backend: Carga modelo .h5, preprocesa imagen, predice.
- Base de Datos: SQLite para guardar nombre, categoría, imagen, fecha.
- Deploy: En Streamlit Cloud (gratis): [https://fashion-e-commerce-classifier-jt8l29wh6fopdjvxnammps.streamlit.app/](https://fashion-e-commerce-classifier-jt8l29wh6fopdjvxnammps.streamlit.app/)
- Funcionalidad: Sube foto → Clasifica → Guarda en inventario → Muestra lista con fotos.

Ejemplo de Tabla de Inventario (generada en app):
| Nombre | Categoría | Fecha | Imagen |
|--------|-----------|-------|--------|
| Ejemplo1 | Camiseta | 2025-11-06 | (Foto) |

## 7. Conclusiones y Mejoras
- El modelo inicial (CNN) funciona bien (~92%), pero MobileNetV2 es más robusto para imágenes reales (~88% test en versión limitada).
- Corregimos errores: append → concat, LeakyReLU como capa, dimensiones en preprocesamiento, OOM con tf.data.
- Exposición: Dataset equilibrado (pie charts muestran ~10% por clase), preprocesamiento clave, Transfer Learning mejora eficiencia, app práctica para e-commerce.
- Mejoras futuras: Fine-tuning completo, dataset personalizado (fotos reales), integración con base de datos cloud.
