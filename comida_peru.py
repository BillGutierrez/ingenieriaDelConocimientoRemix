#!/usr/bin/env python
# coding: utf-8

# # Pipeline CNN completo: clasificación de platos típicos peruanos (PeruFoodNet)
#
# **Universidad Nacional del Centro del Perú**
#
# **Facultad de Ingeniería de Sistemas**
#
# **Curso:** Ingeniería del Conocimiento (ISO56B)
#
# **Docente:** Msc. Jaime Antonio Huaytalla Pariona
#
# **Semestre:** 2026-II
# **Fecha:** 08-11-2026
#
# **Integrantes:**
#
# | Nombre | Código |
# |---|---|
# | Gutierrez Vega Bill Toño | 2012100229F |
#
# ---
# **Estructura del notebook (pipeline de 12 fases):**
# 1. Definición del problema
# 2. Selección y validación del dataset
# 3. EDA de imágenes
# 4. Preprocesamiento
# 5. Arquitecturas CNN
# 6. Entrenamiento
# 7. Evaluación rigurosa
# 8. Análisis de errores
# 9. Comparación sistemática de modelos
# 10. Versionado del proyecto
# 11. Logging y registro de experimentos
# 12. Documentación técnica y despliegue

# # Fase 1: Definición del problema
#
# Antes de escribir código se responden cinco preguntas que condicionan todas las decisiones posteriores: la capa de salida, la función de pérdida, las métricas, el tamaño del modelo y el criterio para saber si el modelo realmente sirve.
#
# ## 1.1 ¿Qué problema resuelvo?
#
# Dada una fotografía de un plato de comida, el modelo debe identificar a cuál de **40 platos típicos peruanos** corresponde (por ejemplo, Ceviche, Anticuchos, Lomo Saltado, Ají de Gallina).
#
# **Contexto y utilidad:** la gastronomía peruana atrae turistas de todo el mundo, pero algunos platos contienen ingredientes que pueden causar reacciones alérgicas. Reconocer el plato a partir de una foto permite asociarlo a su lista de ingredientes y, a futuro, avisar de posibles alérgenos mediante una aplicación o un chatbot. Esa es justamente la motivación de los autores del dataset.
#
# ## 1.2 ¿Es clasificación binaria o multiclase?
#
# **Clasificación multiclase:** cada imagen pertenece a **una sola** de las 40 categorías.
#
# Esto define el diseño del modelo:
# - **Capa de salida:** `Dense(40, activation="softmax")`, una neurona por plato; `softmax` convierte las 40 salidas en probabilidades que suman 1.
# - **Función de pérdida:** entropía cruzada categórica (la variante exacta, con etiquetas enteras o con *label smoothing*, se fija en la Fase 6).
#
# ## 1.3 ¿Cuál es la métrica de éxito?
#
# | Métrica | Para qué se usa |
# |---|---|
# | **Macro F1-score** (principal) | Promedia el F1 de las 40 clases por igual, de modo que ningún plato queda oculto detrás de un promedio global. |
# | Accuracy global | Referencia general e intuitiva. |
# | Accuracy top-3 | Útil con 40 clases parecidas: indica si el plato correcto está entre las 3 predicciones más probables. |
# | AUC macro (One-vs-Rest) | Mide qué tan bien se separa cada plato del resto, sin depender de un umbral. |
# | Matriz de confusión y reporte por clase | Identifican qué platos se confunden entre sí (Fases 7 y 8). |
#
# **Nota:** el dataset está perfectamente balanceado (100 imágenes por plato), por lo que accuracy y Macro F1 serán parecidos; aun así se reporta Macro F1 como métrica principal por ser la que exige el pipeline profesional.
#
# ## 1.4 ¿Cuáles son las restricciones?
#
#
# | Restricción | Valor |
# |---|---|
# | **Tamaño del modelo** | Menor a 50 MB, para poder exportarlo a TFLite y usarlo en un dispositivo móvil. |
# | **Tiempo de inferencia** | Una predicción por imagen en menos de 1 segundo en CPU. |
# | **Hardware y entorno** | GPU local NVIDIA GeForce RTX 5060 Laptop (8 GB), con Keras 3 y PyTorch como backend. |
# | **Datos** | Pocas imágenes por clase (100), por lo que se necesitará *data augmentation* (Fase 4). |
# | **Despliegue** | Modelo `.keras`, API REST de predicción con Django REST Framework y, si es viable, exportación a TFLite/ONNX para móvil (Fase 12). |
#
# ## 1.5 ¿Cuál es la línea base (baseline)?
#
# Un clasificador trivial que adivina al azar entre 40 clases acierta con probabilidad **1/40 = 2.5%**. Cualquier modelo útil debe superar con claridad ese valor. Más adelante la CNN base (Fase 5) servirá como línea base de modelo para comparar las arquitecturas mejoradas.
#
# ## 1.6 Ficha resumen del problema (*Case Study Card*)
#
# | Elemento | Definición |
# |---|---|
# | **Tipo** | Clasificación multiclase (40 categorías) |
# | **Dataset** | PeruFoodNet (4,000 imágenes, 100 por clase) |
# | **Fuente** | Mendeley Data, DOI: 10.17632/hxhbbm497d.3 |
# | **Referencia** | Arzola Gutierrez, Canchari Muñoz y Escobedo Cárdenas (2025), *Data in Brief*, vol. 60. DOI: 10.1016/j.dib.2025.111604 |
# | **Métrica principal** | Macro F1-score |
# | **Baseline** | 2.5% de accuracy (azar: 1/40) |
# | **Restricciones** | Modelo < 50 MB, inferencia < 1 s en CPU, GPU de 8 GB, backend PyTorch |

# # Fase 2: Selección y validación del dataset
#
# ## 2.1 Criterios mínimos de aceptación
#
# Antes de aceptar un dataset se comprueba que cumpla los criterios mínimos de la guía del curso. Los repositorios recomendados son Kaggle, TensorFlow Datasets, Hugging Face y Roboflow Universe; aquí se usó **Mendeley Data**, que aloja datasets publicados junto con artículos científicos.
#
# | Criterio | Requisito | PeruFoodNet | Cumple |
# |---|---|---|---|
# | **Volumen** | ≥ 1,000 imágenes en total | 4,000 imágenes | Sí |
# | **Complejidad** | ≥ 3 clases diferentes | 40 platos | Sí |
# | **Viabilidad** | Contexto claro y aplicabilidad al mundo real | Reconocimiento de platos típicos e inferencia de alérgenos | Sí |
# | **Exclusiones** | No usar MNIST ni Fashion-MNIST | No es ninguno de los dos | Sí |
#
# Los números de volumen y clases se **verifican con código** en la sección 2.6, en lugar de confiar solo en la descripción del autor.
#
# ## 2.2 Dataset seleccionado
#
# **PeruFoodNet** es un dataset de 4,000 imágenes de 40 platos típicos peruanos, con 100 imágenes por plato, tomadas desde distintos ángulos, entornos e iluminaciones. Fue creado por estudiantes de Ingeniería de Sistemas de la Universidad de Lima.
#
# - **Fuente:** https://data.mendeley.com/datasets/hxhbbm497d/3
# - **DOI:** 10.17632/hxhbbm497d.3
# - **Artículo:** Arzola Gutierrez, Canchari Muñoz y Escobedo Cárdenas (2025), *Data in Brief*, vol. 60. DOI: 10.1016/j.dib.2025.111604
#
# **Justificación de la elección:** es un problema de clasificación de imágenes reales con contexto local, cumple todos los criterios mínimos y tiene una aplicación concreta (reconocer el plato para asociarlo a sus ingredientes y posibles alérgenos). Con 40 clases, hay platos visualmente parecidos, lo que permite un análisis de errores significativo en las Fases 8 y 9. Su limitación es el tamaño: 100 imágenes por clase es poco para una CNN entrenada desde cero, por lo que se usará *data augmentation* (Fase 4).
#
# ## 2.3 Descarga y estructura de carpetas
#
# El dataset se descargó manualmente desde la página de Mendeley Data (botón de descarga) y se descomprimió en la carpeta `Dataset_PeruFoodNet/` dentro de la carpeta del proyecto. Hay una carpeta por plato; el nombre de la carpeta es la **clase** de cada imagen.
#
# ```
# proyecto_comida/
# ├── Dataset_PeruFoodNet/
# │   ├── (carpeta de un plato)/   ← 100 imágenes
# │   └── ... (40 carpetas)
# └── comida_peru.ipynb
# ```

# ## 2.4 Configuración del entorno
#
# `KERAS_BACKEND` indica qué motor hace los cálculos de la red (aquí PyTorch, porque soporta la GPU RTX 5060); debe definirse **antes** de importar Keras. `keras.utils.set_random_seed` fija la semilla de Python, NumPy y del backend para que los resultados sean reproducibles. `torch.cuda.is_available()` confirma que la GPU está activa.

# In[1]:


import os
os.environ["KERAS_BACKEND"] = "torch"   # debe ir ANTES de importar keras

import keras, torch

SEED = 42
keras.utils.set_random_seed(SEED)

print("Keras:", keras.__version__, "| backend:", keras.backend.backend())
print("GPU disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguna")


# ## 2.5 Listar las imágenes y sus etiquetas
#
# Se recorre `DATA_DIR` con `os.walk` (visita todas las subcarpetas). El nombre de la carpeta que contiene cada imagen es su **clase**. Se arma una tabla (`DataFrame`) con la ruta y la clase de cada imagen.
#
# - `Path` (de `pathlib`) maneja rutas de archivos de forma independiente del sistema operativo.
# - `PROYECTO` es la carpeta desde la que se abre el notebook (`Path.cwd()`), es decir, `proyecto_comida/`.
# - `EXT` es el conjunto de extensiones que se consideran imágenes; los demás archivos se cuentan en `otros` y se ignoran.
# - `CLASES` es la lista ordenada de platos y `clase_a_id` asigna a cada uno un número entero (0 a 39), que se usará más adelante como etiqueta.

# In[2]:


from pathlib import Path

import numpy as np
import pandas as pd

PROYECTO = Path.cwd()
DATA_DIR = PROYECTO / "Dataset_PeruFoodNet"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset en: {DATA_DIR}\n"
        "Descárgalo desde Mendeley Data y colócalo en la carpeta Dataset_PeruFoodNet."
    )
print("Proyecto:", PROYECTO)

registros, otros = [], 0
for carpeta, _, archivos in os.walk(DATA_DIR):
    for a in archivos:
        if Path(a).suffix.lower() in EXT:
            registros.append((str(Path(carpeta) / a), Path(carpeta).name))
        else:
            otros += 1

df_total = pd.DataFrame(registros, columns=["ruta", "clase"])
CLASES = sorted(df_total["clase"].unique())
clase_a_id = {c: i for i, c in enumerate(CLASES)}

print("Imágenes encontradas:", len(df_total), "| Clases:", len(CLASES))
print("Archivos que no son imagen (ignorados):", otros)
muestras_por_clase = {
    clase: (
        df_total.loc[df_total["clase"].eq(clase), "ruta"]
        .map(lambda ruta: Path(ruta).name)
        .sort_values()
        .head(10)
        .reset_index(drop=True)
    )
    for clase in CLASES
}
tabla_muestras = pd.DataFrame(muestras_por_clase)
display(tabla_muestras)


# ## 2.6 Validación del dataset
#
# Se comprueba con código que el dataset descargado coincide con lo que describe el autor: 4,000 imágenes, 40 clases y 100 imágenes por clase. `value_counts` cuenta las apariciones de cada clase y `reindex` las ordena según `CLASES`. Si alguna comprobación falla, hay que revisar la descarga o la estructura de carpetas antes de continuar.

# In[3]:


conteo = df_total["clase"].value_counts().reindex(CLASES)

verificaciones = {
    "Volumen >= 1,000 imágenes": len(df_total) >= 1000,
    "Complejidad >= 3 clases": len(CLASES) >= 3,
    "Total esperado: 4,000 imágenes": len(df_total) == 4000,
    "Clases esperadas: 40": len(CLASES) == 40,
    "100 imágenes por clase": bool((conteo == 100).all()),
}
for criterio, ok in verificaciones.items():
    print(f"[{'OK' if ok else 'FALLA'}] {criterio}")

print("\nImágenes por clase:")
print(conteo.to_string())

anomalas = conteo[conteo != 100]
if len(anomalas):
    print("\nClases con un número distinto de 100 imágenes:")
    print(anomalas.to_string())


# # Fase 3: EDA para imágenes
#
# El análisis exploratorio (EDA) revisa el dataset **antes** de entrenar, para descubrir problemas que afectarían al modelo: clases desbalanceadas, imágenes dañadas o repetidas, tamaños muy distintos, fotos muy oscuras o desenfocadas. Lo que se encuentre aquí guía las decisiones de preprocesamiento de la Fase 4.
#
# ## 3.1 Distribución de clases
#
# Se cuenta cuántas imágenes tiene cada plato y se calcula el **ratio de desbalance** (clase más grande dividida entre la más pequeña). Un dataset muy desbalanceado puede llevar al modelo a favorecer a las clases mayoritarias y a producir una accuracy global engañosa.

# In[4]:


import time, hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm.auto import tqdm

conteo = df_total["clase"].value_counts().reindex(CLASES)
ratio = conteo.max() / conteo.min()
print(f"Mínimo: {conteo.min()} ({conteo.idxmin()}) | Máximo: {conteo.max()} ({conteo.idxmax()})")
print(f"Ratio de desbalance: {ratio:.1f}:1")

plt.figure(figsize=(14, 5))
plt.bar(CLASES, conteo.values)
plt.xticks(rotation=90)
plt.ylabel("Número de imágenes")
plt.title("Distribución por clase (dataset original)")
plt.tight_layout()
plt.show()


# ## 3.2 Análisis de todas las imágenes (una sola pasada)
#
# Para no abrir cada imagen varias veces, la función `analizar` la abre una sola vez y extrae todo lo que necesitan las secciones siguientes. `ThreadPoolExecutor` procesa varias imágenes en paralelo y `tqdm` muestra una barra de progreso.
#
# | Dato | Para qué sirve |
# |---|---|
# | `h_ex` (hash MD5 del archivo) | Detectar copias **exactas** (mismos bytes). |
# | `formato`, `modo`, `canales`, `ancho`, `alto`, `tam_kb` | Propiedades: dimensiones, canales de color y peso del archivo. |
# | `im.verify()` y lectura completa | Detectar archivos **corruptos o truncados**. |
# | `h_ap` (hash aproximado) | Detectar copias casi idénticas: la imagen se reduce a 16×16 en grises y cada píxel se marca como más claro o más oscuro que el promedio (256 bits). |
# | `brillo`, `contraste`, `nitidez` | Detectar ruido visual: media de grises, desviación estándar y varianza del filtro Laplaciano (a menor valor, más desenfocada). Se calculan sobre una versión reducida de 64×64, así que sirven para **comparar** imágenes entre sí. |
#
# Si una imagen falla, se guarda `ok = False` y el mensaje de error, en lugar de detener el análisis.

# In[5]:


def analizar(ruta):
    r = {"ok": True, "error": ""}
    try:
        with open(ruta, "rb") as f:
            r["h_ex"] = hashlib.md5(f.read()).hexdigest()
        r["tam_kb"] = os.path.getsize(ruta) / 1024

        with Image.open(ruta) as im:
            r["formato"] = im.format
            r["modo"] = im.mode
            r["canales"] = len(im.getbands())
            r["ancho"], r["alto"] = im.size
            im.verify()                      # detecta archivos dañados

        with Image.open(ruta) as im:         # verify() obliga a reabrir
            im.draft("RGB", (256, 256))      # acelera la lectura de JPEG
            g = np.asarray(im.convert("L").resize((64, 64), Image.Resampling.BILINEAR),
                           dtype=np.float32)

        r["brillo"] = float(g.mean())
        r["contraste"] = float(g.std())
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        r["nitidez"] = float(lap.var())

        p = g.reshape(16, 4, 16, 4).mean(axis=(1, 3))          # reducción a 16x16
        r["h_ap"] = np.packbits((p > p.mean()).ravel()).tobytes().hex()
    except Exception as e:
        r["ok"] = False
        r["error"] = f"{type(e).__name__}: {e}"
    return r

t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(tqdm(ex.map(analizar, df_total["ruta"]), total=len(df_total), desc="Analizando"))

df_eda = pd.concat([df_total.reset_index(drop=True), pd.DataFrame(res)], axis=1)
print(f"Análisis terminado en {time.time() - t0:.0f} s | imágenes analizadas: {len(df_eda)}")


# ## 3.3 Imágenes corruptas
#
# Se listan las imágenes que no pudieron abrirse o leerse por completo (`Image.open().verify()` más una lectura completa del archivo). Estas imágenes no pueden usarse para entrenar y se descartan; `df_ok` conserva solo las imágenes sanas.

# In[6]:


corruptas = df_eda[~df_eda["ok"]]
print("Imágenes corruptas o ilegibles:", len(corruptas))
if len(corruptas):
    print(corruptas[["ruta", "clase", "error"]].to_string(index=False))

df_ok = df_eda[df_eda["ok"]].copy()
print("Imágenes sanas:", len(df_ok))


# ## 3.4 Dimensiones, canales y formatos
#
# Se resumen las propiedades de las imágenes **originales**:
#
# - **Ancho y alto:** determinan a qué resolución se reducirán las imágenes (Fase 4). Reducir demasiado pierde detalle; reducir poco encarece el entrenamiento.
# - **Relación de aspecto** (ancho / alto): si varía mucho, al redimensionar a un cuadrado las imágenes se deforman.
# - **Canales:** una CNN con entrada RGB necesita 3 canales; las imágenes en gris (1 canal) o con transparencia (4 canales) deben convertirse.
# - **Formato y peso:** informativos; el peso ayuda a detectar imágenes demasiado pequeñas o comprimidas.
#
# Los tamaños se leen tal como están almacenados en el archivo, antes de corregir una posible rotación guardada en los metadatos de la foto.

# In[7]:


df_ok["aspecto"] = df_ok["ancho"] / df_ok["alto"]

print(df_ok[["ancho", "alto", "aspecto", "tam_kb"]].describe().round(2).to_string())
print("\nCanales :", dict(Counter(df_ok["canales"])))
print("Modos   :", dict(Counter(df_ok["modo"])))
print("Formatos:", dict(Counter(df_ok["formato"])))
print("\nImágenes con el lado menor < 128 px:", int((df_ok[["ancho", "alto"]].min(axis=1) < 128).sum()))
print("Imágenes con el lado menor < 224 px:", int((df_ok[["ancho", "alto"]].min(axis=1) < 224).sum()))

fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
for e, col, tit in zip(ejes, ["ancho", "alto", "aspecto"],
                       ["Ancho (px)", "Alto (px)", "Relación de aspecto (ancho/alto)"]):
    e.hist(df_ok[col], bins=40)
    e.set_title(tit)
    e.grid(alpha=0.3)
plt.suptitle("Propiedades de las imágenes originales", fontsize=14)
plt.tight_layout()
plt.show()


# ## 3.5 Imágenes duplicadas
#
# Si una misma foto está en train y en test, el modelo la memoriza y la accuracy sale inflada. Por eso se detectan los duplicados **antes** de dividir los datos. Se usan dos huellas:
#
# - **Hash exacto (MD5):** mismos bytes, misma huella.
# - **Hash aproximado:** detecta copias casi idénticas (reescaladas, recomprimidas). Puede dar falsos positivos con fotos distintas pero muy parecidas, por eso se **revisan algunos grupos a simple vista**.
#
# Se eliminan (1) las imágenes cuya huella aparece con **etiquetas distintas** (contradictorias) y (2) las copias repetidas, conservando una sola. `CLAVE_DUP` elige qué huella se usa para eliminar: si la revisión visual muestra que el hash aproximado agrupa fotos que **no** son copias, se cambia a `"h_ex"`. El resultado es `df_limpio`, el conjunto que se usará desde la Fase 4. No se detectan fotos distintas del mismo plato.

# In[8]:


def cargar(ruta, lado=256):
    """Abre una imagen para mostrarla (corrige la rotación guardada en los metadatos)."""
    with Image.open(ruta) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((lado, lado))
        return im.copy()

CLAVE_DUP = "h_ap"        # "h_ap" = copias casi idénticas | "h_ex" = solo copias exactas

print("Imágenes con hash exacto repetido    :", int(df_ok.duplicated("h_ex", keep=False).sum()))
print("Imágenes con hash aproximado repetido:", int(df_ok.duplicated("h_ap", keep=False).sum()))

n_cl = df_ok.groupby(CLAVE_DUP)["clase"].nunique()
contradictorias = n_cl[n_cl > 1].index
print(f"Huellas con etiquetas contradictorias ({CLAVE_DUP}):", len(contradictorias))

# Revisión visual de algunos grupos de duplicados
grupos = [g for _, g in df_ok[df_ok.duplicated(CLAVE_DUP, keep=False)].groupby(CLAVE_DUP)][:4]
if grupos:
    fig, ejes = plt.subplots(len(grupos), 3, figsize=(10, 3.2 * len(grupos)), squeeze=False)
    for e in ejes.ravel():
        e.axis("off")
    for i, g in enumerate(grupos):
        for j, (_, fila) in enumerate(g.head(3).iterrows()):
            ejes[i, j].imshow(cargar(fila["ruta"]))
            ejes[i, j].set_title(fila["clase"], fontsize=9)
    plt.suptitle("Grupos de duplicados detectados (cada fila es un grupo)", fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print("No se encontraron duplicados.")

# Limpieza: quitar contradictorias y quedarse con una sola copia
sin_contra = df_ok[~df_ok[CLAVE_DUP].isin(contradictorias)]
df_limpio = sin_contra.drop_duplicates(CLAVE_DUP, keep="first").reset_index(drop=True)
print("\nImágenes originales :", len(df_total))
print("Imágenes limpias    :", len(df_limpio), "| eliminadas:", len(df_total) - len(df_limpio))

comp = pd.DataFrame({
    "original": df_total["clase"].value_counts().reindex(CLASES),
    "limpias": df_limpio["clase"].value_counts().reindex(CLASES),
})
comp["eliminadas"] = comp["original"] - comp["limpias"]
if (comp["eliminadas"] > 0).any():
    print("\nClases afectadas:")
    print(comp[comp["eliminadas"] > 0].to_string())


# ## 3.6 Ruido visual: brillo, contraste y nitidez
#
# Fotos muy oscuras, muy claras o desenfocadas pueden confundir al modelo. Se grafican las distribuciones de las tres medidas calculadas en 3.2 y se muestran los casos extremos (las más oscuras, las más brillantes y las más desenfocadas). La nitidez se grafica en escala logarítmica porque sus valores son muy asimétricos.
#
# Esta sección ayuda a decidir si conviene aplicar *data augmentation* de brillo y contraste en la Fase 4.

# In[9]:


fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
ejes[0].hist(df_limpio["brillo"], bins=40)
ejes[0].set_title("Brillo medio (0 = negro, 255 = blanco)")
ejes[1].hist(df_limpio["contraste"], bins=40)
ejes[1].set_title("Contraste (desviación estándar)")
ejes[2].hist(np.log10(1 + df_limpio["nitidez"]), bins=40)
ejes[2].set_title("Nitidez (log10 de 1 + varianza Laplaciana)")
for e in ejes:
    e.grid(alpha=0.3)
plt.suptitle("Calidad visual de las imágenes", fontsize=14)
plt.tight_layout()
plt.show()

print(df_limpio[["brillo", "contraste", "nitidez"]].describe().round(2).to_string())

def mostrar(df, titulo, n=5):
    fig, ejes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    for e, (_, f) in zip(ejes, df.head(n).iterrows()):
        e.imshow(cargar(f["ruta"]))
        e.set_title(f["clase"], fontsize=9)
        e.axis("off")
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()

mostrar(df_limpio.nsmallest(5, "brillo"), "Las 5 imágenes más oscuras")
mostrar(df_limpio.nlargest(5, "brillo"), "Las 5 imágenes más brillantes")
mostrar(df_limpio.nsmallest(5, "nitidez"), "Las 5 imágenes más desenfocadas")


# ## 3.7 Grilla de ejemplos: una imagen por plato
#
# `plt.subplots` crea una cuadrícula con una celda por clase. De cada plato se elige una imagen al azar (reproducible con `SEED`) y se muestra con `imshow`, con el nombre de la clase como título. Sirve para revisar visualmente que las imágenes sean fotos reales y correspondan a su clase, y para anticipar qué platos se parecen entre sí.

# In[17]:


rng = np.random.default_rng(SEED)
columnas = 5
filas = -(-len(CLASES) // columnas)       # división hacia arriba

fig, ejes = plt.subplots(filas, columnas, figsize=(15, 3 * filas))
for e in ejes.ravel():
    e.axis("off")

for i, nombre in enumerate(CLASES):
    sub = df_limpio[df_limpio["clase"] == nombre]
    fila = sub.iloc[rng.integers(len(sub))]
    ejes.ravel()[i].imshow(cargar(fila["ruta"]))
    ejes.ravel()[i].set_title(nombre, fontsize=10)

plt.suptitle("Una   imagen   de  ejemplo   por   plato", fontsize=19)
plt.tight_layout()
plt.show()


# ## 3.8 Variabilidad dentro de una clase
#
# Una sola imagen por plato no muestra cuánto varía un mismo plato entre fotos (ángulo, iluminación, presentación, fondo). Se muestran 10 imágenes de una misma clase (`CLASE_DEMO`, que se puede cambiar por cualquier otro plato). Mientras mayor sea esa variabilidad y menor la cantidad de imágenes por clase, más difícil será aprender el patrón, y más útil será el *data augmentation*.

# In[18]:


CLASE_DEMO = CLASES[0]        # cambiar por otro plato para explorar

sub = df_limpio[df_limpio["clase"] == CLASE_DEMO].sample(10, random_state=SEED)
fig, ejes = plt.subplots(2, 5, figsize=(15, 6.5))
for e, (_, f) in zip(ejes.ravel(), sub.iterrows()):
    e.imshow(cargar(f["ruta"]))
    e.axis("off")
plt.suptitle(f"10 imágenes de la clase: {CLASE_DEMO}", fontsize=14)
plt.tight_layout()
plt.show()


# # Fase 3a: Preprocesamiento
#
# El preprocesamiento convierte las fotos originales en tensores listos para la red. Se hace en este orden, que importa:
#
# 1. **Cargar y redimensionar** todas las imágenes a un tamaño fijo.
# 2. **Dividir** en entrenamiento, validación y prueba (antes de cualquier aumento).
# 3. **Normalizar** los píxeles al rango [0, 1].
# 4. **Data augmentation**, solo sobre el entrenamiento.
#
# > **Regla de oro:** nunca se aplica *data augmentation* a los conjuntos de validación o prueba; deben medir el rendimiento sobre imágenes reales, sin alterar.
#
# ## 3.9 Parámetros de preprocesamiento
#
# | Parámetro | Valor | Justificación |
# |---|---|---|
# | `IMG_SIZE` | 224 | Según el EDA, solo 4 de las 4,000 imágenes tienen el lado menor por debajo de 224 px, así que casi ninguna se amplía. Es el tamaño que usa la guía del curso. |
# | `AJUSTE` | `"estirar"` | Las fotos tienen aspecto variable (mediana 0.75). Se redimensionan directamente a un cuadrado, sin perder contenido; la deformación se aplica igual a todas las imágenes, incluidas las de producción. La alternativa `"recortar"` toma el centro de la foto. |
# | `TEST_SIZE` | 0.15 | 15% para prueba (15 imágenes por plato). |
# | `VAL_SIZE` | 0.15 | 15% para validación (15 imágenes por plato); el 70% restante queda para entrenamiento. |

# In[19]:


from sklearn.model_selection import train_test_split
from keras import layers

IMG_SIZE = 224
AJUSTE = "estirar"        # "estirar" = redimensiona a cuadrado | "recortar" = recorte central
TEST_SIZE = 0.15
VAL_SIZE = 0.15

N_CLASES = len(CLASES)
FORMA_ENTRADA = (IMG_SIZE, IMG_SIZE, 3)
print("Entrada:", FORMA_ENTRADA, "| Clases:", N_CLASES, "| Ajuste:", AJUSTE)


# ## 3.10 Cargar y redimensionar las imágenes
#
# La función `preparar_imagen` abre cada foto, **corrige la rotación guardada en los metadatos** (`exif_transpose`, para que las fotos tomadas con el celular no queden giradas), la convierte a RGB y la redimensiona a `IMG_SIZE × IMG_SIZE`. El resultado es un arreglo de NumPy de enteros de 8 bits (`uint8`, valores de 0 a 255). `Image.draft` acelera la lectura de JPEG decodificándolos ya a menor tamaño.
#
# Esta misma función se **reutilizará en el despliegue** (Fase 12), para que el servicio prepare las fotos exactamente igual que en el entrenamiento. El resultado se guarda en disco con `np.save` (caché): la primera vez tarda un poco y las siguientes, segundos.

# In[20]:


CACHE_X = PROYECTO / f"cache_perufoodnet_{IMG_SIZE}px_{AJUSTE}.npy"

def preparar_imagen(ruta):
    """Abre, corrige la rotación, convierte a RGB y redimensiona a IMG_SIZE x IMG_SIZE."""
    with Image.open(ruta) as im:
        im.draft("RGB", (IMG_SIZE * 2, IMG_SIZE * 2))
        im = ImageOps.exif_transpose(im).convert("RGB")
        if AJUSTE == "recortar":
            im = ImageOps.fit(im, (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        else:
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)

if CACHE_X.exists():
    X_all = np.load(CACHE_X)
    print("Cargado desde caché")
else:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        X_all = np.stack(list(tqdm(ex.map(preparar_imagen, df_limpio["ruta"]),
                                   total=len(df_limpio), desc="Cargando")))
    np.save(CACHE_X, X_all)
    print(f"Cargadas y guardadas en caché en {time.time() - t0:.0f} s")

assert len(X_all) == len(df_limpio), "La caché no coincide con df_limpio: borra el archivo cache_perufoodnet_*"
y_all = df_limpio["clase"].map(clase_a_id).to_numpy(dtype="int64")
print("X_all:", X_all.shape, X_all.dtype, f"| {X_all.nbytes / 1e6:.0f} MB")


# ## 3.11 División estratificada: entrenamiento, validación y prueba
#
# Se dividen los datos en tres conjuntos con `train_test_split(..., stratify=y)`, que conserva la misma proporción de cada plato en todos los conjuntos:
#
# - **Entrenamiento (70%):** con él la red ajusta sus pesos.
# - **Validación (15%):** se usa durante el entrenamiento para vigilar el sobreajuste y elegir la mejor época. No ajusta pesos.
# - **Prueba (15%):** se reserva hasta la evaluación final (Fase 6). No se toca antes.
#
# La división se hace **antes** de aplicar el *data augmentation*, y en dos pasos: primero se separa la prueba y luego la validación del resto (por eso `VAL_SIZE` se reescala dividiendo entre `1 - TEST_SIZE`). Se guardan también las tablas con la ruta y la clase de cada conjunto, que servirán en el análisis de errores. Como se eliminaron los duplicados en el EDA, ninguna imagen debe estar en dos conjuntos a la vez, y se comprueba comparando sus huellas.

# In[21]:


idx = np.arange(len(X_all))
idx_trval, idx_test = train_test_split(
    idx, test_size=TEST_SIZE, stratify=y_all, random_state=SEED
)
idx_train, idx_val = train_test_split(
    idx_trval, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_all[idx_trval], random_state=SEED
)

X_train, y_train = X_all[idx_train], y_all[idx_train]
X_val, y_val = X_all[idx_val], y_all[idx_val]
X_test, y_test = X_all[idx_test], y_all[idx_test]

df_train = df_limpio.iloc[idx_train].reset_index(drop=True)
df_val = df_limpio.iloc[idx_val].reset_index(drop=True)
df_test = df_limpio.iloc[idx_test].reset_index(drop=True)
del X_all

# Comprobaciones: sin solapamiento entre conjuntos y proporciones por clase
assert not (set(idx_train) & set(idx_val)) and not (set(idx_train) & set(idx_test)) \
    and not (set(idx_val) & set(idx_test))
fuga = len(set(df_train["h_ap"]) & set(df_test["h_ap"])) + len(set(df_val["h_ap"]) & set(df_test["h_ap"]))
print("Imágenes de test con copia en train o validación:", fuga)

for nombre, y in [("Train", y_train), ("Validación", y_val), ("Test", y_test)]:
    c = np.bincount(y, minlength=N_CLASES)
    print(f"{nombre:<11}: {len(y)} imágenes | por clase: mín {c.min()}, máx {c.max()}")


# ## 3.12 Normalización de los píxeles a [0, 1]
#
# Los píxeles son enteros de 0 a 255. Se convierten a decimales (`float32`) y se dividen entre 255 para llevarlos al rango **[0, 1]**: la red entrena de forma más estable con valores pequeños. La misma operación se aplica a los tres conjuntos. Las imágenes ya tienen la forma `(N, alto, ancho, 3)` que espera una CNN, así que no hace falta cambiar su forma; se comprueba con `assert`.

# In[22]:


print("Antes   :", X_train.dtype, "| mín:", X_train.min(), "| máx:", X_train.max())

X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("Después :", X_train.dtype, "| mín:", X_train.min(), "| máx:", X_train.max())
assert X_train.ndim == 4 and X_train.shape[1:] == FORMA_ENTRADA
print("Forma por imagen:", X_train.shape[1:])


# ## 3.13 Data augmentation (solo entrenamiento)
#
# Con apenas 70 imágenes de entrenamiento por plato, la red tiende a memorizarlas. El *data augmentation* genera en cada época variaciones aleatorias de cada foto, de modo que la red nunca ve exactamente la misma imagen. El criterio de la guía es el **principio rectificador**: toda transformación debe producir una imagen que podría existir en el mundo real.
#
# | Técnica | ¿Se aplica? | Justificación para fotos de comida |
# |---|---|---|
# | Giro horizontal | Sí | Un plato no tiene orientación izquierda/derecha "correcta"; no hay texto que se invierta. |
# | Giro vertical | **No** | Una foto de comida de lado, puesta al revés, no es realista (gravedad fija). |
# | Rotación (±36°, `0.1`) | Sí | Las fotos se toman con distintas inclinaciones del celular. |
# | Zoom (15%) | Sí | Hay fotos tomadas a distintas distancias. |
# | Brillo y contraste (suaves) | Sí | El EDA mostró variación de iluminación (brillo medio de 60 a 194). |
# | Cambios de color (matiz) | **No** | El color es una pista clave para distinguir platos. |
#
# Las capas de augmentation de Keras se incluyen **dentro del modelo** (Fase 4): solo actúan cuando `training=True` y se desactivan solas al validar, evaluar o predecir. `value_range=(0, 1)` indica a las capas de brillo y contraste que los píxeles ya están normalizados. Para visualizar, se pasa una misma imagen varias veces por el bloque con `training=True`; cada pasada da una variación distinta.

# In[23]:


def crear_aumento():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),                             # hasta ±36°
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.1, value_range=(0, 1)),
        layers.RandomContrast(0.15, value_range=(0, 1)),
    ], name="aumento")

aug_demo = crear_aumento()
rng = np.random.default_rng(SEED)
ejemplos = [rng.choice(np.where(y_train == c)[0]) for c in (0, N_CLASES // 2)]   # dos platos distintos

fig, ejes = plt.subplots(2, 5, figsize=(15, 6.5))
for fila, j in enumerate(ejemplos):
    ejes[fila, 0].imshow(X_train[j])
    ejes[fila, 0].set_title(f"Original ({CLASES[y_train[j]]})")
    for k in range(1, 5):
        v = aug_demo(X_train[j][None], training=True)
        v = keras.ops.convert_to_numpy(v)[0]
        ejes[fila, k].imshow(np.clip(v, 0, 1))
        ejes[fila, k].set_title(f"Variación {k}")
for e in ejes.ravel():
    e.axis("off")
plt.suptitle("Data Augmentation: original y 4 variaciones", fontsize=14)
plt.tight_layout()
plt.show()


# ## 3.13.1 Verificación de la regla de oro
#
# Se comprueba que las capas de augmentation **solo modifican las imágenes en entrenamiento** (`training=True`) y que dejan intactas las de validación y prueba (`training=False`). Así se garantiza que la validación y la prueba se evalúan sobre imágenes reales, sin alterar.

# In[24]:


x = X_val[:8]
sin_aug = keras.ops.convert_to_numpy(aug_demo(x, training=False))
con_aug = keras.ops.convert_to_numpy(aug_demo(x, training=True))

print("Con training=False (val/test), la imagen queda idéntica:", np.allclose(sin_aug, x))
print("Con training=True (train), la imagen se modifica      :", not np.allclose(con_aug, x))


# ## 3.14 Resumen del preprocesamiento
#
# Tabla con el número de imágenes y la forma final de cada conjunto. `X` contiene las imágenes y `y` las etiquetas enteras (0 a 39).

# In[25]:


resumen = pd.DataFrame({
    "Conjunto": ["Train", "Validación", "Test"],
    "Imágenes": [len(X_train), len(X_val), len(X_test)],
    "Shape X": [X_train.shape, X_val.shape, X_test.shape],
    "Shape y": [y_train.shape, y_val.shape, y_test.shape],
    "Augmentation": ["Sí (dentro del modelo)", "No", "No"],
})
print(resumen.to_string(index=False))
print("\nTotal de imágenes usadas:", len(X_train) + len(X_val) + len(X_test))
print("Clases:", N_CLASES, "| Rango de píxeles: [", X_train.min(), ",", X_train.max(), "]")


# ## 3.15 Conclusiones del EDA
#
# - **Distribución de clases:** las 40 clases tienen exactamente 100 imágenes, con un ratio de desbalance de 1.0:1. El dataset está perfectamente balanceado, por lo que no hace falta submuestrear ni ponderar clases, y la accuracy global será una medida interpretable (el azar da 1/40 = 2.5%). El mínimo y el máximo coinciden, por eso la salida muestra el mismo plato en ambos.
#
# - **Calidad de los archivos:** las 4,000 imágenes se abrieron y leyeron completas; no hay archivos corruptos ni truncados. Todas son JPEG en modo RGB de 3 canales, por lo que no se requiere ninguna conversión de formato ni de canales.
#
# - **Dimensiones:** las imágenes originales son grandes y heterogéneas. La mediana es de 1200 × 1600 px (ancho × alto), pero van desde 265 × 185 hasta 5312 × 4270 px, y su peso va de 12 KB a unos 16 MB (mediana de 168 KB). Ninguna tiene el lado menor por debajo de 128 px y solo 4 lo tienen por debajo de 224 px, de modo que reducirlas a 128 × 128 o a 224 × 224 no obliga a ampliar casi ninguna imagen.
#
# - **Relación de aspecto:** el valor más frecuente es 0.75 (fotos verticales de 3:4), pero el rango llega de 0.36 a 3.18, con una media de 0.90. Hay una mezcla de fotos verticales, cuadradas y horizontales, así que redimensionar todas a un cuadrado las deformará en distinta medida. Esto se tendrá en cuenta en el preprocesamiento.
#
# - **Duplicados:** no se encontraron duplicados exactos ni aproximados, ni imágenes con etiquetas contradictorias. No hubo que eliminar ninguna imagen, por lo que `df_limpio` conserva las 4,000. Esta comprobación no detecta fotos distintas de un mismo plato.
#
# - **Ruido visual:** el brillo medio es de 130 sobre 255 (rango de 60 a 194), el contraste medio es de 50 y la nitidez es muy variable (de 98 a 8,056 en la medida usada, con mediana de 1,604). No hay imágenes extremadamente oscuras ni quemadas, pero sí variación de iluminación y de enfoque entre fotos, coherente con el hecho de que se tomaron en distintos entornos, con distinta iluminación y con distintos dispositivos.
#
# - **Variabilidad visual:** la revisión de las grillas muestra fotos reales de platos, con distintos ángulos, fondos y presentaciones dentro de una misma clase.

# # Fase 4: Modelos (Baseline, CNN Mejorada y CNN Optimizada)
#
# Se diseñan **tres arquitecturas de complejidad creciente** primero se confirma que el pipeline funciona de principio a fin con un modelo simple, y luego invertir en arquitecturas más complejas solo si aportan una mejora real.
#
# | Modelo | Idea principal | Qué agrega respecto al anterior |
# |---|---|---|
# | **1. CNN Baseline** | 3 bloques `Conv + BatchNorm + ReLU + MaxPool` (32, 64, 128 filtros), `GlobalAveragePooling`, `Dense(256)` + `Dropout(0.5)` | Es la línea base de modelo: confirma que el pipeline funciona. |
# | **2. CNN Mejorada (estilo ResNet)** | Bloques residuales con *skip connection* | Combate el desvanecimiento del gradiente y permite redes más profundas. |
# | **3. CNN Optimizada Final** | Bloques residuales más profundos, con *stem* de *stride* agresivo, *data augmentation* dentro del modelo, L2 y Dropout escalonado | Reduce el sobreajuste y la memoria usada, con las optimizaciones de la sesión 06. |
#
# **Decisiones comunes a los tres modelos:**
# - **Entrada:** `(224, 224, 3)`; **salida:** `Dense(40, activation="softmax")`, una neurona por plato (Fase 1).
# - **Batch Normalization antes de la activación:** normaliza la salida lineal de la convolución y `ReLU` recibe valores centrados. Como `BatchNormalization` ya incluye su propio desplazamiento (`beta`), las convoluciones usan `use_bias=False`.
# - **`GlobalAveragePooling2D` (GAP)** en lugar de `Flatten`: promedia cada mapa de rasgos en un solo número, lo que reduce mucho los parámetros del clasificador. En la Actividad 2, `Flatten + Dense(128)` concentraba el 97.8% de los parámetros y causó un sobreajuste severo.
# - Los modelos **solo se construyen aquí**; se compilan y se entrenan en la Fase 5.
#
# ## 4.1 Modelo 1: CNN Baseline
#
# Tres bloques convolucionales que duplican los filtros (32, 64, 128): las primeras capas detectan rasgos simples (bordes, colores) y las siguientes, combinaciones más complejas (texturas, formas del plato). Después, `GlobalAveragePooling2D`, una capa `Dense(256)` con `Dropout(0.5)` y la capa de salida con `softmax`.
#
# - `Conv2D(filtros, 3, padding="same")`: ventanas de 3×3 que recorren la imagen; `padding="same"` conserva el alto y ancho.
# - `BatchNormalization()`: normaliza la salida de la capa por lote (media 0, desviación 1) y estabiliza el entrenamiento.
# - `MaxPooling2D(2)`: reduce a la mitad el alto y ancho, quedándose con el máximo de cada ventana de 2×2.
# - `Dropout(0.5)`: apaga al azar el 50% de las neuronas en cada paso de entrenamiento, para que no memoricen.
#
# Siguiendo las diapositivas, este modelo **no** incluye *data augmentation*; se usa solo para confirmar que el pipeline funciona.

# In[26]:


from keras import regularizers

def crear_cnn_baseline(forma=FORMA_ENTRADA, n_clases=N_CLASES):
    capas = [keras.Input(shape=forma)]
    for filtros in (32, 64, 128):
        capas += [
            layers.Conv2D(filtros, 3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(2),
        ]
    capas += [
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_clases, activation="softmax"),
    ]
    return keras.Sequential(capas, name="cnn_baseline")

modelo_1 = crear_cnn_baseline()
modelo_1.summary()


# ## 4.2 Bloque residual (*skip connection*)
#
# En una red muy profunda, el gradiente (la señal que indica cómo corregir los pesos) se debilita al atravesar muchas capas y las primeras casi no aprenden: es el **desvanecimiento del gradiente**. Un **bloque residual** lo combate sumando a la salida de dos convoluciones la entrada original (el *atajo* o *shortcut*), de modo que el gradiente tiene un camino directo hacia atrás:
#
# ```
# x ─→ Conv → BN → ReLU → Conv → BN ─→ (+) → ReLU
#  └────────────── atajo ───────────────┘
# ```
#
# - `layers.Add()` suma la ruta principal y el atajo; ambos deben tener la misma forma.
# - **Atajo con proyección:** cuando el bloque cambia el número de filtros o reduce el tamaño (`stride=2`), la entrada ya no tiene la misma forma que la salida. En ese caso el atajo pasa por una convolución de 1×1 (con su `BatchNormalization`) que la ajusta. La diapositiva muestra el caso más simple, con `shortcut = x`, y esta variante lo extiende para que el bloque funcione con cualquier cambio de filtros.
# - `stride=2` reduce a la mitad el alto y ancho, y reemplaza al `MaxPooling` dentro de las etapas.
# - `l2` aplica regularización L2 a los pesos de las convoluciones (se usa en el Modelo 3).

# In[28]:


def bloque_residual(x, filtros, stride=1, l2=0.0):
    reg = regularizers.l2(l2) if l2 else None
    atajo = x

    # Ruta principal
    x = layers.Conv2D(filtros, 3, strides=stride, padding="same",
                      use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filtros, 3, padding="same",
                      use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    # Atajo: si cambia la forma, se proyecta con una convolución 1x1
    if stride != 1 or atajo.shape[-1] != filtros:
        atajo = layers.Conv2D(filtros, 1, strides=stride, padding="same",
                              use_bias=False, kernel_regularizer=reg)(atajo)
        atajo = layers.BatchNormalization()(atajo)

    # SUMA (skip connection)
    x = layers.Add()([x, atajo])
    return layers.Activation("relu")(x)


# ## 4.3 Modelo 2: CNN Mejorada (estilo ResNet)
#
# Un *stem* inicial (`Conv 3×3` + BN + ReLU + `MaxPool`) seguido de **cuatro etapas de un bloque residual** (32, 64, 128 y 256 filtros), donde las etapas 2 a 4 reducen el tamaño con `stride=2`. Cierra con `GlobalAveragePooling2D`, `Dense(256)`, `Dropout(0.5)` y la capa `softmax`. Se construye con la **API funcional** de Keras (`keras.Model(entradas, salidas)`), porque la suma del atajo no puede expresarse con `Sequential`.
#
# Igual que el Modelo 1, no incluye *data augmentation*: así, la diferencia entre ambos mide el efecto de los bloques residuales y la mayor profundidad.

# In[29]:


ETAPAS_MEJORADA = [(32, 1), (64, 2), (128, 2), (256, 2)]      # (filtros, stride)

def crear_cnn_mejorada(forma=FORMA_ENTRADA, n_clases=N_CLASES):
    entradas = keras.Input(shape=forma)

    # Stem
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(entradas)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    # Etapas residuales
    for filtros, stride in ETAPAS_MEJORADA:
        x = bloque_residual(x, filtros, stride)

    # Clasificador
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    salidas = layers.Dense(n_clases, activation="softmax")(x)
    return keras.Model(entradas, salidas, name="cnn_mejorada_resnet")

modelo_2 = crear_cnn_mejorada()
modelo_2.summary()


# ## 4.4 Modelo 3: CNN Optimizada Final
#
# Es la arquitectura del Modelo 2, **más profunda** (dos bloques residuales por etapa) y con las optimizaciones de la sesión 06 aplicadas:
#
# 1. **Data augmentation incrustado en el grafo:** `crear_aumento()` (Fase 3) es la primera capa del modelo; solo actúa en entrenamiento y se desactiva sola al validar, evaluar o predecir. Por eso, al entrenar este modelo se le pasan las imágenes **sin** aumentar.
# 2. ***Stem* con *stride* agresivo:** una convolución de 7×7 con `stride=2` seguida de `MaxPooling(3, stride=2)` reduce la imagen de 224×224 a 56×56 de inmediato. Así, las etapas siguientes procesan mapas mucho más pequeños, con menos memoria y menos cómputo.
# 3. **Regularización L2 (1e-4)** en todas las capas convolucionales y densas.
# 4. **Dropout escalonado:** `SpatialDropout2D` (apaga mapas de rasgos completos, más adecuado para convoluciones) con tasas crecientes de 0.10, 0.15 y 0.20 en las etapas 2 a 4, y luego `Dropout(0.5)` antes de la salida. Obliga a la red a repartir la información en varias neuronas (redundancia).
#

# In[30]:


L2 = 1e-4
ETAPAS_OPTIMIZADA = [(32, 1, 0.00), (64, 2, 0.10), (128, 2, 0.15), (256, 2, 0.20)]   # (filtros, stride, dropout)

def crear_cnn_optimizada(forma=FORMA_ENTRADA, n_clases=N_CLASES, l2=L2):
    reg = regularizers.l2(l2)
    entradas = keras.Input(shape=forma)
    x = crear_aumento()(entradas)                      # 1. Data augmentation incrustado

    # 2. Stem con stride agresivo: 224 -> 112 -> 56
    x = layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Etapas residuales (dos bloques por etapa) con L2 y Dropout escalonado
    for filtros, stride, p in ETAPAS_OPTIMIZADA:
        x = bloque_residual(x, filtros, stride, l2)
        x = bloque_residual(x, filtros, 1, l2)
        if p:
            x = layers.SpatialDropout2D(p)(x)

    # Clasificador
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    salidas = layers.Dense(n_clases, activation="softmax", kernel_regularizer=reg)(x)
    return keras.Model(entradas, salidas, name="cnn_optimizada")

modelo_3 = crear_cnn_optimizada()
modelo_3.summary()


# ## 4.5 Comparación de las arquitecturas
#
# Se reúnen en una tabla los **parámetros totales** (los números que la red ajusta al entrenar), los **entrenables** (los que realmente se actualizan; los restantes son las estadísticas de `BatchNormalization`) y el **peso aproximado** del modelo en MB, calculado como parámetros × 4 bytes (`float32`). El peso importa por la restricción de la Fase 1: el modelo debe pesar menos de 50 MB.
#
# Además se hace una **prueba de humo**: se pasan 4 imágenes por cada modelo y se comprueba que la salida tiene la forma `(4, 40)` y que cada fila suma 1 (son probabilidades de `softmax`). Con eso se confirma que las tres arquitecturas están bien conectadas antes de entrenarlas.

# In[31]:


modelos = [modelo_1, modelo_2, modelo_3]
filas = []
for m in modelos:
    total = m.count_params()
    entrenables = sum(int(np.prod(w.shape)) for w in m.trainable_weights)
    filas.append({
        "Arquitectura": m.name,
        "Parámetros totales": total,
        "Parámetros entrenables": entrenables,
        "Peso aprox. (MB)": round(total * 4 / 1e6, 1),
        "Capas": len(m.layers),
        "Cumple < 50 MB": total * 4 / 1e6 < 50,
    })
tabla_modelos = pd.DataFrame(filas)
print(tabla_modelos.to_string(index=False))

# Prueba de humo: forma y suma de probabilidades
x_prueba = X_train[:4]
for m in modelos:
    p = keras.ops.convert_to_numpy(m(x_prueba, training=False))
    print(f"{m.name:<22} salida {p.shape} | suma por fila: {p.sum(axis=1).round(3)}")


# ## 4.6 Conclusiones de la Fase 4
#
# Los tres modelos se construyeron correctamente: aceptan imágenes de 224 × 224 × 3 y devuelven un vector de 40 probabilidades que suma 1 por imagen. Las arquitecturas aún no se han entrenado; sus resultados se compararán en las Fases 5 y 8.
#
# | Arquitectura | Parámetros totales | Entrenables | Peso aprox. | Capas | Cumple < 50 MB |
# |---|---|---|---|---|---|
# | CNN Baseline | 137,224 | 136,776 | 0.5 MB | 16 | Sí |
# | CNN Mejorada (ResNet) | 1,305,352 | 1,302,472 | 5.2 MB | 43 | Sí |
# | CNN Optimizada | 2,879,752 | 2,874,952 | 11.5 MB | 75 | Sí |
#
# - **Modelo 1 (Baseline):** es la arquitectura más pequeña, con 137,224 parámetros (0.5 MB). Tres bloques convolucionales (32, 64 y 128 filtros) reducen la imagen hasta mapas de 28 × 28 × 128, que `GlobalAveragePooling2D` resume en 128 valores. A diferencia de la CNN base de la Actividad 2, donde el clasificador denso concentraba el 97.8% de los parámetros, aquí el clasificador (`Dense(256)` y la salida) suma 43,304 parámetros, el 31.6% del total. Sirve como línea base de modelo para confirmar que el pipeline funciona.
#
# - **Modelo 2 (Mejorada, estilo ResNet):** añade cuatro bloques residuales (32, 64, 128 y 256 filtros) y llega a 1,305,352 parámetros (5.2 MB), unas 9.5 veces el Baseline. Tres de los cuatro bloques cambian el número de filtros y reducen el tamaño, por lo que usan un atajo con proyección (convolución 1×1 de 2,048, 8,192 y 32,768 parámetros). El mapa final es de 14 × 14 × 256 y el clasificador pesa solo el 5.8% del total: casi todo el modelo es extractor de rasgos.
#
# - **Modelo 3 (Optimizada):** duplica los bloques residuales (8 en total) y reúne las optimizaciones de la sesión 06: *data augmentation* como primera capa, *stem* de 7×7 con *stride* 2 y *max pooling* (224 → 56 de entrada a las etapas), L2 de 1e-4, `SpatialDropout2D` escalonado (0.10, 0.15 y 0.20) y `Dropout(0.5)` en el clasificador. Tiene 2,879,752 parámetros (11.5 MB), 2.2 veces el Modelo 2 y 21 veces el Baseline. Sus etapas trabajan con mapas de 56, 28, 14 y 7 píxeles de lado, frente a los 112, 56, 28 y 14 del Modelo 2, por lo que cada capa maneja cuatro veces menos activaciones en la entrada de las etapas, y el entrenamiento necesitará menos memoria de GPU que su tamaño sugiere.
#
# - **Restricción de tamaño (< 50 MB):** los tres modelos la cumplen con margen. El más grande (Modelo 3, 11.5 MB) ocupa alrededor del 23% del límite, de modo que la restricción no limita la elección de la arquitectura. La conversión a TFLite o ONNX y la medición del tiempo de inferencia (< 1 s en CPU) se verifican en la Fase 12.
#
# - **Limitación de la comparación:** el Modelo 3 cambia varias cosas a la vez respecto al Modelo 2 (profundidad, *stem*, augmentation, L2 y Dropout), por lo que una mejora no podrá atribuirse a una sola técnica. Si el tiempo lo permite, se puede entrenar una variante del Modelo 3 sin augmentation para aislar su efecto.

# # Fase 5: Configuración del entrenamiento y callbacks
#
# Esta fase define **cómo** aprenden los modelos (hiperparámetros, pérdida, métricas, callbacks) y entrena las tres arquitecturas de la Fase 4 con exactamente las mismas condiciones, para que la comparación sea justa. Solo se usan los conjuntos de entrenamiento y validación; el de prueba no se toca hasta la Fase 6.
#
# ## 5.1 Configuración centralizada (`CONFIG`)
#
# Todos los hiperparámetros viven en **un solo diccionario**, que funciona como "panel de control": cambiar un valor aquí cambia el entrenamiento de los tres modelos, y el mismo diccionario se guarda junto a cada experimento para poder reproducirlo.
#
# | Parámetro | Valor | Justificación |
# |---|---|---|
# | `learning_rate` | 1e-3 | Tamaño base del paso de Adam; es el valor estándar de la guía. |
# | `epochs` | 50 | Es solo un **tope**: `EarlyStopping` detendrá el entrenamiento antes si deja de mejorar. |
# | `batch_size` | 32 | Imágenes por lote (valor de la guía); con lotes más grandes se agotaría antes la memoria de la GPU. |
# | `optimizer` | adam | Ajusta el paso de cada peso según el historial de sus gradientes. |
# | `label_smoothing` | 0.1 | Suaviza las etiquetas (la clase correcta pasa de 1.0 a ≈ 0.9) para evitar que el modelo se vuelva demasiado confiado. |
# | `weight_decay` | `None` | El Modelo 3 ya aplica L2 en sus capas; usar también *weight decay* aplicaría la regularización dos veces. |
# | `early_stopping_patience` | 10 | Si `val_loss` no mejora en 10 épocas seguidas, se detiene y se restauran los mejores pesos. |
# | `reduce_lr_patience` | 4 | Si `val_loss` no mejora en 4 épocas, se reduce el *learning rate*. |
# | `reduce_lr_factor` / `min_lr` | 0.5 / 1e-6 | Se reduce a la mitad cada vez, sin bajar de 1e-6. |
#
# `ReduceLROnPlateau` (paciencia 4) reacciona antes que `EarlyStopping` (paciencia 10): primero se prueba con pasos más pequeños y solo si tampoco mejora se corta el entrenamiento.
#
# Los resultados se guardan en dos carpetas: `modelos/` (el mejor modelo de cada entrenamiento, en formato `.keras`) y `experimentos/` (el historial en CSV y JSON y los registros de cada corrida).

# In[53]:


CONFIG = {
    "learning_rate": 1e-3,
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "adam",
    "label_smoothing": 0.1,
    "weight_decay": None,            # el Modelo 3 ya tiene L2 en sus capas (no duplicar)
    "early_stopping_patience": 10,
    "reduce_lr_patience": 4,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
    "seed": SEED,
    "img_size": IMG_SIZE,
    "n_clases": N_CLASES,
}

MODELOS_DIR = PROYECTO / "modelos"
EXP_DIR = PROYECTO / "experimentos"
MODELOS_DIR.mkdir(exist_ok=True)
EXP_DIR.mkdir(exist_ok=True)

pasos_por_epoca = -(-len(X_train) // CONFIG["batch_size"])
print("Imágenes de entrenamiento:", len(X_train), "| lotes por época:", pasos_por_epoca)
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# ## 5.2 Etiquetas en formato *one-hot*
#
# Hasta ahora las etiquetas son números enteros (0 a 39). La pérdida con *label smoothing* y la métrica AUC necesitan que cada etiqueta sea un **vector *one-hot*** de 40 posiciones: un 1 en la posición de su clase y 0 en las demás (por ejemplo, la clase 2 es `[0, 0, 1, 0, ..., 0]`). `keras.utils.to_categorical` hace la conversión. Se conservan también las etiquetas enteras (`y_train`, `y_val`, `y_test`), porque las usarán las evaluaciones de las Fases 6 y 7.

# In[54]:


y_train_oh = keras.utils.to_categorical(y_train, N_CLASES)
y_val_oh = keras.utils.to_categorical(y_val, N_CLASES)
y_test_oh = keras.utils.to_categorical(y_test, N_CLASES)

print("y_train:", y_train.shape, "->", y_train_oh.shape)
print("Ejemplo: clase", y_train[0], f"({CLASES[y_train[0]]}) ->", y_train_oh[0].astype(int))
assert (y_train_oh.argmax(axis=1) == y_train).all()


# ## 5.3 Pérdida, optimizador y métricas
#
# - **Optimizador `Adam`:** ajusta los pesos para reducir el error, adaptando el paso de cada peso según cómo han cambiado sus gradientes.
# - **Pérdida `CategoricalCrossentropy(label_smoothing=0.1)`:** mide qué tan lejos están las probabilidades predichas de la clase correcta. Con *label smoothing*, el objetivo deja de ser 1.0 exacto y pasa a ≈ 0.9, repartiendo el resto entre las demás clases. No se usa `from_logits` porque la última capa ya aplica `softmax`.
# - **Métricas:**
#   - `accuracy`: porcentaje de imágenes bien clasificadas.
#   - `top3`: porcentaje de imágenes cuyo plato correcto está entre las 3 predicciones más probables. Con 40 clases parecidas, indica si el modelo al menos "se acerca" cuando falla.
#   - `auc`: área bajo la curva ROC (una curva por clase, `multi_label=True`), que mide qué tan bien se separa cada plato del resto.
#
# > **Cómo leer la pérdida:** con *label smoothing* la pérdida **nunca llega a 0**, ni siquiera con un modelo perfecto, y la pérdida del Modelo 3 incluye además la penalización L2 de sus pesos. Por eso el `loss` **no sirve para comparar entre modelos**; para eso se usan `accuracy`, `top3`, `auc` y el F1 de las Fases 6 y 8. El `loss` solo se compara dentro de un mismo modelo, época tras época.

# In[55]:


def crear_optimizador():
    return keras.optimizers.Adam(
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

def crear_perdida():
    return keras.losses.CategoricalCrossentropy(label_smoothing=CONFIG["label_smoothing"])

def crear_metricas():
    return [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        keras.metrics.AUC(name="auc", multi_label=True),
    ]

print("Optimizador:", crear_optimizador().__class__.__name__)
print("Pérdida    :", crear_perdida().__class__.__name__, "| label_smoothing =", CONFIG["label_smoothing"])
print("Métricas   :", [m.name for m in crear_metricas()])


# ## 5.4 Callbacks de automatización
#
# Un **callback** es un objeto que Keras ejecuta en momentos concretos del entrenamiento (al terminar un lote, una época, etc.). Permiten vigilar, ajustar y registrar el entrenamiento sin intervención humana:
#
# | Callback | Qué hace |
# |---|---|
# | `ModelCheckpoint` | Guarda en disco **solo el mejor modelo** (menor `val_loss`), de modo que un fallo a mitad del entrenamiento no pierde el progreso. |
# | `EarlyStopping` | Detiene el entrenamiento si `val_loss` no mejora y **restaura los pesos de la mejor época**, no los de la última. |
# | `ReduceLROnPlateau` | Reduce el *learning rate* a la mitad cuando `val_loss` se estanca, para afinar el aprendizaje. |
# | `TensorBoardTorch` (propio) | Telemetría interactiva en tiempo real. El callback `TensorBoard` de Keras depende de TensorFlow, que este proyecto no usa (el backend es PyTorch), por lo que se escribe un callback propio que registra las métricas de cada época con `torch.utils.tensorboard`. Requiere el paquete `tensorboard`; si no está instalado, se omite. |
# | `CSVLogger` | Registro persistente en texto plano de cada época, para trazabilidad. |
# | `BarraProgreso` (propio) | Barra de progreso con el tiempo restante y una línea de resumen por época, para ver en todo momento en qué época va el entrenamiento. |
#
# En Windows, TensorBoard guarda sus eventos en una carpeta temporal única para evitar conflictos con archivos existentes y errores con rutas del proyecto que contienen caracteres no ASCII (como `í`). En PowerShell, `%TEMP%` no se expande como en CMD: copia y ejecuta el comando completo `tensorboard --logdir "..."` que imprime la celda de callbacks (incluye la ruta única). Abre `http://localhost:6006`. Los archivos CSV y JSON siguen en `experimentos/`.
#
# Los callbacks se crean **de nuevo para cada entrenamiento** (`crear_callbacks`), porque guardan su propio estado y no deben compartirse entre modelos.

# In[56]:


import importlib.util
import tempfile
from tqdm.auto import tqdm

TENSORBOARD_OK = importlib.util.find_spec("tensorboard") is not None
TENSORBOARD_DIR = Path(tempfile.mkdtemp(prefix="PeruFoodNet_TensorBoard_"))
print("TensorBoard disponible (paquete 'tensorboard' instalado):", TENSORBOARD_OK)
print("Directorio de eventos TensorBoard:", TENSORBOARD_DIR)
print(f'Para abrir TensorBoard: tensorboard --logdir "{TENSORBOARD_DIR}"')

class BarraProgreso(keras.callbacks.Callback):
    """Barra global (todos los lotes de todas las épocas) con tiempo restante
    y una línea de resumen al terminar cada época."""

    def on_train_begin(self, logs=None):
        self.epocas = self.params["epochs"]
        self.pasos = self.params["steps"]
        self.barra = tqdm(total=self.epocas * self.pasos, unit="lote",
                          desc="Entrenando", dynamic_ncols=True)
        self.epoca_actual = 0
        self.ultima_epoca = 0
        self.t_inicio = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoca_actual = epoch + 1
        self.t_epoca = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.barra.set_postfix(
            época=f"{self.epoca_actual}/{self.epocas}",
            loss=f"{float(logs['loss']):.3f}",
            acc=f"{float(logs['accuracy']):.3f}",
        )
        self.barra.update(1)

    def on_test_begin(self, logs=None):
        if hasattr(self, "barra"):
            self.barra.set_postfix_str(f"época {self.epoca_actual}/{self.epocas} | validando...")

    def on_epoch_end(self, epoch, logs=None):
        self.ultima_epoca = epoch + 1
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        tqdm.write(
            f"Época {epoch + 1:>2}/{self.epocas} | {time.time() - self.t_epoca:>4.0f} s | lr {lr:.1e} | "
            f"loss {float(logs['loss']):.3f}  acc {float(logs['accuracy']):.3f} | "
            f"val_loss {float(logs['val_loss']):.3f}  val_acc {float(logs['val_accuracy']):.3f}  "
            f"val_top3 {float(logs.get('val_top3', float('nan'))):.3f}"
        )

    def on_train_end(self, logs=None):
        self.barra.close()
        total = time.time() - self.t_inicio
        print(f"Entrenamiento terminado: {self.ultima_epoca} épocas en {total / 60:.1f} min")

class TensorBoardTorch(keras.callbacks.Callback):
    """TensorBoard sin TensorFlow: registra las métricas de cada época
    con torch.utils.tensorboard."""

    def __init__(self, log_dir):
        super().__init__()
        from torch.utils.tensorboard import SummaryWriter
        self.escritor = SummaryWriter(log_dir=str(log_dir))

    def on_epoch_end(self, epoch, logs=None):
        for nombre, valor in (logs or {}).items():
            self.escritor.add_scalar(nombre, float(valor), epoch + 1)
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        self.escritor.add_scalar("learning_rate", lr, epoch + 1)
        self.escritor.flush()

    def on_train_end(self, logs=None):
        self.escritor.close()

def crear_callbacks(nombre):
    carpeta = EXP_DIR / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    cbs = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELOS_DIR / f"{nombre}.keras"),
            monitor="val_loss", save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=CONFIG["early_stopping_patience"],
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=CONFIG["reduce_lr_factor"],
            patience=CONFIG["reduce_lr_patience"], min_lr=CONFIG["min_lr"], verbose=1),
        keras.callbacks.CSVLogger(str(carpeta / "entrenamiento.csv")),
    ]
    if TENSORBOARD_OK:
        try:
            cbs.append(TensorBoardTorch(TENSORBOARD_DIR / nombre))
        except Exception as e:                       # no detener el entrenamiento por esto
            print("TensorBoard omitido:", type(e).__name__, e)
    cbs.append(BarraProgreso())
    return cbs


# ## 5.5 Función de entrenamiento reutilizable
#
# La función `entrenar` hace, **siempre en el mismo orden y con el mismo `CONFIG`**, todo lo necesario para un modelo:
#
# 1. Fija la semilla (`set_random_seed`), de modo que los tres modelos parten de las mismas condiciones aleatorias.
# 2. Construye el modelo desde cero con su función `crear_...` (así `fit` no continúa desde pesos ya entrenados).
# 3. Lo compila con el optimizador, la pérdida y las métricas de 5.3.
# 4. Entrena con `fit`, usando validación y los callbacks de 5.4. `verbose=0` apaga la salida por defecto de Keras para no duplicar la de `BarraProgreso`.
# 5. Guarda un **resumen en JSON** (configuración, tiempo, mejor época, métricas e historial completo) en `experimentos/<nombre>/`, que servirá para la comparación (Fase 8), el versionado y el registro de experimentos.
#
# Devuelve el modelo (con los pesos de la mejor época restaurados), el historial de Keras y el tiempo total.

# In[57]:


import json

def guardar_resumen(nombre, hist, tiempo):
    h = hist.history
    mejor = int(np.argmin(h["val_loss"]))
    resumen = {
        "modelo": nombre,
        "tiempo_s": round(tiempo, 1),
        "epocas_ejecutadas": len(h["loss"]),
        "mejor_epoca": mejor + 1,
        "mejor_val_loss": float(h["val_loss"][mejor]),
        "val_accuracy_mejor_epoca": float(h["val_accuracy"][mejor]),
        "config": CONFIG,
        "historial": {k: [float(x) for x in v] for k, v in h.items()},
    }
    with open(EXP_DIR / nombre / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    return resumen

def entrenar(constructor, nombre, epocas=None):
    epocas = epocas or CONFIG["epochs"]
    keras.utils.set_random_seed(CONFIG["seed"])

    modelo = constructor()
    modelo.compile(optimizer=crear_optimizador(), loss=crear_perdida(), metrics=crear_metricas())

    t0 = time.time()
    hist = modelo.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=epocas,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        verbose=0,
        callbacks=crear_callbacks(nombre),
    )
    tiempo = time.time() - t0
    guardar_resumen(nombre, hist, tiempo)
    return modelo, hist, tiempo


# ## 5.6 Prueba corta (2 épocas por modelo)
#
# Antes de lanzar los entrenamientos completos se hace una **prueba de 2 épocas** con cada modelo. No sirve para evaluar resultados, sino para comprobar tres cosas:
#
# 1. **Que todo el código funciona** de principio a fin (configuración, métricas, callbacks, guardado) antes de gastar tiempo de GPU.
# 2. **Cuánto tarda cada época**, para estimar el tiempo de las 50 épocas. Es una cota superior: `EarlyStopping` probablemente detendrá el entrenamiento antes, y la primera época es más lenta por la inicialización de la GPU.
# 3. **Cuánta memoria de GPU usa cada modelo** (pico durante el entrenamiento), para confirmar que caben en los 8 GB con `batch_size=32`.
#
# Después de cada modelo se libera la memoria (`clear_session` y `empty_cache`) para que el siguiente parta limpio. Los archivos de la prueba quedan con el prefijo `prueba_` en `modelos/` y `experimentos/`, y se pueden borrar después.

# In[58]:


modelos_prueba = [
    ("cnn_baseline", crear_cnn_baseline),
    ("cnn_mejorada_resnet", crear_cnn_mejorada),
    ("cnn_optimizada", crear_cnn_optimizada),
]
EPOCAS_PRUEBA = 2
filas = []

for nombre, constructor in modelos_prueba:
    print(f"\n=== Prueba corta: {nombre} ===")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    modelo, hist, tiempo = entrenar(constructor, f"prueba_{nombre}", epocas=EPOCAS_PRUEBA)

    pico = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    por_epoca = tiempo / EPOCAS_PRUEBA
    filas.append({
        "Modelo": nombre,
        "s/época": round(por_epoca, 1),
        "Est. 50 épocas (min)": round(por_epoca * CONFIG["epochs"] / 60, 1),
        "Pico GPU (GB)": round(pico, 2),
        "val_acc tras 2 épocas": round(hist.history["val_accuracy"][-1], 3),
    })

    del modelo
    keras.backend.clear_session()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print()
print(pd.DataFrame(filas).to_string(index=False))


# ## 5.7 Entrenamiento de los tres modelos
#
# Se entrena cada arquitectura con la función `entrenar` y el mismo `CONFIG` (máximo 50 épocas; `EarlyStopping` puede detener antes). La función `correr_entrenamiento` envuelve el proceso para que sea cómodo y seguro:
#
# - Si el modelo **ya fue entrenado** (existe su `resumen.json` en `experimentos/`), no se repite: se carga el resultado guardado. Así se puede reiniciar el kernel sin perder trabajo ni repetir horas de entrenamiento.
# - Al terminar, muestra un **resumen**: épocas ejecutadas, mejor época, métricas en esa época, brecha entre entrenamiento y validación, y el tiempo total.
# - Libera la memoria de la GPU para que el siguiente modelo parta limpio. El mejor modelo de cada entrenamiento queda guardado en `modelos/` y se cargará desde ahí en la Fase 6.
#
# Los resultados de cada entrenamiento se guardan en `experimentos/<nombre>/`: historial en CSV y JSON, el registro de TensorBoard y la configuración utilizada.

# In[59]:


def cargar_resumen(nombre):
    with open(EXP_DIR / nombre / "resumen.json", encoding="utf-8") as f:
        return json.load(f)

def mostrar_resumen(r):
    h = r["historial"]
    i = r["mejor_epoca"] - 1
    tope = r["config"]["epochs"]
    corto = " (detenido por EarlyStopping)" if r["epocas_ejecutadas"] < tope else ""
    print(f"\nResumen de {r['modelo']}")
    print(f"  Épocas ejecutadas : {r['epocas_ejecutadas']} de {tope}{corto}")
    print(f"  Mejor época       : {r['mejor_epoca']} (val_loss {r['mejor_val_loss']:.3f})")
    print(f"  En esa época      : train_acc {h['accuracy'][i]:.3f} | val_acc {h['val_accuracy'][i]:.3f} "
          f"| brecha {h['accuracy'][i] - h['val_accuracy'][i]:+.3f}")
    print(f"                      val_top3 {h['val_top3'][i]:.3f} | val_auc {h['val_auc'][i]:.3f}")
    print(f"  Tiempo total      : {r['tiempo_s'] / 60:.1f} min")

def correr_entrenamiento(constructor, nombre, forzar=False):
    ruta = EXP_DIR / nombre / "resumen.json"
    if ruta.exists() and not forzar:
        print(f"'{nombre}' ya está entrenado: se usa {ruta.relative_to(PROYECTO)}. "
              "Para repetirlo, pasa forzar=True.")
    else:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        modelo, hist, tiempo = entrenar(constructor, nombre)
        if torch.cuda.is_available():
            print(f"Pico de memoria de GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        del modelo
        keras.backend.clear_session()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    resumen = cargar_resumen(nombre)
    mostrar_resumen(resumen)
    return resumen


# ## 5.8 Entrenamiento del Modelo 1: CNN Baseline
#
# Línea base de modelo: sin *data augmentation* y con pocos parámetros. Se espera que aprenda rápido y que, al no tener regularización fuerte, muestre signos de sobreajuste: es información útil para juzgar si las arquitecturas más complejas realmente aportan.

# In[60]:


res_1 = correr_entrenamiento(crear_cnn_baseline, "cnn_baseline")


# ## 5.9 Entrenamiento del Modelo 2: CNN Mejorada (estilo ResNet)
#
# Bloques residuales, sin *data augmentation*. Es el modelo que más memoria de GPU puede usar por trabajar con mapas de 112 × 112 en sus primeras etapas.

# In[61]:


res_2 = correr_entrenamiento(crear_cnn_mejorada, "cnn_mejorada_resnet")


# ## 5.10 Entrenamiento del Modelo 3: CNN Optimizada Final
#
# Es el único modelo con *data augmentation* (incrustado como primera capa) y con L2 y Dropout escalonado. Recibe las imágenes **sin aumentar**: la capa de augmentation las modifica dentro del modelo, solo durante el entrenamiento.
#
# > **Cómo leer sus métricas de entrenamiento:** durante el entrenamiento, este modelo ve imágenes alteradas y con neuronas apagadas por Dropout, mientras que la validación se mide sin ninguna alteración. Por eso su `train_acc` puede ser **igual o incluso menor** que su `val_acc` sin que eso sea un error: una brecha negativa o pequeña no indica por sí sola un buen modelo.

# In[62]:


res_3 = correr_entrenamiento(crear_cnn_optimizada, "cnn_optimizada")


# ## 5.11 Diagnóstico visual del entrenamiento
#
# Las **curvas de aprendizaje** (loss y accuracy de entrenamiento y de validación por época) permiten diagnosticar si un modelo está aprendiendo bien. La guía del curso las clasifica así:
#
# | Condición | Síntomas en las curvas | Tratamiento |
# |---|---|---|
# | **Sobreajuste** (*overfitting*) | Brecha creciente entre entrenamiento y validación en accuracy; el `val_loss` sube mientras el `loss` de entrenamiento sigue bajando (memorización). | Más Dropout, más *augmentation*, reducir la complejidad. |
# | **Subajuste** (*underfitting*) | Ambas curvas se estancan en valores malos: el modelo no capta ni los patrones básicos. | Modelo más profundo, más épocas, aumentar el *learning rate*. |
# | **Buen ajuste** | Brecha pequeña y estable; ambas curvas mejoran juntas. | Dejar actuar al `EarlyStopping`. |
#
# Para cada modelo se muestran tres paneles: accuracy, loss y *learning rate* (que permite ver cuándo `ReduceLROnPlateau` lo redujo). La línea punteada marca la **mejor época** según `val_loss`, cuyos pesos se conservaron.
#
# Además se calculan **indicadores numéricos** de la brecha y una **etiqueta orientativa** con umbrales simples (definidos en el código). La etiqueta es solo una ayuda: el diagnóstico final se decide mirando las curvas.

# In[63]:


NOMBRES_MODELOS = ["cnn_baseline", "cnn_mejorada_resnet", "cnn_optimizada"]
RESUMENES = {n: cargar_resumen(n) for n in NOMBRES_MODELOS}

def graficar_curvas(r):
    h = r["historial"]
    ep = np.arange(1, len(h["loss"]) + 1)
    mejor = r["mejor_epoca"]
    tiene_lr = "learning_rate" in h
    fig, ejes = plt.subplots(1, 3 if tiene_lr else 2, figsize=(19 if tiene_lr else 14, 5))

    for eje, (tr, va, nombre) in zip(ejes[:2], [("accuracy", "val_accuracy", "Accuracy"),
                                                 ("loss", "val_loss", "Loss")]):
        eje.plot(ep, h[tr], label="Train", marker="o", markersize=3)
        eje.plot(ep, h[va], label="Validación", marker="o", markersize=3)
        eje.axvline(mejor, color="gray", linestyle="--", label=f"Mín. val_loss (época {mejor})")
        eje.set_title(nombre); eje.set_xlabel("Época"); eje.set_ylabel(nombre)
        eje.legend(); eje.grid(alpha=0.3)

    if tiene_lr:
        ejes[2].plot(ep, h["learning_rate"], marker="o", markersize=3, color="tab:green")
        ejes[2].set_yscale("log")
        ejes[2].set_title("Learning rate"); ejes[2].set_xlabel("Época"); ejes[2].grid(alpha=0.3)

    plt.suptitle(f"{r['modelo']}: train vs validación", fontsize=14)
    plt.tight_layout()
    plt.show()

# Umbrales orientativos para la etiqueta
BRECHA_ALTA = 0.15      # brecha train-val en accuracy a partir de la cual se sospecha sobreajuste
SUBE_VAL_LOSS = 1.10    # val_loss de la última época vs su mínimo (10% más)
ACC_BAJA = 0.40         # train_acc final por debajo de este valor sugiere subajuste

def indicadores(r):
    h = r["historial"]
    i = r["mejor_epoca"] - 1
    tr, va, vl = np.array(h["accuracy"]), np.array(h["val_accuracy"]), np.array(h["val_loss"])
    ind = {
        "Modelo": r["modelo"],
        "Épocas": len(tr),
        "Mejor época": i + 1,
        "train_acc (mejor)": tr[i], "val_acc (mejor)": va[i], "Brecha (mejor)": tr[i] - va[i],
        "train_acc (última)": tr[-1], "val_acc (última)": va[-1], "Brecha (última)": tr[-1] - va[-1],
        "val_loss última / mín.": vl[-1] / vl.min(),
    }
    if ind["Brecha (última)"] > BRECHA_ALTA and ind["val_loss última / mín."] > SUBE_VAL_LOSS:
        ind["Etiqueta orientativa"] = "Sobreajuste"
    elif ind["train_acc (última)"] < ACC_BAJA:
        ind["Etiqueta orientativa"] = "Posible subajuste"
    elif ind["Brecha (última)"] <= BRECHA_ALTA:
        ind["Etiqueta orientativa"] = "Brecha pequeña (buen ajuste)"
    else:
        ind["Etiqueta orientativa"] = "Brecha moderada (revisar curvas)"
    return ind

for nombre in NOMBRES_MODELOS:
    graficar_curvas(RESUMENES[nombre])

tabla_diag = pd.DataFrame([indicadores(RESUMENES[n]) for n in NOMBRES_MODELOS])
print(tabla_diag.round(3).to_string(index=False))


# ## 5.12 Comparación de los tres modelos en validación
#
# Para decidir cómo seguir se superponen las curvas de `val_accuracy` de los tres modelos y se resumen sus métricas en la **mejor época** de cada uno (la que conserva `EarlyStopping`). Se usa accuracy, top-3 y AUC, y no el `loss`, porque este no es comparable entre modelos: incluye *label smoothing* y, en el Modelo 3, la penalización L2.
#
# Esta es una comparación **preliminar sobre validación**. La evaluación imparcial con el conjunto de prueba se hace en la Fase 6, y la comparación sistemática (con intervalos de confianza y el balance entre rendimiento y tamaño) en la Fase 8.

# In[64]:


fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
for n in NOMBRES_MODELOS:
    h = RESUMENES[n]["historial"]
    ep = np.arange(1, len(h["val_accuracy"]) + 1)
    ejes[0].plot(ep, h["val_accuracy"], label=n, marker="o", markersize=3)
    ejes[1].plot(ep, h["val_top3"], label=n, marker="o", markersize=3)
ejes[0].set_title("val_accuracy"); ejes[1].set_title("val_top3")
for e in ejes:
    e.set_xlabel("Época"); e.legend(); e.grid(alpha=0.3)
plt.suptitle("Los tres modelos en validación", fontsize=14)
plt.tight_layout()
plt.show()

filas = []
for n in NOMBRES_MODELOS:
    r = RESUMENES[n]
    h, i = r["historial"], r["mejor_epoca"] - 1
    filas.append({
        "Modelo": n,
        "Mejor época": i + 1,
        "val_acc": h["val_accuracy"][i],
        "val_top3": h["val_top3"][i],
        "val_auc": h["val_auc"][i],
        "Épocas": r["epocas_ejecutadas"],
        "Tiempo (min)": r["tiempo_s"] / 60,
    })
print(pd.DataFrame(filas).round(3).to_string(index=False))
