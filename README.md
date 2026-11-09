# PeruFoodNet — Clasificación de platos peruanos

Pipeline de visión por computadora para clasificar fotografías de **40 platos típicos del Perú**. El proyecto cubre exploración y validación de datos, preprocesamiento, entrenamiento y comparación de CNN, evaluación en un conjunto de prueba y registro de experimentos.

> **Alcance:** el modelo predice una categoría de plato. No identifica ingredientes ni determina si un alimento es seguro para una persona con alergias.

## Resultados principales

El mejor resultado registrado corresponde a **Transfer Learning con EfficientNetB0**. Las métricas se leen de `models/production/model_v1_metadata.json`.

| Métrica | Resultado |
|---|---:|
| Accuracy en validación | 97.83% |
| Accuracy en prueba | 97.00% |
| Macro F1 en prueba | 0.9699 |
| Épocas ejecutadas | 10 |
| Tamaño del modelo guardado | ≈ 21.1 MB |

El conjunto está balanceado: 4,000 imágenes, 40 clases y 100 imágenes por clase. La partición estratificada es 70% entrenamiento, 15% validación y 15% prueba. Las métricas son resultados de una ejecución concreta; pueden variar si cambian los datos, el entorno o la configuración.

## Dataset

Se utiliza **PeruFoodNet**, publicado en Mendeley Data:

- [Página del dataset](https://data.mendeley.com/datasets/hxhbbm497d/3)
- DOI: [10.17632/hxhbbm497d.3](https://doi.org/10.17632/hxhbbm497d.3)

Descarga y descomprime el dataset. **La celda de carga del notebook busca la carpeta `data/raw/Dataset_PeruFoodNet/`**, con una subcarpeta por clase:

```text
proyecto_comida/
├── data/
│   └── raw/
│       └── Dataset_PeruFoodNet/
│           ├── Anticuchos/
│           ├── Ceviche/
│           └── ... una carpeta por clase
├── comida_peru.ipynb
└── requirements.txt
```

Cada carpeta de clase debe contener sus imágenes. Los datos no se incluyen en el repositorio; descárgalos desde la fuente y colócalos en esa ruta antes de ejecutar el notebook.

## Modelos y metodología

Se comparan tres CNN entrenadas desde cero y un modelo de transferencia de aprendizaje:

| Modelo | Enfoque |
|---|---|
| CNN Baseline | CNN pequeña de referencia |
| CNN Mejorada (ResNet) | Bloques residuales con conexiones de salto |
| CNN Optimizada | Bloques residuales y regularización |
| **EfficientNetB0** | Base preentrenada congelada y cabeza clasificadora para 40 clases |

Configuración común documentada en el notebook:

- Imágenes RGB redimensionadas a **224 × 224**.
- División estratificada **70/15/15** para entrenamiento, validación y prueba.
- Aumento de datos durante el entrenamiento.
- Métrica principal: **Macro F1**; se complementa con accuracy, top-3, AUC y análisis por clase.
- Keras 3 con backend **PyTorch**.

## Requisitos

- Windows y Python **3.11** (versión usada para el entorno del proyecto).
- GPU NVIDIA compatible y controladores instalados para ejecutar PyTorch con CUDA. También se puede ejecutar en CPU, pero el entrenamiento será más lento.
- Espacio suficiente para el dataset, los modelos y los archivos temporales de entrenamiento.

El archivo [`requirements.txt`](./requirements.txt) fija PyTorch para CUDA 12.8 en Windows. TensorFlow está incluido para compatibilidad con herramientas como TensorBoard; **el backend de Keras del proyecto sigue siendo PyTorch**. En una instalación nativa de Windows, TensorFlow no proporciona aceleración GPU para este flujo.

## Instalación

Desde PowerShell, en la carpeta del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno, puedes invocar sus herramientas directamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En VS Code, selecciona `.venv\Scripts\python.exe` como intérprete y el entorno `.venv` como kernel del notebook. Reinicia el kernel después de instalar dependencias.

Comprobación opcional del backend y CUDA:

```powershell
$env:KERAS_BACKEND = "torch"
python -c "import keras, torch; print('Keras:', keras.__version__, '| backend:', keras.backend.backend()); print('CUDA disponible:', torch.cuda.is_available())"
```

## Ejecución

1. Coloca el dataset en `data/raw/Dataset_PeruFoodNet/`, según la estructura anterior.
2. Abre `comida_peru.ipynb` en VS Code o Jupyter y selecciona el kernel `.venv`.
3. Ejecuta las celdas en orden. La primera fase de carga valida la presencia y distribución del dataset; el entrenamiento puede tardar y requiere memoria considerable.
4. Los resultados del entrenamiento, modelos y metadatos se guardan bajo `experimentos/` y `models/production/`, respectivamente.

Para consultar las curvas en TensorBoard, ejecuta primero el entrenamiento para que se creen archivos de eventos. Después ejecuta la celda **10.3**: busca automáticamente un directorio temporal que contenga esos archivos e imprime el comando para PowerShell. Copia el comando, ejecútalo en una terminal con `.venv` activa y abre [http://localhost:6006](http://localhost:6006).

La carpeta temporal de logs cambia entre sesiones. Si TensorBoard muestra **“No dashboards are active”**, vuelve a ejecutar la celda 10.3 y comprueba que la ruta impresa corresponda a la sesión que generó los eventos. Los gráficos registrados son escalares (por ejemplo, `loss`, `accuracy`, `val_loss`, `val_accuracy` y `learning_rate`); no se garantiza que TensorBoard muestre una pestaña de arquitectura.

## API gastronómica

El backend en [`backend/`](./backend/) ofrece un sitio Django para explorar los 40 platillos de PeruFoodNet, buscar por platillo o ingrediente y consultar fichas gastronómicas. En la página principal se puede subir una fotografía: el modelo EfficientNetB0 predice la clase y dirige a la ficha correspondiente. La imagen se procesa durante la solicitud y no se conserva después. DRF expone el catálogo en una API paginada; la predicción está disponible desde el formulario web. El modelo se versiona en `models/production/`. El proyecto incluye Docker Compose y Caddy para desplegar el sitio con HTTPS en una VM ARM64; consulta [`backend/README.md`](./backend/README.md) para las instrucciones.

Para una demostración pública sin administrar una VM, el repositorio incluye un despliegue alternativo en Render Free con PostgreSQL de Neon Free. La aplicación web usa una versión LiteRT cuantizada de aproximadamente 4.9 MB para reducir memoria; el modelo Keras original se mantiene para reproducir el entrenamiento. Render puede suspender el servicio por inactividad, y su sistema de archivos no es persistente. Consulta [`backend/README.md`](./backend/README.md) y `render.yaml` para los pasos y limitaciones.

## Estructura del proyecto

```text
proyecto_comida/
├── comida_peru.ipynb            # Pipeline y análisis principal
├── requirements.txt             # Dependencias del entorno
├── backend/                     # Sitio Django, API DRF y catálogo gastronómico
├── data/
│   ├── raw/                     # Datos locales; excluidos de Git
│   └── processed/               # Caché de imágenes procesadas
├── src/
│   ├── models/                  # Espacio para código de modelos
│   └── training/                # Espacio para código de entrenamiento
├── experimentos/                # CSV, JSON y resultados de cada ejecución
├── models/
│   └── production/              # Modelo Keras, modelo LiteRT para despliegue y metadatos
└── .gitignore                   # Excluye datos, modelos y artefactos generados
```

Los datos y experimentos generados se excluyen del control de versiones mediante `.gitignore`. Se versionan explícitamente el modelo Keras de producción, su metadata y el modelo LiteRT compacto usado por el backend.

## Conclusiones

1. En este experimento, EfficientNetB0 con pesos preentrenados obtuvo el mejor resultado registrado con un conjunto relativamente pequeño.
2. Las CNN entrenadas desde cero sirven como referencias para comparar el efecto de arquitectura y regularización.
3. El análisis por clase y las curvas de validación complementan la accuracy y ayudan a detectar errores entre platos visualmente similares.
4. La evaluación depende de la partición y la calidad del dataset; el resultado no garantiza el mismo rendimiento en fotografías de otras fuentes.

## Autoría y referencia académica

- **Autores:** 

| Nombre | Código |
|---|---|
| Gutierrez Vega Bill Toño | 2012100229F |
| Gutierrez Vega Bill Toño | 2012100229F |
| Gutierrez Vega Bill Toño | 2012100229F |
| Gutierrez Vega Bill Toño | 2012100229F |

- **Curso:** Ingeniería del Conocimiento (IS056B)
- **Docente:** Msc. Jaime Antonio Huaytalla Pariona
- **Institución:** Universidad Nacional del Centro del Perú (UNCP)
- **Periodo:** 2026-II

Dataset: PeruFoodNet, Mendeley Data, DOI [10.17632/hxhbbm497d.3](https://doi.org/10.17632/hxhbbm497d.3).