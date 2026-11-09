import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CLASS_NAMES = (
    "Aguadito",
    "Aji de Gallina",
    "Anticuchos",
    "Arroz Chaufa",
    "Arroz Tapado",
    "Arroz con Mariscos",
    "Arroz con Pollo",
    "Bisteck a lo Pobre",
    "Caldo de Gallina",
    "Carapulcra",
    "Cau Cau",
    "Causa Limeña",
    "Ceviche",
    "Chanfainita",
    "Chicharron de Mariscos",
    "Chicharron de cerdo",
    "Choclo con Queso",
    "Choritos a la Chalaca",
    "Escabeche",
    "Frejoles",
    "Humitas",
    "Lentejitas",
    "Locro",
    "Lomo Saltado",
    "Ocopa",
    "Olluquito",
    "Papa Rellena",
    "Papa a la Huancaina",
    "Patasca",
    "Picante de cuy",
    "Pollo a la Brasa",
    "Pollo al horno con puré",
    "Pulpo al Olivo",
    "Rocoto Relleno",
    "Sopa a la Minuta",
    "Tacu Tacu",
    "Tallarin Saltado",
    "Tallarines Rojos",
    "Tallarines Verdes",
    "Tamales",
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "production" / "model_v1.tflite"
INPUT_SIZE = (224, 224)
_INFERENCE_LOCK = threading.Lock()


class PredictionError(Exception):
    pass


@dataclass(frozen=True)
class DishPrediction:
    dish_name: str
    confidence: float


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise PredictionError(
            "No se encontró el modelo de clasificación. Configura PERUFOODNET_MODEL_PATH."
        )

    try:
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_content=path.read_bytes(), num_threads=2)
        interpreter.allocate_tensors()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise PredictionError(
            "No se pudo cargar el modelo. Verifica el archivo y las dependencias de inferencia."
        ) from exc

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or tuple(input_details[0]["shape"]) != (1, 224, 224, 3):
        raise PredictionError(
            "El modelo no tiene la forma de entrada esperada (224, 224, 3)."
        )
    if input_details[0]["dtype"] != np.float32:
        raise PredictionError("El modelo debe recibir imágenes de tipo float32.")
    if len(output_details) != 1 or tuple(output_details[0]["shape"]) != (
        1,
        len(CLASS_NAMES),
    ):
        raise PredictionError(
            f"El modelo debe tener {len(CLASS_NAMES)} salidas para coincidir con las etiquetas."
        )
    return interpreter


def predict_dish(uploaded_image) -> DishPrediction:
    model_path = os.environ.get("PERUFOODNET_MODEL_PATH", str(MODEL_PATH))
    model = _load_model(model_path)

    uploaded_image.seek(0)
    with Image.open(uploaded_image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
        batch = np.asarray(image, dtype=np.float32)[None, ...] / 255.0

    try:
        with _INFERENCE_LOCK:
            input_index = model.get_input_details()[0]["index"]
            output_index = model.get_output_details()[0]["index"]
            model.set_tensor(input_index, batch)
            model.invoke()
            scores = np.asarray(model.get_tensor(output_index))
    except (RuntimeError, ValueError) as exc:
        raise PredictionError("Ocurrió un error al analizar la imagen.") from exc

    if scores.shape != (1, len(CLASS_NAMES)) or not np.isfinite(scores).all():
        raise PredictionError("El modelo devolvió resultados incompatibles con sus 40 clases.")

    index = int(np.argmax(scores[0]))
    confidence = float(scores[0, index])
    if not 0 <= confidence <= 1:
        raise PredictionError("El modelo devolvió una puntuación de confianza inválida.")

    return DishPrediction(CLASS_NAMES[index], confidence)
