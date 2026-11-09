# Sitio y API gastronómica

Backend con Django y Django REST Framework para explorar el catálogo de 40 platillos peruanos, consultar sus fichas gastronómicas y clasificar una fotografía con el modelo PeruFoodNet. Una predicción exitosa lleva directamente a la ficha del platillo más probable. La foto se procesa durante la solicitud y no se conserva después.

## Preparación

Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python backend\manage.py migrate
python backend\manage.py seed_dishes
python backend\manage.py createsuperuser
python backend\manage.py runserver
```

El clasificador usa el archivo local `models/production/model_v1.keras`, generado por el notebook. Los modelos están excluidos de Git; si está en otra ubicación, configura la variable antes de iniciar Django:

```powershell
$env:PERUFOODNET_MODEL_PATH = "D:\ruta\al\modelo\model_v1.keras"
```

La instalación de `backend/requirements.txt` incluye Keras con backend PyTorch, además de Pillow para validar y procesar imágenes.

Abre `http://127.0.0.1:8000/` para el sitio, `http://127.0.0.1:8000/api/` para la API y `http://127.0.0.1:8000/admin/` para administrar el catálogo.

`seed_dishes` carga los 40 nombres exactos de las categorías de PeruFoodNet y sus ingredientes típicos de referencia. Se puede volver a ejecutar sin duplicar registros y no elimina otros datos. Algunas preparaciones tienen variantes regionales o familiares; revisa y completa las fichas desde Admin antes de publicarlas. Las fuentes enlazadas se incluyen cuando se verificó una receta concreta.

## Endpoints iniciales

- `GET /api/health/`: estado del servicio y de la base de datos.
- `GET /api/dishes/`: lista paginada de platillos.
- `GET /api/dishes/<slug>/`: detalle de platillo, con región, ingredientes típicos, notas y fuentes disponibles.
- `POST /predecir/`: recibe una imagen desde el formulario web y redirige a la ficha predicha. Acepta imágenes válidas de hasta 8 MB.

El endpoint de predicción es una vista web protegida con CSRF; no forma parte de la API DRF. La API permanece en solo lectura para clientes.

Los ingredientes representan información gastronómica habitual y no confirman la composición de una porción ni su seguridad frente a alergias.

## Configuración

Para desarrollo se usa SQLite local en `backend/db.sqlite3`, excluida de Git. La configuración de despliegue se controla mediante variables de entorno:

- `DJANGO_SECRET_KEY`: obligatorio al ejecutar con `DJANGO_DEBUG=false`.
- `DJANGO_DEBUG`: `true` para desarrollo; usar `false` en producción.
- `DJANGO_ALLOWED_HOSTS`: hosts permitidos separados por comas.
- `PERUFOODNET_MODEL_PATH`: ruta al modelo Keras; por defecto es `models/production/model_v1.keras`.

Las 40 salidas están asociadas al orden alfabético de las carpetas del dataset usado para entrenar. La app comprueba que ese orden coincida con los 40 nombres del catálogo. La puntuación mostrada es una estimación del modelo, no una garantía de reconocimiento.

Para validar la integración, ejecuta `python backend\manage.py test catalog`.
