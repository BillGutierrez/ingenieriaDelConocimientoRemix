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

El sitio usa el modelo compacto `models/production/model_v1.tflite`. Para el entrenamiento y las métricas completas se conserva el modelo Keras original, `models/production/model_v1.keras`. Si el modelo LiteRT está en otra ubicación, configura la variable antes de iniciar Django:

```powershell
$env:PERUFOODNET_MODEL_PATH = "D:\ruta\al\modelo\model_v1.tflite"
```

La instalación de `backend/requirements.txt` incluye LiteRT para ejecutar la inferencia, además de Pillow para validar y procesar imágenes. El backend web no instala Keras ni PyTorch.

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

Para desarrollo se usa SQLite local en `backend/db.sqlite3`, excluida de Git. Si se configura `DATABASE_URL`, Django usa PostgreSQL; esto permite que la base de datos sobreviva al sistema de archivos efímero de Render. La configuración de despliegue se controla mediante variables de entorno:

- `DJANGO_SECRET_KEY`: obligatorio al ejecutar con `DJANGO_DEBUG=false`.
- `DJANGO_DEBUG`: `true` para desarrollo; usar `false` en producción.
- `DJANGO_ALLOWED_HOSTS`: hosts permitidos separados por comas.
- `DATABASE_URL`: URL de conexión PostgreSQL; si no está definida se usa SQLite local.
- `PERUFOODNET_MODEL_PATH`: ruta al modelo LiteRT; por defecto es `models/production/model_v1.tflite`.

Las 40 salidas están asociadas al orden alfabético de las carpetas del dataset usado para entrenar. La app comprueba que ese orden coincida con los 40 nombres del catálogo. La puntuación mostrada es una estimación del modelo, no una garantía de reconocimiento.

Para validar la integración, ejecuta `python backend\manage.py test catalog`.

## Despliegue gratuito en Render y Neon

El repositorio incluye `render.yaml` para desplegar el contenedor en Render Free y usar PostgreSQL externo de Neon Free. La imagen web usa LiteRT y el modelo cuantizado de aproximadamente 4.9 MB, en vez de cargar Keras y PyTorch. El plan gratuito de Render usa almacenamiento efímero y suspende el servicio cuando está inactivo; Neon Free conserva la base de datos, pero también puede suspender su compute en periodos de inactividad. Por eso el primer acceso después de una pausa puede tardar y no se deben guardar archivos locales permanentes en Render.

1. Crea un proyecto PostgreSQL en Neon. Escoge una región cercana a Oregon, donde se configura el servicio de Render.
2. En la página de conexión de Neon, copia la URL de conexión PostgreSQL (con SSL). Trátala como contraseña: no la subas a Git ni la publiques.
3. En Render, crea un **Blueprint** desde el repositorio de GitHub. Render detectará el `render.yaml`; cuando solicite `DATABASE_URL`, pega la URL de Neon como variable secreta.
4. Espera a que termine el despliegue. Al iniciar, el contenedor aplica migraciones y carga el catálogo de los 40 platillos. Render comprobará `/api/health/`.
5. Abre la URL `onrender.com` que Render asigne. Comprueba el catálogo y prueba una imagen desde la página principal.

El Blueprint genera `DJANGO_SECRET_KEY` de forma privada y configura HTTPS, hosts y CSRF para el subdominio de Render. Para administrar datos en producción, usa Django Admin creando un superusuario mediante Shell de Render (`python manage.py createsuperuser`). Haz copias de seguridad periódicas de la base Neon y revisa sus cuotas gratuitas vigentes.

La conversión cuantizada se comparó con el modelo Keras en una partición estratificada de 600 imágenes: alcanzó 96.83% de accuracy frente a 97.00% del original, con coincidencia de 99.83% entre predicciones. Esta prueba ordenó explícitamente las rutas y no necesariamente usa exactamente los mismos archivos que la partición de prueba del notebook.

La alternativa OCI con Docker Compose y Caddy sigue disponible para quien tenga acceso a una VM; consulta `compose.yaml` y `deploy/Caddyfile`. No es necesaria para el despliegue gratuito descrito arriba.
