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

## Despliegue gratuito en Oracle Cloud

La carpeta `deploy/`, el `Dockerfile` y `compose.yaml` preparan el sitio para una VM Linux ARM64. Oracle Cloud ofrece recursos Always Free que, si están disponibles en tu región, superan la RAM medida para la inferencia local. La cuenta y la capacidad de la región dependen de Oracle; verifica las condiciones actuales de Always Free. La aplicación está configurada para un solo worker porque cargar el modelo consume bastante memoria.

1. Crea una VM Ubuntu ARM64 Always Free y permite tráfico TCP entrante por los puertos **80** y **443** en sus reglas de red. Limita el acceso SSH al origen que administra la VM.
2. Crea un subdominio gratuito en DuckDNS y apunta su registro IPv4 a la dirección pública de la VM.
3. Instala Docker Engine con el complemento Compose siguiendo la documentación oficial de Docker. En la VM:

   ```bash
   sudo mkdir -p /opt/perufoodnet
   sudo chown "$USER":"$USER" /opt/perufoodnet
   git clone https://github.com/BillGutierrez/ingenieriaDelConocimientoRemix.git /opt/perufoodnet
   cd /opt/perufoodnet
   cp .env.example .env
   chmod 600 .env
   ```

4. Genera un secreto con `python3 -c "import secrets; print(secrets.token_urlsafe(64))"` y edita `.env`. Sustituye el dominio de ejemplo en `SITE_DOMAIN`, `DJANGO_ALLOWED_HOSTS` y `DJANGO_CSRF_TRUSTED_ORIGINS`; pega el secreto generado en `DJANGO_SECRET_KEY`. No publiques ni compartas el archivo `.env`.
5. Inicia los servicios:

   ```bash
   docker compose up -d --build
   docker compose ps
   docker compose logs --tail=100 web caddy
   ```

   La primera compilación descarga PyTorch y puede tardar varios minutos. Caddy solicitará un certificado HTTPS automáticamente cuando el DNS ya apunte a la VM y los puertos 80/443 sean accesibles.
6. Abre `https://TU-SUBDOMINIO.duckdns.org/`. Para administrar las fichas, crea una cuenta con `docker compose exec web python manage.py createsuperuser`.

La base SQLite y los certificados se guardan en volúmenes persistentes de Docker. El modelo incluido en Git se empaqueta en la imagen. El catálogo se migra y se completa automáticamente al iniciar el servicio. Antes de administrar datos en producción, configura copias de seguridad del volumen `catalog_data`.

El chequeo `manage.py check --deploy` puede advertir que HSTS preload está desactivado. Se deja así para no solicitar la inclusión de un subdominio gratuito compartido; no actives `DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS` ni preload salvo que controles el dominio y comprendas sus implicaciones.
