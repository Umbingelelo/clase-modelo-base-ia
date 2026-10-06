# ---------------------------------------------------------------------------
# Imagen del microservicio de IA
#
#   docker build -t servicio-ia:1.0 .
#   docker run -d --name servicio-ia -p 8000:8000 servicio-ia:1.0
#
# En la EP2 esta imagen la construye el pipeline (no tu computador) y se
# publica en GHCR. En el compose.yaml de la defensa NO lleva "ports:":
# solo el gateway publica el 8080.
# ---------------------------------------------------------------------------

# Imagen base oficial, "slim": Debian minimo + Python. Sin compiladores ni CUDA.
FROM python:3.12-slim

# Commit del que sale la imagen (lo inyecta el pipeline con --build-arg)
ARG REVISION=dev
LABEL org.opencontainers.image.title="servicio-ia" \
      org.opencontainers.image.description="Prediccion de falla de CAEX en 7 dias" \
      org.opencontainers.image.revision=$REVISION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GIT_SHA=$REVISION \
    MODEL_DIR=/app/model

WORKDIR /app

# 1) Dependencias primero: esta capa se reutiliza (cache) mientras requirements.txt no cambie
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Codigo y modelo despues: cambian seguido, se reconstruyen rapido
COPY app/ ./app/
COPY model/ ./model/

# No correr como root dentro del contenedor
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Docker marca el contenedor como healthy/unhealthy segun /health
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
