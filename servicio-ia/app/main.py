"""
Microservicio de IA · Prediccion de falla de CAEX en 7 dias

Endpoints
    GET  /health          -> salud del servicio (lo usan Docker, el gateway y el balanceador)
    GET  /modelo          -> metadata del modelo servido
    POST /predict         -> una lectura   -> una prediccion
    POST /predict/lote    -> hasta 500 lecturas

Correr en local:
    uvicorn app.main:app --reload --port 8000      (desde la carpeta servicio-ia)
    Documentacion interactiva: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .esquemas import LecturaCAEX, LoteLecturas, Prediccion, RespuestaLote, Salud
from .modelo import modelo

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("servicio-ia")

REVISION = os.getenv("GIT_SHA", "dev")   # commit del que salio la imagen (IE10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El modelo se carga UNA vez al arrancar, no en cada peticion.
    try:
        modelo.cargar()
    except Exception:  # el servicio arranca igual, pero /health responde 503
        log.exception("No se pudo cargar el modelo")
    yield


app = FastAPI(
    title="Servicio IA · Falla CAEX 7 dias",
    description="Minera Pampa Norte (caso ficticio). Predice la probabilidad de que un camion "
                "de extraccion falle en los proximos 7 dias.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def medir_tiempo(request: Request, call_next):
    t0 = time.perf_counter()
    respuesta = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    respuesta.headers["X-Tiempo-ms"] = f"{ms:.2f}"
    respuesta.headers["X-Version-Modelo"] = modelo.version or "sin-modelo"
    log.info("%s %s -> %s en %.2f ms", request.method, request.url.path, respuesta.status_code, ms)
    return respuesta


def _exigir_modelo() -> None:
    if not modelo.cargado:
        raise HTTPException(status_code=503, detail="Modelo no disponible")


@app.get("/", include_in_schema=False)
def raiz():
    return {"servicio": "servicio-ia", "version_modelo": modelo.version, "docs": "/docs"}


@app.get("/health", response_model=Salud, tags=["operacion"])
def health():
    cuerpo = Salud(
        status="ok" if modelo.cargado else "degradado",
        modelo_cargado=modelo.cargado,
        version_modelo=modelo.version,
        revision=REVISION,
    )
    # 503 hace que Docker / el gateway saquen esta replica de circulacion
    return JSONResponse(cuerpo.model_dump(), status_code=200 if modelo.cargado else 503)


@app.get("/modelo", tags=["operacion"])
def info_modelo():
    _exigir_modelo()
    return modelo.metadata


@app.post("/predict", response_model=Prediccion, tags=["prediccion"])
def predict(lectura: LecturaCAEX):
    _exigir_modelo()
    t0 = time.perf_counter()
    p = modelo.predecir([lectura.model_dump()])[0]
    return Prediccion(
        id_prediccion=str(uuid.uuid4()),
        probabilidad_falla=round(p, 4),
        alerta=p >= modelo.umbral,
        nivel_riesgo=modelo.nivel(p),
        umbral=modelo.umbral,
        version_modelo=modelo.version,
        latencia_ms=round((time.perf_counter() - t0) * 1000, 3),
        fecha=datetime.now(timezone.utc),
    )


@app.post("/predict/lote", response_model=RespuestaLote, tags=["prediccion"])
def predict_lote(lote: LoteLecturas):
    _exigir_modelo()
    t0 = time.perf_counter()
    probs = modelo.predecir([l.model_dump() for l in lote.lecturas])
    ms = round((time.perf_counter() - t0) * 1000, 3)
    ahora = datetime.now(timezone.utc)
    preds = [
        Prediccion(id_prediccion=str(uuid.uuid4()), probabilidad_falla=round(p, 4),
                   alerta=p >= modelo.umbral, nivel_riesgo=modelo.nivel(p), umbral=modelo.umbral,
                   version_modelo=modelo.version, latencia_ms=ms, fecha=ahora)
        for p in probs
    ]
    return RespuestaLote(version_modelo=modelo.version, total=len(preds),
                         alertas=sum(p.alerta for p in preds), latencia_ms=ms, predicciones=preds)
