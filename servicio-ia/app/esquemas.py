"""El CONTRATO JSON del microservicio (lo que pregunta el IE6 de la defensa).

Pydantic valida cada peticion antes de que llegue al modelo: tipos, rangos y
categorias. Si algo no cumple, FastAPI responde 422 sin tocar el modelo.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EJEMPLO = {
    "antiguedad_anios": 11,
    "horas_desde_mantencion": 620,
    "carga_promedio_ton": 395,
    "temp_motor_c": 99.5,
    "presion_aceite_psi": 41.0,
    "vibracion_mm_s": 7.8,
    "rpm_promedio": 1980,
    "alertas_24h": 3,
    "flota": "C",
    "turno": "noche",
}


class LecturaCAEX(BaseModel):
    """Una lectura diaria de telemetria de un camion de extraccion."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO]})

    antiguedad_anios: int = Field(..., ge=0, le=40, description="Anios de servicio del camion")
    horas_desde_mantencion: float = Field(..., ge=0, le=5000)
    carga_promedio_ton: Optional[float] = Field(None, ge=0, le=450)
    # Sensores: pueden venir en null (el sensor no reporto). El pipeline del modelo los imputa.
    temp_motor_c: Optional[float] = Field(None, ge=40, le=140)
    presion_aceite_psi: Optional[float] = Field(None, ge=0, le=120)
    vibracion_mm_s: Optional[float] = Field(None, ge=0, le=50)
    rpm_promedio: float = Field(..., ge=0, le=3000)
    alertas_24h: int = Field(..., ge=0, le=50)
    flota: Literal["A", "B", "C"]
    turno: Literal["dia", "noche"]

    # Misma regla R2 de la limpieza: "DIA", " Noche", "día" -> "dia"/"noche"
    @field_validator("turno", mode="before")
    @classmethod
    def normalizar_turno(cls, v):
        if isinstance(v, str):
            return v.strip().lower().replace("í", "i")
        return v

    @field_validator("flota", mode="before")
    @classmethod
    def normalizar_flota(cls, v):
        return v.strip().upper() if isinstance(v, str) else v


class Prediccion(BaseModel):
    id_prediccion: str
    probabilidad_falla: float = Field(..., ge=0, le=1)
    alerta: bool = Field(..., description="True si probabilidad >= umbral -> mandar a inspeccion")
    nivel_riesgo: Literal["bajo", "medio", "alto"]
    umbral: float
    version_modelo: str
    latencia_ms: float
    fecha: datetime


class LoteLecturas(BaseModel):
    lecturas: list[LecturaCAEX] = Field(..., min_length=1, max_length=500)


class RespuestaLote(BaseModel):
    version_modelo: str
    total: int
    alertas: int
    latencia_ms: float
    predicciones: list[Prediccion]


class Salud(BaseModel):
    status: Literal["ok", "degradado"]
    modelo_cargado: bool
    version_modelo: Optional[str]
    revision: str
