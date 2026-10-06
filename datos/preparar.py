"""
Tratamiento de datos · Caso 0 · Mantenimiento predictivo de CAEX

Version "script" de las secciones 2 y 3 del notebook. Es lo que ejecuta el
pipeline de CI (EP2, seccion 2.2): perfila, limpia con reglas escritas y deja
las salidas tratadas.

Entradas : datos/camiones_crudo.csv
Salidas  : datos/camiones_tratado.csv
           datos/reporte_limpieza.json   (cifras antes/despues -> capitulo 8.4)

Uso:
    python datos/preparar.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).parent
ENTRADA = CARPETA / "camiones_crudo.csv"
SALIDA = CARPETA / "camiones_tratado.csv"
REPORTE = CARPETA / "reporte_limpieza.json"

RANGOS = {
    "temp_motor_c": (40, 140),
    "presion_aceite_psi": (0, 120),
    "carga_promedio_ton": (0, 450),
    "vibracion_mm_s": (0, 50),
}


def perfilar(df: pd.DataFrame) -> dict:
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "duplicados_exactos": int(df.duplicated().sum()),
        "nulos": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "categorias": {c: sorted(df[c].dropna().unique().tolist()) for c in ["flota", "turno"]},
    }


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    reglas: dict[str, int] = {}

    # R1 · duplicados por id_lectura -> descartar
    antes = len(df)
    df = df.drop_duplicates(subset="id_lectura")
    reglas["R1_duplicados_descartados"] = antes - len(df)

    # R2 · categorias escritas de distintas formas -> corregir
    df["turno"] = (
        df["turno"].str.strip().str.lower()
        .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    )
    df["flota"] = df["flota"].str.strip().str.upper()

    # R3 · sensor fuera de rango fisico -> marcar como nulo (lo imputa el pipeline del modelo)
    for col, (lo, hi) in RANGOS.items():
        fuera = df[col].notna() & ~df[col].between(lo, hi)
        reglas[f"R3_{col}_fuera_de_rango"] = int(fuera.sum())
        df.loc[fuera, col] = np.nan

    # R4 · etiqueta faltante -> descartar
    antes = len(df)
    df = df.dropna(subset=["falla_7d"])
    df["falla_7d"] = df["falla_7d"].astype(int)
    reglas["R4_sin_etiqueta_descartadas"] = antes - len(df)

    return df, reglas


def validar(df: pd.DataFrame) -> None:
    """Lo mismo que prueba el test del pipeline: sin duplicados ni nulos en columnas clave."""
    assert df["id_lectura"].is_unique, "Quedaron id_lectura duplicados"
    for col in ["id_lectura", "id_camion", "flota", "turno", "falla_7d"]:
        assert df[col].notna().all(), f"Nulos en columna clave {col}"
    assert set(df["turno"]) <= {"dia", "noche"}, f"turno sin normalizar: {set(df['turno'])}"
    assert set(df["flota"]) <= {"A", "B", "C"}, f"flota sin normalizar: {set(df['flota'])}"


def main() -> None:
    crudo = pd.read_csv(ENTRADA)
    tratado, reglas = limpiar(crudo)
    validar(tratado)

    reporte = {"antes": perfilar(crudo), "reglas": reglas, "despues": perfilar(tratado)}
    tratado.to_csv(SALIDA, index=False)
    REPORTE.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Antes : {reporte['antes']['filas']:,} filas")
    for regla, n in reglas.items():
        print(f"  {regla:<40} {n:>5}")
    print(f"Despues: {reporte['despues']['filas']:,} filas  ->  {SALIDA.name}")


if __name__ == "__main__":
    main()
