"""
Generador de datos sinteticos · Caso 0 · Mantenimiento predictivo de CAEX
Minera Pampa Norte (empresa ficticia) · Antofagasta

Simula lecturas diarias de telemetria de una flota de camiones de extraccion
(CAEX) y una etiqueta: si el camion tuvo una falla en los 7 dias siguientes.

Los datos salen "sucios a proposito", igual que los de los casos semestrales:
duplicados, nulos, categorias escritas de distintas formas y lecturas de
sensor fuera de rango. El notebook los perfila y los limpia.

Uso:
    python datos/generar_datos.py                 # genera datos/camiones_crudo.csv
    python datos/generar_datos.py --filas 8000 --semilla 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FLOTAS = {
    # flota: (capacidad nominal en toneladas, sesgo de riesgo)
    "A": (290, 0.00),
    "B": (330, 0.15),
    "C": (360, 0.55),  # la flota mas exigida
}


def _verdad_oculta(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """La 'fisica' que el modelo tiene que descubrir.

    Es no lineal a proposito: hay umbrales (temperatura sobre 95 C, presion
    bajo 45 psi), una interaccion (calor con carga alta) y una curva en U
    (rpm). Un modelo lineal capta una parte; un modelo de arboles, mas.
    """
    sesgo_flota = df["flota"].map({k: v[1] for k, v in FLOTAS.items()}).to_numpy()
    capacidad = df["flota"].map({k: v[0] for k, v in FLOTAS.items()}).to_numpy()
    sobrecarga = np.clip(df["carga_promedio_ton"].to_numpy() / capacidad - 1.0, 0, None)

    logit = (
        -8.2
        + 0.006 * df["horas_desde_mantencion"].to_numpy()
        + 0.55 * np.clip(df["temp_motor_c"].to_numpy() - 93, 0, None)
        + 0.40 * np.clip(48 - df["presion_aceite_psi"].to_numpy(), 0, None)
        + 1.80 * np.clip(df["vibracion_mm_s"].to_numpy() - 6, 0, None)
        + 30.0 * sobrecarga * (df["temp_motor_c"].to_numpy() > 90)
        + 3.0e-5 * (df["rpm_promedio"].to_numpy() - 1650) ** 2
        + 0.60 * df["alertas_24h"].to_numpy()
        + 0.08 * df["antiguedad_anios"].to_numpy()
        + 0.30 * (df["turno"].to_numpy() == "noche")
        + sesgo_flota
    )
    prob = 1 / (1 + np.exp(-logit))
    return (rng.random(len(df)) < prob).astype(int)


def generar(filas: int = 6000, semilla: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    n_camiones = 60
    camiones = pd.DataFrame(
        {
            "id_camion": [f"CAEX-{i:03d}" for i in range(1, n_camiones + 1)],
            "flota": rng.choice(list(FLOTAS), size=n_camiones, p=[0.4, 0.35, 0.25]),
            "antiguedad_anios": rng.integers(1, 15, size=n_camiones),
        }
    )

    idx = rng.integers(0, n_camiones, size=filas)
    df = camiones.iloc[idx].reset_index(drop=True)

    capacidad = df["flota"].map({k: v[0] for k, v in FLOTAS.items()}).to_numpy()
    fechas = pd.Timestamp("2026-03-01") + pd.to_timedelta(rng.integers(0, 180, filas), unit="D")

    df.insert(0, "id_lectura", np.arange(100001, 100001 + filas))
    df["fecha"] = fechas.strftime("%Y-%m-%d")
    df["turno"] = rng.choice(["dia", "noche"], size=filas)
    df["horas_desde_mantencion"] = rng.gamma(2.2, 110, filas).round(0)
    df["carga_promedio_ton"] = (capacidad * rng.normal(0.97, 0.06, filas)).round(1)
    df["temp_motor_c"] = (
        rng.normal(86, 5.5, filas) + 0.012 * df["horas_desde_mantencion"].to_numpy()
    ).round(1)
    df["presion_aceite_psi"] = (
        rng.normal(56, 6, filas) - 0.010 * df["horas_desde_mantencion"].to_numpy()
    ).round(1)
    df["vibracion_mm_s"] = rng.lognormal(np.log(4.2), 0.32, filas).round(2)
    df["rpm_promedio"] = rng.normal(1650, 170, filas).round(0)
    df["alertas_24h"] = rng.poisson(0.35 + 0.0012 * df["horas_desde_mantencion"].to_numpy())

    df["falla_7d"] = _verdad_oculta(df, rng)

    columnas = [
        "id_lectura", "id_camion", "fecha", "flota", "turno", "antiguedad_anios",
        "horas_desde_mantencion", "carga_promedio_ton", "temp_motor_c",
        "presion_aceite_psi", "vibracion_mm_s", "rpm_promedio", "alertas_24h", "falla_7d",
    ]
    df = df[columnas]
    return _ensuciar(df, rng)


def _ensuciar(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Introduce los problemas tipicos de datos reales."""
    df = df.copy()
    n = len(df)
    df["falla_7d"] = df["falla_7d"].astype("Int64")

    # 1. Categorias escritas de distintas formas
    variantes_turno = {"dia": ["Dia", "DIA", "día", "dia "], "noche": ["Noche", "NOCHE", " noche"]}
    for valor, variantes in variantes_turno.items():
        mask = (df["turno"] == valor) & (rng.random(n) < 0.12)
        df.loc[mask, "turno"] = rng.choice(variantes, size=mask.sum())
    mask = rng.random(n) < 0.05
    df.loc[mask, "flota"] = df.loc[mask, "flota"].str.lower() + " "

    # 2. Nulos en sensores (el sensor no reporto)
    for col, tasa in [("temp_motor_c", 0.015), ("presion_aceite_psi", 0.02), ("vibracion_mm_s", 0.01)]:
        df.loc[rng.random(n) < tasa, col] = np.nan

    # 3. Lecturas fuera de rango (sensor descalibrado o codigo de error)
    df.loc[rng.random(n) < 0.004, "temp_motor_c"] = -999.0
    df.loc[rng.random(n) < 0.003, "presion_aceite_psi"] = -1.0
    df.loc[rng.random(n) < 0.003, "carga_promedio_ton"] = 9999.0

    # 4. Etiqueta faltante (no se pudo confirmar si hubo falla)
    df.loc[rng.random(n) < 0.004, "falla_7d"] = pd.NA

    # 5. Filas duplicadas (el ETL de telemetria reenvio el lote)
    dup = df.sample(n=int(n * 0.012), random_state=int(rng.integers(0, 10_000)))
    df = pd.concat([df, dup], ignore_index=True)
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filas", type=int, default=6000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", type=Path, default=Path(__file__).parent / "camiones_crudo.csv")
    args = parser.parse_args()

    datos = generar(args.filas, args.semilla)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(args.salida, index=False)
    print(f"Generado {args.salida} con {len(datos):,} filas y {datos.shape[1]} columnas")
    print(f"Tasa de falla (sin nulos): {datos['falla_7d'].dropna().astype(int).mean():.1%}")
