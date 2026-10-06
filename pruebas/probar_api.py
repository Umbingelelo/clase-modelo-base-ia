"""
Set de pruebas del microservicio de IA (contra la API corriendo)
================================================================

Hace lo mismo que la colección de Postman, pero desde la terminal y sin
instalar nada (solo librería estándar de Python):

  1. Operación: /health y /modelo
  2. Casos conocidos: camión en riesgo, camión sano, sensores nulos, categorías sucias
  3. Contrato: entradas inválidas que deben responder 422
  4. Carga: N lecturas aleatorias → distribución de probabilidades y latencia (p50 / p95)
  5. Lote: /predict/lote debe dar lo mismo que /predict uno por uno

Uso (con la API levantada con docker compose o uvicorn):
    python pruebas/probar_api.py
    python pruebas/probar_api.py --url http://localhost:8000 --n 300

Deja un reporte en pruebas/reporte_pruebas.md y termina con código 1 si alguna prueba falla.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

RIESGO = {"antiguedad_anios": 11, "horas_desde_mantencion": 620, "carga_promedio_ton": 395,
          "temp_motor_c": 99.5, "presion_aceite_psi": 41.0, "vibracion_mm_s": 7.8,
          "rpm_promedio": 1980, "alertas_24h": 3, "flota": "C", "turno": "noche"}
SANO = {"antiguedad_anios": 2, "horas_desde_mantencion": 40, "carga_promedio_ton": 280,
        "temp_motor_c": 84.0, "presion_aceite_psi": 58.0, "vibracion_mm_s": 3.5,
        "rpm_promedio": 1650, "alertas_24h": 0, "flota": "A", "turno": "dia"}


def llamar(url: str, metodo: str = "GET", cuerpo=None, crudo: bytes | None = None):
    datos = crudo if crudo is not None else (json.dumps(cuerpo).encode() if cuerpo is not None else None)
    req = urllib.request.Request(url, data=datos, method=metodo,
                                 headers={"Content-Type": "application/json"} if datos else {})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            estado, texto = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        estado, texto = e.code, e.read().decode()
    ms = (time.perf_counter() - t0) * 1000
    try:
        return estado, json.loads(texto), ms
    except json.JSONDecodeError:
        return estado, texto, ms


def lectura_aleatoria(rng: random.Random) -> dict:
    """Lectura con distribuciones parecidas a la telemetría real del caso (no uniformes)."""
    flota = rng.choice("AABBC")
    capacidad = {"A": 290, "B": 330, "C": 360}[flota]
    horas = rng.gammavariate(2.2, 110)
    return {
        "antiguedad_anios": rng.randint(1, 14),
        "horas_desde_mantencion": round(horas),
        "carga_promedio_ton": round(capacidad * rng.gauss(0.97, 0.06), 1),
        "temp_motor_c": round(rng.gauss(86, 5.5) + 0.012 * horas, 1),
        "presion_aceite_psi": round(rng.gauss(56, 6) - 0.010 * horas, 1),
        "vibracion_mm_s": round(rng.lognormvariate(1.435, 0.32), 2),
        "rpm_promedio": round(rng.gauss(1650, 170)),
        "alertas_24h": min(sum(rng.random() < 0.1 for _ in range(6)), 50),
        "flota": flota,
        "turno": rng.choice(["dia", "noche"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200, help="lecturas aleatorias para la prueba de carga")
    ap.add_argument("--semilla", type=int, default=7)
    args = ap.parse_args()
    base = args.url.rstrip("/")
    resultados: list[tuple[str, str, bool, str]] = []   # (grupo, prueba, ok, detalle)

    def chequear(grupo, nombre, condicion, detalle=""):
        resultados.append((grupo, nombre, bool(condicion), detalle))
        print(f"  {'✔' if condicion else '✘'} {nombre}" + (f"  · {detalle}" if detalle else ""))

    # 1 · Operación ---------------------------------------------------------
    print("1 · Operación")
    try:
        e, salud, _ = llamar(f"{base}/health")
    except urllib.error.URLError as err:
        print(f"No se pudo conectar a {base}: {err.reason}. ¿Está levantada la API?")
        return 2
    version = salud.get("version_modelo") if isinstance(salud, dict) else None
    chequear("Operación", "GET /health responde 200 con modelo cargado",
             e == 200 and salud.get("modelo_cargado"), f"versión {version}, revisión {salud.get('revision')}")
    e, meta, _ = llamar(f"{base}/modelo")
    chequear("Operación", "GET /modelo trae umbral, formato y métricas",
             e == 200 and {"umbral", "formato", "metricas_prueba"} <= set(meta),
             f"formato {meta.get('formato')}, framework {meta.get('framework')}, umbral {meta.get('umbral')}")

    # 2 · Casos conocidos ---------------------------------------------------
    print("2 · Casos conocidos")
    casos = [
        ("Camión en riesgo → alerta", RIESGO, True),
        ("Camión sano → sin alerta", SANO, False),
        ("Sensores en null (los imputa el preprocesador)", {**RIESGO, "temp_motor_c": None, "vibracion_mm_s": None}, True),
        ("Categorías sucias ' NOCHE' y 'c' (las normaliza el contrato)", {**RIESGO, "turno": " NOCHE", "flota": "c"}, True),
    ]
    for nombre, cuerpo, espera_alerta in casos:
        e, r, ms = llamar(f"{base}/predict", "POST", cuerpo)
        ok = (e == 200 and r.get("alerta") is espera_alerta and r.get("version_modelo") == version)
        chequear("Casos conocidos", nombre, ok,
                 f"prob {r.get('probabilidad_falla')}, riesgo {r.get('nivel_riesgo')}, {ms:.1f} ms" if e == 200 else f"HTTP {e}")

    # 3 · Contrato ----------------------------------------------------------
    print("3 · Contrato (deben responder 422)")
    invalidos = [
        ("Flota inexistente", {**SANO, "flota": "Z"}),
        ("Turno inexistente", {**SANO, "turno": "tarde"}),
        ("Sensor con código de error -999", {**SANO, "temp_motor_c": -999}),
        ("Carga imposible 9999", {**SANO, "carga_promedio_ton": 9999}),
        ("Falta un campo obligatorio", {k: v for k, v in SANO.items() if k != "rpm_promedio"}),
        ("Campo que no existe en el contrato", {**SANO, "id_camion": "CAEX-007"}),
        ("Texto donde va un número", {**SANO, "alertas_24h": "muchas"}),
    ]
    for nombre, cuerpo in invalidos:
        e, r, _ = llamar(f"{base}/predict", "POST", cuerpo)
        campo = r["detail"][0]["loc"][-1] if e == 422 and isinstance(r, dict) and r.get("detail") else ""
        chequear("Contrato", nombre, e == 422, f"HTTP {e}, campo: {campo}")
    e, _, _ = llamar(f"{base}/predict", "POST", crudo=b'{"flota": "A", ')
    chequear("Contrato", "JSON mal formado", e == 422, f"HTTP {e}")

    # 4 · Carga -------------------------------------------------------------
    print(f"4 · Carga ({args.n} lecturas aleatorias)")
    rng = random.Random(args.semilla)
    lecturas = [lectura_aleatoria(rng) for _ in range(args.n)]
    probs, tiempos, errores = [], [], 0
    for l in lecturas:
        e, r, ms = llamar(f"{base}/predict", "POST", l)
        if e == 200:
            probs.append(r["probabilidad_falla"]); tiempos.append(ms)
        else:
            errores += 1
    tiempos.sort()
    p50 = statistics.median(tiempos) if tiempos else float("nan")
    p95 = tiempos[int(len(tiempos) * 0.95) - 1] if tiempos else float("nan")
    alertas = sum(p >= meta.get("umbral", 0.5) for p in probs)
    chequear("Carga", f"{args.n} peticiones sin errores", errores == 0, f"{errores} errores")
    chequear("Carga", "Probabilidades entre 0 y 1", all(0 <= p <= 1 for p in probs),
             f"mín {min(probs):.3f} · mediana {statistics.median(probs):.3f} · máx {max(probs):.3f}")
    chequear("Carga", "Latencia p95 bajo 300 ms", p95 < 300, f"p50 {p50:.1f} ms · p95 {p95:.1f} ms")
    print(f"     alertas: {alertas} de {len(probs)} ({alertas / max(len(probs), 1):.0%})")

    # 5 · Lote --------------------------------------------------------------
    print("5 · Lote")
    lote = lecturas[:100]
    e, r, ms = llamar(f"{base}/predict/lote", "POST", {"lecturas": lote})
    ok_lote = e == 200 and r.get("total") == len(lote)
    chequear("Lote", f"/predict/lote con {len(lote)} lecturas", ok_lote, f"{ms:.1f} ms en total")
    if ok_lote:
        dif = max(abs(a["probabilidad_falla"] - b) for a, b in zip(r["predicciones"], probs[:100]))
        chequear("Lote", "Lote da lo mismo que /predict uno por uno", dif < 1e-6, f"diferencia máxima {dif:.1e}")
    e, _, _ = llamar(f"{base}/predict/lote", "POST", {"lecturas": [SANO] * 501})
    chequear("Lote", "Lote de 501 lecturas se rechaza (máximo 500)", e == 422, f"HTTP {e}")

    # Reporte ---------------------------------------------------------------
    total, oks = len(resultados), sum(r[2] for r in resultados)
    print(f"\nResultado: {oks}/{total} pruebas OK")
    filas = "\n".join(f"| {g} | {n} | {'✔' if ok else '✘'} | {d} |" for g, n, ok, d in resultados)
    reporte = f"""# Reporte de pruebas · servicio de IA

- Fecha: {datetime.now():%Y-%m-%d %H:%M}
- URL: {base}
- Modelo: {version} ({meta.get('framework')}, formato {meta.get('formato')}, umbral {meta.get('umbral')})
- Resultado: **{oks}/{total} pruebas OK**
- Carga: {args.n} peticiones · latencia p50 {p50:.1f} ms · p95 {p95:.1f} ms · alertas {alertas}/{len(probs)}

| Grupo | Prueba | OK | Detalle |
|---|---|---|---|
{filas}
"""
    salida = Path(__file__).with_name("reporte_pruebas.md")
    salida.write_text(reporte, encoding="utf-8")
    print(f"Reporte: {salida}")
    return 0 if oks == total else 1


if __name__ == "__main__":
    sys.exit(main())
