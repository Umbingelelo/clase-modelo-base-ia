"""
Entrenamiento + quality gate · microservicio de IA

Version "script" del notebook. Es lo que corre el job de la imagen de IA en el
pipeline (EP2, seccion 2.2):

    1. lee los datos tratados por datos/preparar.py
    2. separa entrenamiento / validacion / prueba ANTES de entrenar
    3. entrena (PyTorch, TensorFlow o scikit-learn) y elige el umbral con validacion
    4. si la PR-AUC de prueba no alcanza el minimo -> exit 1 (no se construye imagen)
    5. exporta a servicio-ia/model/:
         pytorch / tensorflow -> preprocesador.joblib + modelo.onnx + metadata.json
         sklearn              -> modelo.joblib + metadata.json

Las dependencias de ENTRENAMIENTO (torch, tensorflow, tf2onnx) se instalan en
el job de CI o en tu computador, NUNCA en la imagen. La imagen solo lleva
onnxruntime (ver requirements.txt).

Uso:
    python datos/preparar.py
    python servicio-ia/entrenar.py                                  # PyTorch (por defecto)
    python servicio-ia/entrenar.py --framework tensorflow
    python servicio-ia/entrenar.py --framework sklearn --algoritmo rf
    python servicio-ia/entrenar.py --capas 128,64 --dropout 0.3 --epocas 80 --version 1.1.0-xx
    python servicio-ia/entrenar.py --pr-auc-minimo 0.90             # para ver fallar el gate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos" / "camiones_tratado.csv"
REPORTE_LIMPIEZA = RAIZ / "datos" / "reporte_limpieza.json"
SALIDA = Path(__file__).resolve().parent / "model"

SEMILLA = 42
VARIABLES_NUMERICAS = [
    "antiguedad_anios", "horas_desde_mantencion", "carga_promedio_ton", "temp_motor_c",
    "presion_aceite_psi", "vibracion_mm_s", "rpm_promedio", "alertas_24h",
]
VARIABLES_CATEGORICAS = ["flota", "turno"]
OBJETIVO = "falla_7d"

COSTO_FALLA_NO_DETECTADA = 20_000
COSTO_INSPECCION = 2_500


def costo(y_real, y_pred) -> int:
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    fn = int(((y_pred == 0) & (y_real == 1)).sum())
    return fn * COSTO_FALLA_NO_DETECTADA + int(y_pred.sum()) * COSTO_INSPECCION


def mejor_umbral(y_real, prob) -> float:
    candidatos = np.arange(0.02, 0.96, 0.01)
    return float(candidatos[int(np.argmin([costo(y_real, prob >= u) for u in candidatos]))])


def construir_preprocesador() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                          ("escalar", StandardScaler())]), VARIABLES_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), VARIABLES_CATEGORICAS),
    ])


# ---------------------------------------------------------------- PyTorch
def entrenar_pytorch(A_tr, y_tr, capas, dropout, epocas, lr):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEMILLA)
    bloques, entrada = [], A_tr.shape[1]
    for n in capas:
        bloques += [nn.Linear(entrada, n), nn.ReLU(), nn.Dropout(dropout)]
        entrada = n
    red = nn.Sequential(*bloques, nn.Linear(entrada, 1))

    peso_pos = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], dtype=torch.float32)
    perdida = nn.BCEWithLogitsLoss(pos_weight=peso_pos)
    optim = torch.optim.Adam(red.parameters(), lr=lr, weight_decay=1e-4)
    X = torch.from_numpy(A_tr)
    Y = torch.from_numpy(np.asarray(y_tr, dtype=np.float32)).unsqueeze(1)
    gen = torch.Generator().manual_seed(SEMILLA)
    for _ in range(epocas):
        red.train()
        orden = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), 64):
            idx = orden[i:i + 64]
            optim.zero_grad()
            perdida(red(X[idx]), Y[idx]).backward()
            optim.step()
    red.eval()

    class ConProbabilidad(nn.Module):          # el ONNX devuelve probabilidad, no logit
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.sigmoid(self.m(x))

    final = ConProbabilidad(red).eval()

    def predecir(A):
        with torch.no_grad():
            return final(torch.from_numpy(A)).numpy().ravel()

    def exportar(ruta: Path):
        import warnings
        warnings.simplefilter("ignore")   # aviso de exportador "legacy": es el mas compatible con Colab
        torch.onnx.export(final, torch.zeros(1, A_tr.shape[1]), str(ruta),
                          input_names=["x"], output_names=["prob"],
                          dynamic_axes={"x": {0: "n"}, "prob": {0: "n"}},
                          opset_version=17, dynamo=False)

    return predecir, exportar, f"torch {torch.__version__}"


# ---------------------------------------------------------------- TensorFlow / Keras
def entrenar_tensorflow(A_tr, y_tr, capas, dropout, epocas, lr):
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(SEMILLA)
    red = keras.Sequential([keras.Input(shape=(A_tr.shape[1],))])
    for n in capas:
        red.add(keras.layers.Dense(n, activation="relu"))
        red.add(keras.layers.Dropout(dropout))
    red.add(keras.layers.Dense(1, activation="sigmoid"))
    red.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
    peso_pos = float((y_tr == 0).sum() / (y_tr == 1).sum())
    red.fit(A_tr, np.asarray(y_tr), epochs=epocas, batch_size=64,
            class_weight={0: 1.0, 1: peso_pos}, verbose=0)

    def predecir(A):
        return red.predict(A, verbose=0).ravel()

    def exportar(ruta: Path):
        red.export(str(ruta), format="onnx")   # requiere tf2onnx

    return predecir, exportar, f"tensorflow {tf.__version__}"


# ---------------------------------------------------------------- scikit-learn
def entrenar_sklearn(algoritmo):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=SEMILLA),
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              max_leaf_nodes=15, random_state=SEMILLA),
    }[algoritmo]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["pytorch", "tensorflow", "sklearn"], default="pytorch")
    ap.add_argument("--algoritmo", choices=["logreg", "rf", "hgb"], default="logreg",
                    help="solo con --framework sklearn")
    ap.add_argument("--capas", default="64,32", help="neuronas por capa oculta, ej. 128,64")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--version", default=None)
    ap.add_argument("--pr-auc-minimo", type=float, default=0.55,
                    help="quality gate: si PR-AUC en prueba es menor, termina con codigo 1")
    args = ap.parse_args()
    version = args.version or f"1.0.0-{args.framework}"
    capas = [int(n) for n in args.capas.split(",") if n.strip()]

    if not DATOS.exists():
        print(f"No existe {DATOS}. Corre primero: python datos/preparar.py", file=sys.stderr)
        return 2

    df = pd.read_csv(DATOS)
    X = df[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS]
    y = df[OBJETIVO].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=SEMILLA)

    prep = construir_preprocesador().fit(X_tr)          # se ajusta SOLO con entrenamiento
    A = lambda d: prep.transform(d).astype(np.float32)  # noqa: E731
    A_tr, A_val, A_te = A(X_tr), A(X_val), A(X_te)

    if args.framework == "sklearn":
        pipe = Pipeline([("prep", prep), ("clf", entrenar_sklearn(args.algoritmo))]).fit(X_tr, y_tr)
        predecir = lambda d: pipe.predict_proba(d)[:, 1]  # noqa: E731
        p_val, p_te = predecir(X_val), predecir(X_te)
        motor = f"scikit-learn {sklearn.__version__}"
        descripcion = f"scikit-learn {args.algoritmo}"
    else:
        entrenar = entrenar_pytorch if args.framework == "pytorch" else entrenar_tensorflow
        predecir, exportar_onnx, motor = entrenar(A_tr, y_tr, capas, args.dropout, args.epocas, args.lr)
        p_val, p_te = predecir(A_val), predecir(A_te)
        descripcion = f"MLP {capas} dropout={args.dropout} epocas={args.epocas} lr={args.lr}"

    umbral = mejor_umbral(y_val, p_val)                  # decidido SIN mirar prueba
    pred = (p_te >= umbral).astype(int)
    metricas = {
        "roc_auc": round(roc_auc_score(y_te, p_te), 4),
        "pr_auc": round(average_precision_score(y_te, p_te), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "costo_usd": costo(y_te, pred),
    }
    print(f"[{motor}] {descripcion} | umbral={umbral:.2f} {metricas}")

    # ---- Quality gate ---------------------------------------------------
    if metricas["pr_auc"] < args.pr_auc_minimo:
        print(f"✘ GATE: PR-AUC {metricas['pr_auc']} < minimo {args.pr_auc_minimo}. No se exporta.",
              file=sys.stderr)
        return 1
    print(f"✔ GATE: PR-AUC {metricas['pr_auc']} >= {args.pr_auc_minimo}")

    # ---- Exportar ---------------------------------------------------------
    SALIDA.mkdir(parents=True, exist_ok=True)
    for viejo in ["modelo.joblib", "modelo.onnx", "preprocesador.joblib"]:
        (SALIDA / viejo).unlink(missing_ok=True)

    if args.framework == "sklearn":
        formato = "sklearn"
        joblib.dump(pipe, SALIDA / "modelo.joblib", compress=3)
    else:
        formato = "onnx"
        joblib.dump(prep, SALIDA / "preprocesador.joblib")
        exportar_onnx(SALIDA / "modelo.onnx")
        # Verificacion: onnxruntime debe dar lo mismo que el framework
        import onnxruntime as ort
        sesion = ort.InferenceSession(str(SALIDA / "modelo.onnx"), providers=["CPUExecutionProvider"])
        p_onnx = sesion.run(None, {sesion.get_inputs()[0].name: A_te})[0].ravel()
        diferencia = float(np.abs(p_onnx - p_te).max())
        print(f"✔ ONNX verificado con onnxruntime: diferencia maxima {diferencia:.2e}")
        if diferencia > 1e-4:
            print("✘ El ONNX no reproduce al modelo original", file=sys.stderr)
            return 1

    metadata = {
        "nombre": "falla-caex-7d",
        "version": version,
        "formato": formato,
        "framework": motor,
        "descripcion": descripcion,
        "entrenado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "variables_numericas": VARIABLES_NUMERICAS,
        "variables_categoricas": VARIABLES_CATEGORICAS,
        "umbral": round(umbral, 4),
        "filas_entrenamiento": int(len(X_tr)),
        "metricas_prueba": metricas,
    }
    if REPORTE_LIMPIEZA.exists():
        metadata["limpieza"] = json.loads(REPORTE_LIMPIEZA.read_text(encoding="utf-8"))["reglas"]
    (SALIDA / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    tam = sum(f.stat().st_size for f in SALIDA.iterdir()) / 1024
    print(f"✔ Exportado {version} ({formato}) -> {SALIDA}  ({tam:,.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
