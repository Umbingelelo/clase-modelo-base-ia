"""Carga del modelo e inferencia (componente 'modelo' del diagrama C4 nivel 3).

Soporta dos formatos de artefacto, segun metadata.json -> "formato":

  "onnx"    (recomendado)  preprocesador.joblib + modelo.onnx
            La red se entreno con PyTorch o TensorFlow y se exporto a ONNX.
            Se sirve con onnxruntime: la imagen NO necesita torch ni tensorflow.

  "sklearn"                 modelo.joblib (Pipeline completo de scikit-learn)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

log = logging.getLogger("servicio-ia")

CARPETA_MODELO = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))


class Modelo:
    def __init__(self) -> None:
        self.metadata: dict = {}
        self.formato: str | None = None
        self._pipeline = None        # formato sklearn
        self._prep = None            # formato onnx: preprocesamiento (scikit-learn)
        self._sesion = None          # formato onnx: sesion de onnxruntime
        self._entrada = None         # nombre del tensor de entrada del grafo ONNX

    @property
    def cargado(self) -> bool:
        return self._pipeline is not None or self._sesion is not None

    @property
    def version(self) -> str | None:
        return self.metadata.get("version")

    @property
    def umbral(self) -> float:
        return float(self.metadata.get("umbral", 0.5))

    @property
    def columnas(self) -> list[str]:
        return self.metadata["variables_numericas"] + self.metadata["variables_categoricas"]

    def cargar(self, carpeta: Path = CARPETA_MODELO) -> None:
        self.metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
        self.formato = self.metadata.get("formato", "sklearn")

        entrenado_con = self.metadata.get("sklearn_version")
        if entrenado_con and entrenado_con != sklearn.__version__:
            log.warning("Preprocesamiento creado con scikit-learn %s y servido con %s: "
                        "las predicciones pueden diferir", entrenado_con, sklearn.__version__)

        if self.formato == "onnx":
            import onnxruntime as ort  # solo se necesita si el modelo es ONNX

            self._prep = joblib.load(carpeta / "preprocesador.joblib")
            opciones = ort.SessionOptions()
            opciones.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
            self._sesion = ort.InferenceSession(str(carpeta / "modelo.onnx"), opciones,
                                                providers=["CPUExecutionProvider"])
            self._entrada = self._sesion.get_inputs()[0].name
        else:
            self._pipeline = joblib.load(carpeta / "modelo.joblib")

        log.info("Modelo %s (%s, entrenado con %s) cargado desde %s, umbral %.2f",
                 self.version, self.formato, self.metadata.get("framework", "scikit-learn"),
                 carpeta, self.umbral)

    def predecir(self, filas: list[dict]) -> list[float]:
        df = pd.DataFrame(filas, columns=self.columnas)
        if self.formato == "onnx":
            x = np.asarray(self._prep.transform(df), dtype=np.float32)
            prob = self._sesion.run(None, {self._entrada: x})[0]
            return np.asarray(prob, dtype=float).reshape(-1).tolist()
        return self._pipeline.predict_proba(df)[:, 1].tolist()

    def nivel(self, p: float) -> str:
        if p < self.umbral:
            return "bajo"
        return "alto" if p >= max(0.6, self.umbral) else "medio"


modelo = Modelo()
