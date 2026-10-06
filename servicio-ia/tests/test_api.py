"""Tests del microservicio de IA. El pipeline de CI los corre ANTES del docker build.

    cd servicio-ia
    pytest -q
"""

from fastapi.testclient import TestClient

from app.esquemas import EJEMPLO
from app.main import app


def cliente():
    return TestClient(app)  # usado como context manager para que corra el lifespan


def test_health_ok_y_modelo_cargado():
    with cliente() as c:
        r = c.get("/health")
        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["modelo_cargado"] is True
        assert cuerpo["version_modelo"]


def test_predict_devuelve_prediccion_valida_para_entrada_conocida():
    with cliente() as c:
        r = c.post("/predict", json=EJEMPLO)
        assert r.status_code == 200
        cuerpo = r.json()
        assert 0.0 <= cuerpo["probabilidad_falla"] <= 1.0
        assert cuerpo["version_modelo"]
        assert cuerpo["nivel_riesgo"] in {"bajo", "medio", "alto"}
        # Camion viejo, recalentado, sin presion y sobrecargado: debe alertar
        assert cuerpo["alerta"] is True


def test_predict_acepta_sensor_nulo_y_normaliza_categorias():
    with cliente() as c:
        lectura = {**EJEMPLO, "temp_motor_c": None, "turno": " NOCHE", "flota": "c"}
        assert c.post("/predict", json=lectura).status_code == 200


def test_predict_rechaza_contrato_invalido():
    with cliente() as c:
        assert c.post("/predict", json={**EJEMPLO, "flota": "Z"}).status_code == 422
        assert c.post("/predict", json={**EJEMPLO, "temp_motor_c": -999}).status_code == 422
        sin_campo = {k: v for k, v in EJEMPLO.items() if k != "rpm_promedio"}
        assert c.post("/predict", json=sin_campo).status_code == 422


def test_predict_lote():
    with cliente() as c:
        r = c.post("/predict/lote", json={"lecturas": [EJEMPLO, EJEMPLO]})
        assert r.status_code == 200
        assert r.json()["total"] == 2


def test_la_imagen_no_necesita_frameworks_de_entrenamiento():
    """Servir el modelo no debe importar torch ni tensorflow (por eso la imagen pesa poco)."""
    import sys

    with cliente() as c:
        c.post("/predict", json=EJEMPLO)
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules
