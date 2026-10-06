# Del notebook al microservicio · Tu primer modelo servido en un contenedor

**ITY1102 · Arquitectura de Sistemas de IA** · Caso 0 de referencia: *Minera Pampa Norte* (ficticio) · predicción de falla de camiones de extracción (CAEX) en 7 días.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Umbingelelo/clase-modelo-base-ia/blob/main/notebooks/01_modelo_base_colab.ipynb)

## Inicio rápido (en clase)

```powershell
git clone https://github.com/Umbingelelo/clase-modelo-base-ia.git
cd clase-modelo-base-ia
docker compose up -d --build        # construye la imagen con el modelo que ya trae el repo y la levanta
docker compose ps                   # espera a ver (healthy)
```

Luego abre <http://localhost:8000/docs> o importa en Postman `postman/servicio-ia.postman_collection.json` y ejecuta la colección.

Sin construir nada (la imagen que publicó el pipeline en GHCR):

```powershell
docker compose -f compose.ghcr.yaml up -d
```

Set de pruebas desde la terminal (sin Postman, sin instalar nada: solo Python):

```powershell
python pruebas/probar_api.py            # 20 pruebas: operación, casos conocidos, contrato (422), carga y lote
```

Deja el resultado en `pruebas/reporte_pruebas.md`.

Para apagar: `docker compose down`.

---

Hoy construyen, de punta a punta, la pieza **"Microservicio de IA"** de la EP2: una red neuronal entrenada por ustedes con **PyTorch o TensorFlow**, exportada a **ONNX**, servida dentro de una imagen Docker con `POST /predict` y `GET /health`, que responde la predicción **con la versión del modelo**.

```
 notebook ──► preparar.py ──► entrenar.py ───────► model/ ──────────────────► FastAPI ──────► docker build ──► docker run ──► Postman
 PyTorch o TF  (limpiar)     (+ quality gate)     preprocesador.joblib       + onnxruntime    (imagen sin       (contenedor)    (probar)
                                                  modelo.onnx                 (contrato)       torch ni TF)
                                                  metadata.json
```

## La regla de la imagen de IA

**Entrenen con PyTorch o TensorFlow. Sirvan con `onnxruntime`.** El framework de entrenamiento no entra a la imagen.

| Para servir el modelo, la imagen instala… | Tamaño instalado (medido) | |
|---|---|---|
| `torch` desde PyPI en Linux (trae CUDA) | ≈ 4.600 MB | ✘ Prohibido: sin CUDA |
| `tensorflow-cpu` | ≈ 1.500 MB | ✘ Demasiado para servir una red de 13 KB |
| **`onnxruntime` + numpy** | **≈ 120 MB** | ✔ Lo que usa este servicio |

1. Sin CUDA ni librerías de GPU en ninguna imagen: el computador de la defensa no tiene GPU.
2. `torch` / `tensorflow` se instalan en tu computador, en Colab o en el job de CI, **nunca** en el `requirements.txt` de la imagen.
3. La imagen de IA respeta el límite de tamaño del enunciado.

---

## Antes de clase (hazlo en tu casa, no en la red del laboratorio)

| Necesitas | Cómo verificar |
|---|---|
| Python **3.12 o 3.13** (3.11 no sirve: numpy 2.5 ya no lo soporta) | `py --version` |
| Docker Desktop corriendo | `docker version` (debe mostrar *Client* y *Server*) |
| Postman (app de escritorio) | Abrirlo |
| VS Code con extensión *Jupyter* (o JupyterLab), **o** Google Colab | — |

```powershell
# Desde la carpeta del proyecto (PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # si falla: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt       # incluye PyTorch (solo CPU en Windows). Para TensorFlow ver el archivo
docker pull python:3.12-slim          # la imagen base, para no descargarla en clase
```

> Si tu computador no tiene espacio o Python 3.12+, haz la Parte 1 en **Google Colab** (PyTorch y TensorFlow ya vienen instalados) y descarga `modelo.zip` al final.

## Estructura

```
clase-modelo-base-ia/
├── notebooks/01_modelo_base.ipynb     ← Parte 1 en tu computador: explorar, entrenar, evaluar, ACTIVIDAD
├── notebooks/01_modelo_base_colab.ipynb ← la misma Parte 1 para Google Colab
├── compose.yaml                       ← docker compose up -d --build (desde el código)
├── compose.ghcr.yaml                  ← levantar la imagen publicada en GHCR
├── .github/workflows/servicio-ia.yml  ← pipeline: datos → entrenar → gate → tests → imagen → GHCR
├── datos/
│   ├── generar_datos.py               ← crea el dataset sintético (sucio a propósito)
│   ├── camiones_crudo.csv
│   └── preparar.py                    ← perfilado + limpieza (versión script)
├── servicio-ia/                       ← el microservicio
│   ├── app/
│   │   ├── main.py                    ← endpoints FastAPI
│   │   ├── esquemas.py                ← el CONTRATO JSON (Pydantic)
│   │   └── modelo.py                  ← carga del modelo e inferencia
│   ├── model/                         ← preprocesador.joblib + modelo.onnx + metadata.json
│   ├── tests/test_api.py              ← lo que corre el pipeline antes del build
│   ├── entrenar.py                    ← entrenamiento (PyTorch / TensorFlow / scikit-learn) + quality gate
│   ├── requirements.txt               ← dependencias de la IMAGEN: onnxruntime, sin torch ni TF
│   ├── requirements-pytorch.txt       ← dependencias para ENTRENAR con PyTorch (no van en la imagen)
│   ├── requirements-tensorflow.txt    ← dependencias para ENTRENAR con TensorFlow (no van en la imagen)
│   ├── Dockerfile
│   └── .dockerignore
├── pruebas/probar_api.py              ← set de pruebas contra la API corriendo (alternativa a Postman)
└── postman/servicio-ia.postman_collection.json
```

---

## Parte 1 · El modelo en el notebook (≈ 45 min, incluye la actividad)

**En Colab** (sin instalar nada): botón *Abrir en Colab* de arriba → `notebooks/01_modelo_base_colab.ipynb`. Al final descarga `modelo.zip` y copia su contenido en `servicio-ia/model/`.

**En tu computador:** abre `notebooks/01_modelo_base.ipynb`, elige el kernel de `.venv` y **define tu framework** en la celda de imports: `FRAMEWORK = "pytorch"` o `"tensorflow"`. Ejecuta celda por celda. En la sección 9 la red se exporta a ONNX, se verifica con `onnxruntime` y queda en `servicio-ia/model/`.

**Actividad (sección 10, 15 min, en parejas):** baja el **costo operacional en prueba** respecto de la red base cambiando capas, épocas, dropout, peso de clase o tasa de aprendizaje. Exporta tu mejor red con una versión nueva (`1.1.0-<iniciales>`). Al cierre se arma la tabla de posiciones en la pizarra.

## Parte 2 · Del notebook al script (≈ 5 min)

El notebook no va a producción. Lo que corre el pipeline de CI son scripts:

```powershell
python datos/preparar.py                              # perfila y limpia → datos/camiones_tratado.csv
python servicio-ia/entrenar.py                        # PyTorch: entrena, exporta a ONNX y verifica si pasa el gate
python servicio-ia/entrenar.py --framework tensorflow --capas 128,64 --version 1.1.0-xx
python servicio-ia/entrenar.py --framework sklearn --algoritmo rf     # alternativa clásica (formato joblib)
python servicio-ia/entrenar.py --pr-auc-minimo 0.90   # mira cómo FALLA el gate (exit code 1)
echo $LASTEXITCODE
```

> Si exportaste tu modelo desde el notebook en la actividad, **no corras `entrenar.py` después** o lo vas a sobrescribir.

## Parte 3 · El microservicio con FastAPI (≈ 20 min)

> **El repositorio ya trae un modelo y los datos**: los entrenó y guardó el pipeline (commit del bot), no se subieron a mano. Si quieres regenerarlos tú, corre el notebook completo o estos tres comandos desde la raíz:
> ```powershell
> python datos/generar_datos.py
> python datos/preparar.py
> python servicio-ia/entrenar.py
> ```

```powershell
cd servicio-ia
uvicorn app.main:app --reload --port 8000
```

- Documentación interactiva (Swagger): <http://localhost:8000/docs>
- Salud: <http://localhost:8000/health>

| Endpoint | Para qué |
|---|---|
| `GET /health` | Docker, el gateway y el balanceador preguntan si esta réplica puede recibir tráfico. `200` sí, `503` no |
| `GET /modelo` | Metadata del modelo: versión, umbral, métricas, cifras de limpieza |
| `POST /predict` | Una lectura → probabilidad, alerta, nivel de riesgo, **versión del modelo**, latencia |
| `POST /predict/lote` | Hasta 500 lecturas |

**El contrato** (`app/esquemas.py`): si un campo falta, sobra, viene fuera de rango o con una categoría que no existe, FastAPI responde `422` **sin tocar el modelo**. Los sensores pueden venir en `null`: los imputa el `preprocesador.joblib` antes de pasar a la red.

**Cómo predice** (`app/modelo.py`): lee `metadata.json`; si `"formato": "onnx"`, carga `preprocesador.joblib` y abre `modelo.onnx` con `onnxruntime`. Hay un test que comprueba que el servicio **no importa** `torch` ni `tensorflow`.

Tests (lo mismo que hará el pipeline antes del `docker build`):

```powershell
pip install -r requirements-dev.txt    # ya viene si instalaste el requirements.txt de la raíz
pytest -q
```

## Parte 4 · Contenedorizar (≈ 20 min)

Detén el `uvicorn` local (Ctrl+C) y, desde `servicio-ia/`:

```powershell
docker build -t servicio-ia:1.0 --build-arg REVISION=local-1 .
docker images servicio-ia                       # ¿cuánto pesa la imagen? (compárala con la tabla de arriba)
docker run -d --name servicio-ia -p 8000:8000 servicio-ia:1.0
docker ps                                       # espera a ver (healthy) en STATUS
docker logs -f servicio-ia                      # Ctrl+C para salir de los logs
```

Preguntas que la EP2 te va a hacer sobre este contenedor:

```powershell
# ¿De qué commit salió la imagen? (IE10)
docker image inspect servicio-ia:1.0 --format '{{json .Config.Labels}}'

# ¿Cuánta memoria y CPU usa? (IE2) → déjalo abierto mientras haces la Parte 5
docker stats servicio-ia
```

## Parte 5 · Probar con Postman (≈ 15 min)

1. **Import** → `postman/servicio-ia.postman_collection.json`.
2. Ejecuta las carpetas en orden: **1 · Operación**, **2 · Predicción**, **3 · Contrato**. Mira la pestaña *Test Results*: cada petición trae sus pruebas.
3. **Carga:** clic derecho en la carpeta **4 · Carga** → *Run folder* → 500 iteraciones, delay 0. Mientras corre, mira `docker stats`.
   - Anota el **pico de memoria** → de ahí sale tu `mem_limit` (no de un ejemplo copiado).
   - Anota el **tiempo de respuesta promedio** → tu latencia (IE8).

Si el servicio corre en otro puerto, cambia la variable `baseUrl` de la colección.

## Parte 6 · Cambiar el modelo = imagen nueva

El modelo va **dentro** de la imagen. Si cambias el modelo, construyes otra imagen con otro tag:

```powershell
# después de exportar tu modelo mejorado desde el notebook
docker rm -f servicio-ia
docker build -t servicio-ia:1.1 --build-arg REVISION=local-2 .
docker run -d --name servicio-ia -p 8000:8000 servicio-ia:1.1
```

Vuelve a correr Postman: `version_modelo` cambió. Compara la memoria y el tamaño de la imagen con la versión 1.0. ¿Cambia tu `mem_limit`?

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `Activate.ps1 cannot be loaded` | Política de ejecución de PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `/health` responde 503 / contenedor `unhealthy` | Falta algún archivo de `model/` o no se pudo cargar | Corre la sección 9 del notebook o `entrenar.py`; mira `docker logs servicio-ia` |
| `torch.onnx.export() got an unexpected keyword argument 'dynamo'` | PyTorch antiguo (< 2.5) | Borra `dynamo=False` de la celda, o actualiza PyTorch |
| `ModuleNotFoundError: tf2onnx` al exportar con Keras | Falta el convertidor | `pip install tf2onnx` (en Colab va en la primera celda de código) |
| `AssertionError: El ONNX no reproduce a la red` | La exportación cambió el cálculo | Revisa que la red esté en modo evaluación (`eval()`) y que exportes con `sigmoid` |
| La imagen pesa varios GB | Pusiste `torch` o `tensorflow` en `servicio-ia/requirements.txt` | Sácalos: la imagen solo necesita `onnxruntime` |
| `InconsistentVersionWarning` o error al cargar el modelo | Entrenaste con otra versión de scikit-learn | Usa el `requirements.txt` del proyecto (versiones fijas) |
| `port is already allocated` | Quedó otro contenedor o `uvicorn` en el 8000 | `docker rm -f servicio-ia` o cierra el uvicorn local |
| `AttributeError: Can't get attribute 'agregar_variables'` al arrancar | Usaste una función propia (`FunctionTransformer`) que la API no conoce | Mueve esa función a `app/` e impórtala desde ahí en el notebook y en la API |
| `docker: error during connect` | Docker Desktop no está corriendo | Ábrelo y espera a que diga *Engine running* |

---

## Puente a la EP2

Lo de hoy es **una** de las siete piezas. Así se vería en el `compose.yaml` de la defensa:

```yaml
services:
  ia:
    image: ghcr.io/<cuenta>/servicio-ia:ep2      # la publica el pipeline, sin "build:"
    # sin "ports:": solo el gateway publica el 8080
    deploy:
      replicas: 2                                # si eligen replicar este servicio
    cpus: "0.50"                                 # ← de tu medición con docker stats (capítulo 7.2)
    mem_limit: 384m                              # ← idem. Este número es un EJEMPLO: mide el tuyo
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]
      interval: 15s
      timeout: 3s
      retries: 3
```

Checklist para tu caso:

- [ ] Mi `datos/preparar.py` perfila, limpia con reglas escritas y deja cifras antes/después (8.4).
- [ ] Mi `entrenar.py` separa train/test antes de entrenar y falla si la métrica no alcanza el umbral.
- [ ] Mi job de CI instala PyTorch/TensorFlow **solo CPU** para entrenar (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), y mi imagen solo lleva `onnxruntime`.
- [ ] Mi `/predict` responde la predicción **y la versión del modelo**; mi `/health` responde 503 si no hay modelo.
- [ ] Tengo un test que prueba `/predict` con una entrada conocida.
- [ ] Mi imagen pesa lo que dice el capítulo 7.1, y mi `mem_limit` sale de `docker stats`.
- [ ] Sé la latencia de una predicción y cómo la bajaría.

---

## El pipeline de este repositorio (GitHub Actions)

`.github/workflows/servicio-ia.yml` es el mismo patrón que pide la EP2 (sección 2.5):

| Evento | Qué hace |
|---|---|
| Pull request a `dev` o `main` | Genera y limpia datos → entrena con PyTorch **solo CPU** → verifica el ONNX → *quality gate* (PR-AUC ≥ 0,55) → `pytest` → `docker build` → prueba de humo del contenedor (`/health`, `/predict`, tamaño < 1 GB). **No publica** |
| Push a `main` | Todo lo anterior y publica en GHCR: `ghcr.io/<cuenta>/servicio-ia-caex` con tags `sha-<commit>` y `latest` |
| Tag (`ep2`, `v1.0`…) | Publica además con el nombre del tag |

Fíjate en tres cosas:
1. **PyTorch se instala en el job, no en la imagen.** El job pesa; la imagen no.
2. **El modelo lo produce el pipeline, no una persona:** `entrenar.py` lo entrena en cada ejecución; si la métrica baja del mínimo, el job falla y no se publica nada. En `main`, el bot guarda ese modelo y los datos en el repositorio (commit con `[skip ci]`) para que se pueda clonar y levantar directo.
3. **La imagen lleva el commit** en la etiqueta `org.opencontainers.image.revision`: `docker image inspect` responde "¿qué commit está corriendo?" (IE10).

Gitflow mínimo: se trabaja en `feat/<algo>` → PR a `dev` → PR de `dev` a `main` → tag `ep2`.
