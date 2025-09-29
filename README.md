# 🚦 Traffic Flow Forecasting Package 

A modular Python package for **short‑term traffic speed forecasting** built around a clean, extensible pipeline: data loading & cleaning → feature engineering → model training/tuning (XGBoost) → evaluation → packaging a **single joblib artifact** for **easy deployment** as an HTTP API (Flask + Gunicorn, with Docker support).

> **Examples**: A usage example notebook lives in **`scripts/compare_xgb_vs_lstm.ipynb`** and runnable examples live in the `scripts/` folder (e.g., `train_xgb.py`, `make_payload.py`, `compare_api_vs_local.py`).

---

## Table of Contents
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Expectations](#data-expectations)
- [Quickstart](#quickstart)
  - [Hybrid (GMAN) workflow](#hybrid-gman-workflow)
  - [1) Train & Bundle an XGBoost Model](#1-train--bundle-an-xgboost-model)
  - [2) Serve the Model via HTTP API](#2-serve-the-model-via-http-api)
  - [3) Call the API](#3-call-the-api)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Development Notes](#development-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Key Features

- **End‑to‑end tabular pipeline** for multi‑sensor, time‑indexed traffic speed data.
- **Hybrid (GMAN) support**: use **GMAN** predictions as *features* for the tabular/XGBoost model (hybrid approach), distinct from the **Deep** baselines (e.g., LSTM/GRU).
- **Feature engineering**: lags, calendar/cyclical time features, congestion/outlier flags, adjacent‑sensor features, optional **GMAN** hybrid inputs, and more.
- **Modeling**: XGBoost with grid/timed search, K‑fold or time‑series split; GPU toggle.
- **Consistent outputs**: single `y_test` (from smallest horizon) + per‑horizon predictions.
- **Artifact bundling**: trains and saves a single **`.joblib`** containing preprocessor + model + states.
- **Serving**: Flask **app factory** + Gunicorn, with a **Dockerfile** and **docker‑compose**.
- **Utilities**: plotting helpers, residual correction, parity checks, payload generator, etc.
- **Examples**: see **`scripts/compare_xgb_vs_lstm.ipynb`** and Python scripts in `scripts/`.

> ℹ️ **Deep** (LSTM/GRU) modules are provided as baselines, while **Hybrid (GMAN)** is a first‑class option where GMAN predictions feed the tabular/XGBoost model. The serving path still focuses on deploying the **XGBoost** artifact.

---

## Repository Structure

```
traffic-flow-package-src/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── gunicorn.conf.py
├── scripts/
│   ├── train_xgb.py                  # Train + bundle XGB → joblib
│   ├── make_payload.py               # Build JSON payload for /predict
│   ├── compare_api_vs_local.py       # Parity check (API vs local)
│   └── compare_xgb_vs_lstm.ipynb     # Example usage notebook  ✅
├── src/traffic_flow/
│   ├── tabular/
│   │   ├── pipeline/                 # Orchestrators (XGB across horizons)
│   │   ├── features/                 # Lag/cyclical/adjacent/congestion...
│   │   ├── modeling/                 # XGB tuner (GridSearchCV, TSCV)
│   │   ├── inference/                # Inference pipeline & protocol
│   │   └── service/                  # Flask app factory + runtime
│   ├── preprocessing/                # Cleaning/smoothing/utils
│   ├── post_processing/              # Residual correction, etc.
│   ├── deep/                         # Deep baselines (e.g., LSTM/GRU/Conv1D)
│   ├── hybrid/                       # GMAN hybrid integration (GMAN as features for XGB)
│   └── utils/                        # Logging, plotting helpers, dicts
├── pyproject.toml                    # PEP 621, setuptools build
├── environment.yml                   # Conda env (>= Python 3.10,<3.12)
└── constraints.txt                   # Version pinning for Docker install
```

---

## Requirements

- **Python**: `>=3.10,<3.12`
- **OS**: Linux, macOS, or Windows
- **Core libs**: `numpy`, `pandas`, `scikit-learn`, `xgboost (>=2.0)`, `joblib`, `requests`, `nbformat`
- **Optional**:
  - `flask`, `gunicorn` (serving)
  - `pyarrow` or `fastparquet` (Parquet I/O)
  - `plotly`, `matplotlib` (visualization)
  - Deep extras (e.g., TensorFlow) are only needed for the **deep/hybrid** experiments

> A ready‑to‑use Conda environment is provided: **`environment.yml`**.

---

## Installation

### Option A — Conda (recommended for reproducibility)

```bash
# from repo root
conda env create -f environment.yml
conda activate traffic-flow-environment

# install your package in editable mode
pip install -e .[service,parquet,vis]
```

### Option B — pip (system/virtualenv)

```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .[service,parquet,vis]
```

> The `[service]` extra brings in Flask + Gunicorn. The `[parquet]` extra installs `pyarrow`/`fastparquet` so that Parquet files work out‑of‑the‑box.

---

## Data Expectations

The pipeline expects **long‑format** sensor time series with at least:

- `date` — timestamp (`datetime64[ns]`), evenly spaced per sensor
- `sensor_id` — sensor identifier (string or int)
- `value` — observed speed (float)

Optionally supported columns include **location** (`longitude`, `latitude`) and a wide set of **weather** variables. See `traffic_flow/tabular/constants/constants.py` → `WEATHER_COLUMNS` for the full list.

> All sensors should share the same timestamp index. The pipeline can compute **relative lags**, **adjacent‑sensor** features (using provided upstream/downstream dictionaries), **congestion flags**, and more.

---

## Quickstart

### 1) Train & Bundle an XGBoost Model

Train on a Parquet file and produce a **timestamped** artifact (e.g., `250929_traffic_pipeline_h-15.joblib`) in `artifacts/`:

```bash
python scripts/train_xgb.py   --file /path/to/your_data.parquet   --out-dir artifacts   --filter-extreme-changes   --smooth-speeds   --relative-threshold 0.7   --window-size 5   --horizon 15   --use-ts-split   --n-splits 3   --use-gpu   --objective reg:pseudohubererror
```

**Notes**

- Run `python scripts/train_xgb.py -h` to see the full list of flags:
  - general: `--test-size`, `--use-ts-split/--n-splits`, `--disable-logs`
  - cleaning: `--filter-extreme-changes`, `--smooth-speeds`, `--diagnose-extreme-changes`
  - features: `--relative-lags`, `--window-size`, `--add-gman-predictions`, `--add-previous-weekday-feature`, etc.
  - modeling: `--use-gpu/--no-use-gpu`, `--params` (JSON grid), `--objective`, `--n-jobs`
- Output file pattern: **`{YYMMDD}_traffic_pipeline_h-{H}.joblib`** under `--out-dir`.

### 2) Serve the Model via HTTP API

#### Option A — Gunicorn (local)

```bash
export ARTIFACT_PATH=/absolute/path/to/artifacts/250929_traffic_pipeline_h-15.joblib
gunicorn -c docker/gunicorn.conf.py "traffic_flow.tabular.service.app:create_app()"
# Binds to 0.0.0.0:8080 by default (see gunicorn.conf.py)
```

#### Option B — Docker

Build and run the API with the provided Dockerfile and compose file.

```bash
# from repo root
docker build -t traffic-flow:local -f docker/Dockerfile .

# Ensure your trained artifact is available on the host in ./artifacts
# docker-compose mounts it read-only into the container at /app/artifacts.
docker compose -f docker/docker-compose.yaml up --build
```

> The container expects `ARTIFACT_PATH=/app/artifacts/traffic_pipeline_h-15.joblib` by default. You can override it via `docker-compose.yaml` or `-e ARTIFACT_PATH=...`

### 3) Call the API

Create a payload and POST to `/predict`:

```bash
# Build a JSON payload from raw data
python scripts/make_payload.py --raw /path/to/your_data.parquet --out payload.json --n-rows 5000

# Health check
curl -s localhost:8080/healthz

# Predict
curl -X POST "http://localhost:8080/predict"   -H "Content-Type: application/json"   --data-binary @payload.json | jq . | head
```


### Hybrid (GMAN) workflow

> **What is Hybrid?** In this repository, **Hybrid** means using a **GMAN** (Graph Multi‑Attention Network) model to generate predictions that are then consumed as **input features** by the **tabular XGBoost** pipeline. This is **different from Deep baselines** like LSTM/GRU, which are standalone end‑to‑end predictors.

**Typical steps**

1. **Train GMAN and export predictions** for the time window/sensors of interest.

   - Example script patterns live under `scripts/` (look for `gman` in the filename). These produce a Parquet/CSV with keys like `date`, `sensor_id` and one or more columns such as `gman_pred_h{H}`.

2. **Train XGBoost with GMAN features** by pointing to the GMAN outputs.

   - The tabular trainer supports a flag like `--add-gman-predictions` (and optionally `--gman-file`) to merge GMAN predictions on `date` & `sensor_id` and include them as a **feature group**.

3. **Inference**

   - To reproduce hybrid behavior at inference time, ensure the **same GMAN features** are present for the incoming window. Two common patterns:

     - **Sidecar precompute (recommended):** run your GMAN model to produce `gman_pred_*` columns and either (a) include them directly in the **API payload** or (b) configure the runtime to read a sidecar file and merge by keys.

     - **Inline service call:** stand up a GMAN microservice and enrich the payload by calling it before hitting the XGB API (requires custom glue; not provided here by default).


**CLI example (training with GMAN features)**

```bash
python scripts/train_xgb.py \
  --file /path/to/traffic.parquet \
  --out-dir artifacts \
  --horizon 15 \
  --window-size 5 \
  --add-gman-predictions \
  --gman-file /path/to/gman_preds_h15.parquet
```


**API payload note**


If the bundled artifact was trained with GMAN features, the runtime expects the **same GMAN feature columns** at inference. Include them in your request body (example below) or configure a merge step inside your runtime.

```json
{
  "records": [
    {
      "date": "2024-01-01T00:00:00Z",
      "sensor_id": 1001,
      "value": 63.5,
      "gman_pred_h15": 0.42
    }
  ]
}
```

> If GMAN features are **not** provided at inference while the model expects them, predictions may degrade or the runtime may raise a schema error.

---

## API Reference

> The app uses a **Flask app factory** (`create_app`), loaded by Gunicorn. Endpoints below reflect the intended interface; names may be adjusted as the code evolves.

### `GET /healthz`
- Returns `{"status": "ok"}` if the server is up and the artifact path is readable.

### `POST /predict`
- **Body**: JSON object with a `records` array of raw rows (all columns you have; the runtime will select/clean what it needs). Example:
  ```json
  {
    "records": [
      {"date": "2024-01-01T00:00:00Z", "sensor_id": 1001, "value": 63.5, "longitude": 4.9, "latitude": 52.3, "...": "..."}
    ]
  }
  ```
- **Response**: JSON with the serving horizon and predictions. Depending on configuration, fields include:
  - `horizon` (minutes)
  - `n` (number of predictions)
  - `predictions`: list of objects with (at minimum) `sensor_id`, timestamps, and **`y_pred_delta`**. If `add_total=True`, you will also get **`y_pred_total`** (baseline + delta).

> See `scripts/compare_api_vs_local.py` for a parity check workflow (local inference vs API).

---

## Configuration

- **Environment variables**
  - `ARTIFACT_PATH` — absolute path to the `.joblib` bundle to load at startup.
- **Training flags**
  - Use `python scripts/train_xgb.py -h` for the authoritative list.
- **Adjacency dictionaries**
  - Located under `src/traffic_flow/utils/` as `upstream_dict.json` and `downstream_dict.json`.
- **Horizon handling**
  - The *smallest* horizon drives the canonical `y_test` shape; higher horizons merge on `date`/`sensor_id`.

---

## Development Notes

- Build & install locally:
  ```bash
  pip install -e .[service,parquet,vis]
  ```
- Package layout follows **PEP 621** (`pyproject.toml`) with **setuptools** and `src/` layout.
- Minimal supported Python from `pyproject.toml`: **>=3.10,<3.12**.
- Notebook examples live in **`scripts/`** (see `compare_xgb_vs_lstm.ipynb`).

> **Heads‑up** for v1 → v2 adopters:
> - Some legacy scripts import `traffic_flow.service` or `traffic_flow.pipeline`. In this repository the serving code lives under **`traffic_flow.tabular.service`** and the orchestrators under **`traffic_flow.tabular.pipeline`**. Update your imports and Docker Gunicorn target to:
>
> ```bash
> gunicorn "traffic_flow.tabular.service.app:create_app()"
> ```

---

## Troubleshooting

- **Docker fails to start app / Gunicorn cannot import app**  
  Ensure you are targeting the correct module path: `traffic_flow.tabular.service.app:create_app()`.
- **Parquet read errors**  
  Install the `[parquet]` extra: `pip install -e .[parquet]`.
- **Out of memory / slow training**  
  Reduce `--window-size`, disable some feature groups (e.g., adjacent/congestion), or lower `--n_estimators` in your XGBoost grid. Consider `--use-gpu` on capable hardware.
- **Mismatched timestamps across sensors**  
  The pipeline assumes aligned timestamps; pre‑align or aggregate before training.
- **API predictions look off**  
  Verify your payload rows cover enough warm‑up history for lag features; see `--window-size` and runtime trimming logic.

---

## Roadmap

- Extend API response schema and OpenAPI spec.
- Expand tests and CI.
- Add packaged demo dataset + fully reproducible notebook.
- Optional GPU Docker image for XGBoost + TF (Linux).

---

## Acknowledgements

Developed as part of research activities at **Data Science Lab, University of Piraeus** and the EMERALDS initiative. Contributions and feedback are welcome.
