



# ---- base runtime ----------------------------------------------------
FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# xgboost needs libgomp
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage build cache: copy metadata first
COPY pyproject.toml ./

# Copy source
COPY traffic_flow/ ./traffic_flow/

# Install  package (and its deps from pyproject)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[service]"

RUN pip install --no-cache-dir flask "gunicorn>=20.1.0"

# ---- runtime dependencies -------------------------------------------

COPY artifacts/ ./artifacts/
# or mount it at runtime (recommended to swap models without rebuild).
ENV ARTIFACT_PATH=/app/artifacts/traffic_pipeline_h-15.joblib

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--factory", "traffic_flow.service.app:create_app"]