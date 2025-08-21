# 1) Base image
FROM python:3.10-slim-bullseye

# 2) Env hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# 3) System libs (xgboost needs OpenMP runtime)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 4) Workdir
WORKDIR /app

# 5) Copy pins first (best cache)
COPY constraints.txt /tmp/constraints.txt

# 6) Copy project metadata and source
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/

# 7) Install: pin with constraints, install your package + [service] extras
#    (Gunicorn is already in your [project.optional-dependencies].)
RUN python -m pip install --upgrade pip \
 && pip install -c /tmp/constraints.txt ".[service]"

# 8) Model artifact location (can be overridden at runtime)
ENV ARTIFACT_PATH=/app/artifacts/traffic_pipeline_h-15.joblib

# 9) Gunicorn config
COPY docker/gunicorn.conf.py /app/gunicorn.conf.py

# 10) Network & entrypoint
EXPOSE 8080
CMD ["gunicorn", "-c", "/app/gunicorn.conf.py", "traffic_flow.service.app:create_app()"]