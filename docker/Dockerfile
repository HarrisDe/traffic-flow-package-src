
# 1) BASE IMAGE – we start from an official slim Python interpreter   #
#######################################################################
FROM python:3.10-slim-bullseye   
# -----------------------------------------------------------------------------
# 2) BASIC ENVIRONMENT FLAGS
#    - PYTHONDONTWRITEBYTECODE : don’t create *.pyc files (they bloat layers)
#    - PYTHONUNBUFFERED        : get logs immediately, not buffered
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# 3) SYSTEM LIBS THAT THE MODEL NEEDS
#    xgboost’s binary wheels link against libgomp (OpenMP).
# -----------------------------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

#######################################################################
# 4) COPY + INSTALL PYTHON PACKAGE                              
#######################################################################
WORKDIR /app             

# 4-a) copy only metadata first (this lets Docker cache pip install
#      as long as pyproject.toml doesn’t change)
COPY pyproject.toml ./

# 4-b) copy the actual source code
COPY traffic_flow/ ./traffic_flow/

# 4-c) install package plus the “service” extras (Flask+Gunicorn)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir '.[service]' \
 && pip install --no-cache-dir --force-reinstall --upgrade "gunicorn==21.2.0"

#######################################################################
# 5) COPY THE TRAINED MODEL (artifact)                               #
#######################################################################
# ▸ If you want one immutable image per model version, **add it here**.
# ▸ If you prefer to swap models without rebuilding the image,
#   comment this line below and mount -v /host/artifacts:/app/artifacts
#COPY artifacts/ ./artifacts/

# path that the Flask code expects (can be overridden with -e)
ENV ARTIFACT_PATH=/app/artifacts/traffic_pipeline_h-15.joblib

#######################################################################
# 6) NETWORK & PROCESS ENTRYPOINT                                    #
#######################################################################
# 6-a) Docker runtime will publish container port → host port
EXPOSE 8080          

# 6-b) Start Gunicorn, **factory style**.
#      `traffic_flow.service.app:create_app()` means
#      “import create_app from that module and CALL it”.
CMD ["gunicorn","--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "--access-logfile", "-","traffic_flow.service.app:create_app()"]