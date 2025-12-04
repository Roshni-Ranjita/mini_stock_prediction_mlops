# ========= base image =========
FROM python:3.11-slim

# System deps (scipy/statsmodels need BLAS/LAPACK); tini for proper signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev liblapack-dev \
    git curl tini \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy only dependency files first (better build cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Now copy app code
COPY streamlit_sarimax_app.py drift_check.py build_manifest.py /app/

# Default dirs created inside the container (we'll mount host volumes over these)
RUN mkdir -p /app/models /app/Stock /app/logs /app/mlruns /app/artifacts

# Environment so the app finds things in /app/*
ENV MODELS_DIR=/app/models \
    STOCK_DIR=/app/Stock \
    LOG_DIR=/app/logs \
    MLFLOW_URI=file:/app/mlruns \
    PYTHONUNBUFFERED=1

# Healthcheck: ping Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit runs on 8501
EXPOSE 8501

# Use tini as entrypoint for proper signal handling
ENTRYPOINT ["/usr/bin/tini","--"]

# Launch the dashboard
CMD ["streamlit","run","/app/streamlit_sarimax_app.py","--server.address=0.0.0.0","--server.port=8501","--server.headless=true"]