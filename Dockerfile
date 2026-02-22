# ============================================================
# Somaliland Drought Prediction — Dockerfile
# ============================================================
# Multi-stage build for a reproducible environment.
#
# Usage:
#   Build:
#       docker build -t somaliland-drought:latest .
#
#   Run Jupyter notebook:
#       docker run -p 8888:8888 \
#           -v $(pwd)/data:/app/data \
#           -v $(pwd)/models:/app/models \
#           somaliland-drought:latest
#
#   Run Streamlit dashboard:
#       docker run -p 8501:8501 \
#           -v $(pwd)/data:/app/data \
#           -v $(pwd)/models:/app/models \
#           somaliland-drought:latest \
#           streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0
#
#   Run with ERA5 CDS API credentials:
#       docker run -p 8888:8888 \
#           -v ~/.cdsapirc:/root/.cdsapirc:ro \
#           -v $(pwd)/data:/app/data \
#           somaliland-drought:latest
# ============================================================

# ── Base image ────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System dependencies for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgdal-dev \
        libproj-dev \
        libgeos-dev \
        libspatialindex-dev \
        libhdf5-dev \
        libnetcdf-dev \
        curl \
        wget \
        git \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Set GDAL environment
ENV GDAL_VERSION=3.6 \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# ── Builder stage ─────────────────────────────────────────────
FROM base AS builder

WORKDIR /tmp

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Final stage ───────────────────────────────────────────────
FROM base AS final

# Create non-root user for security
RUN useradd -m -u 1000 drought && \
    mkdir -p /app/data/raw /app/data/processed /app/models /app/reports/figures && \
    chown -R drought:drought /app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY --chown=drought:drought . /app/

# Install project package
RUN pip install -e . --no-deps

USER drought

# ── Environment ───────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app \
    JUPYTER_ENABLE_LAB=yes

# Expose ports
EXPOSE 8888 8501

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8888/api || curl -f http://localhost:8501/_stcore/health || exit 1

# ── Default command — Jupyter Lab ────────────────────────────
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/app/notebooks"]

# ============================================================
# Docker Compose (alternative):
#
# docker-compose.yml:
#
# version: '3.8'
# services:
#   notebook:
#     build: .
#     ports:
#       - "8888:8888"
#     volumes:
#       - ./data:/app/data
#       - ./models:/app/models
#       - ~/.cdsapirc:/root/.cdsapirc:ro
#
#   streamlit:
#     build: .
#     command: streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0
#     ports:
#       - "8501:8501"
#     volumes:
#       - ./data:/app/data
#       - ./models:/app/models
#     depends_on:
#       - notebook
# ============================================================
