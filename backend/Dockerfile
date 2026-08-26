# ==============================================================================
# Phase 2 — Production-Ready Backend Dockerfile (FastAPI + RAG Pipeline)
# ==============================================================================

# 1. BASE IMAGE: Use official Python 3.11 slim image for a small, secure footprint
FROM python:3.11-slim

# 2. ENVIRONMENT VARIABLES:
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk inside container
# - PYTHONUNBUFFERED: Forces stdout/stderr streams to log instantly without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STORAGE_DIR=/app/storage

# 3. SYSTEM DEPENDENCIES: Install curl (needed for Docker HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. WORKING DIRECTORY: Set application directory inside container
WORKDIR /app

# 5. DEPENDENCY LAYER: Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch FIRST from official CPU wheel repository (140MB vs 2.6GB CUDA bundle).
# This cuts image download and build time from 70+ minutes down to < 2 minutes!
RUN pip install --no-cache-dir torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model at image build time so container starts instantly without downloading
RUN python -m spacy download en_core_web_sm

# 6. APPLICATION CODE: Copy application source code
COPY app /app/app

# 7. PERSISTENT DATA VOLUME:
# Declare /app/storage as a mount point for host volumes (FAISS index, SQLite metadata, PDF uploads)
VOLUME ["/app/storage"]

# 8. PORT EXPOSURE: Document that FastAPI runs on port 8000
EXPOSE 8000

# 9. HEALTH CHECK: Verify container readiness via FastAPI /health route
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 10. CONTAINER COMMAND: Launch FastAPI application with uvicorn listening on all interfaces (0.0.0.0)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
