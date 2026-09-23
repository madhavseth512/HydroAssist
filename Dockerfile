# HydroAssist: the FastAPI app serving the API and the built React workspace.
#
# Multi-stage, and multi-architecture by construction: both base images publish
# arm64 builds, and every Python dependency resolves to a Linux aarch64 wheel,
# so the same file builds on an Oracle Ampere VM and on an x86 laptop.
#
#   docker compose up -d --build        (see DEPLOY.md)

# ── 1. Build the browser workspace ───────────────────────────────────────────
FROM node:22-slim AS frontend

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ── 2. The application ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # default.yaml, not production.yaml: every test, including the retrieval
    # regression, ran against default. See DEPLOY.md.
    HYDROASSIST_ENV=default

RUN useradd --create-home --uid 10001 hydroassist
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the app reads at runtime. The publisher PDFs under "knowledge
# source/" are deliberately absent: the app needs the built index, not the papers.
COPY --chown=hydroassist:hydroassist src/ src/
COPY --chown=hydroassist:hydroassist configs/ configs/
COPY --chown=hydroassist:hydroassist data/vectordb/ data/vectordb/
COPY --chown=hydroassist:hydroassist --from=frontend /build/frontend/dist frontend/dist

USER hydroassist

# Download chromadb's ONNX embedding model (~80 MB) now, so the first visitor
# does not wait for it and the running container needs no access to its host.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm the embedding cache'])"

EXPOSE 8000

# PORT, where the platform sets one. Render and most container hosts assign the
# port and expect the process to bind it; docker-compose sets nothing, so 8000
# stays the default and the Caddy setup is unaffected.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, sys, urllib.request; port = os.getenv('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=4).status == 200 else 1)"

# One worker: sessions (a visitor's project and conversation) live in memory.
# --proxy-headers: something else terminates TLS in front of this container —
# Caddy on a VM, the platform's own router on a managed host.
# `exec` so uvicorn is PID 1 and receives the stop signal directly.
CMD ["sh", "-c", "exec uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips '*'"]
