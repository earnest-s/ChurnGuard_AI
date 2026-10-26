# syntax=docker/dockerfile:1
#
# Multi-stage build:
#   builder  -> compiles/installs the Python dependencies into a virtualenv
#   runtime  -> slim image with ONLY the virtualenv + application code
#
# The heavy training stack (PyCaret, matplotlib, xgboost) is intentionally not
# part of the runtime image; the exported model is a plain scikit-learn pipeline.

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build toolchain for any dependency without a prebuilt wheel. It stays in this
# stage, so it never reaches the shipped image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CHURNGUARD_MODEL_DIR=/app/models \
    CHURNGUARD_DATA_DIR=/app/data

WORKDIR /app

# Unprivileged user: the API never needs to write outside /app/data and /app/models.
RUN groupadd --gid 1001 app \
    && useradd --uid 1001 --gid 1001 --create-home app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app . .

RUN mkdir -p /app/data /app/models \
    && chown -R app:app /app/data /app/models

USER app

EXPOSE 8000

# DevOps monitoring: fails the container health check if the API stops answering.
# Uses Python (already installed) instead of curl, so the runtime image stays slim.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
