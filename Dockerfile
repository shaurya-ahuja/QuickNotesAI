# syntax=docker/dockerfile:1
#
# QuickNotes-AI — production image for Streamlit on Render / Railway / Cloud Run.
# Binds to the platform-provided $PORT (defaults to 8501 locally).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache HuggingFace models (embeddings + cross-encoder reranker) inside the
    # image dir so re-ranking works on read-only-rootfs platforms.
    HF_HOME=/app/.cache/huggingface \
    PORT=8501

# System libraries:
#   ffmpeg     — decode webm/mp3/m4a uploads to WAV for Whisper
#   libsndfile1 — required by soundfile
#   curl       — container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code.
COPY . .

# Writable dirs for the SQLite DB, uploads, transcripts, and model cache.
RUN mkdir -p data uploads records .cache/huggingface

# Run as an unprivileged user (defense-in-depth); own the writable dirs.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f "http://localhost:${PORT}/_stcore/health" || exit 1

# Shell form so ${PORT} is expanded at runtime (Cloud Run/Railway inject it).
# Note: we intentionally do NOT pass --server.enableCORS=false — with XSRF
# protection enabled (see .streamlit/config.toml), disabling CORS makes
# Streamlit silently turn XSRF protection off.
CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true
