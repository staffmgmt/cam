## Docker runtime for Hugging Face GPU Space (A10G) in Docker mode
## Single-stage image on Ubuntu 22.04 (Python 3.10) with CUDA 11.8 + cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_CACHE_PATH=/tmp/cuda_cache \
    TORCH_CUDA_ARCH_LIST="8.6" \
    CUDA_LAUNCH_BLOCKING=0 \
    CUDA_VISIBLE_DEVICES=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    pkg-config \
    ca-certificates \
    ffmpeg \
    libopus0 \
    libsrtp2-1 \
    libsrtp2-dev \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    cuda-nvrtc-11-8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 11.8 first to avoid resolver overriding
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools \
 && pip3 install --no-cache-dir \
    torch==2.2.2+cu118 \
    torchvision==0.17.2+cu118 \
    torchaudio==2.2.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install remaining Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source
COPY . /app

# --- Pre-cache InsightFace analysis models (buffalo_l) to avoid cold-start ---
# This is best-effort: failure will not break build; runtime can still lazy-download.
ENV MIRAGE_ANALYSIS_MODEL=buffalo_l \
        MIRAGE_PRECACHE_ANALYSIS=1
RUN if [ "$MIRAGE_PRECACHE_ANALYSIS" = "1" ]; then \
            echo "[build] Pre-caching InsightFace model: $MIRAGE_ANALYSIS_MODEL" && \
            python3 precache_insightface.py || echo "[build] Warning: precache failed (will attempt at runtime)" ; \
        else \
            echo "[build] Skipping InsightFace precache" ; \
        fi

## Create only required directories (face swap: inswapper + optional codeformer)
RUN mkdir -p \
        /app/models/inswapper \
        /app/models/codeformer \
        /app/.cache/huggingface/hub \
        /app/.cache/huggingface/transformers \
        /app/.cache/insightface \
        /app/.insightface \
        /tmp/matplotlib \
        /tmp/cuda_cache \
    && chmod -R 777 /app/models /app/.cache /app/.insightface /tmp/cuda_cache /tmp/matplotlib

ARG MIRAGE_DOWNLOAD_MODELS=1
ENV MIRAGE_DOWNLOAD_MODELS=${MIRAGE_DOWNLOAD_MODELS}

# Expose HTTP port
EXPOSE 7860

# Default port (Hugging Face Spaces injects PORT env; fallback to 7860)
ENV PORT=7860

# Configure cache locations to avoid writing to '/.cache' when HOME is unset by the platform
ENV HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    INSIGHTFACE_HOME=/app/.insightface \
    MPLCONFIGDIR=/tmp/matplotlib \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    MIRAGE_ORT_DISABLE_SHAPE_INFERENCE=1 \
    MIRAGE_REQUIRE_GPU=1 \
    MIRAGE_DL_TAG=build \
    MIRAGE_WEBRTC_VERBOSE=1 \
    MIRAGE_WEBRTC_STATS_INTERVAL=5000 \
    MIRAGE_WEBRTC_FORCE_RELAY=0

# Face swap enforced (no reenactment stack)
ENV MIRAGE_FACE_SWAP_ONLY=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c 'curl -fsS http://localhost:${PORT:-7860}/health || exit 1'

# Add entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Run via entrypoint to guarantee model provisioning & integrity checks
CMD ["/app/entrypoint.sh"]
