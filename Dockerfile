## Docker runtime for Hugging Face GPU Space (A10G) in Docker mode
## Single-stage image on Ubuntu 22.04 (Python 3.10) with CUDA 12.1 + cuDNN 8
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.1 first to avoid resolver overriding
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools \
 && pip3 install --no-cache-dir \
    torch==2.3.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install remaining Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source
COPY . /app

# Create directories for models and checkpoints (if not already present)
RUN mkdir -p /app/models/liveportrait /app/models/rvc /app/models/hubert /app/models/rmvpe /app/checkpoints /tmp/cuda_cache

# Optional model downloader configuration (example URLs)
ARG MIRAGE_DOWNLOAD_MODELS=0
ARG MIRAGE_LP_APPEARANCE_URL="https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/appearance_feature_extractor.onnx"
ARG MIRAGE_LP_MOTION_URL="https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/motion_extractor.onnx"
ENV MIRAGE_DOWNLOAD_MODELS=${MIRAGE_DOWNLOAD_MODELS} \
    MIRAGE_LP_APPEARANCE_URL=${MIRAGE_LP_APPEARANCE_URL} \
    MIRAGE_LP_MOTION_URL=${MIRAGE_LP_MOTION_URL}
# Skip model download during build - only download at runtime if needed
# RUN python3 /app/model_downloader.py || true

# Expose HTTP port
EXPOSE 7860

# Default port (Hugging Face Spaces injects PORT env; fallback to 7860)
ENV PORT=7860

# Feature flags for safe model integration (can be overridden in Space settings)
# Enable SCRFD face detection by default for better reliability; keep LivePortrait safe path off initially.
# Landmark reenactor is also off by default to avoid MediaPipe dependency issues
ENV MIRAGE_ENABLE_SCRFD=1 \
    MIRAGE_ENABLE_LIVEPORTRAIT=0 \
    MIRAGE_ENABLE_LANDMARK_REENACTOR=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c 'curl -fsS http://localhost:${PORT:-7860}/health || exit 1'

# Run FastAPI app with uvicorn (WebRTC endpoints + static UI), binding to PORT
CMD ["sh", "-c", "uvicorn original_fastapi_app:app --host 0.0.0.0 --port ${PORT:-7860}"]
