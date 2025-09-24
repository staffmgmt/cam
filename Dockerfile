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

# Create directories for models and checkpoints (if not already present) and make them writable at runtime
RUN mkdir -p \
        /app/models/liveportrait \
        /app/models/rvc \
        /app/models/hubert \
        /app/models/rmvpe \
        /app/checkpoints \
        /app/.cache/huggingface/hub \
        /app/.cache/huggingface/transformers \
        /app/.cache/insightface \
        /app/.insightface \
        /tmp/matplotlib \
        /tmp/cuda_cache \
    && chmod -R 777 /app/models /app/checkpoints /app/.cache /app/.insightface /tmp/cuda_cache /tmp/matplotlib

# Optional model downloader configuration (example URLs)
ARG MIRAGE_DOWNLOAD_MODELS=1
ARG MIRAGE_LP_APPEARANCE_URL="https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx/appearance_feature_extractor.onnx"
ARG MIRAGE_LP_MOTION_URL="https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx/motion_extractor.onnx"
# Use myn0908 generator with grid fix (opset 20); requires ORT >= 1.18
ARG MIRAGE_LP_GENERATOR_URL="https://huggingface.co/myn0908/Live-Portrait-ONNX/resolve/main/generator_fix_grid.onnx"
# Do NOT set GRID_PLUGIN_URL to avoid TensorRT dependency in this image
# Optional custom ops plugin is disabled by default (TensorRT not present in this image)
# ARG MIRAGE_LP_GRID_PLUGIN_URL=""
ARG MIRAGE_LP_STITCHING_URL="https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx/stitching.onnx"
ENV MIRAGE_DOWNLOAD_MODELS=${MIRAGE_DOWNLOAD_MODELS} \
    MIRAGE_LP_APPEARANCE_URL=${MIRAGE_LP_APPEARANCE_URL} \
    MIRAGE_LP_MOTION_URL=${MIRAGE_LP_MOTION_URL} \
    MIRAGE_LP_GENERATOR_URL=${MIRAGE_LP_GENERATOR_URL} \
    MIRAGE_LP_STITCHING_URL=${MIRAGE_LP_STITCHING_URL}
# Skip model download during build - only download at runtime if needed
# RUN python3 /app/model_downloader.py || true

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
    MIRAGE_FORCE_DOWNLOAD_GENERATOR=1 \
    MIRAGE_FORCE_DOWNLOAD_APPEARANCE=1 \
    MIRAGE_FORCE_DOWNLOAD_MOTION=1

# Enforce single neural path (SCRFD + LivePortrait generator)
ENV MIRAGE_REQUIRE_NEURAL=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c 'curl -fsS http://localhost:${PORT:-7860}/health || exit 1'

# Add entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Run via entrypoint to guarantee model provisioning & integrity checks
CMD ["/app/entrypoint.sh"]
