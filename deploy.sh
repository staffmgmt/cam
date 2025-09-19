#!/bin/bash
# Deployment script for Mirage Real-time AI Avatar System

set -e

echo "🎭 Mirage Real-time AI Avatar - Deployment Script"
echo "=================================================="

# Check if we're deploying to HuggingFace Spaces
if [[ "${SPACE_ID}" ]]; then
    echo "📡 Deploying to HuggingFace Spaces: ${SPACE_ID}"
    DEPLOYMENT_TARGET="huggingface"
else
    echo "🐳 Local Docker deployment"
    DEPLOYMENT_TARGET="local"
fi

# Set environment variables for optimal A10G performance
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"  # A10G architecture
export CUDA_LAUNCH_BLOCKING=0
export MIRAGE_VOICE_ENABLE=1
export MIRAGE_CHUNK_MS=160
export MIRAGE_VIDEO_MAX_FPS=20

echo "🔧 Environment configured for A10G GPU"

# Download required models
echo "📥 Downloading AI models..."

# Create model directories
mkdir -p models/{liveportrait,rvc,hubert,rmvpe}
mkdir -p checkpoints

# Function to download from HuggingFace with retry
download_hf_model() {
    local repo=$1
    local filename=$2
    local output_dir=$3
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if python3 -c "
from huggingface_hub import hf_hub_download
import os
try:
    hf_hub_download('$repo', '$filename', local_dir='$output_dir', local_dir_use_symlinks=False)
    print('✅ Downloaded $filename')
except Exception as e:
    print(f'❌ Failed to download $filename: {e}')
    exit(1)
        "; then
            break
        fi
        
        retry_count=$((retry_count + 1))
        echo "⏳ Retry $retry_count/$max_retries for $filename"
        sleep 2
    done
    
    if [ $retry_count -eq $max_retries ]; then
        echo "❌ Failed to download $filename after $max_retries retries"
        return 1
    fi
}

# Download LivePortrait models (if available)
if python3 -c "from huggingface_hub import HfApi; api = HfApi(); print('✅ HuggingFace available')" 2>/dev/null; then
    echo "🎨 Attempting to download LivePortrait models..."
    # Note: These would be the actual model files when available
    # download_hf_model "KwaiVGI/LivePortrait" "appearance_feature_extractor.pth" "models/liveportrait"
    # download_hf_model "KwaiVGI/LivePortrait" "motion_extractor.pth" "models/liveportrait"
    # download_hf_model "KwaiVGI/LivePortrait" "warping_module.pth" "models/liveportrait"
    # download_hf_model "KwaiVGI/LivePortrait" "spade_generator.pth" "models/liveportrait"
    echo "ℹ️ LivePortrait models will be downloaded on first use"
else
    echo "⚠️ HuggingFace Hub not available, models will be downloaded at runtime"
fi

# Verify GPU availability
echo "🔍 Checking GPU configuration..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️ GPU not available - running in CPU mode')
"

# Setup virtual camera (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📹 Setting up virtual camera (v4l2loopback)..."
    
    # Check if v4l2loopback is available
    if modprobe v4l2loopback devices=1 video_nr=10 card_label="Mirage Virtual Camera" 2>/dev/null; then
        echo "✅ Virtual camera device created: /dev/video10"
    else
        echo "⚠️ Could not create virtual camera device (requires sudo)"
        echo "💡 Run: sudo modprobe v4l2loopback devices=1 video_nr=10 card_label='Mirage Virtual Camera'"
    fi
fi

# Start the application
echo "🚀 Starting Mirage AI Avatar System..."

if [[ "${DEPLOYMENT_TARGET}" == "huggingface" ]]; then
    # HuggingFace Spaces deployment
    echo "🤗 Running on HuggingFace Spaces with A10G GPU"
    exec python3 -u app.py
else
    # Local deployment
    echo "💻 Running locally"
    
    # Check if port 7860 is available
    if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null; then
        echo "⚠️ Port 7860 is already in use"
        PORT=7861
    else
        PORT=7860
    fi
    
    echo "🌐 Server will be available at: http://localhost:${PORT}"
    export PORT=${PORT}
    exec python3 -u app.py
fi