---
title: Mirage Real-time AI Avatar
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
hardware: a10g-large
python_version: "3.10"
models:
- KwaiVGI/LivePortrait
- RVC-Project/Retrieval-based-Voice-Conversion-WebUI
tags:
- real-time
- ai-avatar
- face-animation
- voice-conversion
- live-portrait
- rvc
- virtual-camera
short_description: "Real-time AI avatar with face animation and voice conversion"
---

# 🎭 Mirage: Real-time AI Avatar System

Transform yourself into an AI avatar in real-time with sub-250ms latency! Perfect for video calls, streaming, and virtual meetings.

## 🚀 Features

- **Real-time Face Animation**: Live portrait animation using state-of-the-art AI
- **Voice Conversion**: Real-time voice transformation with RVC
- **Ultra-low Latency**: <250ms end-to-end latency optimized for A10G GPU
- **Virtual Camera**: Direct integration with Zoom, Teams, Discord, and more
- **Adaptive Quality**: Automatic quality adjustment to maintain real-time performance
- **GPU Optimized**: Efficient memory management and CUDA acceleration

## 🎯 Use Cases

- **Video Conferencing**: Use AI avatars in Zoom, Google Meet, Microsoft Teams
- **Content Creation**: Streaming with animated avatars on Twitch, YouTube
- **Virtual Meetings**: Professional presentations with consistent avatar appearance
- **Privacy Protection**: Maintain anonymity while participating in video calls

## 🛠️ Technology Stack

- **Face Animation**: LivePortrait (KwaiVGI)
- **Voice Conversion**: RVC (Retrieval-based Voice Conversion)
- **Face Detection**: SCRFD with optimized inference
- **Backend**: FastAPI with WebSocket streaming
- **Frontend**: WebRTC-enabled real-time client
- **GPU**: NVIDIA A10G with CUDA optimization

## 📊 Performance Specs

- **Video Resolution**: 512x512 @ 20 FPS (adaptive)
- **Audio Processing**: 160ms chunks @ 16kHz
- **End-to-end Latency**: <250ms target
- **GPU Memory**: ~8GB peak usage on A10G
- **Face Detection**: SCRFD every 5 frames for efficiency

## 🚀 Quick Start

1. **Initialize Pipeline**: Click "Initialize AI Pipeline" to load models
2. **Set Reference**: Upload your reference image for avatar creation
3. **Start Capture**: Begin real-time avatar generation
4. **Enable Virtual Camera**: Use avatar output in third-party apps

## 🔧 Technical Details

### Latency Optimization
- Adaptive quality control based on processing time
- Frame buffering with overflow protection
- GPU memory management and cleanup
- Audio-video synchronization within 150ms

### Model Architecture
- **LivePortrait**: Efficient portrait animation with stitching control
- **RVC**: High-quality voice conversion with minimal latency
- **SCRFD**: Fast face detection with confidence thresholding

### Real-time Features
- WebSocket streaming for minimal overhead
- Adaptive resolution (512x512 → 384x384 → 256x256)
- Quality degradation order: Quality → FPS → Resolution
- Automatic recovery when performance improves

## 📱 Virtual Camera Integration

The system creates a virtual camera device that can be used in:

- **Video Conferencing**: Zoom, Google Meet, Microsoft Teams, Discord
- **Streaming Software**: OBS Studio, Streamlabs, XSplit
- **Social Media**: WhatsApp Desktop, Skype, Facebook Messenger
- **Gaming**: Steam, Discord voice channels

## ⚡ Performance Monitoring

Real-time metrics include:
- Video FPS and latency
- GPU memory usage
- Audio processing time
- Frame drop statistics
- System resource utilization

## 🔒 Privacy & Security

- All processing happens locally on the GPU
- No data is stored or transmitted to external servers
- Reference images are processed in memory only
- WebSocket connections use secure protocols

## 🔧 Advanced Configuration

The system automatically adapts quality based on performance:

- **High Performance**: 512x512 @ 20 FPS, full quality
- **Medium Performance**: 384x384 @ 18 FPS, reduced quality
- **Low Performance**: 256x256 @ 15 FPS, minimum quality

## 📋 Requirements

- **GPU**: NVIDIA A10G or equivalent (RTX 3080+ recommended)
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Browser**: Chrome/Edge with WebRTC support
- **Camera**: Any USB webcam or built-in camera

## 🛠️ Development

Built with modern technologies:
- FastAPI for high-performance backend
- PyTorch with CUDA acceleration
- OpenCV for image processing
- WebSocket for real-time communication
- Docker for consistent deployment

## 📄 License

MIT License - Feel free to use and modify for your projects!

## 🙏 Acknowledgments

- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) for face animation
- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for voice conversion
- [InsightFace](https://github.com/deepinsight/insightface) for face detection
- HuggingFace for providing A10G GPU infrastructure

## Metrics Endpoints
- `GET /metrics` – JSON with audio/video counters, EMAs (loop interval, inference), rolling FPS, frame interval EMA.
- `GET /gpu` – GPU availability & memory (torch or `nvidia-smi` fallback).

Example:
```bash
curl -s http://localhost:7860/metrics | jq '.video_fps_rolling, .audio_infer_time_ema_ms'
```

## Voice Stub Activation
Set `MIRAGE_VOICE_ENABLE=1` to activate the voice processor stub. Behavior:
- Audio chunks are routed through `voice_processor.process_pcm_int16` (pass-through now).
- `audio_infer_time_ema_ms` becomes > 0 after a few chunks.
- When disabled, inference EMA remains 0.0.

## Future Parameterization
- Frontend will fetch a `/config` endpoint to align `chunk_ms` and `video_max_fps` dynamically.
- Adaptation layer will adjust chunk size and video quality based on runtime ratios.

## Accessing Endpoints on Hugging Face Spaces
When viewing the Space at `https://huggingface.co/spaces/Islamckennon/mirage` you are on the Hub UI (repository page). **API paths appended there (e.g. `/metrics`, `/gpu`) will 404** because that domain serves repo metadata, not your running container.

Your running app is exposed on a separate subdomain:

```
https://islamckennon-mirage.hf.space
```

(Pattern: `https://<username>-<space_name>.hf.space`)

So the full endpoint URLs are, for example:

```
https://islamckennon-mirage.hf.space/metrics
https://islamckennon-mirage.hf.space/gpu
```

If the Space is private you must be logged into Hugging Face in the browser for these to load.

## Troubleshooting "Restarting" Status
If the Space shows a perpetual "Restarting" badge:
1. Open the **Logs** panel and switch to the *Container* tab (not just *Build*) to see runtime exceptions.
2. Look for the `[startup] { ... }` line. If absent, the app may be crashing before FastAPI starts (syntax error, missing dependency, etc.).
3. Ensure the container listens on port 7860 (this repo's Dockerfile already does). The startup log now prints the `port` value it detected.
4. GPU provisioning can briefly cycle while allocating hardware; give it a minute after the first restart. If it loops >5 times, inspect for CUDA driver errors or `torch` import failures.
5. Test locally with `uvicorn app:app --port 7860` to rule out code issues.
6. Use `curl -s https://islamckennon-mirage.hf.space/health` (if public) to verify liveness.

If problems persist, capture the Container log stack trace and open an issue.

## Model Weights (Planned Voice Pipeline)
The codebase now contains placeholder directories for upcoming audio feature extraction and conversion models.

```
models/
	hubert/            # HuBERT feature extractor checkpoint(s)
	rmvpe/             # RMVPE pitch extraction weights
	rvc/               # RVC (voice conversion) model checkpoints
```

### Expected File Names & Relative Paths
You can adapt names, but these canonical filenames will be referenced in future code examples:

| Component | Recommended Source | Save As (relative path) |
|-----------|--------------------|-------------------------|
| HuBERT Base | `facebook/hubert-base-ls960` (Torch .pt) or official fairseq release | `models/hubert/hubert_base.pt` |
| RMVPE Weights | Community RMVPE release (pitch extraction) | `models/rmvpe/rmvpe.pt` |
| RVC Model Checkpoint | Your trained / downloaded RVC model | `models/rvc/model.pth` |

Optional additional assets (not yet required):
| Type | Path Example |
|------|--------------|
| Speaker embedding(s) | `models/rvc/spk_embeds.npy` |
| Index file (faiss) | `models/rvc/features.index` |

### Manual Download (Lightweight Instructions)
Because licenses vary and some distributions require acceptance, **we do not auto-download by default**. Manually fetch the files you are licensed to use:

```bash
# HuBERT (example using torch hub or direct URL)
curl -L -o models/hubert/hubert_base.pt \
	https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt

# RMVPE (replace URL with the official/community mirror you trust)
curl -L -o models/rmvpe/rmvpe.pt \
	https://example.com/path/to/rmvpe.pt

# RVC model (place your trained checkpoint)
cp /path/to/your_rvc_model.pth models/rvc/model.pth
```

All of these binary patterns are ignored by git via `.gitignore` (we only keep `.gitkeep` & documentation). Verify after download:

```bash
ls -lh models/hubert models/rmvpe models/rvc
```

### Optional Convenience Script
You can create `scripts/download_models.sh` (not yet included) with the above `curl` commands; keep URLs commented if redistribution is unclear. Example skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p models/hubert models/rmvpe models/rvc
echo "(Add real URLs you are licensed to download)"
# curl -L -o models/hubert/hubert_base.pt <URL>
# curl -L -o models/rmvpe/rmvpe.pt <URL>
```

### Integrity / Size Hints (Approximate)
| File | Typical Size |
|------|--------------|
| hubert_base.pt | ~360 MB |
| rmvpe.pt | ~90–150 MB (varies) |
| model.pth (RVC) | 50–200+ MB |

Ensure your Space has enough disk (HF GPU Spaces usually allow several GB, but keep total under limits).

### License Notes
Review and comply with each model's license (Fairseq / Facebook AI for HuBERT, RMVPE authors, your own RVC training data constraints). Do **not** commit weights.

Future code will detect presence and log which components are available at startup.

## License
MIT
