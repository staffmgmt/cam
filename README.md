---
title: Mirage Real-time AI Avatar
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
hardware: a10g-large
python_version: "3.10"
tags:
- real-time
- ai-avatar
- face-swap
- voice-conversion
- virtual-camera
short_description: "Real-time AI avatar with face swap + voice conversion"
---

# 🎭 Mirage: Real-time AI Avatar System

Mirage performs real-time identity-preserving face swap plus optional facial enhancement and (stub) voice conversion, streaming back a virtual camera + microphone feed with sub‑250ms target latency. Designed for live calls, streaming overlays, and privacy where you want a consistent alternate appearance.

## 🚀 Features

- **Real-time Face Swap (InSwapper)**: Identity transfer from a single reference image to your live video.
- **Enhancement (Optional)**: CodeFormer restoration (fidelity‑controllable) if weights present.
- **Low Latency WebRTC**: Bi-directional streaming via aiortc (camera + mic) with adaptive frame scaling.
- **Voice Conversion Stub**: Pluggable path ready for RVC / HuBERT integration (currently pass-through by default).
- **Virtual Camera**: Output suitable for Zoom, Meet, Discord, OBS (via local virtual camera module).
- **Model Auto-Provisioning**: Deterministic downloader for required swap + enhancer weights.
- **Metrics & Health**: JSON endpoints for latency, FPS, GPU memory, and pipeline stats.

## 🎯 Use Cases

- **Video Conferencing Privacy**: Appear as a consistent alternate identity.
- **Streaming / VTubing**: Lightweight swap + enhancement pipeline for overlays.
- **A/B Creative Experiments**: Rapid prototyping of face identity transforms.
- **Data Minimization**: Keep original face private while communicating.

## 🛠️ Technology Stack

- **Face Detection & Embedding**: InsightFace `buffalo_l` (SCRFD + embedding).
- **Face Swap Core**: `inswapper_128_fp16.onnx` (InSwapper) via InsightFace model zoo.
- **Enhancer (optional)**: CodeFormer 0.1 (fidelity controllable).
- **Backend**: FastAPI + aiortc (WebRTC) + asyncio.
- **Metrics**: Custom endpoints (`/metrics`, `/gpu`) with rolling latency/FPS stats.
- **Downloader**: Atomic, lock-protected model fetcher (`model_downloader.py`).
- **Frontend**: Minimal WebRTC client (`static/`).

## 📊 Performance Targets

- **Processing Window**: <50ms typical swap @ 512px (A10G) w/ single face.
- **End-to-end Latency Goal**: <250ms (capture → swap → enhancement → return).
- **Adaptive Scale**: Frames >512px longest side are downscaled before inference.
- **Enhancement Overhead**: CodeFormer ~18–35ms (A10G, single face, 512px) – approximate; adjust fidelity to trade quality vs latency.

## 🚀 Quick Start (Hugging Face Space)

1. Open the Space UI and allow camera/microphone.
2. Click **Initialize** – triggers model download (if not already cached) & pipeline load.
3. Upload a clear, front-facing reference image (only largest face is used).
4. Start streaming – swapped frames appear in the preview.
5. (Optional) Provide CodeFormer weights (`models/codeformer/codeformer.pth`) for enhancement.
6. Use the virtual camera integration locally (if running self-hosted) to broadcast swapped output to Zoom/OBS.

## 🔧 Technical Details

### Latency Optimization
- Adaptive quality control based on processing time
- Frame buffering with overflow protection
- GPU memory management and cleanup
- Audio-video synchronization within 150ms

### Model Flow
1. Capture frame → optional downscale to <=512 max side
2. InsightFace detector+embedding obtains face bboxes + identity vectors
3. InSwapper ONNX performs identity replacement using source embedding
4. Optional CodeFormer enhancer refines facial region
5. Frame returned to WebRTC outbound track

### Real-time Features
- WebRTC (aiortc) low-latency transport.
- Asynchronous frame processing (background tasks) to avoid blocking capture.
- Adaptive pre-inference downscale heuristic (cap largest dimension to 512).
- Metrics-driven latency tracking for dynamic future pacing.

## 📱 Virtual Camera Integration

The system creates a virtual camera device that can be used in:

- **Video Conferencing**: Zoom, Google Meet, Microsoft Teams, Discord
- **Streaming Software**: OBS Studio, Streamlabs, XSplit
- **Social Media**: WhatsApp Desktop, Skype, Facebook Messenger
- **Gaming**: Steam, Discord voice channels

## ⚡ Metrics & Observability

Key endpoints (base URL: running server root):

| Endpoint | Description |
|----------|-------------|
| `/metrics` | Core video/audio latency & FPS stats |
| `/gpu` | GPU presence + memory usage (torch / nvidia-smi) |
| `/webrtc/ping` | WebRTC router availability & TURN status |
| `/pipeline_status` (if implemented) | High-level pipeline readiness |

Pipeline stats (subset) from swap pipeline:
```json
{
	"frames": 240,
	"avg_latency_ms": 42.7,
	"swap_faces_last": 1,
	"enhanced_frames": 180,
	"enhancer": "codeformer",
	"codeformer_fidelity": 0.75,
	"codeformer_loaded": true
}
```

## 🔒 Privacy & Security

- No reference image persisted to disk (processed in-memory).
- Only model weights are cached; media frames are transient.
- Optional API key enforcement via `MIRAGE_API_KEY` + `MIRAGE_REQUIRE_API_KEY=1`.

## 🔧 Environment Variables (Face Swap & Enhancers)

| Variable | Purpose | Default |
|----------|---------|---------|
| `MIRAGE_DOWNLOAD_MODELS` | Auto download required models on startup | `1` |
| `MIRAGE_INSWAPPER_URL` | Override InSwapper ONNX URL | internal default |
| `MIRAGE_CODEFORMER_URL` | Override CodeFormer weight URL | 0.1 release |
| `MIRAGE_CODEFORMER_FIDELITY` | 0.0=more detail recovery, 1.0=preserve input | `0.75` |
| `MIRAGE_MAX_FACES` | Swap up to N largest faces per frame | `1` |
| `MIRAGE_CUDA_ONLY` | Restrict ONNX to CUDA EP + CPU fallback | unset |
| `MIRAGE_API_KEY` | Shared secret for control / TURN token | unset |
| `MIRAGE_REQUIRE_API_KEY` | Enforce API key if set | `0` |
| `MIRAGE_TOKEN_TTL` | Signed token lifetime (seconds) | `300` |
| `MIRAGE_STUN_URLS` | Comma list of STUN servers | Google defaults |
| `MIRAGE_TURN_URL` | TURN URI(s) (comma separated) | unset |
| `MIRAGE_TURN_USER` | TURN username | unset |
| `MIRAGE_TURN_PASS` | TURN credential | unset |
| `MIRAGE_FORCE_RELAY` | Force relay-only traffic | `0` |
| `MIRAGE_TURN_TLS_ONLY` | Filter TURN to TLS/TCP | `1` |
| `MIRAGE_PREFER_H264` | Prefer H264 codec in SDP munging | `0` |
| `MIRAGE_VOICE_ENABLE` | Enable voice processor stub | `0` |
| `MIRAGE_PERSIST_MODELS` | Persist models under /data and symlink /app/models | `1` |
| `MIRAGE_PERSIST_MODELS` | Persist models in /data (HF Space) via symlink | `1` |

CodeFormer fidelity example:
```bash
MIRAGE_CODEFORMER_FIDELITY=0.6
```

## 📋 Requirements

- **GPU**: NVIDIA (Ampere+ recommended). CPU-only will be extremely slow.
- **VRAM**: ~3–4GB baseline (swap + detector) + optional enhancer overhead.
- **RAM**: 8GB+ (12–16GB recommended for multitasking).
- **Browser**: Chromium-based / Firefox with WebRTC.
- **Reference Image**: Clear, frontal, good lighting, minimal occlusions.

## 🛠️ Development / Running Locally

Download models & start server:
```bash
python model_downloader.py  # or set MIRAGE_DOWNLOAD_MODELS=1 and let startup handle
uvicorn app:app --port 7860 --host 0.0.0.0
```
Open the browser client at `http://localhost:7860`.

Set a reference image via UI (Base64 upload path) then begin WebRTC session. Inspect `/metrics` for swap latency and `webrtc/debug_state` for connection internals.

## 📄 License

MIT License - Feel free to use and modify for your projects!

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) (detection + swap)
- [CodeFormer](https://github.com/sczhou/CodeFormer) (fidelity-controllable enhancement)
- Hugging Face (inference infra)

## Metrics Endpoints (Current Subset)
- `GET /metrics`
- `GET /gpu`
- `GET /webrtc/ping`
- `GET /webrtc/debug_state`
- (Legacy endpoints referenced in SPEC may be pruned in future refactors.)

## Voice Stub Activation
Set `MIRAGE_VOICE_ENABLE=1` to route audio through the placeholder voice processor. Current behavior is pass‑through while preserving structural hooks for future RVC model integration.

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

## Model Auto-Download
`model_downloader.py` manages required weights with atomic file locks. It supports overriding sources via env variables and gracefully continues if optional enhancers fail to download.

### Persistent Storage Strategy (Hugging Face Spaces)

By default (`MIRAGE_PERSIST_MODELS=1`), the container will:

1. Create (if missing) a persistent directory: `/data/mirage_models`.
2. Migrate any existing files from an earlier ephemeral `/app/models` (first run only, if the persistent dir is empty).
3. Symlink `/app/models -> /data/mirage_models`.
4. Run integrity checks each startup: if the sentinel `.provisioned` exists but any required model (currently `inswapper/inswapper_128_fp16.onnx`) is missing, the downloader is re-invoked automatically.

Disable persistence with:
```bash
MIRAGE_PERSIST_MODELS=0
```
This forces models to re-download on each cold start (not recommended for production latency / rate-limits).

Sentinel files:
- `.provisioned` – marker indicating a successful prior provisioning.
- `.provisioned_meta.json` – sizes and metadata of required models at provisioning time (informational).

If you set `MIRAGE_PROVISION_FRESH=1`, the sentinel is removed and a full re-download is attempted (useful when updating model versions or clearing partial/corrupt files).

Troubleshooting missing models:
- Call `/debug/models` – it now reports symlink status, sentinel presence, and sizes.
- If `inswapper` is missing but sentinel present, integrity logic should already trigger re-provision. If not, trigger manually with `MIRAGE_PROVISION_FRESH=1`.


### Persistence Strategy (Hugging Face Spaces)
By default (`MIRAGE_PERSIST_MODELS=1`) the container will:
1. Create a persistent directory at `/data/mirage_models`.
2. Symlink `/app/models -> /data/mirage_models` so model downloads survive restarts.
3. Maintain a sentinel file `.provisioned` plus a meta JSON summarizing required model sizes.
4. On startup, if the sentinel exists but required files are missing (stale / manual deletion), it forces a re-download.

Disable persistence (always ephemeral) with:
```bash
MIRAGE_PERSIST_MODELS=0
```

You can also force a fresh provisioning ignoring the sentinel:
```bash
MIRAGE_PROVISION_FRESH=1
```

Debug current model status:
```bash
curl -s https://<space-subdomain>.hf.space/debug/models | jq
```

Example response:
```json
{
	"inswapper": {"exists": true, "size": 87916544},
	"codeformer": {"exists": true, "size": 178140560},
	"sentinel": {"exists": true, "meta_exists": true},
	"storage": {
		"root_is_symlink": true,
		"root_path": "/app/models",
		"target": "/data/mirage_models",
		"persist_mode_env": "1"
	},
	"pipeline_initialized": false
}
```

### Endpoints Recap
See Metrics Endpoints section above. Typical usage examples:

```bash
curl -s http://localhost:7860/metrics/async | jq
curl -s http://localhost:7860/metrics/pacing | jq '.latency_ema_ms, .pacing_hint'
curl -s http://localhost:7860/metrics/motion | jq '.recent_motion[-5:]'
```

### Pacing Hint Logic
`pacing_hint` is derived from a latency exponential moving average vs target frame time:
- ~1.0: Balanced.
- <0.85: System overloaded – consider lowering capture FPS or resolution.
- >1.15: Headroom available – you may increase FPS modestly.

### Motion Magnitude
Aggregated from per-frame keypoint motion vectors; higher values trigger more frequent face detection to avoid drift. Low motion stretches automatically reduce detection frequency to save compute.

### Enhancer Fidelity (CodeFormer)
Fidelity weight (`w`):
- Lower (e.g. 0.3–0.5): More aggressive restoration, may alter identity details.
- Higher (0.7–0.9): Preserve more original swapped structure, less smoothing.
Tune with `MIRAGE_CODEFORMER_FIDELITY`.

### Latency Histogram Snapshots
`/metrics/stage_histogram` exposes periodic snapshots (e.g. every N frames) of stage latency distribution to help identify tail regressions. Use to tune pacing thresholds or decide on model quantization.

## Security Notes
If exposing publicly:
- Set `MIRAGE_API_KEY` and `MIRAGE_REQUIRE_API_KEY=1`.
- Serve behind TLS (reverse proxy like Caddy / Nginx for certificate management).
- Optionally restrict TURN server usage or enforce relay only for stricter NAT traversal control.

## Planned Voice Pipeline (Future)
Placeholder directories exist for future real-time voice conversion integration.

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
