---
title: Mirage
emoji: 👀
colorFrom: indigo
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# Mirage

Phase 1–2 FastAPI + WebSocket echo scaffold (no ML models yet).

## Current Status
- GPU-backed metrics endpoint (`/metrics`, `/gpu`)
- Voice stub integrated (pass-through timing)
- Audio & Video echo functioning
- Frontend governed: audio chunk 160ms, video max 10 FPS
- Static client operational

## Planned Phases
- GPU switch
- Metrics
- Voice skeleton
- Video skeleton
- Adaptation
- Security

## Local Run
```bash
pip install -r requirements.txt
uvicorn app:app --port 7860
```

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MIRAGE_CHUNK_MS` | `160` | Target audio capture & processing chunk duration (ms). Frontend currently hard-set; future: fetched dynamically. |
| `MIRAGE_VOICE_ENABLE` | `0` | Enable voice processing stub path (adds inference timing EMA). |
| `MIRAGE_VIDEO_MAX_FPS` | `10` | Target maximum outbound video frame send rate (frontend governed). |
| `MIRAGE_METRICS_FPS_WINDOW` | `30` | Rolling window size for FPS calculation. |

Export before launching uvicorn or set in Space settings:
```bash
export MIRAGE_VOICE_ENABLE=1
export MIRAGE_CHUNK_MS=160
uvicorn app:app --port 7860
```

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

## License
MIT
