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
- CPU baseline
- `/audio` & `/video` echo
- Static client

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

## License
MIT
