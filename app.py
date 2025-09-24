"""Unified FastAPI application for Mirage system (face swap pipeline).

All obsolete LivePortrait / reenactment / Gradio demo code removed.
This file supersedes original_fastapi_app.py and prior app.py stub.
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import cv2
import numpy as np
from typing import Any, Dict
from metrics import metrics as _metrics_singleton, Metrics
from config import config
from voice_processor import voice_processor
from swap_pipeline import get_pipeline

app = FastAPI(title="Mirage Real-time AI Avatar System")

pipeline = get_pipeline()
pipeline_initialized = False

if config.metrics_fps_window != 30:
    metrics = Metrics(fps_window=config.metrics_fps_window)
else:
    metrics = _metrics_singleton

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

WEBRTC_ROUTER_LOADED = False
WEBRTC_IMPORT_ERROR = None
try:
    from webrtc_server import router as webrtc_router  # type: ignore
    app.include_router(webrtc_router)
    WEBRTC_ROUTER_LOADED = True
except Exception as e:  # pragma: no cover
    WEBRTC_IMPORT_ERROR = str(e)
    from fastapi import HTTPException as _HTTPException

    @app.get("/webrtc/token")
    async def _fallback_token():  # type: ignore[override]
        raise _HTTPException(status_code=503, detail=f"WebRTC unavailable: {WEBRTC_IMPORT_ERROR or 'router not loaded'}")

    @app.post("/webrtc/offer")
    async def _fallback_offer():  # type: ignore[override]
        raise _HTTPException(status_code=503, detail=f"WebRTC unavailable: {WEBRTC_IMPORT_ERROR or 'router not loaded'}")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = static_dir / "index.html"
    try:
        content = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = "<html><body><h1>Mirage AI Avatar System</h1><p>FastAPI unified app.</p></body></html>"
    return HTMLResponse(content)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "system": "real-time-ai-avatar",
        "pipeline_loaded": pipeline_initialized,
        "gpu_available": pipeline.app is not None,  # coarse indicator
        "webrtc_router_loaded": WEBRTC_ROUTER_LOADED,
        "webrtc_import_error": WEBRTC_IMPORT_ERROR,
    }


@app.post("/initialize")
async def initialize_pipeline():
    global pipeline_initialized
    if pipeline_initialized:
        return {"status": "already_initialized"}
    try:
        ok = pipeline.initialize()
        pipeline_initialized = ok
        return {"status": "success" if ok else "error"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/set_reference")
async def set_reference_image(file: UploadFile = File(...)):
    global pipeline_initialized
    try:
        contents = await file.read()
        import cv2, numpy as _np
        arr = _np.frombuffer(contents, _np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        if not pipeline_initialized:
            pipeline.set_source_image(frame)
            return {"status": "queued"}
        ok = pipeline.set_source_image(frame)
        return {"status": "success" if ok else "error"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/metrics")
async def get_metrics():
    snap = metrics.snapshot()
    if pipeline_initialized:
        snap["ai_pipeline"] = pipeline.get_stats()
    return snap


@app.get("/pipeline_status")
async def pipeline_status():
    if not pipeline_initialized:
        return {"initialized": False}
    return {"initialized": True, "stats": pipeline.get_stats(), "source_set": pipeline.source_face is not None}


@app.get("/gpu")
async def gpu():
    # Minimal GPU presence check
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return {"available": True, "name": torch.cuda.get_device_name(0)}
    except Exception:
        pass
    return {"available": False}


@app.get("/debug/models")
async def debug_models():
    root = Path(__file__).parent / 'models'
    ins = root / 'inswapper' / 'inswapper_128_fp16.onnx'
    codef = root / 'codeformer' / 'codeformer.pth'
    return {
        'inswapper': {'exists': ins.exists(), 'size': ins.stat().st_size if ins.exists() else 0},
        'codeformer': {'exists': codef.exists(), 'size': codef.stat().st_size if codef.exists() else 0},
        'pipeline_initialized': pipeline_initialized
    }


@app.on_event("startup")
async def startup_log():
    print("[startup] unified app starting")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)