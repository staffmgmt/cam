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
# Since get_pipeline() now initializes immediately, mark initialized if pipeline.initialized True
pipeline_initialized = getattr(pipeline, 'initialized', False)

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
    # webrtc_server router is defined WITHOUT a prefix; mount under /webrtc here so
    # endpoints resolve at /webrtc/ping, /webrtc/offer, etc.
    app.include_router(webrtc_router, prefix="/webrtc")
    WEBRTC_ROUTER_LOADED = True

    # --- Compatibility layer ---
    # In some deployed revisions the underlying router still carried an internal
    # '/webrtc' prefix, yielding effective paths like /webrtc/webrtc/ping.
    # The production JS client calls /webrtc/ping, /webrtc/token, /webrtc/ice_config, /webrtc/offer.
    # To avoid breaking older images (or double-prefix drift during hot reload),
    # expose lightweight pass-through wrappers at the expected single-prefix paths
    # ONLY if those paths are currently missing (best-effort). Since FastAPI does
    # not provide a simple public API to query registered routes before definition
    # without introspection, we always register wrappers; duplicates are avoided
    # because underlying double-prefixed versions have different paths.
    from typing import Optional as _Opt
    from fastapi import Body as _Body, Header as _Header
    import inspect as _inspect
    
    try:  # pragma: no cover - defensive
        # Import underlying handlers for direct reuse
        from webrtc_server import webrtc_ping as _rt_webrtc_ping, _ice_configuration as _rt_ice_conf, webrtc_offer as _rt_webrtc_offer, _mint_token as _rt_mint_token  # type: ignore
    except Exception:  # noqa: BLE001
        _rt_webrtc_ping = None  # type: ignore
        _rt_ice_conf = None  # type: ignore
        _rt_webrtc_offer = None  # type: ignore
        _rt_mint_token = None  # type: ignore

    @app.get("/webrtc/ping")
    async def _compat_webrtc_ping():  # type: ignore
        if _rt_webrtc_ping:
            return await _rt_webrtc_ping()  # type: ignore
        return {"router": False, "error": "webrtc_ping unavailable"}

    @app.get("/webrtc/ice_config")
    async def _compat_webrtc_ice_config():  # type: ignore
        if _rt_ice_conf:
            cfg = _rt_ice_conf()  # returns RTCConfiguration
            # Convert to serializable form similar to original endpoint
            servers = []
            for s in getattr(cfg, 'iceServers', []) or []:
                entry = {"urls": s.urls}
                if getattr(s, 'username', None):
                    entry["username"] = s.username
                if getattr(s, 'credential', None):
                    entry["credential"] = s.credential
                servers.append(entry)
            return {"iceServers": servers}
        return {"iceServers": []}

    @app.get("/webrtc/token")
    async def _compat_webrtc_token():  # type: ignore
        if _rt_mint_token and WEBRTC_ROUTER_LOADED:
            try:
                return {"token": _rt_mint_token()}
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)}
        return {"error": "token_unavailable"}

    @app.post("/webrtc/offer")
    async def _compat_webrtc_offer(
        offer: dict = _Body(...),
        x_api_key: _Opt[str] = _Header(default=None, alias="x-api-key"),
        x_auth_token: _Opt[str] = _Header(default=None, alias="x-auth-token"),
    ):  # type: ignore
        if _rt_webrtc_offer:
            return await _rt_webrtc_offer(offer=offer, x_api_key=x_api_key, x_auth_token=x_auth_token)  # type: ignore
        raise HTTPException(status_code=503, detail="WebRTC offer handler unavailable")
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
    """Set (or reset) the source reference face.

    Behavior:
      - Ensures pipeline initialized (synchronous) before setting reference.
      - Returns detailed status including number of faces detected in reference.
    """
    global pipeline_initialized
    try:
        contents = await file.read()
        import cv2, numpy as _np
        arr = _np.frombuffer(contents, _np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        if not pipeline_initialized or not getattr(pipeline, 'initialized', False):
            pipeline.initialize()
            pipeline_initialized = True
        ok = pipeline.set_source_image(frame)
        meta = getattr(pipeline, 'source_img_meta', {}) if ok else {}
        return {"status": "success" if ok else "error", "meta": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/debug/pipeline")
async def debug_pipeline():
    """Return detailed pipeline diagnostics for debugging."""
    exists = pipeline is not None
    if not exists:
        return {"exists": False}
    try:
        stats = pipeline.get_stats()
    except Exception as e:  # pragma: no cover
        stats = {"error": str(e)}
    return {
        "exists": True,
        "initialized": getattr(pipeline, 'initialized', False),
        "loaded": getattr(pipeline, 'loaded', False),
        "source_set": getattr(pipeline, 'source_face', None) is not None,
        "stats": stats,
    }


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
    sentinel = root / '.provisioned'
    meta = root / '.provisioned_meta.json'
    # Detect symlink & target
    is_symlink = root.is_symlink()
    target = None
    if is_symlink:
        try:
            target = root.resolve()
        except Exception:
            target = None
    storage_mode = os.environ.get('MIRAGE_PERSIST_MODELS', '1')
    return {
        'inswapper': {'exists': ins.exists(), 'size': ins.stat().st_size if ins.exists() else 0},
        'codeformer': {'exists': codef.exists(), 'size': codef.stat().st_size if codef.exists() else 0},
        'sentinel': {'exists': sentinel.exists(), 'meta_exists': meta.exists()},
        'storage': {
            'root_is_symlink': is_symlink,
            'root_path': str(root),
            'target': str(target) if target else None,
            'persist_mode_env': storage_mode
        },
        'pipeline_initialized': pipeline_initialized
    }


@app.on_event("startup")
async def startup_log():
    print("[startup] unified app starting")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)