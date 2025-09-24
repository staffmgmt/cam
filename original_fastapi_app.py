from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import traceback
import time
import subprocess
import json
import os
import asyncio
import numpy as np
import cv2
from typing import Any, Dict, List
import hashlib
from metrics import metrics as _metrics_singleton, Metrics
from config import config
from voice_processor import voice_processor
from avatar_pipeline import get_pipeline
try:
    import model_downloader  # optional runtime downloader
except Exception:
    model_downloader = None

app = FastAPI(title="Mirage Real-time AI Avatar System")

# Initialize AI pipeline
pipeline = get_pipeline()
pipeline_initialized = False

# Potentially reconfigure metrics based on config
if config.metrics_fps_window != 30:  # default in metrics module
    metrics = Metrics(fps_window=config.metrics_fps_window)
else:
    metrics = _metrics_singleton

# Mount the static directory
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount WebRTC router (aiortc based)
WEBRTC_ROUTER_LOADED = False
WEBRTC_IMPORT_ERROR = None
try:
    from webrtc_server import router as webrtc_router  # type: ignore
    app.include_router(webrtc_router)
    WEBRTC_ROUTER_LOADED = True
except Exception as e:  # pragma: no cover
    WEBRTC_IMPORT_ERROR = str(e)
    print(f"[WARN] WebRTC router not loaded: {e}")
    # Provide diagnostic fallbacks so the frontend sees 503 instead of 404
    from fastapi import HTTPException as _HTTPException  # local alias to avoid shadowing

    @app.get("/webrtc/token")
    async def _fallback_token():  # type: ignore[override]
        raise _HTTPException(status_code=503, detail=f"WebRTC unavailable: {WEBRTC_IMPORT_ERROR or 'router not loaded'}")

    @app.post("/webrtc/offer")
    async def _fallback_offer():  # type: ignore[override]
        raise _HTTPException(status_code=503, detail=f"WebRTC unavailable: {WEBRTC_IMPORT_ERROR or 'router not loaded'}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the static/index.html file contents as HTML."""
    index_path = static_dir / "index.html"
    try:
        content = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Minimal fallback to satisfy route even if file not yet present.
        content = "<html><body><h1>Mirage AI Avatar System</h1><p>Real-time AI avatar with face animation and voice conversion.</p></body></html>"
    return HTMLResponse(content)


@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "system": "real-time-ai-avatar",
        "pipeline_loaded": pipeline_initialized,
        "gpu_available": pipeline.config.device == "cuda",
        "webrtc_router_loaded": WEBRTC_ROUTER_LOADED,
        "webrtc_import_error": WEBRTC_IMPORT_ERROR,
    }


@app.get("/healthz")
async def healthz():
    # Minimal liveness/readiness check; returns 200 if app is serving
    return {"ok": True}


@app.post("/initialize")
async def initialize_pipeline():
    """Initialize the AI pipeline"""
    global pipeline_initialized
    
    if pipeline_initialized:
        return {"status": "already_initialized", "message": "Pipeline already loaded"}
    
    try:
        # Downloader removed from initialize; provisioning handled exclusively by entrypoint.
        # This preserves deterministic model presence and avoids duplicate downloads.
        success = await pipeline.initialize()
        if success:
            pipeline_initialized = True
            return {"status": "success", "message": "Pipeline initialized successfully"}
        else:
            # Provide more detail for debugging
            try:
                stats = pipeline.get_performance_stats()
            except Exception:
                stats = {}
            # Also include model file presence for quick diagnosis
            try:
                from pathlib import Path as _Path
                lp_dir = _Path(__file__).parent / 'models' / 'liveportrait'
                files = {}
                for name in ("appearance_feature_extractor.onnx", "motion_extractor.onnx", "generator.onnx", "stitching.onnx"):
                    p = lp_dir / name
                    files[name] = {"exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0}
                stats["model_files"] = files
            except Exception:
                pass
            return {"status": "error", "message": "Failed to initialize pipeline", "details": stats}
    except Exception as e:
        return {"status": "error", "message": f"Initialization error: {str(e)}"}


    @app.get("/debug/models")
    async def debug_models(limit: int = 20):
        """Report presence, size, sha256 (if small enough) and recent audit entries.
        limit: number of recent audit log lines to return (tail)."""
        lp_dir = Path(__file__).parent / 'models' / 'liveportrait'
        models = {}
        names = [
            'appearance_feature_extractor.onnx',
            'motion_extractor.onnx',
            'generator.onnx',
            'stitching.onnx',
            'libgrid_sample_3d_plugin.so'
        ]
        hash_disabled = os.getenv('MIRAGE_HASH_MODELS', '1').lower() in ('0','false','no','off')
        max_hash_bytes = int(os.getenv('MIRAGE_HASH_MAX_BYTES', '52428800'))  # default 50MB
        for name in names:
            p = lp_dir / name
            info: Dict[str, Any] = {"exists": p.exists()}
            if p.exists():
                try:
                    size = p.stat().st_size
                    info['size_bytes'] = size
                    if not hash_disabled and size <= max_hash_bytes:
                        h = hashlib.sha256()
                        with p.open('rb') as f:
                            for chunk in iter(lambda: f.read(8192), b''):
                                h.update(chunk)
                        info['sha256'] = h.hexdigest()
                    else:
                        info['sha256'] = None if hash_disabled else 'skipped_large'
                except Exception as e:  # pragma: no cover
                    info['error'] = str(e)
            models[name] = info
        # Sentinel / audit
        sentinel = (lp_dir / '.provisioned').exists()
        audit_path = lp_dir / '_download_audit.jsonl'
        audit_entries: List[Dict[str, Any]] = []
        if audit_path.exists():
            try:
                lines = audit_path.read_text(encoding='utf-8').strip().splitlines()
                for line in lines[-limit:]:
                    try:
                        audit_entries.append(json.loads(line))
                    except Exception:
                        pass
            except Exception:
                pass
        return {
            'models': models,
            'sentinel_present': sentinel,
            'audit_tail': audit_entries,
            'audit_count': len(audit_entries),
            'limit': limit,
        }


    @app.get("/debug/webrtc_config")
    async def debug_webrtc_config():
        """Expose selected WebRTC-related runtime configuration to assist debugging disconnections."""
        env = os.environ
        cfg = {
            'force_relay_param_supported': True,
            'MIRAGE_WEBRTC_VERBOSE': env.get('MIRAGE_WEBRTC_VERBOSE', ''),
            'MIRAGE_WEBRTC_FORCE_RELAY': env.get('MIRAGE_WEBRTC_FORCE_RELAY', ''),
            'MIRAGE_WEBRTC_STATS_INTERVAL': env.get('MIRAGE_WEBRTC_STATS_INTERVAL', ''),
            'MIRAGE_DL_TAG': env.get('MIRAGE_DL_TAG', ''),
        }
        return cfg


@app.post("/set_reference")
async def set_reference_image(file: UploadFile = File(...)):
    """Set reference image for avatar.
    If pipeline isn't initialized yet, queue the image so the user sees it as soon as video starts.
    """
    global pipeline_initialized

    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Basic EXIF orientation handling: if image is portrait and very large, rotate or resize for detection
        if frame is not None:
            h, w = frame.shape[:2]
            # Resize overly large images to speed detection and avoid memory limits
            max_dim = 1280
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            # Heuristic: if image is rotated (common on mobile), try rotating copies until a face is found (done in pipeline)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        if not pipeline_initialized:
            # Queue the frame so the UI gets immediate visual confirmation even before init
            pipeline.reference_frame = frame.copy()
            return {"status": "success", "message": "Reference queued (pipeline not initialized yet)"}

        # Pipeline initialized: attempt to validate/set via detector
        success = pipeline.set_reference_frame(frame)

        if success:
            return {"status": "success", "message": "Reference image set successfully"}
        else:
            return {"status": "error", "message": "No suitable face found in image"}

    except Exception as e:
        return {"status": "error", "message": f"Error setting reference: {str(e)}"}


# Note: Legacy WebSocket streaming endpoints removed in production.


@app.get("/metrics")
async def get_metrics():
    base_metrics = metrics.snapshot()
    
    # Add AI pipeline metrics if available
    if pipeline_initialized:
        pipeline_stats = pipeline.get_performance_stats()
        base_metrics.update({
            "ai_pipeline": pipeline_stats
        })
    
    return base_metrics


@app.get("/pipeline_status")
async def get_pipeline_status():
    """Get detailed pipeline status"""
    if not pipeline_initialized:
        return {
            "initialized": False,
            "message": "Pipeline not initialized"
        }
    
    try:
        stats = pipeline.get_performance_stats()
        return {
            "initialized": True,
            "stats": stats,
            "reference_set": pipeline.reference_frame is not None
        }
    except Exception as e:
        return {
            "initialized": False,
            "error": str(e)
        }


@app.get("/gpu")
async def gpu_info():
    """Return basic GPU availability and memory statistics.

    Priority order:
    1. torch (if installed and CUDA available) for detailed stats per device.
    2. nvidia-smi (if executable present) for name/total/used.
    3. Fallback: available false.
    """
    # Response scaffold
    resp: Dict[str, Any] = {
        "available": False,
        "provider": None,
        "device_count": 0,
        "devices": [],  # type: ignore[list-item]
    }

    # Try torch first (lazy import)
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            resp["available"] = True
            resp["provider"] = "torch"
            count = torch.cuda.device_count()
            resp["device_count"] = count
            devices: List[Dict[str, Any]] = []
            for idx in range(count):
                name = torch.cuda.get_device_name(idx)
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)  # type: ignore[arg-type]
                except TypeError:
                    # Older PyTorch versions take no index
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                allocated = torch.cuda.memory_allocated(idx)
                reserved = torch.cuda.memory_reserved(idx)
                # Estimate free including unallocated reserved as reclaimable
                est_free = free_bytes + max(reserved - allocated, 0)
                to_mb = lambda b: round(b / (1024 * 1024), 2)
                devices.append({
                    "index": idx,
                    "name": name,
                    "total_mb": to_mb(total_bytes),
                    "allocated_mb": to_mb(allocated),
                    "reserved_mb": to_mb(reserved),
                    "free_mem_get_info_mb": to_mb(free_bytes),
                    "free_estimate_mb": to_mb(est_free),
                })
            resp["devices"] = devices
            return resp
    except Exception:  # noqa: BLE001
        # Torch not installed or failed; fall through to nvidia-smi
        pass

    # Try nvidia-smi fallback
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=2).decode("utf-8").strip()
        lines = [l for l in out.splitlines() if l.strip()]
        if lines:
            resp["available"] = True
            resp["provider"] = "nvidia-smi"
            resp["device_count"] = len(lines)
            devices: List[Dict[str, Any]] = []
            for idx, line in enumerate(lines):
                # Expect: name, total, used
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    name, total_str, used_str = parts[:3]
                    try:
                        total = float(total_str)
                        used = float(used_str)
                        free = max(total - used, 0)
                    except ValueError:
                        total = used = free = 0.0
                    devices.append({
                        "index": idx,
                        "name": name,
                        "total_mb": total,
                        "allocated_mb": used,  # approximate
                        "reserved_mb": None,
                        "free_estimate_mb": free,
                    })
            resp["devices"] = devices
            return resp
    except Exception:  # noqa: BLE001
        pass

    return resp


@app.on_event("startup")
async def log_config():
    # Enhanced startup logging: core config + GPU availability summary.
    cfg = config.as_dict()
    # GPU probe (reuse gpu_info logic minimally without full device list to keep log concise)
    gpu_available = False
    gpu_name = None
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
        else:
            # Fallback quick nvidia-smi single line
            try:
                out = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"
                ], stderr=subprocess.STDOUT, timeout=1).decode("utf-8").strip().splitlines()
                if out:
                    gpu_available = True
                    gpu_name = out[0].strip()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    # Honor dynamic PORT if provided (HF Spaces usually fixed at 7860 for docker, but logging helps debugging)
    listen_port = int(os.getenv("PORT", "7860"))
    startup_line = {
        "chunk_ms": cfg.get("chunk_ms"),
        "voice_enabled": cfg.get("voice_enable"),
        "metrics_fps_window": cfg.get("metrics_fps_window"),
        "video_fps_limit": cfg.get("video_max_fps"),
        "port": listen_port,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
    }
    print("[startup]", startup_line)
    # Model download: ensure presence once, then also fire background attempt
    # Downloader invocation removed (entrypoint is now single source of truth)


@app.get("/debug/models")
async def debug_models():
    """Return presence, sizes, optional sha256 prefixes and pipeline load status."""
    from pathlib import Path
    import hashlib
    lp_dir = Path(__file__).parent / 'models' / 'liveportrait'
    files = {}
    hash_enabled = os.getenv('MIRAGE_HASH_MODELS','0').lower() in ('1','true','yes','on')
    for name in ("appearance_feature_extractor.onnx", "motion_extractor.onnx", "generator.onnx", "stitching.onnx"):
        p = lp_dir / name
        info = {"exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0}
        if p.exists() and hash_enabled:
            try:
                h = hashlib.sha256()
                with open(p, 'rb') as fh:
                    for chunk in iter(lambda: fh.read(1024*512), b''):
                        h.update(chunk)
                info["sha256"] = h.hexdigest()
            except Exception as e:
                info["hash_error"] = str(e)
        files[name] = info
    try:
        stats = pipeline.get_performance_stats()
    except Exception as e:
        stats = {"error": str(e)}
    flags = {
        "MIRAGE_DOWNLOAD_MODELS": os.getenv("MIRAGE_DOWNLOAD_MODELS"),
        "MIRAGE_ALLOW_WARP_FALLBACK": os.getenv("MIRAGE_ALLOW_WARP_FALLBACK"),
        "MIRAGE_HASH_MODELS": os.getenv("MIRAGE_HASH_MODELS"),
    }
    all_required = all(files[f]["exists"] for f in ("appearance_feature_extractor.onnx","motion_extractor.onnx","generator.onnx"))
    return {"files": files, "flags": flags, "all_required_present": all_required, "pipeline_stats": stats}


# Removed /debug/download_models endpoint to avoid triggering duplicate downloads after provisioning.


# Note: The Dockerfile / README launch with: uvicorn app:app --port 7860
if __name__ == "__main__":  # Optional direct run helper
    import uvicorn  # type: ignore

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)