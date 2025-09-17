from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import traceback
import time
from metrics import metrics as _metrics_singleton, Metrics
from config import config
from voice_processor import voice_processor

app = FastAPI(title="Mirage Phase 1+2 Scaffold")

# Potentially reconfigure metrics based on config
if config.metrics_fps_window != 30:  # default in metrics module
    metrics = Metrics(fps_window=config.metrics_fps_window)
else:
    metrics = _metrics_singleton

# Mount the static directory
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the static/index.html file contents as HTML."""
    index_path = static_dir / "index.html"
    try:
        content = index_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Minimal fallback to satisfy route even if file not yet present.
        content = "<html><body><h1>Mirage Scaffold</h1><p>Place an index.html in /static.</p></body></html>"
    return HTMLResponse(content)


@app.get("/health")
async def health():
    return {"status": "ok", "phase": "baseline"}


async def _echo_websocket(websocket: WebSocket, kind: str):
    await websocket.accept()
    last_ts = time.time() * 1000.0 if kind == "audio" else None
    while True:
        try:
            data = await websocket.receive_bytes()
            size = len(data)
            if kind == "audio":
                now = time.time() * 1000.0
                interval = None
                if last_ts is not None:
                    interval = now - last_ts

                infer_ms = None
                if config.voice_enable:
                    # Run through voice processor (pass-through currently)
                    processed_view, infer_ms = voice_processor.process_pcm_int16(data, sample_rate=16000)
                    # Use processed bytes for echo (still original length)
                    data = processed_view.tobytes()
                metrics.record_audio_chunk(size_bytes=size, loop_interval_ms=interval, infer_time_ms=infer_ms)
                last_ts = now
            elif kind == "video":
                metrics.record_video_frame(size_bytes=size)
            # Echo straight back (audio maybe processed)
            await websocket.send_bytes(data)
        except WebSocketDisconnect:
            # Silent disconnect
            break
        except Exception:  # noqa: BLE001
            # Print traceback for unexpected errors, then break loop
            print(f"[{kind} ws] Unexpected error:")
            traceback.print_exc()
            break


@app.websocket("/audio")
async def audio_ws(websocket: WebSocket):
    await _echo_websocket(websocket, "audio")


@app.websocket("/video")
async def video_ws(websocket: WebSocket):
    await _echo_websocket(websocket, "video")


@app.get("/metrics")
async def get_metrics():
    return metrics.snapshot()


@app.on_event("startup")
async def log_config():
    # Simple startup log of configuration
    print("[config]", config.as_dict())


# Note: The Dockerfile / README launch with: uvicorn app:app --port 7860
if __name__ == "__main__":  # Optional direct run helper
    import uvicorn  # type: ignore

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)