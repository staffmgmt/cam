from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import traceback

app = FastAPI(title="Mirage Phase 1+2 Scaffold")

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
    while True:
        try:
            data = await websocket.receive_bytes()
            # Echo straight back
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


# Note: The Dockerfile / README launch with: uvicorn app:app --port 7860
if __name__ == "__main__":  # Optional direct run helper
    import uvicorn  # type: ignore

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)