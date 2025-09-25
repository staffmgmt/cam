"""Pre-cache InsightFace analysis models (e.g., buffalo_l) during image build.

This script is invoked at Docker build time to eliminate first-run latency
and remove runtime network dependency for detector / recognition models
used by FaceAnalysis.

Behavior:
  * Determines target model name via MIRAGE_ANALYSIS_MODEL (default 'buffalo_l').
  * Uses INSIGHTFACE_HOME as root (set in Dockerfile to /app/.insightface).
  * Attempts GPU providers first; falls back cleanly to CPU-only.
  * Emits a JSON summary line to stdout for auditing.

Idempotent: If the model directory already contains .onnx files, it will skip
loading unless MIRAGE_PRECACHE_FORCE=1.
"""
from __future__ import annotations

import os
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List


def _available_providers() -> List[str]:
    try:
        import onnxruntime as ort  # type: ignore
        return list(ort.get_available_providers())
    except Exception:
        return []


def precache() -> Dict[str, Any]:
    model_name = os.getenv("MIRAGE_ANALYSIS_MODEL", "buffalo_l").strip() or "buffalo_l"
    root = Path(os.getenv("INSIGHTFACE_HOME", str(Path.home() / ".insightface")))
    models_dir = root / "models" / model_name
    force = os.getenv("MIRAGE_PRECACHE_FORCE", "0").lower() in {"1", "true", "yes", "on"}
    result: Dict[str, Any] = {
        "model": model_name,
        "root": str(root),
        "models_dir": str(models_dir),
        "force": force,
    }

    try:
        existing = list(models_dir.glob("*.onnx"))
        if existing and not force:
            result.update(
                {
                    "skipped": True,
                    "reason": "already_present",
                    "onnx_files": [p.name for p in existing],
                    "count": len(existing),
                }
            )
            return result

        from insightface.app import FaceAnalysis  # type: ignore

        providers = _available_providers()
        requested: List[str]
        ctx_id: int
        if "CUDAExecutionProvider" in providers:
            requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            # ctx_id=0 leverages first GPU
            ctx_id = 0
        else:
            requested = ["CPUExecutionProvider"]
            # ctx_id=-1 signals CPU in insightface
            ctx_id = -1

        app = FaceAnalysis(name=model_name, providers=requested)
        # det_size large enough to force detector weight load
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # Re-scan directory for confirmation
        onnx_files = list(models_dir.glob("*.onnx"))
        result.update(
            {
                "skipped": False,
                "providers": providers,
                "used_providers": requested,
                "ctx_id": ctx_id,
                "onnx_files": [p.name for p in onnx_files],
                "count": len(onnx_files),
                "success": True,
            }
        )
        return result
    except Exception as e:  # noqa: BLE001
        result.update(
            {
                "error": str(e),
                "trace": traceback.format_exc(limit=6),
                "success": False,
            }
        )
        return result


if __name__ == "__main__":
    summary = precache()
    print(json.dumps({"event": "precache_insightface", **summary}))
    # Non-fatal: build should not fail if network hiccup; runtime will retry
    if not summary.get("success", True) and not summary.get("skipped", False):
        # Exit with 0 to avoid aborting image build; log handled in JSON
        pass
