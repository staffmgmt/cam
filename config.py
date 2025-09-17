from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    val_lower = val.lower().strip()
    if val_lower in {"1", "true", "yes", "on"}:
        return True
    if val_lower in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Config:
    chunk_ms: int = 160
    voice_enable: bool = False
    video_max_fps: int = 10
    metrics_fps_window: int = 30

    @staticmethod
    def load() -> "Config":
        return Config(
            chunk_ms=_get_int("MIRAGE_CHUNK_MS", 160),
            voice_enable=_get_bool("MIRAGE_VOICE_ENABLE", False),
            video_max_fps=_get_int("MIRAGE_VIDEO_MAX_FPS", 10),
            metrics_fps_window=_get_int("MIRAGE_METRICS_FPS_WINDOW", 30),
        )

    def as_dict(self) -> dict[str, Any]:  # JSON-friendly
        return {
            "chunk_ms": self.chunk_ms,
            "voice_enable": self.voice_enable,
            "video_max_fps": self.video_max_fps,
            "metrics_fps_window": self.metrics_fps_window,
        }


# Eagerly loaded singleton pattern (can be reloaded manually if needed)
config = Config.load()
