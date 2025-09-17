from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Any


class Metrics:
    """Simple in-process metrics accumulator with lightweight locking.

    Tracks audio/video counts, sizes, EMAs, and rolling FPS.
    """

    def __init__(self, fps_window: int = 30, ema_alpha: float = 0.2) -> None:
        self._lock = threading.Lock()
        # Audio counters
        self.audio_chunks = 0
        self.audio_bytes = 0
        self.audio_avg_chunk_size = 0.0
        self.audio_loop_interval_ema = 0.0  # ms between loop ticks (if fed)
        self.audio_infer_time_ema = 0.0  # ms processing (placeholder for future inference)
        self._last_audio_ts = None  # type: ignore

        # Video counters
        self.video_frames = 0
        self.video_bytes = 0
        self.video_avg_frame_size = 0.0
        self._fps_window = fps_window
        self._frame_times: Deque[float] = deque(maxlen=fps_window)
        self.video_frame_interval_ema = 0.0
        self._last_video_ts = None  # type: ignore

        self.ema_alpha = ema_alpha

    # ---------------- Audio -----------------
    def record_audio_chunk(self, size_bytes: int, loop_interval_ms: float | None = None, infer_time_ms: float | None = None) -> None:
        with self._lock:
            self.audio_chunks += 1
            self.audio_bytes += size_bytes
            # Running average chunk size
            self.audio_avg_chunk_size = ((self.audio_avg_chunk_size * (self.audio_chunks - 1)) + size_bytes) / self.audio_chunks
            now = time.time() * 1000.0
            if loop_interval_ms is None and self._last_audio_ts is not None:
                loop_interval_ms = now - self._last_audio_ts
            if loop_interval_ms is not None:
                if self.audio_loop_interval_ema == 0.0:
                    self.audio_loop_interval_ema = loop_interval_ms
                else:
                    self.audio_loop_interval_ema = (self.ema_alpha * loop_interval_ms) + (1 - self.ema_alpha) * self.audio_loop_interval_ema
            self._last_audio_ts = now
            if infer_time_ms is not None:
                if self.audio_infer_time_ema == 0.0:
                    self.audio_infer_time_ema = infer_time_ms
                else:
                    self.audio_infer_time_ema = (self.ema_alpha * infer_time_ms) + (1 - self.ema_alpha) * self.audio_infer_time_ema

    # ---------------- Video -----------------
    def record_video_frame(self, size_bytes: int) -> None:
        with self._lock:
            self.video_frames += 1
            self.video_bytes += size_bytes
            self.video_avg_frame_size = ((self.video_avg_frame_size * (self.video_frames - 1)) + size_bytes) / self.video_frames
            now = time.time()
            self._frame_times.append(now)
            # Frame interval EMA (ms)
            if self._last_video_ts is not None:
                interval_ms = (now - self._last_video_ts) * 1000.0
                if self.video_frame_interval_ema == 0.0:
                    self.video_frame_interval_ema = interval_ms
                else:
                    self.video_frame_interval_ema = (self.ema_alpha * interval_ms) + (1 - self.ema_alpha) * self.video_frame_interval_ema
            self._last_video_ts = now

    # --------------- Report ------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            fps = 0.0
            if len(self._frame_times) > 1:
                span = self._frame_times[-1] - self._frame_times[0]
                if span > 0:
                    fps = (len(self._frame_times) - 1) / span
            return {
                "audio_chunks": self.audio_chunks,
                "audio_bytes": self.audio_bytes,
                "audio_avg_chunk_size": self.audio_avg_chunk_size,
                "audio_loop_interval_ema_ms": self.audio_loop_interval_ema,
                "audio_infer_time_ema_ms": self.audio_infer_time_ema,
                "video_frames": self.video_frames,
                "video_bytes": self.video_bytes,
                "video_avg_frame_size": self.video_avg_frame_size,
                "video_fps_rolling": fps,
                "video_frame_interval_ema_ms": self.video_frame_interval_ema,
                "fps_window": self._fps_window,
            }

# Global singleton for simple use
metrics = Metrics()
