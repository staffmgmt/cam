"""
Enhanced Performance Metrics for Existing Pipeline
Adds p50/p95/p99 latency tracking and GPU monitoring
Drop-in compatible with existing get_performance_stats()
"""

import time
import psutil
import numpy as np
from collections import deque
from typing import Dict, Any, List


class EnhancedMetrics:
    """Enhanced metrics collection with percentiles"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Timing collections
        self.video_times = deque(maxlen=window_size)
        self.audio_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)

        # Component timing (for debugging)
        self.component_times = {
            'face_detection': deque(maxlen=window_size),
            'animation': deque(maxlen=window_size),
            'voice_processing': deque(maxlen=window_size),
            'webrtc_encode': deque(maxlen=window_size)
        }

        # FPS tracking
        self.frame_timestamps = deque(maxlen=window_size)

        # System monitoring
        self.last_gpu_check = 0
        self.gpu_memory_mb = 0

    def record_video_timing(self, elapsed_ms: float):
        self.video_times.append(elapsed_ms)
        self.frame_timestamps.append(time.time())

    def record_audio_timing(self, elapsed_ms: float):
        self.audio_times.append(elapsed_ms)

    def record_component_timing(self, component: str, elapsed_ms: float):
        if component in self.component_times:
            self.component_times[component].append(elapsed_ms)

    def record_total_timing(self, elapsed_ms: float):
        self.total_times.append(elapsed_ms)

    def get_percentiles(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
        arr = np.array(values)
        return {
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99))
        }

    def get_fps(self) -> float:
        if len(self.frame_timestamps) < 2:
            return 0.0
        timestamps = list(self.frame_timestamps)
        time_span = timestamps[-1] - timestamps[0]
        if time_span <= 0:
            return 0.0
        return (len(timestamps) - 1) / time_span

    def get_gpu_memory(self) -> float:
        current_time = time.time()
        if current_time - self.last_gpu_check > 2.0:
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                else:
                    self.gpu_memory_mb = 0
            except ImportError:
                self.gpu_memory_mb = 0
            self.last_gpu_check = current_time
        return self.gpu_memory_mb

    def get_enhanced_stats(self) -> Dict[str, Any]:
        video_list = list(self.video_times)
        audio_list = list(self.audio_times)
        total_list = list(self.total_times)
        stats = {
            "avg_video_latency_ms": float(np.mean(video_list)) if video_list else 0.0,
            "avg_audio_latency_ms": float(np.mean(audio_list)) if audio_list else 0.0,
            "video_fps": self.get_fps(),
            "gpu_memory_used_mb": self.get_gpu_memory(),
            "video_latency": {
                "mean": float(np.mean(video_list)) if video_list else 0.0,
                "std": float(np.std(video_list)) if video_list else 0.0,
                **self.get_percentiles(video_list)
            },
            "audio_latency": {
                "mean": float(np.mean(audio_list)) if audio_list else 0.0,
                "std": float(np.std(audio_list)) if audio_list else 0.0,
                **self.get_percentiles(audio_list)
            },
            "total_latency": {
                "mean": float(np.mean(total_list)) if total_list else 0.0,
                "std": float(np.std(total_list)) if total_list else 0.0,
                **self.get_percentiles(total_list)
            },
            "components": {}
        }
        for component, times in self.component_times.items():
            times_list = list(times)
            if times_list:
                stats["components"][component] = {
                    "mean": float(np.mean(times_list)),
                    **self.get_percentiles(times_list)
                }
        stats["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "active_connections": 1
        }
        return stats


_enhanced_metrics = EnhancedMetrics()


def get_enhanced_metrics() -> EnhancedMetrics:
    return _enhanced_metrics


def enhance_existing_stats(existing_stats: Dict[str, Any]) -> Dict[str, Any]:
    enhanced = get_enhanced_metrics().get_enhanced_stats()
    result = existing_stats.copy()
    result.update(enhanced)
    return result
