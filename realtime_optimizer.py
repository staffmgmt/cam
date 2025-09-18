"""
Real-time Optimization Module
Implements latency reduction, frame buffering, and GPU optimization
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
import queue
import logging
from collections import deque
from typing import Dict, Any, Optional, Tuple
import psutil
import gc

logger = logging.getLogger(__name__)

class LatencyOptimizer:
    """Optimizes processing pipeline for minimal latency"""
    
    def __init__(self, target_latency_ms: float = 250.0):
        self.target_latency_ms = target_latency_ms
        self.latency_history = deque(maxlen=100)
        self.processing_times = {}
        
        # Adaptive parameters
        self.current_quality = 1.0  # 0.5 to 1.0
        self.current_resolution = (512, 512)
        self.current_fps = 20
        
        # Performance thresholds
        self.latency_threshold_high = target_latency_ms * 0.8  # 200ms
        self.latency_threshold_low = target_latency_ms * 0.6   # 150ms
        
        # Adaptation counters
        self.high_latency_count = 0
        self.low_latency_count = 0
        self.adaptation_threshold = 5  # consecutive frames
        
    def record_latency(self, stage: str, latency_ms: float):
        """Record latency for a processing stage"""
        self.processing_times[stage] = latency_ms
        
        # Calculate total latency
        total_latency = sum(self.processing_times.values())
        self.latency_history.append(total_latency)
        
        # Trigger adaptation if needed
        self._adapt_quality(total_latency)
    
    def _adapt_quality(self, total_latency: float):
        """Adapt quality based on latency"""
        if total_latency > self.latency_threshold_high:
            self.high_latency_count += 1
            self.low_latency_count = 0
            
            if self.high_latency_count >= self.adaptation_threshold:
                self._degrade_quality()
                self.high_latency_count = 0
                
        elif total_latency < self.latency_threshold_low:
            self.low_latency_count += 1
            self.high_latency_count = 0
            
            if self.low_latency_count >= self.adaptation_threshold * 2:  # Be more conservative with upgrades
                self._improve_quality()
                self.low_latency_count = 0
        else:
            self.high_latency_count = 0
            self.low_latency_count = 0
    
    def _degrade_quality(self):
        """Degrade quality to improve latency"""
        if self.current_quality > 0.7:
            self.current_quality -= 0.1
            logger.info(f"Reduced quality to {self.current_quality:.1f}")
        elif self.current_fps > 15:
            self.current_fps -= 2
            logger.info(f"Reduced FPS to {self.current_fps}")
        elif self.current_resolution[0] > 384:
            self.current_resolution = (384, 384)
            logger.info(f"Reduced resolution to {self.current_resolution}")
    
    def _improve_quality(self):
        """Improve quality when latency allows"""
        if self.current_resolution[0] < 512:
            self.current_resolution = (512, 512)
            logger.info(f"Increased resolution to {self.current_resolution}")
        elif self.current_fps < 20:
            self.current_fps += 2
            logger.info(f"Increased FPS to {self.current_fps}")
        elif self.current_quality < 1.0:
            self.current_quality += 0.1
            logger.info(f"Increased quality to {self.current_quality:.1f}")
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current adaptive settings"""
        return {
            "quality": self.current_quality,
            "resolution": self.current_resolution,
            "fps": self.current_fps,
            "avg_latency_ms": np.mean(self.latency_history) if self.latency_history else 0
        }

class FrameBuffer:
    """Thread-safe frame buffer with overflow protection"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.buffer = queue.Queue(maxsize=max_size)
        self.dropped_frames = 0
        self.total_frames = 0
        
    def put_frame(self, frame: np.ndarray, timestamp: float) -> bool:
        """Add frame to buffer, returns False if dropped"""
        self.total_frames += 1
        
        try:
            self.buffer.put_nowait((frame, timestamp))
            return True
        except queue.Full:
            # Drop oldest frame and add new one
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait((frame, timestamp))
                self.dropped_frames += 1
                return True
            except queue.Empty:
                return False
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get next frame from buffer"""
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics"""
        return {
            "size": self.buffer.qsize(),
            "max_size": self.max_size,
            "dropped_frames": self.dropped_frames,
            "total_frames": self.total_frames,
            "drop_rate": self.dropped_frames / max(self.total_frames, 1)
        }

class GPUMemoryManager:
    """Manages GPU memory for optimal performance"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_threshold = 0.9  # 90% of GPU memory
        self.cleanup_interval = 50  # frames
        self.frame_count = 0
        
    def optimize_memory(self):
        """Optimize GPU memory usage"""
        if not torch.cuda.is_available():
            return
        
        self.frame_count += 1
        
        # Periodic cleanup
        if self.frame_count % self.cleanup_interval == 0:
            self._cleanup_memory()
        
        # Emergency cleanup if memory usage is high
        if self._get_memory_usage() > self.memory_threshold:
            self._emergency_cleanup()
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage ratio"""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return allocated / total
    
    def _cleanup_memory(self):
        """Perform memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup"""
        logger.warning("High GPU memory usage, performing emergency cleanup")
        self._cleanup_memory()
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        return {
            "available": True,
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "total_gb": total / (1024**3),
            "usage_ratio": allocated / total
        }

class AudioSyncManager:
    """Manages audio-video synchronization"""
    
    def __init__(self, max_drift_ms: float = 150.0):
        self.max_drift_ms = max_drift_ms
        self.audio_timestamps = deque(maxlen=100)
        self.video_timestamps = deque(maxlen=100)
        self.sync_offset = 0.0
        
    def add_audio_timestamp(self, timestamp: float):
        """Add audio timestamp"""
        self.audio_timestamps.append(timestamp)
        self._calculate_sync_offset()
    
    def add_video_timestamp(self, timestamp: float):
        """Add video timestamp"""
        self.video_timestamps.append(timestamp)
        self._calculate_sync_offset()
    
    def _calculate_sync_offset(self):
        """Calculate current sync offset"""
        if len(self.audio_timestamps) == 0 or len(self.video_timestamps) == 0:
            return
        
        # Calculate average timestamp difference
        audio_avg = np.mean(list(self.audio_timestamps)[-10:])  # Last 10 samples
        video_avg = np.mean(list(self.video_timestamps)[-10:])
        
        self.sync_offset = audio_avg - video_avg
    
    def should_drop_video_frame(self, video_timestamp: float) -> bool:
        """Check if video frame should be dropped for sync"""
        if len(self.audio_timestamps) == 0:
            return False
        
        latest_audio = self.audio_timestamps[-1]
        drift = video_timestamp - latest_audio
        
        return abs(drift) > self.max_drift_ms
    
    def get_sync_stats(self) -> Dict[str, float]:
        """Get synchronization statistics"""
        return {
            "sync_offset_ms": self.sync_offset,
            "audio_samples": len(self.audio_timestamps),
            "video_samples": len(self.video_timestamps)
        }

class PerformanceProfiler:
    """Profiles system performance for optimization"""
    
    def __init__(self):
        self.cpu_usage = deque(maxlen=60)  # 1 minute at 1 Hz
        self.memory_usage = deque(maxlen=60)
        self.gpu_utilization = deque(maxlen=60)
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_system(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
                # GPU utilization (if available)
                if torch.cuda.is_available():
                    # Approximate GPU utilization based on memory usage
                    gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                    self.gpu_utilization.append(gpu_memory_used * 100)
                else:
                    self.gpu_utilization.append(0)
                    
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
            
            time.sleep(1)
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        return {
            "cpu_usage_avg": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "cpu_usage_max": np.max(self.cpu_usage) if self.cpu_usage else 0,
            "memory_usage_avg": np.mean(self.memory_usage) if self.memory_usage else 0,
            "memory_usage_max": np.max(self.memory_usage) if self.memory_usage else 0,
            "gpu_utilization_avg": np.mean(self.gpu_utilization) if self.gpu_utilization else 0,
            "gpu_utilization_max": np.max(self.gpu_utilization) if self.gpu_utilization else 0
        }

class RealTimeOptimizer:
    """Main real-time optimization controller"""
    
    def __init__(self, target_latency_ms: float = 250.0):
        self.latency_optimizer = LatencyOptimizer(target_latency_ms)
        self.frame_buffer = FrameBuffer()
        self.gpu_manager = GPUMemoryManager()
        self.audio_sync = AudioSyncManager()
        self.profiler = PerformanceProfiler()
        
        self.stats = {}
        self.last_stats_update = time.time()
    
    def process_frame(self, frame: np.ndarray, timestamp: float, stage: str = "video") -> bool:
        """Process a frame with optimization"""
        start_time = time.time()
        
        # Check if frame should be dropped for sync
        if stage == "video" and self.audio_sync.should_drop_video_frame(timestamp):
            return False
        
        # Add to buffer
        success = self.frame_buffer.put_frame(frame, timestamp)
        
        # Record processing time
        processing_time = (time.time() - start_time) * 1000
        self.latency_optimizer.record_latency(stage, processing_time)
        
        # Update timestamps for sync
        if stage == "video":
            self.audio_sync.add_video_timestamp(timestamp)
        elif stage == "audio":
            self.audio_sync.add_audio_timestamp(timestamp)
        
        # Optimize GPU memory
        self.gpu_manager.optimize_memory()
        
        return success
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get next frame from buffer"""
        return self.frame_buffer.get_frame()
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get current optimization settings"""
        return self.latency_optimizer.get_current_settings()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        now = time.time()
        
        # Update stats every 2 seconds
        if now - self.last_stats_update > 2.0:
            self.stats = {
                "latency": self.latency_optimizer.get_current_settings(),
                "buffer": self.frame_buffer.get_stats(),
                "gpu": self.gpu_manager.get_memory_stats(),
                "sync": self.audio_sync.get_sync_stats(),
                "system": self.profiler.get_system_stats()
            }
            self.last_stats_update = now
        
        return self.stats
    
    def cleanup(self):
        """Cleanup optimizer resources"""
        self.profiler.stop_monitoring()

# Global optimizer instance
_optimizer_instance = None

def get_realtime_optimizer() -> RealTimeOptimizer:
    """Get or create global optimizer instance"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = RealTimeOptimizer()
    return _optimizer_instance