# Generate targeted enhancements that align with their existing code

# 2. Enhanced metrics that can be dropped into their existing pipeline
enhanced_metrics = '''"""
Enhanced Performance Metrics for Existing Pipeline
Adds p50/p95 latency tracking and GPU monitoring
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
        """Record video processing timing"""
        self.video_times.append(elapsed_ms)
        self.frame_timestamps.append(time.time())
        
    def record_audio_timing(self, elapsed_ms: float):
        """Record audio processing timing"""  
        self.audio_times.append(elapsed_ms)
        
    def record_component_timing(self, component: str, elapsed_ms: float):
        """Record individual component timing"""
        if component in self.component_times:
            self.component_times[component].append(elapsed_ms)
    
    def record_total_timing(self, elapsed_ms: float):
        """Record end-to-end timing"""
        self.total_times.append(elapsed_ms)
    
    def get_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentiles for a list of values"""
        if not values:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
            
        arr = np.array(values)
        return {
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)), 
            'p99': float(np.percentile(arr, 99))
        }
    
    def get_fps(self) -> float:
        """Calculate current FPS from frame timestamps"""
        if len(self.frame_timestamps) < 2:
            return 0.0
            
        timestamps = list(self.frame_timestamps)
        time_span = timestamps[-1] - timestamps[0]
        
        if time_span <= 0:
            return 0.0
            
        return (len(timestamps) - 1) / time_span
    
    def get_gpu_memory(self) -> float:
        """Get GPU memory usage (cached for performance)"""
        current_time = time.time()
        
        # Update GPU memory every 2 seconds
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
        """Get comprehensive performance statistics"""
        
        # Convert deques to lists for percentile calculation
        video_list = list(self.video_times)
        audio_list = list(self.audio_times)  
        total_list = list(self.total_times)
        
        stats = {
            # Basic metrics (compatible with existing)
            "avg_video_latency_ms": float(np.mean(video_list)) if video_list else 0.0,
            "avg_audio_latency_ms": float(np.mean(audio_list)) if audio_list else 0.0,
            "video_fps": self.get_fps(),
            "gpu_memory_used_mb": self.get_gpu_memory(),
            
            # Enhanced percentile metrics
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
            
            # Component breakdown
            "components": {}
        }
        
        # Add component timings
        for component, times in self.component_times.items():
            times_list = list(times)
            if times_list:
                stats["components"][component] = {
                    "mean": float(np.mean(times_list)),
                    **self.get_percentiles(times_list)
                }
        
        # System metrics
        stats["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "active_connections": 1  # Placeholder - you can update this
        }
        
        return stats

# Global metrics instance (compatible with existing singleton pattern)
_enhanced_metrics = EnhancedMetrics()

def get_enhanced_metrics() -> EnhancedMetrics:
    """Get the enhanced metrics instance"""
    return _enhanced_metrics

# Compatibility wrapper for existing code
def enhance_existing_stats(existing_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance existing stats dict with percentiles"""
    enhanced = get_enhanced_metrics().get_enhanced_stats()
    
    # Merge enhanced metrics into existing structure
    result = existing_stats.copy()
    result.update(enhanced)
    
    return result
'''

with open('enhanced_metrics.py', 'w') as f:
    f.write(enhanced_metrics)

# 3. Optional model download utility (not baked into Docker)
optional_downloader = '''#!/usr/bin/env python3
"""
Optional Model Downloader - On-demand only
Safe utility for pre-downloading models when needed
Does NOT run automatically in Docker build
"""

import os
import sys
import logging
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionalModelDownloader:
    """Optional model downloader for on-demand use"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        
        # Conservative model list - only what we actually need
        self.available_models = {
            "scrfd": {
                "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.zip",
                "dir": self.models_dir / "scrfd",
                "description": "SCRFD face detection model",
                "size_mb": 17,
                "required_by": "MIRAGE_ENABLE_SCRFD"
            },
            "liveportrait_appearance": {
                "url": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/appearance_feature_extractor.onnx",
                "dir": self.models_dir / "liveportrait",
                "description": "LivePortrait appearance extractor",
                "size_mb": 85,
                "required_by": "MIRAGE_ENABLE_LIVEPORTRAIT"
            },
            "liveportrait_motion": {
                "url": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/motion_extractor.onnx", 
                "dir": self.models_dir / "liveportrait",
                "description": "LivePortrait motion extractor",
                "size_mb": 28,
                "required_by": "MIRAGE_ENABLE_LIVEPORTRAIT"
            }
        }
    
    def check_model_status(self) -> Dict[str, Dict[str, any]]:
        """Check which models are available and which features are enabled"""
        status = {}
        
        for model_key, config in self.available_models.items():
            # Check if feature is enabled
            feature_flag = config["required_by"]
            is_enabled = os.getenv(feature_flag, "0").lower() in ("1", "true", "yes")
            
            # Check if model files exist
            model_dir = config["dir"]
            if model_key == "scrfd":
                model_exists = (model_dir / "scrfd_10g_bnkps.onnx").exists()
            else:
                filename = config["url"].split("/")[-1]
                model_exists = (model_dir / filename).exists()
            
            status[model_key] = {
                "feature_enabled": is_enabled,
                "model_downloaded": model_exists,
                "needed": is_enabled and not model_exists,
                "description": config["description"],
                "size_mb": config["size_mb"]
            }
        
        return status
    
    def download_model(self, model_key: str, force: bool = False) -> bool:
        """Download a specific model"""
        if model_key not in self.available_models:
            logger.error(f"Unknown model: {model_key}")
            return False
            
        config = self.available_models[model_key]
        
        # Check if already exists (unless force)
        if not force:
            status = self.check_model_status()
            if status[model_key]["model_downloaded"]:
                logger.info(f"Model {model_key} already downloaded")
                return True
        
        try:
            # Create directory
            config["dir"].mkdir(parents=True, exist_ok=True)
            
            # Download
            logger.info(f"Downloading {model_key} ({config['size_mb']}MB)...")
            
            response = requests.get(config["url"], stream=True, timeout=60)
            response.raise_for_status()
            
            # Determine filename and path
            if model_key == "scrfd":
                # Special handling for ZIP file
                zip_path = config["dir"] / "scrfd.zip"
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Extract ZIP
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(config["dir"])
                zip_path.unlink()  # Remove ZIP after extraction
                
            else:
                # Direct file download
                filename = config["url"].split("/")[-1]
                filepath = config["dir"] / filename
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            logger.info(f"✅ Downloaded {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_key}: {e}")
            return False
    
    def download_needed_models(self) -> bool:
        """Download only models needed by enabled features"""
        status = self.check_model_status()
        
        needed_models = [k for k, v in status.items() if v["needed"]]
        
        if not needed_models:
            logger.info("No model downloads needed")
            return True
        
        logger.info(f"Downloading needed models: {needed_models}")
        
        success_count = 0
        for model_key in needed_models:
            if self.download_model(model_key):
                success_count += 1
        
        logger.info(f"Downloaded {success_count}/{len(needed_models)} models")
        return success_count == len(needed_models)
    
    def print_status(self):
        """Print current model status"""
        status = self.check_model_status()
        
        print("\\n=== Model Status ===")
        for model_key, info in status.items():
            enabled_icon = "🟢" if info["feature_enabled"] else "⚪"
            downloaded_icon = "✅" if info["model_downloaded"] else "❌"
            needed_icon = "📥" if info["needed"] else "  "
            
            print(f"{enabled_icon} {downloaded_icon} {needed_icon} {model_key:<25} - {info['description']}")
        
        print("\\n🟢 = Feature enabled")  
        print("✅ = Model downloaded")
        print("📥 = Download needed")

def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optional model downloader")
    parser.add_argument("--status", action="store_true", help="Show model status")
    parser.add_argument("--download", help="Download specific model")
    parser.add_argument("--download-needed", action="store_true", help="Download all needed models")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    downloader = OptionalModelDownloader(args.models_dir)
    
    if args.status:
        downloader.print_status()
        return
    
    if args.download:
        success = downloader.download_model(args.download, force=args.force)
        sys.exit(0 if success else 1)
    
    if args.download_needed:
        success = downloader.download_needed_models()
        sys.exit(0 if success else 1)
    
    # Default: show status
    downloader.print_status()

if __name__ == "__main__":
    main()
'''

with open('scripts/optional_download_models.py', 'w') as f:
    f.write(optional_downloader)

print("✅ Generated enhanced_metrics.py")
print("   - Drop-in p50/p95/p99 percentile tracking")
print("   - Compatible with existing get_performance_stats()")
print("   - Component-level timing breakdown")
print("   - GPU memory monitoring with caching")

print("✅ Generated scripts/optional_download_models.py") 
print("   - On-demand model downloading only")
print("   - NOT baked into Docker build")
print("   - Respects feature flags (MIRAGE_ENABLE_*)")
print("   - Conservative model list (SCRFD + LivePortrait basics)")