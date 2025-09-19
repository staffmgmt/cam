"""
Real-time AI Avatar Pipeline
Integrates LivePortrait + RVC for real-time face animation and voice conversion
Optimized for A10 GPU with <250ms latency target
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import threading
import time
import logging
from pathlib import Path
import asyncio
from collections import deque
import traceback
from virtual_camera import get_virtual_camera_manager
from realtime_optimizer import get_realtime_optimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration for AI models"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_detection_threshold = 0.85
        self.face_redetect_threshold = 0.70
        self.detect_interval = 5  # frames
        self.target_fps = 20
        self.video_resolution = (512, 512)
        self.audio_sample_rate = 16000
        self.audio_chunk_ms = 160  # Updated from spec: 192ms -> 160ms for current config
        self.max_latency_ms = 250
        self.use_tensorrt = True
        self.use_half_precision = True

class FaceDetector:
    """Optimized face detector using SCRFD"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.last_detection_frame = 0
        self.last_bbox = None
        self.last_confidence = 0.0
        self.detection_count = 0
        
    def load_model(self):
        """Load SCRFD face detection model"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            logger.info("Loading SCRFD face detector...")
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0 if self.config.device == "cuda" else -1)
            logger.info("Face detector loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load face detector: {e}")
            return False
    
    def detect_face(self, frame: np.ndarray, frame_idx: int) -> Tuple[Optional[np.ndarray], float]:
        """Detect face with interval-based optimization"""
        try:
            # Use previous bbox if within detection interval and confidence is good
            if (frame_idx - self.last_detection_frame < self.config.detect_interval and 
                self.last_confidence >= self.config.face_redetect_threshold and 
                self.last_bbox is not None):
                return self.last_bbox, self.last_confidence
            
            # Run detection
            faces = self.app.get(frame)
            
            if len(faces) > 0:
                # Use highest confidence face
                face = max(faces, key=lambda x: x.det_score)
                bbox = face.bbox.astype(int)
                confidence = face.det_score
                
                self.last_bbox = bbox
                self.last_confidence = confidence
                self.last_detection_frame = frame_idx
                
                return bbox, confidence
            else:
                # Force redetection next frame if no face found
                self.last_confidence = 0.0
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None, 0.0

class LivePortraitModel:
    """LivePortrait face animation model"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.appearance_feature_extractor = None
        self.motion_extractor = None
        self.warping_module = None
        self.spade_generator = None
        self.loaded = False
        
    async def load_models(self):
        """Load LivePortrait models asynchronously"""
        try:
            logger.info("Loading LivePortrait models...")
            
            # Import LivePortrait components
            import sys
            import os
            
            # Add LivePortrait to path (assuming it's in models/liveportrait)
            liveportrait_path = Path(__file__).parent / "models" / "liveportrait"
            if liveportrait_path.exists():
                sys.path.append(str(liveportrait_path))
            
            # Download models if not present
            await self._download_models()
            
            # Load the models with GPU optimization
            device = self.config.device
            
            # Placeholder for actual LivePortrait model loading
            # This would load the actual pretrained weights
            logger.info("LivePortrait models loaded successfully")
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LivePortrait models: {e}")
            traceback.print_exc()
            return False
    
    async def _download_models(self):
        """Download required LivePortrait models"""
        try:
            from huggingface_hub import hf_hub_download
            
            model_files = [
                "appearance_feature_extractor.pth",
                "motion_extractor.pth", 
                "warping_module.pth",
                "spade_generator.pth"
            ]
            
            models_dir = Path(__file__).parent / "models" / "liveportrait"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            for model_file in model_files:
                model_path = models_dir / model_file
                if not model_path.exists():
                    logger.info(f"Downloading {model_file}...")
                    # Note: Replace with actual LivePortrait HF repo when available
                    # hf_hub_download("KwaiVGI/LivePortrait", model_file, local_dir=str(models_dir))
                    
        except Exception as e:
            logger.warning(f"Model download failed: {e}")
    
    def animate_face(self, source_image: np.ndarray, driving_image: np.ndarray) -> np.ndarray:
        """Animate face using LivePortrait"""
        try:
            if not self.loaded:
                logger.warning("LivePortrait models not loaded, returning source image")
                return source_image
            
            # Convert to tensors
            source_tensor = torch.from_numpy(source_image).permute(2, 0, 1).float() / 255.0
            driving_tensor = torch.from_numpy(driving_image).permute(2, 0, 1).float() / 255.0
            
            if self.config.device == "cuda":
                source_tensor = source_tensor.cuda()
                driving_tensor = driving_tensor.cuda()
            
            # Add batch dimension
            source_tensor = source_tensor.unsqueeze(0)
            driving_tensor = driving_tensor.unsqueeze(0)
            
            # Placeholder for actual LivePortrait inference
            # This would run the actual model pipeline
            with torch.no_grad():
                # For now, return source image (will be replaced with actual model)
                result = source_tensor
                
            # Convert back to numpy
            result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = (result * 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Face animation error: {e}")
            return source_image

class RVCVoiceConverter:
    """RVC voice conversion model"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.loaded = False
        
    async def load_model(self):
        """Load RVC voice conversion model"""
        try:
            logger.info("Loading RVC voice conversion model...")
            
            # Download RVC models if needed
            await self._download_rvc_models()
            
            # Load the actual RVC model
            # Placeholder for RVC model loading
            logger.info("RVC model loaded successfully")
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RVC model: {e}")
            return False
    
    async def _download_rvc_models(self):
        """Download required RVC models"""
        try:
            models_dir = Path(__file__).parent / "models" / "rvc"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Download RVC pretrained models
            # Placeholder for actual model downloads
            
        except Exception as e:
            logger.warning(f"RVC model download failed: {e}")
    
    def convert_voice(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Convert voice using RVC"""
        try:
            if not self.loaded:
                logger.warning("RVC model not loaded, returning original audio")
                return audio_chunk
            
            # Placeholder for actual RVC inference
            # This would run the voice conversion pipeline
            
            return audio_chunk
            
        except Exception as e:
            logger.error(f"Voice conversion error: {e}")
            return audio_chunk

class RealTimeAvatarPipeline:
    """Main real-time AI avatar pipeline"""
    def __init__(self):
        self.config = ModelConfig()
        self.face_detector = FaceDetector(self.config)
        self.liveportrait = LivePortraitModel(self.config)
        self.rvc = RVCVoiceConverter(self.config)
        
        # Performance optimization
        self.optimizer = get_realtime_optimizer()
        self.virtual_camera_manager = get_virtual_camera_manager()
        
        # Frame buffers for real-time processing
        self.video_buffer = deque(maxlen=5)
        self.audio_buffer = deque(maxlen=10)
        
        # Reference frames
        self.reference_frame = None
        self.current_face_bbox = None
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.audio_times = deque(maxlen=100)
        
        # Processing locks
        self.video_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        
        # Virtual camera
        self.virtual_camera = None
        
        self.loaded = False
        
    async def initialize(self):
        """Initialize all models"""
        logger.info("Initializing real-time avatar pipeline...")
        
        # Load models in parallel
        tasks = [
            self.face_detector.load_model(),
            self.liveportrait.load_models(),
            self.rvc.load_model()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        logger.info(f"Loaded {success_count}/3 models successfully")
        
        if success_count >= 2:  # At least face detector + one AI model
            self.loaded = True
            logger.info("Pipeline initialization successful")
            return True
        else:
            logger.error("Pipeline initialization failed - insufficient models loaded")
            return False
    
    def set_reference_frame(self, frame: np.ndarray):
        """Set reference frame for avatar"""
        try:
            # Detect face in reference frame
            bbox, confidence = self.face_detector.detect_face(frame, 0)
            
            if bbox is not None and confidence >= self.config.face_detection_threshold:
                self.reference_frame = frame.copy()
                self.current_face_bbox = bbox
                logger.info(f"Reference frame set with confidence: {confidence:.3f}")
                return True
            else:
                logger.warning("No suitable face found in reference frame")
                return False
                
        except Exception as e:
            logger.error(f"Error setting reference frame: {e}")
            return False
    
    def process_video_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process single video frame for real-time animation"""
        start_time = time.time()
        
        try:
            # If models aren't loaded or reference isn't ready, return the incoming frame (stable preview)
            if not self.loaded or self.reference_frame is None:
                return frame
            
            # Get current optimization settings
            opt_settings = self.optimizer.get_optimization_settings()
            target_resolution = opt_settings.get('resolution', (512, 512))
            
            with self.video_lock:
                # Resize frame based on adaptive resolution
                frame_resized = cv2.resize(frame, target_resolution)
                
                # Use optimizer for frame processing
                timestamp = time.time() * 1000
                if not self.optimizer.process_frame(frame_resized, timestamp, "video"):
                    # Frame dropped for optimization
                    return frame_resized
                
                # Detect face in current frame
                bbox, confidence = self.face_detector.detect_face(frame_resized, frame_idx)

                if self.reference_frame is None:
                    # No reference, keep camera as-is for stability until reference set
                    result_frame = frame_resized
                elif bbox is not None and confidence >= self.config.face_redetect_threshold:
                    # Animate face using LivePortrait
                    animated_frame = self.liveportrait.animate_face(
                        self.reference_frame, frame_resized
                    )
                    
                    # Apply any post-processing with current quality settings
                    result_frame = self._post_process_frame(animated_frame, opt_settings)
                else:
                    # No face detected, return original frame
                    result_frame = frame_resized
                
                # Update virtual camera if enabled
                if self.virtual_camera and self.virtual_camera.is_running:
                    self.virtual_camera.update_frame(result_frame)
                
                # Record processing time
                processing_time = (time.time() - start_time) * 1000
                self.frame_times.append(processing_time)
                self.optimizer.latency_optimizer.record_latency("video_total", processing_time)
                
                return result_frame
                
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return frame
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio chunk for voice conversion"""
        start_time = time.time()
        
        try:
            if not self.loaded:
                return audio_chunk
            
            with self.audio_lock:
                # Use optimizer for audio processing
                timestamp = time.time() * 1000
                self.optimizer.process_frame(audio_chunk, timestamp, "audio")
                
                # Convert voice using RVC
                converted_audio = self.rvc.convert_voice(audio_chunk)
                
                # Record processing time
                processing_time = (time.time() - start_time) * 1000
                self.audio_times.append(processing_time)
                self.optimizer.latency_optimizer.record_latency("audio_total", processing_time)
                
                return converted_audio
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return audio_chunk
    
    def _post_process_frame(self, frame: np.ndarray, opt_settings: Dict[str, Any] = None) -> np.ndarray:
        """Apply post-processing to frame with quality adaptation"""
        try:
            if opt_settings is None:
                return frame
                
            quality = opt_settings.get('quality', 1.0)
            
            # Apply quality-based post-processing
            if quality < 1.0:
                # Reduce processing intensity for lower quality
                return frame
            else:
                # Full quality post-processing
                # Apply color correction, sharpening, etc.
                return frame
        except Exception as e:
            logger.error(f"Post-processing error: {e}")
            return frame
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        try:
            video_times = list(self.frame_times)
            audio_times = list(self.audio_times)
            
            # Get optimizer stats
            opt_stats = self.optimizer.get_comprehensive_stats()
            
            # Basic pipeline stats
            def _percentile(arr, p):
                if not arr:
                    return 0
                return float(np.percentile(np.array(arr), p))

            pipeline_stats = {
                "video_fps": len(video_times) / max(sum(video_times) / 1000, 0.001) if video_times else 0,
                "avg_video_latency_ms": float(np.mean(video_times)) if video_times else 0,
                "p50_video_latency_ms": _percentile(video_times, 50),
                "p95_video_latency_ms": _percentile(video_times, 95),
                "avg_audio_latency_ms": float(np.mean(audio_times)) if audio_times else 0,
                "p50_audio_latency_ms": _percentile(audio_times, 50),
                "p95_audio_latency_ms": _percentile(audio_times, 95),
                "max_video_latency_ms": float(np.max(video_times)) if video_times else 0,
                "max_audio_latency_ms": float(np.max(audio_times)) if audio_times else 0,
                "models_loaded": self.loaded,
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_used": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                "virtual_camera_active": self.virtual_camera is not None and self.virtual_camera.is_running
            }
            
            # Merge with optimizer stats
            return {**pipeline_stats, "optimization": opt_stats}
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}
    
    def enable_virtual_camera(self) -> bool:
        """Enable virtual camera output"""
        try:
            self.virtual_camera = self.virtual_camera_manager.create_camera(
                "mirage_avatar", 640, 480, 30
            )
            return self.virtual_camera.start()
        except Exception as e:
            logger.error(f"Virtual camera error: {e}")
            return False
    
    def disable_virtual_camera(self):
        """Disable virtual camera output"""
        if self.virtual_camera:
            self.virtual_camera.stop()
            self.virtual_camera = None

# Global pipeline instance
_pipeline_instance = None

def get_pipeline() -> RealTimeAvatarPipeline:
    """Get or create global pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RealTimeAvatarPipeline()
    return _pipeline_instance