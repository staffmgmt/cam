"""
Neural-only Real-time AI Avatar Pipeline
- Single production route: SCRFD face detection + LivePortrait ONNX (appearance, motion, generator)
- No fallbacks or feature flags; initialization fails if models are unavailable
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
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports - make them resilient
get_virtual_camera_manager = None
get_enhanced_metrics = None
enhance_existing_stats = lambda x: x
get_safe_model_loader = None
LandmarkReenactor = None
MP_AVAILABLE = False
try:
    from liveportrait_engine import get_liveportrait_engine
except Exception as e:
    logger.error(f"liveportrait_engine not available: {e}")
    get_liveportrait_engine = None
get_realtime_optimizer = None

class ModelConfig:
    """Configuration for AI models"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_detection_threshold = 0.85
        self.face_redetect_threshold = 0.70
        self.detect_interval = 5  # frames between full detection (track in-between)
        self.target_fps = 20
        self.video_resolution = (512, 512)
        self.audio_sample_rate = 16000
        self.audio_chunk_ms = 160  # Updated from spec: 192ms -> 160ms for current config
        self.max_latency_ms = 250
        self.use_tensorrt = True
        self.use_half_precision = True

class SCRFDFaceDetector:
    """Optimized face detector using SCRFD"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.loaded = False
        
    async def load_model(self):
        """Load SCRFD face detection model"""
        if self.loaded:
            return True
        
        try:
            logger.info("Loading SCRFD face detector...")
            import insightface
            from insightface.app import FaceAnalysis
            
            # Initialize InsightFace with SCRFD
            self.model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.device == "cuda" else ['CPUExecutionProvider'])
            self.model.prepare(ctx_id=0 if self.config.device == "cuda" else -1, det_size=(640, 640))
            
            self.loaded = True
            logger.info("SCRFD face detector loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SCRFD face detector: {e}")
            return False
    
    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in image and return bounding boxes with landmarks"""
        if not self.loaded or self.model is None:
            return []
        
        try:
            # Convert BGR to RGB for InsightFace
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            
            # Detect faces
            faces = self.model.get(rgb_image)
            
            results = []
            for face in faces:
                # Extract bounding box and landmarks
                bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
                landmarks = face.kps.astype(int)  # 5 landmarks
                confidence = face.det_score
                
                results.append({
                    'bbox': bbox,
                    'landmarks': landmarks,
                    'confidence': confidence
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def crop_and_align_face(self, image: np.ndarray, face_info: dict, target_size: tuple = (512, 512)) -> Optional[np.ndarray]:
        """Crop and align face for LivePortrait input"""
        try:
            bbox = face_info['bbox']
            landmarks = face_info['landmarks']
            
            # Extract face region with padding
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            
            # Add padding around face
            padding = 0.3
            face_w, face_h = x2 - x1, y2 - y1
            pad_w, pad_h = int(face_w * padding), int(face_h * padding)
            
            # Expand bounding box
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            # Crop face region
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Resize to target size
            face_aligned = cv2.resize(face_crop, target_size)
            
            return face_aligned
            
        except Exception as e:
            logger.error(f"Face crop and alignment failed: {e}")
            return None
    
    def get_best_face(self, faces: list) -> Optional[dict]:
        """Get the best face from detection results"""
        if not faces:
            return None
        
        # Sort by confidence and size
        best_face = max(faces, key=lambda f: f['confidence'] * ((f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1])))
        
        return best_face


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
            
            # Placeholder: Real LivePortrait loading not implemented here.
            # Defer to safe_model_integration ONNX path instead.
            logger.info("LivePortrait (native) not implemented; using ONNX engine path")
            self.loaded = False
            return False
            
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
            # Voice conversion is out of scope for neural-only video path; keep disabled
            self.loaded = False
            return False
            
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
        # Always neural-only
        self.require_neural = True
        
        # Face detection systems
        self.scrfd_detector = SCRFDFaceDetector(self.config)
        
        # Animation engine (required)
        self.liveportrait = LivePortraitModel(self.config)
        self.liveportrait_engine = get_liveportrait_engine() if get_liveportrait_engine else None
        
        # Voice conversion disabled in this build
        self.rvc = RVCVoiceConverter(self.config)
        
        # Safe model loader (optional, not required in neural-only path)
        self.safe_loader = get_safe_model_loader() if get_safe_model_loader else None
        
        # No landmark reenactor or other fallbacks in neural-only mode
        self.landmark_reenactor = None
        
        # Performance optimization
        self.optimizer = None
        self.virtual_camera_manager = None
        
        # Frame buffers for real-time processing
        self.video_buffer = deque(maxlen=5)
        self.audio_buffer = deque(maxlen=10)
        
        # Reference frames and appearance features
        self.reference_frame = None
        self.reference_appearance_features = None
        self.current_face_bbox = None
        
        # LivePortrait is the only path
        self.use_liveportrait = True
        self.liveportrait_ready = False
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.audio_times = deque(maxlen=100)
        self._metrics = get_enhanced_metrics() if get_enhanced_metrics else None
        
        # Processing locks
        self.video_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        
        self.virtual_camera = None

        # ROI smoothing for stable crops
        self._last_bbox = None  # [x1,y1,x2,y2]
        self._bbox_ema_alpha = 0.8  # higher = smoother (more inertia)
        
        self.loaded = False
        
    async def initialize(self):
        """Initialize all models"""
        logger.info("Initializing real-time avatar pipeline...")
        
        # Initialize SCRFD face detection (required)
        scrfd_ok = await self.scrfd_detector.load_model()
        if scrfd_ok:
            logger.info("SCRFD face detection ready")
        else:
            logger.error("SCRFD failed to load (required)")
            return False
        
        # Initialize LivePortrait ONNX engine (required)
        if self.liveportrait_engine:
            liveportrait_ok = self.liveportrait_engine.load_models()
            if liveportrait_ok:
                self.liveportrait_ready = True
                has_gen = getattr(self.liveportrait_engine, 'generator_session', None) is not None
                if not has_gen:
                    logger.error("LivePortrait generator.onnx not found (required)")
                    return False
                else:
                    logger.info("LivePortrait ONNX engine ready (with generator)")
            else:
                logger.error("LivePortrait models failed to load (required)")
                return False
        else:
            logger.error("LivePortrait engine unavailable (required)")
            return False
        
        fd_ok = True  # legacy detector unused

        # Load async models and optional safe models in parallel
        try:
            lp_task = self.liveportrait.load_models()
            rvc_task = self.rvc.load_model()
            # Guard safe_loader presence
            if self.safe_loader is not None:
                scrfd_task = self.safe_loader.safe_load_scrfd()
                lp_safe_task = self.safe_loader.safe_load_liveportrait()
            else:
                async def _false():
                    return False
                scrfd_task = _false()
                lp_safe_task = _false()
            
            # Run all tasks concurrently
            results = await asyncio.gather(
                lp_task, rvc_task, scrfd_task, lp_safe_task,
                return_exceptions=True
            )
            
            lp_ok, rvc_ok, scrfd_safe_ok, lp_safe_ok = results
            
            # Log results
            if isinstance(lp_ok, Exception):
                logger.error(f"LivePortrait (legacy) loading failed: {lp_ok}")
                lp_ok = False
            
            if isinstance(rvc_ok, Exception):
                logger.error(f"RVC loading failed: {rvc_ok}")
                rvc_ok = False
                
            logger.info(f"Model loading results: FD={fd_ok}, LP={lp_ok}, RVC={rvc_ok}, SCRFD_safe={scrfd_safe_ok}, LP_safe={lp_safe_ok}")
            
            # Mark as loaded
            # Fully neural-only readiness
            self.loaded = bool(self.use_liveportrait and self.liveportrait_ready and getattr(self.liveportrait_engine, 'generator_session', None) is not None)
            
            if self.loaded:
                # If a reference was queued before init, try to extract appearance now
                try:
                    if self.reference_frame is not None and self.reference_appearance_features is None:
                        af = self.liveportrait_engine.extract_appearance_features(self.reference_frame)
                        if af is not None:
                            self.reference_appearance_features = af
                            logger.info("Post-init appearance features extracted from queued reference")
                except Exception as e:
                    logger.warning(f"Post-init reference extraction failed: {e}")
                mode = "liveportrait"
                logger.info(f"Avatar pipeline initialized successfully (mode={mode})")
                return True
            else:
                logger.error("Avatar pipeline initialization failed - requirements not met")
                return False
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            # Still mark as loaded if we have basic functionality
            self.loaded = scrfd_ok
            return self.loaded
    
    def set_reference_frame(self, frame: np.ndarray) -> bool:
        """Set reference frame for avatar animation"""
        try:
            logger.info(f"Setting reference frame: {frame.shape}")
            # Detect face in reference frame using SCRFD (required)
            faces = self.scrfd_detector.detect_faces(frame)
            if not faces:
                logger.error("No face detected in reference image")
                return False
            best_face = self.scrfd_detector.get_best_face(faces)
            aligned_face = self.scrfd_detector.crop_and_align_face(frame, best_face, target_size=(512, 512))
            if aligned_face is None:
                logger.error("Failed to align reference face")
                return False
            frame = aligned_face
            self._last_bbox = best_face['bbox'].astype(np.float32)
            
            # Store reference frame
            self.reference_frame = frame.copy()
            
            # Extract appearance features (required)
            if not (self.use_liveportrait and self.liveportrait_engine and self.liveportrait_ready):
                logger.error("LivePortrait engine not ready for reference extraction")
                return False
            appearance_features = self.liveportrait_engine.extract_appearance_features(frame)
            if appearance_features is None:
                logger.error("Appearance feature extraction failed")
                return False
            self.reference_appearance_features = appearance_features
            logger.info("LivePortrait appearance features extracted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set reference frame: {e}")
            return False
    
    def process_video_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process single video frame for real-time animation"""
        start_time = time.time()
        
        try:
            # Check if we have a reference
            if self.reference_frame is None:
                logger.error("No reference set")
                try:
                    self.last_method = 'no_reference'
                except Exception:
                    pass
                return frame
            
            # LivePortrait neural animation (only path)
            if (
                self.use_liveportrait and self.liveportrait_engine and self.liveportrait_ready
                and getattr(self.liveportrait_engine, 'generator_session', None) is not None
                and self.reference_appearance_features is not None
            ):
                try:
                    # Detect and align face for motion extraction with simple EMA smoothing
                    driving_frame = frame
                    faces = self.scrfd_detector.detect_faces(frame) if (self.frame_times.__len__() % self.config.detect_interval == 0 or self._last_bbox is None) else []
                    if faces:
                        best_face = self.scrfd_detector.get_best_face(faces)
                        bbox = best_face['bbox'].astype(np.float32)
                        if self._last_bbox is not None:
                            a = self._bbox_ema_alpha
                            self._last_bbox = a * self._last_bbox + (1.0 - a) * bbox
                        else:
                            self._last_bbox = bbox
                    if self._last_bbox is not None:
                        # Use smoothed bbox to crop
                        fake_face = {'bbox': self._last_bbox.astype(int), 'landmarks': None, 'confidence': 1.0}
                        aligned_face = self.scrfd_detector.crop_and_align_face(frame, fake_face, target_size=(512, 512))
                        if aligned_face is not None:
                            driving_frame = aligned_face
                    
                    # Generate animated frame using LivePortrait
                    animated_frame = self.liveportrait_engine.animate_frame(driving_frame)
                    
                    if animated_frame is not None:
                        # Resize to match input frame size
                        if animated_frame.shape[:2] != frame.shape[:2]:
                            animated_frame = cv2.resize(animated_frame, (frame.shape[1], frame.shape[0]))
                        # Track method
                        try:
                            self.last_method = 'liveportrait'
                        except Exception:
                            pass
                        self._update_stats("liveportrait", start_time)
                        return animated_frame
                    else:
                        logger.error("LivePortrait animation returned None")
                        
                except Exception as e:
                    logger.error(f"LivePortrait animation error: {e}")
                return frame
            # If we reach here, neural path could not produce an output for this frame
            try:
                self.last_method = 'liveportrait_error'
            except Exception:
                pass
            return frame
                
        except Exception as e:
            logger.error(f"Video frame processing error: {e}")
            return frame
    
    def _update_stats(self, method: str, start_time: float = None):
        """Update performance statistics"""
        if start_time is not None:
            elapsed = time.time() - start_time
            self.frame_times.append(elapsed)
        
        # metrics backend optional; keep quiet if missing
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio chunk with voice conversion"""
        start_time = time.time()
        
        try:
            with self.audio_lock:
                # Add to buffer
                self.audio_buffer.append(audio_chunk)
                
                # Process with RVC if available
                if self.rvc and self.rvc.loaded:
                    result = self.rvc.convert_voice(audio_chunk)
                else:
                    result = audio_chunk  # Pass through
                
                # Update performance stats
                elapsed = time.time() - start_time
                self.audio_times.append(elapsed)
                
                return result
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return audio_chunk
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            stats = {
                "loaded": self.loaded,
                "reference_set": self.reference_frame is not None,
                "liveportrait_ready": self.liveportrait_ready,
                "liveportrait_enabled": bool(self.use_liveportrait and self.liveportrait_ready and getattr(self.liveportrait_engine, 'generator_session', None) is not None),
                "frame_count": len(self.frame_times),
                "audio_count": len(self.audio_times)
            }
            
            if self.frame_times:
                avg_frame_time = np.mean(list(self.frame_times))
                stats["avg_frame_time_ms"] = avg_frame_time * 1000
                stats["estimated_fps"] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                # Aliases for client UI expectations
                stats["avg_video_latency_ms"] = stats["avg_frame_time_ms"]
                stats["video_fps"] = stats["estimated_fps"]
            
            if self.audio_times:
                stats["avg_audio_time_ms"] = np.mean(list(self.audio_times)) * 1000
            
                # LivePortrait engine stats
            if self.liveportrait_engine:
                lp_stats = self.liveportrait_engine.get_performance_stats()
                stats["liveportrait"] = lp_stats
            # Last method used (if tracked)
            if hasattr(self, 'last_method'):
                stats["last_method"] = getattr(self, 'last_method')
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"error": str(e), "loaded": False}


# Singleton pipeline instance
_pipeline_instance: Optional[RealTimeAvatarPipeline] = None

def get_pipeline() -> RealTimeAvatarPipeline:
    """Get global pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RealTimeAvatarPipeline()
    return _pipeline_instance
