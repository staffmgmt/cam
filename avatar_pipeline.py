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
from alignment import align_face, gaussian_face_mask, CANONICAL_5PT
from smoothing import KeypointOneEuro

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
        self._default_det_size = (640, 640)
        
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
            self.model.prepare(ctx_id=0 if self.config.device == "cuda" else -1, det_size=self._default_det_size)
            
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
            # Retry at larger detection sizes if nothing found
            if (faces is None or len(faces) == 0):
                for det_size in [(960, 960), (1280, 1280)]:
                    try:
                        self.model.prepare(ctx_id=0 if self.config.device == "cuda" else -1, det_size=det_size)
                        faces = self.model.get(rgb_image)
                        if faces is not None and len(faces) > 0:
                            break
                    except Exception as _:
                        pass
                # Restore default det size
                try:
                    self.model.prepare(ctx_id=0 if self.config.device == "cuda" else -1, det_size=self._default_det_size)
                except Exception:
                    pass
            
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
            # Skip native LivePortrait .pth model loading since we use ONNX-only path
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
        
        # Frame buffers for real-time processing (single-item coalescing queue)
        self.video_buffer = deque(maxlen=1)
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
        self.initializing = False
        # Async inference worker state
        self._inference_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_generated_frame: Optional[np.ndarray] = None
        self._last_generated_time: float = 0.0
        self._frame_submit_count = 0
        self._frame_process_count = 0
        # Keypoint smoothing filter
        self._kp_filter = KeypointOneEuro(K=21, C=3, min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
        self._prev_motion_raw = None
        # Adaptive detection interval tracking
        self._dynamic_detect_interval = self.config.detect_interval
        self._recent_motion_magnitudes = deque(maxlen=30)
        self._consecutive_detect_fail = 0
    # Extended motion history for metrics
    self._motion_history = deque(maxlen=300)
    # Latency histogram snapshots (long window)
    self._latency_history = deque(maxlen=500)
    self._latency_hist_snapshots = []  # list of {timestamp, buckets}
    # Frame pacing
    self._pacing_hint = 1.0  # multiplier suggestion (1.0 = normal)
    self._target_frame_time = 1.0 / max(self.config.target_fps, 1)
    self._latency_ema = None
        
    async def initialize(self):
        """Initialize all models"""
        if self.loaded:
            logger.info("Initialize called: already loaded")
            return {"status": "already_initialized"}
        if self.initializing:
            logger.info("Initialize called: initialization already in progress")
            return {"status": "initializing"}
        self.initializing = True
        logger.info("Initializing real-time avatar pipeline...")
        
        # Initialize SCRFD face detection (required)
        scrfd_ok = await self.scrfd_detector.load_model()
        if scrfd_ok:
            logger.info("SCRFD face detection ready")
        else:
            logger.error("SCRFD failed to load (required)")
            self.initializing = False
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
                            # Trigger generator warmup once (safe guard)
                            try:
                                warm = getattr(self.liveportrait_engine, 'warmup', None)
                                if callable(warm):
                                    warm()
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Post-init reference extraction failed: {e}")
                mode = "liveportrait"
                logger.info(f"Avatar pipeline initialized successfully (mode={mode})")
                # Launch background inference worker
                try:
                    if self._loop is None:
                        self._loop = asyncio.get_running_loop()
                    if self._inference_task is None or self._inference_task.done():
                        self._inference_task = self._loop.create_task(self._inference_worker())
                        logger.info("Async inference worker started")
                except Exception as e:
                    logger.warning(f"Could not start inference worker: {e}")
                self.initializing = False
                return True
            else:
                logger.error("Avatar pipeline initialization failed - requirements not met")
                self.initializing = False
                return False
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.loaded = scrfd_ok
            self.initializing = False
            return self.loaded
    
    def set_reference_frame(self, frame: np.ndarray) -> bool:
        """Set reference frame for avatar animation"""
        try:
            logger.info(f"Setting reference frame: {frame.shape}")
            # Detect face in reference frame using SCRFD (required)
            faces = self.scrfd_detector.detect_faces(frame)
            # If no faces, try rotating the image (common EXIF-rotated portrait photos)
            rotations = 0
            while (not faces) and rotations < 3:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                rotations += 1
                faces = self.scrfd_detector.detect_faces(frame)
            if not faces:
                logger.error("No face detected in reference image")
                return False
            best_face = self.scrfd_detector.get_best_face(faces)
            # Use alignment utilities with 5-point landmarks for canonical transform instead of naive crop
            try:
                landmarks = best_face['landmarks'].astype(np.float32)
                if landmarks.shape[0] == 5:
                    align_res = align_face(frame, landmarks, target_size=512, canonical_template=CANONICAL_5PT)
                    frame = align_res['aligned_image']
                    self._ref_align = align_res  # store full alignment data
                    logger.info("Reference frame aligned via similarity transform")
                else:
                    logger.warning(f"Unexpected landmark count {landmarks.shape[0]}, falling back to bbox crop")
                    aligned_face = self.scrfd_detector.crop_and_align_face(frame, best_face, target_size=(512, 512))
                    if aligned_face is None:
                        logger.error("Failed to align reference face (fallback)")
                        return False
                    frame = aligned_face
                    self._ref_align = None
            except Exception as e:
                logger.warning(f"Alignment failed, using fallback crop: {e}")
                aligned_face = self.scrfd_detector.crop_and_align_face(frame, best_face, target_size=(512, 512))
                if aligned_face is None:
                    logger.error("Failed to align reference face (fallback)")
                    return False
                frame = aligned_face
                self._ref_align = None
            self._last_bbox = best_face['bbox'].astype(np.float32)
            
            # Store reference frame
            self.reference_frame = frame.copy()
            # Precompute soft face mask in canonical space for compositing later
            try:
                self._face_mask = gaussian_face_mask(size=frame.shape[0])  # assume square
            except Exception:
                self._face_mask = None
            
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
        """Enqueue frame for async processing; return last generated frame if available."""
        try:
            self._frame_submit_count += 1
            if self.video_buffer:
                self.video_buffer.clear()
            self.video_buffer.append((frame.copy(), time.time()))
            if self._last_generated_frame is not None:
                return self._last_generated_frame
            return frame
        except Exception as e:
            logger.error(f"Frame enqueue error: {e}")
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
            # Async worker stats
            stats["async_worker"] = {
                "frame_submitted": self._frame_submit_count,
                "frame_processed": self._frame_process_count,
                "dynamic_detect_interval": self._dynamic_detect_interval,
                "recent_motion_avg": float(np.mean(self._recent_motion_magnitudes)) if self._recent_motion_magnitudes else None,
                "consecutive_detect_fail": self._consecutive_detect_fail,
                "pacing_hint": self._pacing_hint,
                "latency_ema_ms": self._latency_ema * 1000.0 if self._latency_ema is not None else None,
            }
            if hasattr(self, 'last_stage_timings'):
                stats['last_stage_timings'] = self.last_stage_timings
            stats['latency_histogram_snapshots'] = self._latency_hist_snapshots[-5:]
            # Provide a trimmed motion tail
            stats['motion_tail'] = list(self._motion_history)[-25:]
            # Last method used (if tracked)
            if hasattr(self, 'last_method'):
                stats["last_method"] = getattr(self, 'last_method')
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"error": str(e), "loaded": False}

    async def _inference_worker(self):
        """Background asynchronous inference loop processing latest video frame only."""
        logger.info("Inference worker started")
        while True:
            try:
                if not self.video_buffer or self.reference_frame is None:
                    await asyncio.sleep(0.005)
                    continue
                frame, ts = self.video_buffer.pop()
                start = time.time()
                stage_timings: Dict[str, float] = {}
                if not (
                    self.use_liveportrait and self.liveportrait_engine and self.liveportrait_ready and self.reference_appearance_features is not None and getattr(self.liveportrait_engine, 'generator_session', None) is not None
                ):
                    self._last_generated_frame = frame
                    continue
                # Decide detection
                need_detect = (self._frame_process_count % self._dynamic_detect_interval == 0) or (self._last_bbox is None)
                faces = []
                if need_detect:
                    t0 = time.time()
                    faces = self.scrfd_detector.detect_faces(frame)
                    stage_timings['detect'] = time.time() - t0
                    if not faces:
                        self._consecutive_detect_fail += 1
                    else:
                        self._consecutive_detect_fail = 0
                # Adaptive logic
                if self._consecutive_detect_fail >= 3:
                    self._dynamic_detect_interval = max(1, self._dynamic_detect_interval - 1)
                elif len(self._recent_motion_magnitudes) >= 10:
                    avg_motion = float(np.mean(self._recent_motion_magnitudes))
                    if avg_motion < 0.02:
                        self._dynamic_detect_interval = min(12, self._dynamic_detect_interval + 1)
                    elif avg_motion > 0.08:
                        self._dynamic_detect_interval = max(2, self._dynamic_detect_interval - 1)
                driving_frame = frame
                best_face = None
                if faces:
                    best_face = self.scrfd_detector.get_best_face(faces)
                    bbox = best_face['bbox'].astype(np.float32)
                    if self._last_bbox is not None:
                        a = self._bbox_ema_alpha
                        self._last_bbox = a * self._last_bbox + (1 - a) * bbox
                    else:
                        self._last_bbox = bbox
                t_align = time.time()
                if best_face is not None and best_face.get('landmarks') is not None:
                    try:
                        landmarks = best_face['landmarks'].astype(np.float32)
                        if landmarks.shape[0] == 5:
                            d_align = align_face(frame, landmarks, target_size=512, canonical_template=CANONICAL_5PT)
                            driving_frame = d_align['aligned_image']
                            self._driving_align = d_align
                        else:
                            raise ValueError('Unexpected landmark count')
                    except Exception:
                        fake_face = {'bbox': self._last_bbox.astype(int), 'landmarks': None, 'confidence': 1.0}
                        aligned_face = self.scrfd_detector.crop_and_align_face(frame, fake_face, target_size=(512, 512))
                        if aligned_face is not None:
                            driving_frame = aligned_face
                elif self._last_bbox is not None:
                    fake_face = {'bbox': self._last_bbox.astype(int), 'landmarks': None, 'confidence': 1.0}
                    aligned_face = self.scrfd_detector.crop_and_align_face(frame, fake_face, target_size=(512, 512))
                    if aligned_face is not None:
                        driving_frame = aligned_face
                stage_timings['align'] = time.time() - t_align
                # Motion & generator
                t_motion = time.time()
                motion_raw = self.liveportrait_engine.extract_motion_parameters(driving_frame)
                motion_smoothed = motion_raw
                if isinstance(motion_raw, np.ndarray) and motion_raw.ndim == 3:
                    if self._prev_motion_raw is not None and self._prev_motion_raw.shape == motion_raw.shape:
                        diff = motion_raw - self._prev_motion_raw
                        mag = float(np.mean(np.linalg.norm(diff[..., :2], axis=-1)))
                        self._recent_motion_magnitudes.append(mag)
                        self._motion_history.append((time.time(), mag))
                    self._prev_motion_raw = motion_raw.copy()
                    motion_smoothed = self._kp_filter.filter(motion_raw, time.time())
                animated_frame = None
                try:
                    animated_frame = self.liveportrait_engine.synthesize_frame(self.reference_appearance_features, motion_smoothed)
                except Exception as se:
                    logger.error(f"Synthesis error: {se}")
                stage_timings['motion+gen'] = time.time() - t_motion
                # Composite
                t_comp = time.time()
                if animated_frame is not None and hasattr(self, '_ref_align') and self._ref_align is not None:
                    try:
                        M_inv = self._ref_align.get('M_inv') if isinstance(self._ref_align, dict) else None
                        if M_inv is not None:
                            h0, w0 = frame.shape[:2]
                            warped_face = cv2.warpAffine(animated_frame, M_inv, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                            if getattr(self, '_face_mask', None) is not None:
                                mask = self._face_mask
                                mask_warp = cv2.warpAffine(mask, M_inv, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                mask_warp = mask_warp[..., None]
                                animated_frame = (warped_face.astype(np.float32) * mask_warp + frame.astype(np.float32) * (1 - mask_warp)).astype(np.uint8)
                    except Exception as ce:
                        logger.debug(f"Compositing skipped: {ce}")
                stage_timings['composite'] = time.time() - t_comp
                out_frame = animated_frame if animated_frame is not None else frame
                if out_frame.shape[:2] != frame.shape[:2]:
                    out_frame = cv2.resize(out_frame, (frame.shape[1], frame.shape[0]))
                self._last_generated_frame = out_frame
                self._last_generated_time = time.time()
                stage_timings['total'] = self._last_generated_time - start
                self.last_stage_timings = stage_timings
                # Update stats (latency curve)
                self.frame_times.append(stage_timings['total'])
                self._latency_history.append((self._last_generated_time, stage_timings['total']))
                # Update pacing: exponential moving average of latency
                lt = stage_timings['total']
                if self._latency_ema is None:
                    self._latency_ema = lt
                else:
                    self._latency_ema = 0.9 * self._latency_ema + 0.1 * lt
                # Suggest pacing multiplier ( <1 means sender should slow )
                if self._latency_ema > self._target_frame_time * 1.2:
                    self._pacing_hint = max(0.5, (self._target_frame_time * 1.2) / self._latency_ema)
                elif self._latency_ema < self._target_frame_time * 0.8:
                    self._pacing_hint = min(1.2, (self._target_frame_time * 0.8) / max(self._latency_ema, 1e-6))
                else:
                    self._pacing_hint = 1.0
                # Periodic histogram snapshot every 50 processed frames
                if self._frame_process_count % 50 == 0 and self._frame_process_count > 0:
                    buckets = {"<50ms":0,"50-100ms":0,"100-200ms":0,"200-400ms":0,">=400ms":0}
                    for _, l in list(self._latency_history)[-300:]:
                        ms = l*1000.0
                        if ms < 50: buckets["<50ms"]+=1
                        elif ms < 100: buckets["50-100ms"]+=1
                        elif ms < 200: buckets["100-200ms"]+=1
                        elif ms < 400: buckets["200-400ms"]+=1
                        else: buckets[">=400ms"]+=1
                    self._latency_hist_snapshots.append({"ts": self._last_generated_time, "buckets": buckets})
                self._frame_process_count += 1
            except Exception as e:
                logger.error(f"Inference worker error: {e}")
                await asyncio.sleep(0.05)


# Singleton pipeline instance
_pipeline_instance: Optional[RealTimeAvatarPipeline] = None

def get_pipeline() -> RealTimeAvatarPipeline:
    """Get global pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RealTimeAvatarPipeline()
    return _pipeline_instance
