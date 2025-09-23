"""
LivePortrait ONNX Engine - Complete Neural Face Animation Pipeline
Implements appearance feature extraction and motion-driven synthesis for real-time avatar animation
"""

import numpy as np
import cv2
import torch
import onnxruntime as ort
import os
import onnx  # type: ignore
from onnx import version_converter  # type: ignore
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

# Try to import onnxruntime-extensions to register extra custom ops (e.g., GridSample)
_ORTEXT_LIB = None
try:
    from onnxruntime_extensions import get_library_path as _ortext_get_library_path  # type: ignore
    _ORTEXT_LIB = _ortext_get_library_path()
except Exception:
    _ORTEXT_LIB = None

class LivePortraitONNX:
    """
    Complete LivePortrait ONNX pipeline for neural face animation
    """
    
    def __init__(self, 
                 models_dir: str = "models/liveportrait",
                 target_size: Tuple[int, int] = (512, 512),
                 device: str = "cuda"):
        
        self.models_dir = Path(models_dir)
        # Default desired size (width, height); will be overridden by model's required size if present
        self.target_size = target_size
        self.device = device
        
        # Model paths
        self.appearance_model_path = self.models_dir / "appearance_feature_extractor.onnx"
        self.motion_model_path = self.models_dir / "motion_extractor.onnx"
        # Single required generator path; no filename fallbacks
        self.generator_model_path = self.models_dir / "generator.onnx"
        
        # ONNX Runtime sessions
        self.appearance_session: Optional[ort.InferenceSession] = None
        self.motion_session: Optional[ort.InferenceSession] = None
        self.generator_session: Optional[ort.InferenceSession] = None

        # Model-specific input sizes (width, height), inferred from ONNX model inputs when loaded
        self.appearance_input_size: Optional[Tuple[int, int]] = None
        self.motion_input_size: Optional[Tuple[int, int]] = None
        
        # Cached appearance features
        self.reference_appearance: Optional[np.ndarray] = None
        self.reference_image: Optional[np.ndarray] = None
        self.reference_kp: Optional[np.ndarray] = None
        
        # Performance tracking
        self.inference_times = []
        
    def _get_onnx_providers(self) -> list[str]:
        """Get optimal ONNX execution providers. Enforce GPU if required."""
        avail = ort.get_available_providers()
        logger.info(f"ONNX Runtime available providers: {avail}")
        require_gpu = os.getenv("MIRAGE_REQUIRE_GPU", "0") in ("1", "true", "True")
        providers: list[str] = []
        if self.device == "cuda" and "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        if require_gpu:
            if "CUDAExecutionProvider" not in providers:
                # Fail fast with explicit error to avoid CPU fallback
                raise RuntimeError("GPU is required but CUDAExecutionProvider is unavailable. Check CUDA/cuDNN/ORT GPU alignment.")
            # When GPU is required, do not append CPU provider to prevent silent fallback
            return providers
        # Otherwise allow CPU fallback
        providers.append("CPUExecutionProvider")
        return providers

    def _register_custom_ops(self):
        """Register optional custom ops library if present."""
        try:
            lib_path = self.models_dir / "libgrid_sample_3d_plugin.so"
            if lib_path.exists():
                logger.info(f"Registering custom ops library: {lib_path}")
                # onnxruntime automatically loads custom opsets when passed via session options - set env var as a fallback
                os.environ["ORT_LOAD_CUSTOM_OP_LIBS"] = str(lib_path)
        except Exception as e:
            logger.warning(f"Failed to register custom ops library: {e}")
    
    def load_models(self) -> bool:
        """Load all available ONNX models"""
        try:
            providers = self._get_onnx_providers()
            # Optionally register custom ops shared library
            self._register_custom_ops()
            
            # Set session options for performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            # Register custom ops library directly if present
            try:
                lib_path = self.models_dir / "libgrid_sample_3d_plugin.so"
                if lib_path.exists():
                    sess_options.register_custom_ops_library(str(lib_path))
                    logger.info("Custom ops library registered via SessionOptions")
                # Prefer onnxruntime-extensions custom ops if available
                if _ORTEXT_LIB:
                    sess_options.register_custom_ops_library(_ORTEXT_LIB)
                    logger.info("onnxruntime-extensions custom ops registered")
            except Exception as e:
                logger.warning(f"Failed to register custom ops via SessionOptions: {e}")
            # Allow disabling shape inference if environment indicates issues
            if os.getenv("MIRAGE_ORT_DISABLE_SHAPE_INFERENCE", "0") in ("1", "true", "True"):
                try:
                    sess_options.add_session_config_entry("session.disable_shape_inference", "1")
                    logger.info("ONNX Runtime: shape inference disabled via session config entry")
                except Exception:
                    pass
            
            # Helper to ensure opset <= 19 by converting if required
            def _ensure_opset_compat(path: Path) -> Path:
                try:
                    model = onnx.load(str(path), load_external_data=True)
                    max_opset = max((imp.version for imp in model.opset_import), default=0)
                    if max_opset > 19:
                        logger.info(f"Converting ONNX opset from {max_opset} to 19 for {path.name}")
                        converted = version_converter.convert_version(model, 19)
                        out_path = path.with_name(path.stem + "_op19.onnx")
                        onnx.save(converted, str(out_path))
                        return out_path
                except Exception as ce:
                    logger.warning(f"Opset conversion skipped for {path.name}: {ce}")
                return path

            # Load appearance feature extractor (required)
            if self.appearance_model_path.exists():
                app_path = _ensure_opset_compat(self.appearance_model_path)
                logger.info(f"Loading appearance model: {app_path}")
                try:
                    self.appearance_session = ort.InferenceSession(
                        str(app_path), 
                        providers=providers,
                        sess_options=sess_options
                    )
                except Exception as e:
                    logger.warning(f"Appearance model failed with tuned providers, retrying basic: {e}")
                    # Retry with default provider list only
                    basic_providers = [p for p in providers]
                    self.appearance_session = ort.InferenceSession(
                        str(app_path),
                        providers=basic_providers
                    )
                # Infer expected input size (assume NCHW: [N, C, H, W])
                try:
                    a_in = self.appearance_session.get_inputs()[0]
                    shape = a_in.shape
                    if isinstance(shape, (list, tuple)) and len(shape) == 4:
                        h = shape[2]
                        w = shape[3]
                        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                            self.appearance_input_size = (int(w), int(h))
                            logger.info(f"Appearance model expects input (HxW): {h}x{w}")
                except Exception as e:
                    logger.warning(f"Unable to infer appearance input size: {e}")
            else:
                logger.error(f"Appearance model not found: {self.appearance_model_path}")
                return False
            
            # Load motion extractor (required)
            if self.motion_model_path.exists():
                mot_path = _ensure_opset_compat(self.motion_model_path)
                logger.info(f"Loading motion model: {mot_path}")
                try:
                    self.motion_session = ort.InferenceSession(
                        str(mot_path),
                        providers=providers,
                        sess_options=sess_options
                    )
                except Exception as e:
                    logger.warning(f"Motion model failed with tuned providers, retrying basic: {e}")
                    basic_providers = [p for p in providers]
                    self.motion_session = ort.InferenceSession(
                        str(mot_path),
                        providers=basic_providers
                    )
                # Infer expected input size for motion model
                try:
                    m_in = self.motion_session.get_inputs()[0]
                    shape = m_in.shape
                    if isinstance(shape, (list, tuple)) and len(shape) == 4:
                        h = shape[2]
                        w = shape[3]
                        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                            self.motion_input_size = (int(w), int(h))
                            logger.info(f"Motion model expects input (HxW): {h}x{w}")
                except Exception as e:
                    logger.warning(f"Unable to infer motion input size: {e}")
            else:
                logger.error(f"Motion model not found: {self.motion_model_path}")
                return False
            
            # Load generator (required)
            if self.generator_model_path.exists():
                gen_path = _ensure_opset_compat(self.generator_model_path)
                logger.info(f"Loading generator model: {gen_path}")
                try:
                    self.generator_session = ort.InferenceSession(
                        str(gen_path),
                        providers=providers,
                        sess_options=sess_options
                    )
                except Exception as e:
                    logger.warning(f"Generator failed with tuned providers, retrying basic: {e}")
                    basic_providers = [p for p in providers]
                    # Try again with minimal options
                    try:
                        # Register extensions on a fresh SessionOptions too
                        sess_basic = ort.SessionOptions()
                        try:
                            if _ORTEXT_LIB:
                                sess_basic.register_custom_ops_library(_ORTEXT_LIB)
                        except Exception:
                            pass
                        if os.getenv("MIRAGE_ORT_DISABLE_SHAPE_INFERENCE", "0") in ("1", "true", "True"):
                            try:
                                sess_basic.add_session_config_entry("session.disable_shape_inference", "1")
                            except Exception:
                                pass
                        self.generator_session = ort.InferenceSession(
                            str(gen_path),
                            providers=basic_providers,
                            sess_options=sess_basic
                        )
                    except Exception as e2:
                        logger.error(f"Generator failed to load with basic providers as well: {e2}")
                        raise

                # Log expected generator inputs for mapping
                try:
                    gin = [i.name for i in self.generator_session.get_inputs()]
                    logger.info(f"Generator input names: {gin}")
                except Exception:
                    pass
            else:
                logger.error("LivePortrait generator.onnx not found (required)")
                return False
            
            logger.info("LivePortrait ONNX models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LivePortrait models: {e}")
            return False
    
    def _preprocess_image(self, image: np.ndarray, *, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Preprocess image for ONNX model input"""
        # Determine target (width, height)
        target_wh = size or self.target_size
        tw, th = target_wh
        # Resize if needed; image.shape[:2] is (H, W)
        h, w = image.shape[:2]
        if (w, h) != (tw, th):
            image = cv2.resize(image, (tw, th))
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)    # Add batch dim
        
        return image
    
    def extract_appearance_features(self, reference_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract appearance features from reference image"""
        if self.appearance_session is None:
            logger.error("Appearance model not loaded")
            return None
        
        try:
            # Preprocess image
            input_tensor = self._preprocess_image(reference_image, size=self.appearance_input_size or self.target_size)
            
            # Get input name
            input_name = self.appearance_session.get_inputs()[0].name
            
            # Run inference
            start_time = time.time()
            outputs = self.appearance_session.run(None, {input_name: input_tensor})
            inference_time = time.time() - start_time
            
            # Cache the appearance features
            appearance_features = outputs[0]  # Assume first output is appearance vector
            self.reference_appearance = appearance_features
            self.reference_image = reference_image.copy()

            # Also compute source keypoints using motion model if available
            if self.motion_session is not None:
                try:
                    kp_src = self._run_motion_for_image(reference_image)
                    self.reference_kp = kp_src
                except Exception as e:
                    logger.warning(f"Failed to compute source keypoints: {e}")
            
            logger.info(f"Appearance features extracted in {inference_time*1000:.1f}ms, shape: {appearance_features.shape}")
            return appearance_features
            
        except Exception as e:
            logger.error(f"Appearance feature extraction failed: {e}")
            return None
    
    def extract_motion_parameters(self, driving_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract motion parameters from driving image"""
        if self.motion_session is None:
            logger.error("Motion model not loaded")
            return None
        
        try:
            kp_drive = self._run_motion_for_image(driving_image)
            return kp_drive
            
        except Exception as e:
            logger.error(f"Motion parameter extraction failed: {e}")
            return None

    def _run_motion_for_image(self, img: np.ndarray) -> np.ndarray:
        """Helper: run motion/keypoint extractor on an image and return output tensor."""
        # Preprocess image to motion model input size
        input_tensor = self._preprocess_image(img, size=self.motion_input_size or self.target_size)
        m_in = self.motion_session.get_inputs()[0].name
        start_time = time.time()
        outputs = self.motion_session.run(None, {m_in: input_tensor})
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-50:]
        # Assume first output contains keypoints/motion representation
        return outputs[0]
    
    def synthesize_frame(self, appearance_features: np.ndarray, motion_params: np.ndarray) -> Optional[np.ndarray]:
        """Synthesize animated frame from appearance + motion"""
        if self.generator_session is not None:
            # Use neural generator if available
            return self._neural_synthesis(appearance_features, motion_params)
        else:
            # Use motion-based warping fallback
            return self._motion_warping_synthesis(motion_params)
    
    def _neural_synthesis(self, appearance_features: np.ndarray, motion_params: np.ndarray) -> Optional[np.ndarray]:
        """Neural synthesis using generator model"""
        try:
            # Get input names
            inputs = self.generator_session.get_inputs()
            input_names = [inp.name for inp in inputs]
            
            # Prepare inputs based on names commonly used in LivePortrait ONNX variants
            feed_dict: Dict[str, np.ndarray] = {}
            name_set = set(n.lower() for n in input_names)

            # Map appearance to appropriate input
            if 'feature_3d' in name_set:
                feed_dict[[n for n in input_names if n.lower() == 'feature_3d'][0]] = appearance_features
            elif 'appearance' in name_set:
                feed_dict[[n for n in input_names if n.lower() == 'appearance'][0]] = appearance_features
            else:
                # fallback: first input
                feed_dict[input_names[0]] = appearance_features

            # Map driving motion/keypoints
            if 'kp_driving' in name_set:
                feed_dict[[n for n in input_names if n.lower() == 'kp_driving'][0]] = motion_params
            elif 'driving' in name_set:
                feed_dict[[n for n in input_names if n.lower() == 'driving'][0]] = motion_params
            else:
                # If only two inputs and first is appearance, second should be motion
                if len(input_names) >= 2:
                    feed_dict[input_names[1]] = motion_params

            # Map source keypoints if required
            if 'kp_source' in name_set or 'source_kp' in name_set:
                kp_name = 'kp_source' if 'kp_source' in name_set else 'source_kp'
                if self.reference_kp is None:
                    # Attempt to compute from cached reference image
                    if self.reference_image is not None and self.motion_session is not None:
                        try:
                            self.reference_kp = self._run_motion_for_image(self.reference_image)
                        except Exception as e:
                            logger.error(f"Failed to compute kp_source on demand: {e}")
                            return None
                    else:
                        logger.error("kp_source required but reference keypoints are unavailable")
                        return None
                feed_dict[[n for n in input_names if n.lower() == kp_name][0]] = self.reference_kp

            # Helper to normalize keypoint tensors (ensure [B, K, C])
            def _normalize_kp(name: str, arr: np.ndarray) -> np.ndarray:
                # Ensure dtype float32
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                # Add batch dim if missing (rank 2 -> [1, K, C])
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, 0)
                # If rank > 3 and squeezable batch/channel dims, try to reduce to 3D conservatively
                if arr.ndim > 3:
                    # Common case: [B, K, C, 1] -> squeeze last
                    if arr.shape[-1] == 1:
                        arr = np.squeeze(arr, axis=-1)
                # Optionally pad last dim from 2->3 if model expects 3
                try:
                    meta = next((m for m in inputs if m.name == name), None)
                    if meta is not None and isinstance(meta.shape, (list, tuple)) and len(meta.shape) == 3:
                        exp_last = meta.shape[2]
                        if isinstance(exp_last, int) and exp_last in (2, 3):
                            if arr.ndim == 3 and isinstance(arr.shape[-1], (int, np.integer)):
                                cur_last = int(arr.shape[-1])
                                if cur_last == 2 and exp_last == 3:
                                    # Pad a zero z-dimension
                                    pad = np.zeros((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
                                    arr = np.concatenate([arr, pad], axis=-1)
                                # If cur_last==3 and exp_last==2, slice to first 2 dims
                                elif cur_last == 3 and exp_last == 2:
                                    arr = arr[..., :2]
                except Exception:
                    pass
                return arr

            # Normalize kp_driving and kp_source shapes/dtypes
            # Find real names (case-preserving) in feed
            lower_to_real = {n.lower(): n for n in feed_dict.keys()}
            if 'kp_driving' in lower_to_real:
                real = lower_to_real['kp_driving']
                feed_dict[real] = _normalize_kp(real, feed_dict[real])
            if 'driving' in lower_to_real:
                real = lower_to_real['driving']
                feed_dict[real] = _normalize_kp(real, feed_dict[real])
            if 'kp_source' in lower_to_real:
                real = lower_to_real['kp_source']
                feed_dict[real] = _normalize_kp(real, feed_dict[real])
            if 'source_kp' in lower_to_real:
                real = lower_to_real['source_kp']
                feed_dict[real] = _normalize_kp(real, feed_dict[real])

            # Log shapes for debugging
            try:
                shapes = {k: list(v.shape) for k, v in feed_dict.items()}
                logger.info(f"Generator feed shapes: {shapes}")
            except Exception:
                pass
            
            # Run inference
            start_time = time.time()
            outputs = self.generator_session.run(None, feed_dict)
            inference_time = time.time() - start_time
            
            # Post-process output
            generated_image = outputs[0]  # Assume first output is generated image
            
            # Convert from CHW to HWC and denormalize
            if len(generated_image.shape) == 4:
                generated_image = generated_image[0]  # Remove batch dim
            
            generated_image = np.transpose(generated_image, (1, 2, 0))  # CHW -> HWC
            generated_image = np.clip(generated_image * 255.0, 0, 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)
            
            return generated_image
            
        except Exception as e:
            logger.error(f"Neural synthesis failed: {e}")
            return None
    
    def _motion_warping_synthesis(self, motion_params: np.ndarray) -> Optional[np.ndarray]:
        """Fallback: Motion-based warping of reference image"""
        if self.reference_image is None:
            return None
        
        try:
            # Extract relevant motion parameters
            # This is a simplified interpretation - actual LivePortrait motion format may differ
            
            # Assume motion_params contains transformation parameters
            # For now, create a simple affine transformation from motion parameters
            
            # Get reference image
            ref_image = cv2.resize(self.reference_image, self.target_size)
            
            # Create transformation matrix from motion parameters
            # This is a placeholder - actual implementation depends on LivePortrait's motion format
            if motion_params.size >= 6:
                # Extract affine parameters (assuming they exist in motion vector)
                motion_flat = motion_params.flatten()
                
                # Create affine transformation matrix
                # Scale motion parameters appropriately
                scale = 0.1  # Reduce motion magnitude
                dx, dy = motion_flat[0] * scale * 10, motion_flat[1] * scale * 10
                angle = motion_flat[2] * scale
                scale_x, scale_y = 1 + motion_flat[3] * scale * 0.1, 1 + motion_flat[4] * scale * 0.1
                
                # Create transformation matrix
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                M = np.array([
                    [scale_x * cos_a, -scale_y * sin_a, dx + self.target_size[0] * 0.05],
                    [scale_x * sin_a, scale_y * cos_a, dy + self.target_size[1] * 0.05]
                ], dtype=np.float32)
                
                # Apply transformation
                warped = cv2.warpAffine(ref_image, M, self.target_size, 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_REFLECT)
                
                return warped
            else:
                # Fallback: return reference image
                return ref_image
                
        except Exception as e:
            logger.error(f"Motion warping synthesis failed: {e}")
            return self.reference_image if self.reference_image is not None else None
    
    def animate_frame(self, driving_image: np.ndarray) -> Optional[np.ndarray]:
        """Complete animation pipeline: extract motion and synthesize frame"""
        if self.reference_appearance is None:
            logger.warning("No reference appearance features - call extract_appearance_features first")
            return None
        
        # Extract motion from driving image
        motion_params = self.extract_motion_parameters(driving_image)
        if motion_params is None:
            return None
        
        # Synthesize animated frame
        result = self.synthesize_frame(self.reference_appearance, motion_params)
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {"avg_inference_ms": 0, "fps_estimate": 0}
        
        avg_time = np.mean(self.inference_times)
        fps_estimate = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_inference_ms": avg_time * 1000,
            "fps_estimate": fps_estimate,
            "total_inferences": len(self.inference_times),
            "models_loaded": {
                "appearance": self.appearance_session is not None,
                "motion": self.motion_session is not None,
                "generator": self.generator_session is not None
            }
        }
    
    def is_ready(self) -> bool:
        """Check if the engine is ready for animation"""
        return (
            self.appearance_session is not None and
            self.motion_session is not None and
            self.generator_session is not None and
            self.reference_appearance is not None
        )


# Global instance
_liveportrait_engine: Optional[LivePortraitONNX] = None

def get_liveportrait_engine() -> LivePortraitONNX:
    """Get global LivePortrait ONNX engine instance"""
    global _liveportrait_engine
    if _liveportrait_engine is None:
        _liveportrait_engine = LivePortraitONNX()
    return _liveportrait_engine