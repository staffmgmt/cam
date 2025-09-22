"""
LivePortrait ONNX Engine - Complete Neural Face Animation Pipeline
Implements appearance feature extraction and motion-driven synthesis for real-time avatar animation
"""

import numpy as np
import cv2
import torch
import onnxruntime as ort
import os
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class LivePortraitONNX:
    """
    Complete LivePortrait ONNX pipeline for neural face animation
    """
    
    def __init__(self, 
                 models_dir: str = "models/liveportrait",
                 target_size: Tuple[int, int] = (512, 512),
                 device: str = "cuda"):
        
        self.models_dir = Path(models_dir)
        self.target_size = target_size
        self.device = device
        
        # Model paths
        self.appearance_model_path = self.models_dir / "appearance_feature_extractor.onnx"
        self.motion_model_path = self.models_dir / "motion_extractor.onnx"
        # The upstream repo provides generator as warping_spade.onnx (or warping_spade-fix.onnx).
        # Prefer a local alias 'generator.onnx' but fall back to these names if missing.
        self.generator_model_path = self.models_dir / "generator.onnx"
        self._alt_generator_paths = [
            self.models_dir / "warping_spade.onnx",
            self.models_dir / "warping_spade-fix.onnx",
        ]
        
        # ONNX Runtime sessions
        self.appearance_session: Optional[ort.InferenceSession] = None
        self.motion_session: Optional[ort.InferenceSession] = None
        self.generator_session: Optional[ort.InferenceSession] = None
        
        # Cached appearance features
        self.reference_appearance: Optional[np.ndarray] = None
        self.reference_image: Optional[np.ndarray] = None
        
        # Performance tracking
        self.inference_times = []
        
    def _get_onnx_providers(self) -> list[str]:
        """Get optimal ONNX execution providers (with safe defaults)."""
        avail = ort.get_available_providers()
        logger.info(f"ONNX Runtime available providers: {avail}")
        providers: list = []
        if self.device == "cuda" and "CUDAExecutionProvider" in avail:
            # Start with a minimal, widely compatible config
            providers.append("CUDAExecutionProvider")
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
            except Exception as e:
                logger.warning(f"Failed to register custom ops via SessionOptions: {e}")
            # Allow disabling shape inference if environment indicates issues
            if os.getenv("MIRAGE_ORT_DISABLE_SHAPE_INFERENCE", "0") in ("1", "true", "True"):
                try:
                    sess_options.add_session_config_entry("session.disable_shape_inference", "1")
                    logger.info("ONNX Runtime: shape inference disabled via session config entry")
                except Exception:
                    pass
            
            # Load appearance feature extractor (required)
            if self.appearance_model_path.exists():
                logger.info(f"Loading appearance model: {self.appearance_model_path}")
                try:
                    self.appearance_session = ort.InferenceSession(
                        str(self.appearance_model_path), 
                        providers=providers,
                        sess_options=sess_options
                    )
                except Exception as e:
                    logger.warning(f"Appearance model failed with tuned providers, retrying basic: {e}")
                    # Retry with default provider list only
                    basic_providers = [p for p in providers]
                    self.appearance_session = ort.InferenceSession(
                        str(self.appearance_model_path),
                        providers=basic_providers
                    )
            else:
                logger.error(f"Appearance model not found: {self.appearance_model_path}")
                return False
            
            # Load motion extractor (required)
            if self.motion_model_path.exists():
                logger.info(f"Loading motion model: {self.motion_model_path}")
                try:
                    self.motion_session = ort.InferenceSession(
                        str(self.motion_model_path),
                        providers=providers,
                        sess_options=sess_options
                    )
                except Exception as e:
                    logger.warning(f"Motion model failed with tuned providers, retrying basic: {e}")
                    basic_providers = [p for p in providers]
                    self.motion_session = ort.InferenceSession(
                        str(self.motion_model_path),
                        providers=basic_providers
                    )
            else:
                logger.error(f"Motion model not found: {self.motion_model_path}")
                return False
            
            # Load generator if available (optional - can use warping fallback)
            gen_path = None
            if self.generator_model_path.exists():
                gen_path = self.generator_model_path
            else:
                for p in self._alt_generator_paths:
                    if p.exists():
                        gen_path = p
                        break
            if gen_path is not None:
                # Optionally create an alias at generator.onnx for diagnostics endpoints
                try:
                    if gen_path != self.generator_model_path and not self.generator_model_path.exists():
                        # create a lightweight copy to keep tooling simple
                        try:
                            import shutil
                            shutil.copyfile(gen_path, self.generator_model_path)
                        except Exception:
                            pass
                except Exception:
                    pass
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
                        self.generator_session = ort.InferenceSession(
                            str(gen_path),
                            providers=basic_providers
                        )
                    except Exception as e2:
                        logger.error(f"Generator failed to load with basic providers as well: {e2}")
                        raise
            else:
                logger.info("Generator model not available - using motion-based warping")
            
            logger.info("LivePortrait ONNX models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LivePortrait models: {e}")
            return False
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model input"""
        # Resize to target size
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size)
        
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
            input_tensor = self._preprocess_image(reference_image)
            
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
            # Preprocess image
            input_tensor = self._preprocess_image(driving_image)
            
            # Get input name
            input_name = self.motion_session.get_inputs()[0].name
            
            # Run inference
            start_time = time.time()
            outputs = self.motion_session.run(None, {input_name: input_tensor})
            inference_time = time.time() - start_time
            
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-50:]  # Keep recent times
            
            motion_params = outputs[0]  # Assume first output is motion vector
            
            return motion_params
            
        except Exception as e:
            logger.error(f"Motion parameter extraction failed: {e}")
            return None
    
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
            
            # Prepare inputs - exact format depends on model architecture
            feed_dict = {}
            if len(input_names) >= 2:
                feed_dict[input_names[0]] = appearance_features
                feed_dict[input_names[1]] = motion_params
            else:
                # Concatenate features if single input expected
                combined_features = np.concatenate([appearance_features, motion_params], axis=-1)
                feed_dict[input_names[0]] = combined_features
            
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