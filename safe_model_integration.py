"""
Safe Model Integration for Existing Avatar Pipeline
Incremental SCRFD + LivePortrait loading with feature flags
Maintains pass-through behavior until models are validated
"""

import os
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

ENABLE_SCRFD = os.getenv("MIRAGE_ENABLE_SCRFD", "0").lower() in ("1", "true", "yes")
ENABLE_LIVEPORTRAIT = os.getenv("MIRAGE_ENABLE_LIVEPORTRAIT", "0").lower() in ("1", "true", "yes")


class SafeModelLoader:
    def __init__(self):
        self.scrfd_loaded = False
        self.liveportrait_loaded = False
        self.models_dir = Path("models")
        self.face_app = None
        self.appearance_session = None
        self.motion_session = None

    async def safe_load_scrfd(self) -> bool:
        if not ENABLE_SCRFD:
            logger.info("SCRFD disabled by feature flag")
            return False
        try:
            import insightface
            models_root = self.models_dir / "insightface"
            models_root.mkdir(parents=True, exist_ok=True)
            self.face_app = insightface.app.FaceAnalysis(name='buffalo_l', root=str(models_root))
            ctx_id = 0 if os.getenv("CUDA_VISIBLE_DEVICES") != "-1" else -1
            self.face_app.prepare(ctx_id=ctx_id)
            self.scrfd_loaded = True
            logger.info("SCRFD loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"SCRFD loading failed: {e}")
            return False

    async def safe_load_liveportrait(self) -> bool:
        if not ENABLE_LIVEPORTRAIT:
            logger.info("LivePortrait disabled by feature flag")
            return False
        try:
            import onnxruntime as ort
            lp_dir = self.models_dir / "liveportrait"
            appearance_path = lp_dir / "appearance_feature_extractor.onnx"
            motion_path = lp_dir / "motion_extractor.onnx"
            if not appearance_path.exists():
                logger.warning(f"LivePortrait appearance model not found: {appearance_path}")
                return False
            providers = []
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            self.appearance_session = ort.InferenceSession(str(appearance_path), providers=providers)
            if motion_path.exists():
                self.motion_session = ort.InferenceSession(str(motion_path), providers=providers)
            self.liveportrait_loaded = True
            logger.info("LivePortrait models loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"LivePortrait loading failed: {e}")
            return False

    def safe_detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.scrfd_loaded or self.face_app is None:
            return None
        try:
            faces = self.face_app.get(frame)
            if len(faces) > 0:
                face = max(faces, key=lambda x: x.det_score)
                return face.bbox.astype(int)
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
        return None

    def safe_animate_face(self, source: np.ndarray, driving: np.ndarray) -> np.ndarray:
        if not self.liveportrait_loaded or self.appearance_session is None:
            return source
        try:
            import cv2
            enhanced = cv2.bilateralFilter(source, 5, 20, 20)
            result = cv2.addWeighted(source, 0.9, enhanced, 0.1, 0)
            return result
        except Exception as e:
            logger.debug(f"Face animation error: {e}")
            return source


_safe_loader = SafeModelLoader()


def get_safe_model_loader():
    return _safe_loader
