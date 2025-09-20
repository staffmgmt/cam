"""
Lightweight landmark-based reenactor using MediaPipe FaceMesh.
Goal: Real-time motion transfer (pose/expression approximation) as a fallback
when generative model is unavailable. Uses global similarity transform from
reference landmarks to driving landmarks. Not as rich as LivePortrait/FOMM,
but fast and stable for demos.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

try:
    import mediapipe as mp  # type: ignore
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


class LandmarkReenactor:
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.ref_img: Optional[np.ndarray] = None
        self.ref_landmarks: Optional[np.ndarray] = None  # (N,2)
        self._mp_face_mesh = None
        if MP_AVAILABLE:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def _detect_landmarks(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._mp_face_mesh is None:
            return None
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self._mp_face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        face = res.multi_face_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in face.landmark], dtype=np.float32)
        return pts

    def set_reference(self, ref_bgr: np.ndarray) -> bool:
        try:
            ref = cv2.resize(ref_bgr, self.target_size)
            lm = self._detect_landmarks(ref)
            if lm is None or lm.shape[0] < 10:
                return False
            self.ref_img = ref
            self.ref_landmarks = lm
            return True
        except Exception:
            return False

    def reenact(self, driving_bgr: np.ndarray) -> np.ndarray:
        if self.ref_img is None or self.ref_landmarks is None:
            return driving_bgr
        drv = cv2.resize(driving_bgr, self.target_size)
        drv_lm = self._detect_landmarks(drv)
        if drv_lm is None or drv_lm.shape[0] != self.ref_landmarks.shape[0]:
            return drv
        # Estimate global similarity transform from ref->drv landmarks
        try:
            M, _ = cv2.estimateAffinePartial2D(self.ref_landmarks, drv_lm, method=cv2.LMEDS)
            if M is None:
                return drv
            warped = cv2.warpAffine(self.ref_img, M, self.target_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return warped
        except Exception:
            return drv
