"""Face Alignment Utilities

Provides lightweight similarity transform alignment between detected facial landmarks and
canonical template. Designed to produce a stable, canonical (e.g., 256x256 or 512x512)
face crop plus forward / inverse transforms for later compositing.

We deliberately keep dependencies minimal (NumPy + OpenCV). If dlib or mediapipe is
introduced later for higher‑fidelity landmarks, this module's interface should continue
working as long as the landmark arrays are (N,2).
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

# A sparse canonical landmark template (normalized) for similarity alignment.
# These can be refined; using a 5-point template (e.g., eyes, nose tip, mouth corners)
# scaled into a 0..1 box then later scaled to target_size.
CANONICAL_5PT = np.array([
    [0.30, 0.35],  # left eye
    [0.70, 0.35],  # right eye
    [0.50, 0.55],  # nose
    [0.35, 0.75],  # left mouth
    [0.65, 0.75],  # right mouth
], dtype=np.float32)


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate 2x3 similarity (affine) transform mapping src -> dst and its inverse.

    src: (N,2) source landmarks
    dst: (N,2) destination landmarks
    returns (M, M_inv) where M is 2x3 affine matrix and M_inv its inverse.
    """
    if src.shape[0] != dst.shape[0] or src.shape[0] < 2:
        raise ValueError("Need at least 2 corresponding points for similarity transform")
    # Use estimateAffinePartial2D to constrain to similarity (rotation+scale+translation)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("Failed to estimate similarity transform")
    # Build 3x3 for inverse
    M3 = np.vstack([M, [0,0,1]])
    M_inv = np.linalg.inv(M3)[:2]
    return M.astype(np.float32), M_inv.astype(np.float32)


def align_face(image: np.ndarray,
               landmarks: np.ndarray,
               target_size: int = 512,
               canonical_template: np.ndarray = CANONICAL_5PT) -> Dict[str, Any]:
    """Align face to canonical space.

    landmarks: (N,2) in original image pixel coords.
    canonical_template: (M,2) normalized 0..1 template. We'll map subset of detected
                        landmarks (5 chosen points) to this template scaled to target size.

    Returns dict:
      aligned_image: (target_size,target_size,3) BGR
      M: forward affine 2x3 (original -> aligned)
      M_inv: inverse affine 2x3 (aligned -> original)
      aligned_landmarks: landmarks transformed into aligned space
    """
    h, w = image.shape[:2]
    # For now expect user supplies a 5-point subset matching template ordering.
    # Later: implement selection from dense landmark set based on semantic indices.
    if landmarks.shape[0] != canonical_template.shape[0]:
        raise ValueError("Provided landmarks count mismatch canonical template size (expected 5)")
    dst = canonical_template * target_size
    M, M_inv = estimate_similarity_transform(landmarks.astype(np.float32), dst.astype(np.float32))
    aligned = cv2.warpAffine(image, M, (target_size, target_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # Transform all provided landmarks into aligned space (optional)
    lm_h = np.hstack([landmarks, np.ones((landmarks.shape[0],1), dtype=np.float32)])
    aligned_lm = (M @ lm_h.T).T
    return {
        'aligned_image': aligned,
        'M': M,
        'M_inv': M_inv,
        'aligned_landmarks': aligned_lm,
    }


def gaussian_face_mask(size: int = 512, sigma: float = 0.22, feather: float = 0.05) -> np.ndarray:
    """Generate a soft elliptical / gaussian face mask in canonical aligned space.

    sigma: relative spread of gaussian (fraction of size)
    feather: extra smoothing on edges
    returns mask (size,size) float32 in [0,1]
    """
    ax = np.linspace(-1,1,size)
    xx, yy = np.meshgrid(ax, ax)
    r2 = (xx**2 + (yy*1.15)**2)  # slightly stretch vertical for face aspect
    g = np.exp(-r2 / (2*(sigma**2)))
    # Normalize
    g = g / g.max()
    # Feather by raising power (softens edges) + blur
    g = g ** (1 + feather*5)
    g = cv2.GaussianBlur(g.astype(np.float32), (0,0), size*0.01)
    return np.clip(g, 0, 1).astype(np.float32)

__all__ = [
    'align_face', 'estimate_similarity_transform', 'gaussian_face_mask', 'CANONICAL_5PT'
]
