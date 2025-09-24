"""Keypoint smoothing utilities (One Euro Filter)

Implements a per-dimension One Euro filter adapted for batched facial keypoints.
Reference: Casiez et al. "The One Euro Filter" (2012)
"""
from __future__ import annotations
import numpy as np
import math
from typing import Optional


def _exp_smoothing_factor(dt: float, cutoff: float) -> float:
    if cutoff <= 0:
        return 1.0  # no smoothing
    r = 2 * math.pi * cutoff * dt
    return r / (r + 1)


class OneEuroFilter:
    """One Euro filter for a single scalar signal."""
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: Optional[float] = None
        self._t_prev: Optional[float] = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x: float, t: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x
        dt = max(t - self._t_prev, 1e-6)
        # Derivative
        dx = (x - self._x_prev) / dt
        # Smooth derivative
        a_d = _exp_smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * (self._dx_prev if self._dx_prev is not None else 0.0)
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _exp_smoothing_factor(dt, cutoff)
        x_hat = a * x + (1 - a) * self._x_prev
        # Update state
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class KeypointOneEuro:
    """Apply One Euro filtering to a keypoint tensor shaped (1,K,C)."""
    def __init__(self, K: int = 21, C: int = 3, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.K = K
        self.C = C
        self.filters = [[OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(C)] for _ in range(K)]

    def reset(self):
        for row in self.filters:
            for f in row:
                f.reset()

    def filter(self, kp: np.ndarray, t: float) -> np.ndarray:
        if kp is None:
            return kp
        if not isinstance(kp, np.ndarray):
            kp = np.asarray(kp, dtype=np.float32)
        if kp.ndim != 3:
            return kp
        B, K, C = kp.shape
        if B != 1:
            # Only single-batch expected; operate on first
            kp = kp[:1]
        out = kp.copy()
        K_use = min(K, self.K)
        C_use = min(C, self.C)
        for i in range(K_use):
            for j in range(C_use):
                out[0, i, j] = self.filters[i][j](float(out[0, i, j]), t)
        return out

__all__ = ["OneEuroFilter", "KeypointOneEuro"]
