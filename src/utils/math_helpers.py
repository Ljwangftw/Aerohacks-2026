"""
math_helpers.py — Small math utilities used across the project.

Keeps control/vision code clean by centralizing clamp, rate-limit, etc.
"""

import numpy as np


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def rate_limit(current: float, previous: float, max_delta: float) -> float:
    """Limit how fast a value can change between ticks.

    Returns a value within [previous - max_delta, previous + max_delta].
    """
    delta = clamp(current - previous, -max_delta, max_delta)
    return previous + delta


def deadband(value: float, threshold: float) -> float:
    """Zero out values smaller than threshold (reduces jitter near target)."""
    if abs(value) < threshold:
        return 0.0
    return value


def normalize_angle(deg: float) -> float:
    """Wrap angle to [-180, 180] degrees."""
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def distance_3d(a, b) -> float:
    """Euclidean distance between two 3D points (lists or arrays)."""
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2))


def ema_update(old: float, new: float, alpha: float) -> float:
    """Exponential moving average update.

    alpha close to 1 → fast response (noisy).
    alpha close to 0 → slow response (smooth).
    """
    return alpha * new + (1.0 - alpha) * old
