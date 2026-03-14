"""
estimator.py — State estimator for 3D drone position.

Smooths noisy triangulated positions and provides velocity estimates.
Starts with EMA (simple), can upgrade to Kalman later.
"""

import time
import numpy as np
from src.utils.math_helpers import ema_update


class PositionEstimator:
    """Smoothed 3D position and velocity estimator.

    Uses exponential moving average filtering to reduce noise from
    triangulation. Also computes velocity via finite differences.
    """

    def __init__(self, alpha: float = 0.3, max_jump_m: float = 0.15):
        """
        Args:
            alpha: EMA smoothing factor (0 = very smooth, 1 = raw). 0.3 is a
                   good starting point for ~20 Hz updates.
            max_jump_m: reject position updates that jump more than this
                        many meters from the previous estimate.
        """
        self.alpha = alpha
        self.max_jump_m = max_jump_m

        # Smoothed state
        self.position = None       # [x, y, z] in meters, or None if not initialized
        self.velocity = [0.0, 0.0, 0.0]  # [vx, vy, vz] in m/s

        # Tracking info
        self._last_update_time = None
        self._initialized = False
        self._frames_since_update = 0
        self._reliable = False

    def update(self, raw_position: list, confidence: float) -> dict:
        """Update the state estimate with a new triangulated position.

        Args:
            raw_position: [x, y, z] in meters from triangulation, or None.
            confidence: 0-1 from triangulator.

        Returns:
            dict with:
              - "position": smoothed [x, y, z]
              - "velocity": estimated [vx, vy, vz]
              - "reliable": bool
              - "frames_stale": int
        """
        now = time.time()

        # No measurement available
        if raw_position is None or confidence <= 0:
            self._frames_since_update += 1
            self._reliable = self._frames_since_update < 5
            return self._make_result()

        raw = np.array(raw_position, dtype=float)

        # First measurement — initialize directly
        if not self._initialized:
            self.position = raw.tolist()
            self.velocity = [0.0, 0.0, 0.0]
            self._last_update_time = now
            self._initialized = True
            self._frames_since_update = 0
            self._reliable = True
            return self._make_result()

        pos = np.array(self.position)

        # Reject outlier jumps (likely detection errors)
        jump = np.linalg.norm(raw - pos)
        if jump > self.max_jump_m:
            self._frames_since_update += 1
            self._reliable = self._frames_since_update < 5
            return self._make_result()

        # Compute dt for velocity estimation
        dt = now - self._last_update_time if self._last_update_time else 0.033
        dt = max(dt, 0.001)  # prevent division by zero

        # EMA filter for position
        new_pos = [
            ema_update(pos[0], raw[0], self.alpha),
            ema_update(pos[1], raw[1], self.alpha),
            ema_update(pos[2], raw[2], self.alpha),
        ]

        # Velocity via finite difference on smoothed position
        self.velocity = [
            (new_pos[0] - self.position[0]) / dt,
            (new_pos[1] - self.position[1]) / dt,
            (new_pos[2] - self.position[2]) / dt,
        ]

        self.position = new_pos
        self._last_update_time = now
        self._frames_since_update = 0
        self._reliable = True

        return self._make_result()

    def _make_result(self) -> dict:
        return {
            "position": self.position,
            "velocity": self.velocity,
            "reliable": self._reliable,
            "frames_stale": self._frames_since_update,
        }

    def get_position(self) -> list:
        """Return current smoothed position, or None if not initialized."""
        return self.position

    def get_velocity(self) -> list:
        """Return current velocity estimate."""
        return self.velocity

    @property
    def is_reliable(self) -> bool:
        return self._reliable

    @property
    def is_initialized(self) -> bool:
        return self._initialized
