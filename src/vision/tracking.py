"""
tracking.py — Temporal tracking with ROI prediction.

Uses a simple Kalman-like prediction to set a region of interest (ROI)
around the expected LED position, reducing search area and false positives.
"""


class LEDTracker:
    """Tracks the LED position across frames using simple prediction.

    Maintains a predicted position based on velocity. When detection succeeds,
    updates the prediction. When detection fails, uses prediction to shrink
    the search ROI.
    """

    def __init__(self, roi_padding: int = 60, max_lost_frames: int = 5):
        """
        Args:
            roi_padding: pixels around predicted position for search ROI
            max_lost_frames: after this many missed frames, reset prediction
        """
        self.roi_padding = roi_padding
        self.max_lost_frames = max_lost_frames

        # State
        self._last_centroid = None  # (u, v)
        self._velocity = (0.0, 0.0)  # pixel velocity (du, dv) per frame
        self._lost_count = 0
        self._initialized = False

    def update(self, centroid) -> dict:
        """Update tracker with new detection result.

        Args:
            centroid: (u, v) pixel coords or None if not detected

        Returns:
            dict with:
              - "predicted": (u, v) predicted position
              - "roi": (x, y, w, h) for next detection call, or None
              - "tracking": bool — whether we have a valid track
        """
        if centroid is not None:
            if self._initialized and self._last_centroid is not None:
                # Update velocity estimate
                du = centroid[0] - self._last_centroid[0]
                dv = centroid[1] - self._last_centroid[1]
                # Smooth velocity with EMA
                alpha = 0.4
                self._velocity = (
                    alpha * du + (1 - alpha) * self._velocity[0],
                    alpha * dv + (1 - alpha) * self._velocity[1],
                )

            self._last_centroid = centroid
            self._lost_count = 0
            self._initialized = True

            # Build ROI around current position for next frame
            roi = self._build_roi(centroid)

            return {
                "predicted": centroid,
                "roi": roi,
                "tracking": True,
            }
        else:
            # Detection failed — use prediction
            self._lost_count += 1

            if not self._initialized or self._lost_count > self.max_lost_frames:
                # Lost for too long or never tracked — no ROI (full frame search)
                return {
                    "predicted": self._last_centroid,
                    "roi": None,
                    "tracking": False,
                }

            # Predict next position using velocity
            predicted = (
                int(self._last_centroid[0] + self._velocity[0]),
                int(self._last_centroid[1] + self._velocity[1]),
            )
            self._last_centroid = predicted

            # Wider ROI when lost (more uncertainty)
            wider_padding = self.roi_padding * (1 + self._lost_count * 0.5)
            roi = self._build_roi(predicted, int(wider_padding))

            return {
                "predicted": predicted,
                "roi": roi,
                "tracking": True,
            }

    def _build_roi(self, center, padding=None):
        """Build (x, y, w, h) ROI rectangle around center."""
        p = padding or self.roi_padding
        x = int(center[0] - p)
        y = int(center[1] - p)
        w = int(2 * p)
        h = int(2 * p)
        return (x, y, w, h)

    def reset(self):
        """Reset tracker state."""
        self._last_centroid = None
        self._velocity = (0.0, 0.0)
        self._lost_count = 0
        self._initialized = False
