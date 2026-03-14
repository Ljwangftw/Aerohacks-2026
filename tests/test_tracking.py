"""
test_tracking.py — Unit tests for LED tracking and state estimation.

Tests the LEDTracker (ROI prediction) and PositionEstimator (EMA filtering).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.tracking import LEDTracker
from src.state.estimator import PositionEstimator


# ── LEDTracker Tests ──────────────────────────────────────────────

class TestLEDTracker:
    """Tests for temporal LED tracking with ROI prediction."""

    def test_first_detection_initializes(self):
        """First detection should initialize the tracker."""
        tracker = LEDTracker(roi_padding=60)
        result = tracker.update((320, 240))

        assert result["tracking"] is True
        assert result["predicted"] == (320, 240)
        assert result["roi"] is not None

    def test_roi_centered_on_detection(self):
        """ROI should be centered on the detected position."""
        tracker = LEDTracker(roi_padding=50)
        result = tracker.update((200, 150))

        roi = result["roi"]
        # ROI should be (200-50, 150-50, 100, 100)
        assert roi[0] == 150  # x
        assert roi[1] == 100  # y
        assert roi[2] == 100  # width
        assert roi[3] == 100  # height

    def test_prediction_on_loss(self):
        """When detection is lost, tracker should predict forward."""
        tracker = LEDTracker(roi_padding=60, max_lost_frames=5)

        # Feed a few detections to build velocity
        tracker.update((100, 100))
        tracker.update((110, 100))  # moving right at 10 px/frame
        tracker.update((120, 100))

        # Now lose detection
        result = tracker.update(None)
        assert result["tracking"] is True
        # Should predict ~(130, 100) based on velocity
        assert result["predicted"] is not None
        assert result["predicted"][0] > 120  # predicted forward

    def test_lost_too_long_resets(self):
        """After max_lost_frames, tracking should be marked as lost."""
        tracker = LEDTracker(roi_padding=60, max_lost_frames=3)
        tracker.update((100, 100))

        # Lose detection for 4 frames (> max_lost=3)
        for _ in range(4):
            result = tracker.update(None)

        assert result["tracking"] is False
        assert result["roi"] is None  # no ROI when fully lost

    def test_recovery_after_loss(self):
        """Tracker should recover when detection resumes after loss."""
        tracker = LEDTracker(roi_padding=60, max_lost_frames=3)
        tracker.update((100, 100))

        # Lose for 2 frames (within limit)
        tracker.update(None)
        tracker.update(None)

        # Re-detect
        result = tracker.update((115, 100))
        assert result["tracking"] is True
        assert result["predicted"] == (115, 100)

    def test_no_detection_before_init(self):
        """Tracker with no initial detection should not crash."""
        tracker = LEDTracker()
        result = tracker.update(None)

        assert result["tracking"] is False
        assert result["predicted"] is None

    def test_reset_clears_state(self):
        """Reset should clear all tracking state."""
        tracker = LEDTracker()
        tracker.update((100, 100))
        tracker.update((200, 200))
        tracker.reset()

        result = tracker.update(None)
        assert result["tracking"] is False
        assert result["predicted"] is None

    def test_wider_roi_when_lost(self):
        """ROI should grow wider with each frame of lost detection."""
        tracker = LEDTracker(roi_padding=50, max_lost_frames=5)
        tracker.update((200, 200))

        # First lost frame
        r1 = tracker.update(None)
        # Second lost frame
        r2 = tracker.update(None)

        # ROI width on second lost frame should be wider
        assert r2["roi"][2] > r1["roi"][2]


# ── PositionEstimator Tests ───────────────────────────────────────

class TestPositionEstimator:
    """Tests for EMA position filtering and velocity estimation."""

    def test_first_update_initializes(self):
        """First position update should initialize directly (no smoothing)."""
        est = PositionEstimator(alpha=0.3)
        result = est.update([0.5, 0.5, 0.5], confidence=1.0)

        assert result["reliable"] is True
        assert result["position"] == [0.5, 0.5, 0.5]
        assert est.is_initialized is True

    def test_ema_smoothing(self):
        """EMA should smooth a step change, not jump immediately."""
        est = PositionEstimator(alpha=0.3)
        est.update([0.0, 0.0, 0.0], confidence=1.0)

        # Big step to [1.0, 1.0, 1.0] — but within max_jump
        est.max_jump_m = 2.0  # allow big jumps for this test
        result = est.update([1.0, 1.0, 1.0], confidence=1.0)

        # With alpha=0.3, new position should be 0.3*1.0 + 0.7*0.0 = 0.3
        assert abs(result["position"][0] - 0.3) < 0.01
        assert abs(result["position"][1] - 0.3) < 0.01

    def test_outlier_rejection(self):
        """Position jumps > max_jump_m should be rejected."""
        est = PositionEstimator(alpha=0.3, max_jump_m=0.1)
        est.update([0.5, 0.5, 0.5], confidence=1.0)

        # Big jump (0.5m), should be rejected
        result = est.update([1.0, 0.5, 0.5], confidence=1.0)

        # Position should remain at [0.5, 0.5, 0.5] (unchanged)
        assert abs(result["position"][0] - 0.5) < 0.01

    def test_zero_confidence_ignored(self):
        """Updates with confidence=0 should not change the estimate."""
        est = PositionEstimator(alpha=0.3)
        est.update([0.5, 0.5, 0.5], confidence=1.0)

        result = est.update([0.8, 0.8, 0.8], confidence=0.0)

        # Should still be near initial position
        assert abs(result["position"][0] - 0.5) < 0.01

    def test_none_position_increments_stale(self):
        """None position should increment frames_stale counter."""
        est = PositionEstimator(alpha=0.3)
        est.update([0.5, 0.5, 0.5], confidence=1.0)

        result = est.update(None, confidence=0.0)
        assert result["frames_stale"] == 1

        result = est.update(None, confidence=0.0)
        assert result["frames_stale"] == 2

    def test_becomes_unreliable_after_lost_frames(self):
        """After 5+ frames without update, should be marked unreliable."""
        est = PositionEstimator(alpha=0.3, max_jump_m=0.15)
        est.update([0.5, 0.5, 0.5], confidence=1.0)

        # Lose for 6 frames
        for _ in range(6):
            result = est.update(None, confidence=0.0)

        assert result["reliable"] is False

    def test_velocity_estimated(self):
        """Velocity should be nonzero after two position updates."""
        est = PositionEstimator(alpha=1.0)  # alpha=1 for no smoothing
        est.update([0.0, 0.0, 0.0], confidence=1.0)

        import time
        time.sleep(0.05)  # small delay so dt > 0

        result = est.update([0.05, 0.0, 0.0], confidence=1.0)

        # Velocity in x should be positive (moved +0.05m in ~0.05s = ~1 m/s)
        assert result["velocity"][0] > 0.0

    def test_recovery_after_stale(self):
        """Estimator should recover and become reliable again."""
        est = PositionEstimator(alpha=0.3, max_jump_m=0.5)
        est.update([0.5, 0.5, 0.5], confidence=1.0)

        # Lose for a few frames
        for _ in range(3):
            est.update(None, confidence=0.0)

        # Recover with a nearby position
        result = est.update([0.52, 0.5, 0.5], confidence=1.0)
        assert result["reliable"] is True
        assert result["frames_stale"] == 0
