"""
test_detection.py — Unit tests for LED detection module.

Tests HSV thresholding, contour filtering, and centroid computation
using synthetic images (no camera needed).
"""

import sys
import os
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.detection import LEDDetector, DetectionResult


# ── Helpers ────────────────────────────────────────────────────────

def make_blank_frame(width=640, height=480):
    """Create a black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def draw_green_circle(frame, center, radius=15):
    """Draw a bright green circle (simulates green LED) on a BGR frame."""
    # Bright green in BGR
    cv2.circle(frame, center, radius, (0, 255, 0), -1)
    return frame


def draw_blue_circle(frame, center, radius=15):
    """Draw a bright blue circle on a BGR frame."""
    cv2.circle(frame, center, radius, (255, 0, 0), -1)
    return frame


# ── Detector Fixture ──────────────────────────────────────────────

@pytest.fixture
def green_detector():
    """LEDDetector configured for green LED detection."""
    return LEDDetector(
        hsv_lower=[35, 100, 100],
        hsv_upper=[85, 255, 255],
        blur_kernel=5,
        min_area=30,
        max_area=5000,
        morph_kernel=3,
    )


# ── Tests ─────────────────────────────────────────────────────────

class TestLEDDetection:
    """Tests for basic LED detection correctness."""

    def test_detect_green_led_center(self, green_detector):
        """Green circle at frame center should be detected."""
        frame = make_blank_frame()
        draw_green_circle(frame, (320, 240))
        result = green_detector.detect(frame)

        assert result.detected is True
        assert result.centroid is not None
        # Centroid should be near (320, 240) within a few pixels
        assert abs(result.centroid[0] - 320) < 5
        assert abs(result.centroid[1] - 240) < 5

    def test_detect_green_led_corner(self, green_detector):
        """Green circle near corner should still be detected."""
        frame = make_blank_frame()
        draw_green_circle(frame, (50, 50))
        result = green_detector.detect(frame)

        assert result.detected is True
        assert abs(result.centroid[0] - 50) < 5
        assert abs(result.centroid[1] - 50) < 5

    def test_no_detection_on_blank_frame(self, green_detector):
        """Blank black frame should yield no detection."""
        frame = make_blank_frame()
        result = green_detector.detect(frame)

        assert result.detected is False
        assert result.centroid is None

    def test_no_detection_wrong_color(self, green_detector):
        """Blue circle should NOT be detected by green detector."""
        frame = make_blank_frame()
        draw_blue_circle(frame, (320, 240))
        result = green_detector.detect(frame)

        assert result.detected is False

    def test_none_frame_returns_not_detected(self, green_detector):
        """Passing None should return detected=False gracefully."""
        result = green_detector.detect(None)
        assert result.detected is False

    def test_contour_area_reported(self, green_detector):
        """Detected circle should have nonzero area."""
        frame = make_blank_frame()
        draw_green_circle(frame, (200, 200), radius=20)
        result = green_detector.detect(frame)

        assert result.detected is True
        assert result.area > 0

    def test_too_small_blob_rejected(self, green_detector):
        """Tiny circle (area < min_area) should be rejected."""
        frame = make_blank_frame()
        # Radius 2 → area ≈ 12 px², below min_area=30
        draw_green_circle(frame, (320, 240), radius=2)
        result = green_detector.detect(frame)

        assert result.detected is False

    def test_roi_detection(self, green_detector):
        """Detection within ROI should return centroid in full-frame coords."""
        frame = make_blank_frame()
        draw_green_circle(frame, (400, 300), radius=15)

        # ROI around the circle
        roi = (350, 250, 100, 100)
        result = green_detector.detect(frame, roi=roi)

        assert result.detected is True
        # Centroid should be in full-frame coords, near (400, 300)
        assert abs(result.centroid[0] - 400) < 10
        assert abs(result.centroid[1] - 300) < 10

    def test_roi_missing_led_returns_not_detected(self, green_detector):
        """ROI that doesn't contain the LED → no detection."""
        frame = make_blank_frame()
        draw_green_circle(frame, (400, 300))

        # ROI far from the circle
        roi = (0, 0, 100, 100)
        result = green_detector.detect(frame, roi=roi)

        assert result.detected is False

    def test_multiple_blobs_picks_largest(self, green_detector):
        """When multiple blobs are present, the largest should be chosen."""
        frame = make_blank_frame()
        draw_green_circle(frame, (100, 100), radius=8)   # small
        draw_green_circle(frame, (400, 300), radius=25)   # large

        result = green_detector.detect(frame)

        assert result.detected is True
        # Should pick the larger circle at (400, 300)
        assert abs(result.centroid[0] - 400) < 10
        assert abs(result.centroid[1] - 300) < 10

    def test_update_thresholds(self, green_detector):
        """Updating thresholds at runtime should take effect."""
        frame = make_blank_frame()
        draw_blue_circle(frame, (320, 240))

        # Green detector shouldn't find blue
        result = green_detector.detect(frame)
        assert result.detected is False

        # Switch to blue thresholds
        green_detector.update_thresholds([100, 100, 100], [130, 255, 255])
        result = green_detector.detect(frame)
        assert result.detected is True
