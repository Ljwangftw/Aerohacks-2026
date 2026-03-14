"""
detection.py — LED detection using HSV color thresholding.

Finds the drone's LED in a camera frame and returns its pixel centroid.
Uses classical CV: blur → HSV convert → threshold → morphology → contour → centroid.
"""

import cv2
import numpy as np


class DetectionResult:
    """Result of LED detection in a single frame."""

    def __init__(self, detected: bool, centroid=None, area: float = 0.0,
                 contour=None, mask=None):
        self.detected = detected       # True if LED was found
        self.centroid = centroid        # (u, v) pixel coords or None
        self.area = area               # contour area in pixels^2
        self.contour = contour         # raw contour for debug drawing
        self.mask = mask               # binary mask for debug display


class LEDDetector:
    """Detects a colored LED in a BGR frame using HSV thresholding.

    Usage:
        det = LEDDetector(hsv_lower=[35,100,100], hsv_upper=[85,255,255])
        result = det.detect(frame)
        if result.detected:
            print(result.centroid)  # (u, v)
    """

    def __init__(self, hsv_lower: list, hsv_upper: list,
                 blur_kernel: int = 5,
                 min_area: float = 30,
                 max_area: float = 5000,
                 morph_kernel: int = 3):
        # HSV range for the LED color
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)

        # Processing parameters
        self.blur_kernel = blur_kernel
        self.min_area = min_area
        self.max_area = max_area
        self.morph_kernel = morph_kernel

        # Morphological structuring element
        self._morph_element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )

    def detect(self, frame: np.ndarray, roi=None) -> DetectionResult:
        """Detect the LED in a BGR frame.

        Args:
            frame: BGR image (numpy array).
            roi: optional (x, y, w, h) region of interest to restrict search.

        Returns:
            DetectionResult with centroid in FULL-FRAME coordinates.
        """
        if frame is None:
            return DetectionResult(detected=False)

        # If ROI is provided, crop to that region (but remember offset)
        x_offset, y_offset = 0, 0
        if roi is not None:
            rx, ry, rw, rh = roi
            # Clamp ROI to frame bounds
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, frame.shape[1] - rx)
            rh = min(rh, frame.shape[0] - ry)
            if rw <= 0 or rh <= 0:
                return DetectionResult(detected=False)
            frame = frame[ry:ry+rh, rx:rx+rw]
            x_offset, y_offset = rx, ry

        # Step 1: Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)

        # Step 2: Convert BGR → HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Step 3: Threshold for LED color
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Step 4: Morphological opening to clean up noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_element)

        # Step 5: Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return DetectionResult(detected=False, mask=mask)

        # Step 6: Filter by area and pick the best candidate
        best_contour = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                if area > best_area:
                    best_area = area
                    best_contour = cnt

        if best_contour is None:
            return DetectionResult(detected=False, mask=mask)

        # Step 7: Compute centroid using moments
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return DetectionResult(detected=False, mask=mask)

        cx = int(M["m10"] / M["m00"]) + x_offset
        cy = int(M["m01"] / M["m00"]) + y_offset

        return DetectionResult(
            detected=True,
            centroid=(cx, cy),
            area=best_area,
            contour=best_contour,
            mask=mask,
        )

    def update_thresholds(self, hsv_lower: list, hsv_upper: list):
        """Update HSV thresholds at runtime (for live tuning)."""
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
