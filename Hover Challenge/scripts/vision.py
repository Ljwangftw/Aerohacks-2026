# vision.py
# Computer vision pipeline — camera-based drone position estimation.
# Stub implementation used during Test 1 (comms only).
# Full implementation to be wired in for Test 2.
#
# Architecture:
#   Two USB cameras provide stereo-ish coverage of the flight cage.
#   The drone's LED indicator is used as the primary detection target —
#   more robust than body detection under variable lighting.
#   Detected pixel positions are mapped to real-world coordinates via
#   camera calibration or a known-geometry homography.
#   A Kalman filter smooths the raw position estimate before it reaches
#   the control loop.

import cv2
import numpy as np
from config import TARGET_ALTITUDE


# ---------------------------------------------------------------------------
# Stub — used during Test 1 when CV is not yet active
# ---------------------------------------------------------------------------

def get_altitude_stub() -> float:
    """
    Returns the target altitude so the altitude PID sees zero error and
    outputs ~0 correction. Drone hovers at BASELINE_THRUST only.
    Replace the get_altitude reference in controller.run_hover() with
    get_altitude (below) once CV is validated.
    """
    return TARGET_ALTITUDE


# ---------------------------------------------------------------------------
# LED detection helpers
# ---------------------------------------------------------------------------

# HSV bounds for the drone's LED indicator — tune these during Test 2 setup.
# Use the led_calibration() utility below to find the right values.
LED_HSV_LOWER = np.array([90, 150, 150])   # default: blue-ish
LED_HSV_UPPER = np.array([130, 255, 255])


def detect_led(frame: np.ndarray) -> tuple[int, int] | None:
    """
    Detect the drone LED in a single camera frame.
    Returns (cx, cy) pixel centroid of the largest matching blob, or None.

    HSV thresholding is used instead of BGR — HSV is significantly more
    robust to lighting intensity changes since hue and saturation are
    separated from brightness.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LED_HSV_LOWER, LED_HSV_UPPER)

    # Morphological open removes small noise blobs before contour detection
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def led_calibration(camera_index: int = 0):
    """
    Interactive utility to find the correct HSV bounds for the drone's LED.
    Run this standalone before Test 2: `python -c "from vision import led_calibration; led_calibration()"`
    Adjust trackbars until only the LED is visible in the mask window, then
    update LED_HSV_LOWER / LED_HSV_UPPER above.
    """
    cap = cv2.VideoCapture(camera_index)

    cv2.namedWindow("Calibration")
    cv2.createTrackbar("H Low",  "Calibration",  90, 179, lambda x: None)
    cv2.createTrackbar("S Low",  "Calibration", 150, 255, lambda x: None)
    cv2.createTrackbar("V Low",  "Calibration", 150, 255, lambda x: None)
    cv2.createTrackbar("H High", "Calibration", 130, 179, lambda x: None)
    cv2.createTrackbar("S High", "Calibration", 255, 255, lambda x: None)
    cv2.createTrackbar("V High", "Calibration", 255, 255, lambda x: None)

    print("[VISION] Calibration mode. Press 'q' to quit and print final values.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hl = cv2.getTrackbarPos("H Low",  "Calibration")
        sl = cv2.getTrackbarPos("S Low",  "Calibration")
        vl = cv2.getTrackbarPos("V Low",  "Calibration")
        hh = cv2.getTrackbarPos("H High", "Calibration")
        sh = cv2.getTrackbarPos("S High", "Calibration")
        vh = cv2.getTrackbarPos("V High", "Calibration")

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask",  mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\nLED_HSV_LOWER = np.array([{hl}, {sl}, {vl}])")
            print(f"LED_HSV_UPPER = np.array([{hh}, {sh}, {vh}])")
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Position estimation
# ---------------------------------------------------------------------------

class PositionEstimator:
    """
    Estimates 3D drone position (x, y, z) in meters within the flight cage
    using two camera feeds.

    Approach (Test 2):
      - Camera 0 mounted to give X/Z view of cage
      - Camera 1 mounted to give Y/Z view of cage
      - Pixel coordinates from LED detection mapped to real-world via
        a per-camera homography (computed from known cage geometry markers)
      - Kalman filter smooths noisy estimates and provides velocity

    For Test 2 setup, call calibrate() with the cage corner pixel coordinates
    before starting the hover loop.
    """

    def __init__(self):
        self.cap0 = None
        self.cap1 = None

        # Homography matrices — set by calibrate(), maps pixel -> meters
        self._H0 = None  # camera 0: maps to (x, z) plane
        self._H1 = None  # camera 1: maps to (y, z) plane

        # Kalman filter for (x, y, z) position smoothing
        # State: [x, y, z, vx, vy, vz] — position + velocity
        self._kf = self._build_kalman()
        self._kf_initialized = False

    def open_cameras(self, cam0_index: int = 0, cam1_index: int = 1):
        """Open both USB camera feeds."""
        self.cap0 = cv2.VideoCapture(cam0_index)
        self.cap1 = cv2.VideoCapture(cam1_index)
        if not self.cap0.isOpened() or not self.cap1.isOpened():
            raise RuntimeError("[VISION] Failed to open one or both cameras.")
        print(f"[VISION] Cameras opened: indices {cam0_index}, {cam1_index}")

    def calibrate(
        self,
        cage_pixels_cam0: list[tuple],
        cage_pixels_cam1: list[tuple],
        cage_world_xz: list[tuple],
        cage_world_yz: list[tuple],
    ):
        """
        Compute homographies from known cage corner correspondences.

        cage_pixels_cam0: 4 pixel coords of cage corners as seen by camera 0
        cage_world_xz:    corresponding real-world (x, z) coords in meters
        (same pattern for cam1 / yz plane)

        Call this once at startup during Test 2 before the hover loop.
        """
        src0 = np.array(cage_pixels_cam0, dtype=np.float32)
        dst0 = np.array(cage_world_xz,    dtype=np.float32)
        self._H0, _ = cv2.findHomography(src0, dst0)

        src1 = np.array(cage_pixels_cam1, dtype=np.float32)
        dst1 = np.array(cage_world_yz,    dtype=np.float32)
        self._H1, _ = cv2.findHomography(src1, dst1)

        print("[VISION] Homographies computed. Position estimation active.")

    def get_altitude(self) -> float:
        """
        Returns estimated drone altitude (z) in meters.
        This is the function to inject into controller.run_hover().

        Falls back to TARGET_ALTITUDE if cameras aren't ready or detection fails,
        so the altitude PID outputs zero correction rather than garbage.
        """
        pos = self._estimate_position()
        if pos is None:
            return TARGET_ALTITUDE
        return pos[2]  # z component

    def get_position(self) -> tuple[float, float, float] | None:
        """Returns full (x, y, z) estimate in meters, or None on failure."""
        return self._estimate_position()

    def _estimate_position(self) -> tuple[float, float, float] | None:
        if self.cap0 is None or self.cap1 is None:
            return None
        if self._H0 is None or self._H1 is None:
            return None

        ret0, frame0 = self.cap0.read()
        ret1, frame1 = self.cap1.read()
        if not ret0 or not ret1:
            return None

        pt0 = detect_led(frame0)
        pt1 = detect_led(frame1)
        if pt0 is None or pt1 is None:
            return None

        # Map pixel detections through homographies to real-world coords
        xz = cv2.perspectiveTransform(
            np.array([[[pt0[0], pt0[1]]]], dtype=np.float32), self._H0
        )[0][0]
        yz = cv2.perspectiveTransform(
            np.array([[[pt1[0], pt1[1]]]], dtype=np.float32), self._H1
        )[0][0]

        raw = np.array([xz[0], yz[0], (xz[1] + yz[1]) / 2.0])  # x, y, z (z averaged)

        # Feed into Kalman filter for smoothed estimate
        smoothed = self._kalman_update(raw)
        return (smoothed[0], smoothed[1], smoothed[2])

    def _build_kalman(self) -> cv2.KalmanFilter:
        """
        6-state Kalman filter: [x, y, z, vx, vy, vz]
        Constant velocity motion model.
        Measurement: [x, y, z] only (no direct velocity measurement)
        """
        kf = cv2.KalmanFilter(6, 3)
        dt = 1.0 / 20.0  # matches CONTROL_HZ

        # State transition: position += velocity * dt
        kf.transitionMatrix = np.array([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1],
        ], dtype=np.float32)

        # Measurement maps [x,y,z,vx,vy,vz] -> [x,y,z]
        kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)

        kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 1e-4
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        kf.errorCovPost        = np.eye(6, dtype=np.float32)

        return kf

    def _kalman_update(self, measurement: np.ndarray) -> np.ndarray:
        if not self._kf_initialized:
            # Seed state with first measurement to avoid transient at startup
            self._kf.statePost = np.array(
                [measurement[0], measurement[1], measurement[2], 0, 0, 0],
                dtype=np.float32
            ).reshape(6, 1)
            self._kf_initialized = True

        self._kf.predict()
        corrected = self._kf.correct(measurement.astype(np.float32).reshape(3, 1))
        return corrected.flatten()

    def release(self):
        """Release camera resources on shutdown."""
        if self.cap0:
            self.cap0.release()
        if self.cap1:
            self.cap1.release()
        print("[VISION] Cameras released.")


# ---------------------------------------------------------------------------
# Module-level estimator instance — imported by controller.py
# ---------------------------------------------------------------------------
estimator = PositionEstimator()
