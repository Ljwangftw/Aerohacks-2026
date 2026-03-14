"""
tune_hsv.py — Interactive HSV threshold tuner for LED detection.

Opens both cameras and provides trackbar sliders to adjust HSV thresholds
in real time. Use this at the venue to find the best thresholds for
the drone's LED color under actual lighting conditions.

Run: python -m scripts.tune_hsv
  or: python scripts/tune_hsv.py
"""

import cv2
import numpy as np
import sys
import os

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.config import load_config


def nothing(x):
    """Callback for trackbar (required by OpenCV but unused)."""
    pass


def main():
    cfg = load_config("default")
    det_cfg = cfg["detection"]

    # Open cameras
    cam_id_left = cfg["cameras"]["left"]["id"]
    cam_id_right = cfg["cameras"]["right"]["id"]

    cap_left = cv2.VideoCapture(cam_id_left)
    cap_right = cv2.VideoCapture(cam_id_right)

    if not cap_left.isOpened():
        print(f"Could not open left camera (device {cam_id_left})")
        return
    if not cap_right.isOpened():
        print(f"Could not open right camera (device {cam_id_right})")
        # Continue with just left camera
        cap_right = None

    # Create control window with HSV trackbars
    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)

    # Initial values from config
    h_lo, s_lo, v_lo = det_cfg["hsv_lower"]
    h_hi, s_hi, v_hi = det_cfg["hsv_upper"]

    cv2.createTrackbar("H Low", "HSV Tuner", h_lo, 179, nothing)
    cv2.createTrackbar("H High", "HSV Tuner", h_hi, 179, nothing)
    cv2.createTrackbar("S Low", "HSV Tuner", s_lo, 255, nothing)
    cv2.createTrackbar("S High", "HSV Tuner", s_hi, 255, nothing)
    cv2.createTrackbar("V Low", "HSV Tuner", v_lo, 255, nothing)
    cv2.createTrackbar("V High", "HSV Tuner", v_hi, 255, nothing)

    print("=" * 50)
    print("  HSV Threshold Tuner")
    print("  Adjust sliders until LED is cleanly detected")
    print("  Press 'p' to print current values")
    print("  Press 'q' to quit")
    print("=" * 50)

    while True:
        # Read trackbar values
        h_lo = cv2.getTrackbarPos("H Low", "HSV Tuner")
        h_hi = cv2.getTrackbarPos("H High", "HSV Tuner")
        s_lo = cv2.getTrackbarPos("S Low", "HSV Tuner")
        s_hi = cv2.getTrackbarPos("S High", "HSV Tuner")
        v_lo = cv2.getTrackbarPos("V Low", "HSV Tuner")
        v_hi = cv2.getTrackbarPos("V High", "HSV Tuner")

        lower = np.array([h_lo, s_lo, v_lo])
        upper = np.array([h_hi, s_hi, v_hi])

        # Process left camera
        ret_l, frame_l = cap_left.read()
        if ret_l and frame_l is not None:
            _process_and_show(frame_l, lower, upper, "Left Camera", "Left Mask")

        # Process right camera
        if cap_right is not None:
            ret_r, frame_r = cap_right.read()
            if ret_r and frame_r is not None:
                _process_and_show(frame_r, lower, upper, "Right Camera", "Right Mask")

        # Handle keyboard
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            print(f"\n  hsv_lower: [{h_lo}, {s_lo}, {v_lo}]")
            print(f"  hsv_upper: [{h_hi}, {s_hi}, {v_hi}]")
            print(f"  (Copy these into configs/default.yaml)\n")

    # Cleanup
    cap_left.release()
    if cap_right is not None:
        cap_right.release()
    cv2.destroyAllWindows()

    # Print final values
    print(f"\nFinal HSV thresholds:")
    print(f"  hsv_lower: [{h_lo}, {s_lo}, {v_lo}]")
    print(f"  hsv_upper: [{h_hi}, {s_hi}, {v_hi}]")


def _process_and_show(frame, lower, upper, window_name, mask_name):
    """Apply HSV threshold to a frame and display results."""
    # Blur
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # Threshold
    mask = cv2.inRange(hsv, lower, upper)
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours and draw the largest one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    display = frame.copy()

    if contours:
        # Sort by area, draw the largest
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 10:  # minimum area threshold
            cv2.drawContours(display, [largest], -1, (0, 255, 0), 2)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawMarker(display, (cx, cy), (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display, f"({cx},{cy}) area={area:.0f}",
                            (cx + 15, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(window_name, display)
    cv2.imshow(mask_name, mask)


if __name__ == "__main__":
    main()
