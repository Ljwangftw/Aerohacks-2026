"""
visualization.py — Debug overlays for camera frames.

Draws LED detection results, position info, and safety zone status
on camera frames for real-time monitoring.
"""

import cv2


def draw_debug_overlay(frame, detection_result, camera_name: str = "",
                       position_3d=None):
    """Draw detection results and position info on a camera frame.

    Args:
        frame: BGR image (modified in place)
        detection_result: DetectionResult from detection.py
        camera_name: label for display
        position_3d: [x, y, z] 3D position estimate (optional)
    """
    if detection_result is None:
        return

    h, w = frame.shape[:2]

    # Camera label
    cv2.putText(frame, f"Cam: {camera_name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if detection_result.detected and detection_result.centroid is not None:
        cx, cy = detection_result.centroid

        # Draw crosshair at centroid
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2)

        # Draw circle around detection
        radius = max(10, int(detection_result.area ** 0.5))
        cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)

        # Draw contour
        if detection_result.contour is not None:
            cv2.drawContours(frame, [detection_result.contour], -1,
                             (0, 255, 255), 1)

        # Centroid text
        cv2.putText(frame, f"({cx}, {cy})", (cx + 15, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Area text
        cv2.putText(frame, f"area={detection_result.area:.0f}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

        # Status
        cv2.putText(frame, "TRACKING", (w - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # No detection
        cv2.putText(frame, "LOST", (w - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 3D position overlay if available
    if position_3d is not None:
        pos_text = f"3D: ({position_3d[0]:.3f}, {position_3d[1]:.3f}, {position_3d[2]:.3f})"
        cv2.putText(frame, pos_text, (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
