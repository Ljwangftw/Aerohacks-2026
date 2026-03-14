"""
triangulation.py — Estimate 3D position from two camera views.

Supports two methods:
  1. "orthogonal" (recommended for quick bring-up):
     Camera 0 is front-facing → gives (x, z) in cage frame.
     Camera 1 is side-facing → gives (y, z) in cage frame.
     z is averaged from both.

  2. "stereo" (for higher accuracy after calibration):
     Uses full camera matrices + cv2.triangulatePoints().

The orthogonal method requires only a simple pixel-to-meters calibration
(place LED at known positions and record pixel coords).
"""

import numpy as np


class OrthogonalTriangulator:
    """Convert two 2D detections from orthogonal cameras into 3D cage coords.

    Assumes:
      - Camera left (index 0) is front-facing: pixel x → cage X, pixel y → cage Z
      - Camera right (index 1) is side-facing: pixel x → cage Y, pixel y → cage Z

    Calibration parameters (from config):
      - origin_px: pixel coordinate where cage origin (0,0,0) appears
      - scale_x/y/z: meters per pixel in each direction
    """

    def __init__(self, cam_left_cfg: dict, cam_right_cfg: dict):
        # Left camera: maps pixels to (x, z)
        self.left_origin = np.array(cam_left_cfg["origin_px"], dtype=float)
        self.left_scale_x = cam_left_cfg["scale_x"]
        self.left_scale_z = cam_left_cfg["scale_z"]

        # Right camera: maps pixels to (y, z)
        self.right_origin = np.array(cam_right_cfg["origin_px"], dtype=float)
        self.right_scale_y = cam_right_cfg["scale_y"]
        self.right_scale_z = cam_right_cfg["scale_z"]

    def triangulate(self, left_centroid, right_centroid) -> dict:
        """Compute 3D position from two centroids.

        Args:
            left_centroid: (u, v) pixel coords from left camera, or None
            right_centroid: (u, v) pixel coords from right camera, or None

        Returns:
            dict with:
              - "position": [x, y, z] in meters (cage frame), or None
              - "confidence": float 0-1 indicating reliability
              - "source": "both", "left_only", "right_only", or "none"
        """
        have_left = left_centroid is not None
        have_right = right_centroid is not None

        if not have_left and not have_right:
            return {"position": None, "confidence": 0.0, "source": "none"}

        x, y, z = None, None, None
        z_values = []

        if have_left:
            lu, lv = left_centroid
            # Convert left camera pixel → cage x and z
            x = (lu - self.left_origin[0]) * self.left_scale_x
            z_left = (lv - self.left_origin[1]) * self.left_scale_z
            z_values.append(z_left)

        if have_right:
            ru, rv = right_centroid
            # Convert right camera pixel → cage y and z
            y = (ru - self.right_origin[0]) * self.right_scale_y
            z_right = (rv - self.right_origin[1]) * self.right_scale_z
            z_values.append(z_right)

        # Average z from both cameras when available
        z = float(np.mean(z_values))

        # Determine source and confidence
        if have_left and have_right:
            source = "both"
            confidence = 1.0
            # If z estimates disagree strongly, lower confidence
            if len(z_values) == 2:
                z_diff = abs(z_values[0] - z_values[1])
                if z_diff > 0.05:  # >5cm disagreement
                    confidence = max(0.3, 1.0 - z_diff * 5)
        elif have_left:
            source = "left_only"
            confidence = 0.5
            # y is unknown — use last known or cage center
            if y is None:
                y = 0.5  # default to cage center
        else:
            source = "right_only"
            confidence = 0.5
            # x is unknown — use last known or cage center
            if x is None:
                x = 0.5

        position = [float(x), float(y), float(z)]

        return {
            "position": position,
            "confidence": confidence,
            "source": source,
        }


class StereoTriangulator:
    """Full stereo triangulation using calibrated camera matrices.

    Use this if you have time to do proper stereo calibration with a checkerboard.
    For hackathon, OrthogonalTriangulator is faster to set up.
    """

    def __init__(self, P_left: np.ndarray, P_right: np.ndarray):
        """
        Args:
            P_left: 3x4 projection matrix for left camera
            P_right: 3x4 projection matrix for right camera
        """
        self.P_left = P_left
        self.P_right = P_right

    def triangulate(self, left_centroid, right_centroid) -> dict:
        """Triangulate 3D point from matched 2D points using DLT.

        Both centroids must be provided for stereo to work.
        """
        import cv2

        if left_centroid is None or right_centroid is None:
            return {"position": None, "confidence": 0.0, "source": "none"}

        # Prepare points as 2xN float arrays
        pts_left = np.array([[left_centroid[0]], [left_centroid[1]]], dtype=np.float64)
        pts_right = np.array([[right_centroid[0]], [right_centroid[1]]], dtype=np.float64)

        # Triangulate → 4D homogeneous coordinates
        points_4d = cv2.triangulatePoints(self.P_left, self.P_right,
                                          pts_left, pts_right)

        # Convert to 3D
        point_3d = points_4d[:3, 0] / points_4d[3, 0]

        return {
            "position": point_3d.tolist(),
            "confidence": 1.0,
            "source": "both",
        }
