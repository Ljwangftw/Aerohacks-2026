"""
hover_controller.py — PID controller for 3D position hold.

Maps position errors to drone commands:
  X error → roll setpoint   (onboard PID stabilizes roll)
  Y error → pitch setpoint  (onboard PID stabilizes pitch)
  Z error → thrust adjustment (added to base hover thrust)

All outputs are clamped and rate-limited for safety.
"""

import time
from src.utils.math_helpers import clamp, rate_limit


class PIDAxis:
    """Single-axis PID controller with integral windup protection."""

    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit

        # Internal state
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, error: float, dt: float = None) -> float:
        """Compute PID output for a given error.

        Args:
            error: target - measured value
            dt: time step in seconds (auto-computed if None)

        Returns:
            PID output (unclamped — caller should clamp)
        """
        now = time.time()

        if dt is None:
            if self._prev_time is None:
                dt = 0.05  # default ~20 Hz
            else:
                dt = now - self._prev_time
            dt = max(dt, 0.001)  # prevent division by zero

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self._integral += error * dt
        self._integral = clamp(self._integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self._integral

        # Derivative (on error, not on measurement — simpler)
        d_term = self.kd * (error - self._prev_error) / dt

        # Store for next iteration
        self._prev_error = error
        self._prev_time = now

        return p_term + i_term + d_term

    def reset(self):
        """Reset the PID internal state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None


class HoverController:
    """3-axis position hold controller.

    Takes smoothed 3D position + target → outputs pitch, roll, thrust commands.
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: the "control" section of default.yaml
        """
        self.cfg = cfg

        # Target hover position
        target = cfg["target"]
        self.target = [target["x"], target["y"], target["z"]]

        # Create PID controllers for each axis
        px = cfg["pid_x"]
        self.pid_x = PIDAxis(px["kp"], px["ki"], px["kd"], px["integral_limit"])

        py = cfg["pid_y"]
        self.pid_y = PIDAxis(py["kp"], py["ki"], py["kd"], py["integral_limit"])

        pz = cfg["pid_z"]
        self.pid_z = PIDAxis(pz["kp"], pz["ki"], pz["kd"], pz["integral_limit"])

        # Output limits from config
        self.max_pitch = cfg["max_pitch"]
        self.max_roll = cfg["max_roll"]
        self.base_thrust = cfg["base_thrust"]
        self.min_thrust = cfg["min_thrust"]
        self.max_thrust = cfg["max_thrust"]
        self.max_thrust_delta = cfg["max_thrust_delta"]
        self.max_angle_delta = cfg["max_angle_delta"]

        # Axis sign mapping (invert if drone moves the wrong direction)
        self.sign_x = cfg.get("axis_sign_x", 1)
        self.sign_y = cfg.get("axis_sign_y", 1)
        self.sign_z = cfg.get("axis_sign_z", 1)

        # Previous outputs for rate limiting
        self._prev_pitch = 0.0
        self._prev_roll = 0.0
        self._prev_thrust = self.base_thrust

    def compute(self, position: list, velocity: list = None) -> dict:
        """Compute control commands from current position.

        Args:
            position: smoothed [x, y, z] in meters
            velocity: estimated [vx, vy, vz] in m/s (optional, for D term)

        Returns:
            dict with "pitch", "roll", "thrust", "yaw"
        """
        # Compute errors (target - current)
        ex = self.target[0] - position[0]
        ey = self.target[1] - position[1]
        ez = self.target[2] - position[2]

        # PID outputs
        roll_cmd = self.sign_x * self.pid_x.compute(ex)
        pitch_cmd = self.sign_y * self.pid_y.compute(ey)
        thrust_adj = self.sign_z * self.pid_z.compute(ez)

        # Clamp outputs
        roll_cmd = clamp(roll_cmd, -self.max_roll, self.max_roll)
        pitch_cmd = clamp(pitch_cmd, -self.max_pitch, self.max_pitch)
        thrust = clamp(self.base_thrust + thrust_adj, self.min_thrust, self.max_thrust)

        # Rate-limit to avoid sudden jerks
        roll_cmd = rate_limit(roll_cmd, self._prev_roll, self.max_angle_delta)
        pitch_cmd = rate_limit(pitch_cmd, self._prev_pitch, self.max_angle_delta)
        thrust = rate_limit(thrust, self._prev_thrust, self.max_thrust_delta)

        # Store for next rate-limiting step
        self._prev_pitch = pitch_cmd
        self._prev_roll = roll_cmd
        self._prev_thrust = thrust

        return {
            "pitch": round(pitch_cmd, 3),
            "roll": round(roll_cmd, 3),
            "thrust": int(thrust),
            "yaw": 0,  # keep yaw neutral unless correction needed
        }

    def apply_warning_clamps(self, commands: dict, warning_pitch: float,
                              warning_roll: float, warning_rate: float,
                              center_bias: list = None) -> dict:
        """Apply tighter limits when drone is in the warning zone.

        Args:
            commands: dict from compute()
            warning_pitch: tighter pitch clamp
            warning_roll: tighter roll clamp
            warning_rate: tighter rate limit
            center_bias: optional [bx, by] bias toward center (added to pitch/roll)
        """
        cmd = dict(commands)  # copy

        # Tighter clamp
        cmd["pitch"] = clamp(cmd["pitch"], -warning_pitch, warning_pitch)
        cmd["roll"] = clamp(cmd["roll"], -warning_roll, warning_roll)

        # Add centering bias if provided
        if center_bias is not None:
            cmd["pitch"] = clamp(cmd["pitch"] + center_bias[1],
                                 -warning_pitch, warning_pitch)
            cmd["roll"] = clamp(cmd["roll"] + center_bias[0],
                                -warning_roll, warning_roll)

        return cmd

    def neutralize(self) -> dict:
        """Return neutral commands (level, base thrust, no yaw)."""
        self._prev_pitch = 0.0
        self._prev_roll = 0.0
        return {
            "pitch": 0.0,
            "roll": 0.0,
            "thrust": int(self._prev_thrust),  # maintain current thrust
            "yaw": 0,
        }

    def reset(self):
        """Reset all PID states."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self._prev_pitch = 0.0
        self._prev_roll = 0.0
        self._prev_thrust = self.base_thrust
