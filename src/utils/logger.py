"""
logger.py — Flight data logger for post-flight analysis and judging.

Writes CSV with all state variables each control tick.
Also prints key values to console in real time.
"""

import os
import csv
import time


class FlightLogger:
    """Logs flight data to CSV and optionally to console.

    Each row in the CSV contains a timestamp + all sensor/control values.
    """

    # CSV column headers
    COLUMNS = [
        "time_s",
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "target_x", "target_y", "target_z",
        "err_x", "err_y", "err_z",
        "cmd_pitch", "cmd_roll", "cmd_thrust", "cmd_yaw",
        "imu_pitch", "imu_roll",
        "vision_source", "vision_confidence",
        "safety_zone", "safety_action",
        "loop_dt_ms",
    ]

    def __init__(self, log_dir: str = "logs", enabled: bool = True,
                 log_to_console: bool = True, log_to_csv: bool = True):
        self.enabled = enabled
        self.log_to_console = log_to_console
        self.log_to_csv = log_to_csv

        self._csv_file = None
        self._csv_writer = None
        self._start_time = time.time()
        self._tick_count = 0

        if self.log_to_csv and self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(log_dir, f"flight_{timestamp}.csv")
            self._csv_file = open(filepath, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self.COLUMNS)
            print(f"[Logger] Writing CSV to: {filepath}")

    def log_tick(self, position, velocity, target, commands,
                 imu_pitch=0.0, imu_roll=0.0,
                 vision_source="both", vision_confidence=1.0,
                 safety_zone=0, safety_action="normal",
                 loop_dt_ms=0.0):
        """Log one control tick.

        Args:
            position: [x, y, z] or None
            velocity: [vx, vy, vz] or None
            target: [tx, ty, tz]
            commands: dict with pitch, roll, thrust, yaw
            imu_pitch, imu_roll: from drone IMU
            vision_source: "both", "left_only", etc.
            vision_confidence: 0-1
            safety_zone: SafetyZone constant
            safety_action: "normal", "warning", "neutralize", "estop"
            loop_dt_ms: actual loop iteration time
        """
        if not self.enabled:
            return

        self._tick_count += 1
        t = time.time() - self._start_time

        pos = position or [0, 0, 0]
        vel = velocity or [0, 0, 0]
        err = [target[i] - pos[i] for i in range(3)]

        row = [
            f"{t:.3f}",
            f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
            f"{vel[0]:.4f}", f"{vel[1]:.4f}", f"{vel[2]:.4f}",
            f"{target[0]:.3f}", f"{target[1]:.3f}", f"{target[2]:.3f}",
            f"{err[0]:.4f}", f"{err[1]:.4f}", f"{err[2]:.4f}",
            f"{commands.get('pitch', 0):.3f}",
            f"{commands.get('roll', 0):.3f}",
            commands.get("thrust", 0),
            commands.get("yaw", 0),
            f"{imu_pitch:.2f}", f"{imu_roll:.2f}",
            vision_source, f"{vision_confidence:.2f}",
            safety_zone, safety_action,
            f"{loop_dt_ms:.1f}",
        ]

        # Write to CSV
        if self.log_to_csv and self._csv_writer:
            self._csv_writer.writerow(row)
            # Flush every 10 ticks to avoid data loss on crash
            if self._tick_count % 10 == 0:
                self._csv_file.flush()

        # Print to console (compact format, every tick)
        if self.log_to_console:
            print(f"[{t:6.1f}s] pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
                  f"err=({err[0]:+.3f},{err[1]:+.3f},{err[2]:+.3f}) "
                  f"cmd=(P:{commands.get('pitch',0):+.2f} R:{commands.get('roll',0):+.2f} "
                  f"T:{commands.get('thrust',0):3d}) "
                  f"src={vision_source:>5s} zone={safety_zone} "
                  f"dt={loop_dt_ms:.0f}ms")

    def close(self):
        """Flush and close the log file."""
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            print(f"[Logger] Closed CSV ({self._tick_count} ticks logged)")
