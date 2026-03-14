"""
drone_client.py — Safe wrapper around organizer_lib for drone communication.

Handles:
  - Lazy connection (does not connect on import)
  - Timeouts on send/receive
  - Emergency stop as first-class operation
  - All commands used by our control system

This wraps organizer_lib.py which the organizers provided.
We never modify organizer_lib.py itself.
"""

import socket
import time


class DroneClient:
    """Manages TCP communication with the drone over Wi-Fi.

    Usage:
        client = DroneClient(host="192.168.4.1", port=8080)
        client.connect()
        client.set_mode(2)
        client.set_pitch(0)
        client.set_roll(0)
        client.manual_thrusts(140, 140, 140, 140)
        client.emergency_stop()
        client.disconnect()
    """

    def __init__(self, host: str = "192.168.4.1", port: int = 8080,
                 connect_timeout: float = 5.0, recv_timeout: float = 0.1):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout

        self._socket = None
        self._connected = False

    # ── Connection Management ──────────────────────────────────────

    def connect(self) -> bool:
        """Establish TCP connection to the drone.

        Returns True on success, False on failure.
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.connect_timeout)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(self.recv_timeout)
            self._connected = True
            print(f"[DroneClient] Connected to {self.host}:{self.port}")
            return True
        except (socket.error, socket.timeout) as e:
            print(f"[DroneClient] Connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Close the TCP connection."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._connected = False
        print("[DroneClient] Disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Low-Level Message Protocol ─────────────────────────────────

    def _msg(self, tx: str) -> str:
        """Send a message and receive the response (newline-delimited).

        Raises RuntimeError on communication failure.
        """
        if not self._connected or self._socket is None:
            raise RuntimeError("DroneClient not connected")

        try:
            self._socket.sendall((tx + "\n").encode("ASCII"))
            rx = ""
            while not rx.endswith("\n"):
                chunk = self._socket.recv(1).decode("ASCII")
                if not chunk:
                    raise RuntimeError("Socket closed by drone")
                rx += chunk
            return rx[:-1]  # strip trailing newline
        except socket.timeout:
            raise RuntimeError(f"DroneClient timeout on command: {tx}")
        except socket.error as e:
            self._connected = False
            raise RuntimeError(f"DroneClient socket error: {e}")

    # ── Emergency Stop (highest priority) ──────────────────────────

    def emergency_stop(self):
        """Immediately cut all motors. Always try, even if connection is shaky."""
        try:
            self._msg("mode0")
        except Exception:
            # Last-resort: try raw send without waiting for response
            try:
                if self._socket:
                    self._socket.sendall(b"mode0\n")
            except Exception:
                pass
        print("[DroneClient] *** EMERGENCY STOP ***")

    # ── Mode Control ───────────────────────────────────────────────

    def set_mode(self, mode: int):
        """Set drone mode: 0=off, 1=manual, 2=PID pitch/roll."""
        self._msg(f"mode{mode}")

    def get_mode(self) -> str:
        """Get current drone mode."""
        return self._msg("gMode")

    # ── Thrust Control ─────────────────────────────────────────────

    def manual_thrusts(self, a: int, b: int, c: int, d: int):
        """Set absolute motor thrusts (0-250 each).

        In mode 2, this sets the baseline that PID corrections are added to.
        """
        # Clamp to valid range
        a = max(0, min(250, int(a)))
        b = max(0, min(250, int(b)))
        c = max(0, min(250, int(c)))
        d = max(0, min(250, int(d)))
        self._msg(f"manT\n{a},{b},{c},{d}\n")

    def set_thrust_uniform(self, thrust: int):
        """Set all four motors to the same thrust value."""
        self.manual_thrusts(thrust, thrust, thrust, thrust)

    # ── Attitude Control (Mode 2) ──────────────────────────────────

    def set_pitch(self, target: float):
        """Set target pitch angle for onboard PID (mode 2)."""
        self._msg(f"gx{target}")

    def set_roll(self, target: float):
        """Set target roll angle for onboard PID (mode 2)."""
        self._msg(f"gy{target}")

    def set_yaw(self, value: float):
        """Set yaw motor differential directly."""
        self._msg(f"yaw{value}")

    # ── IMU Readings ───────────────────────────────────────────────

    def get_pitch(self) -> float:
        """Read current pitch angle (~degrees, not exact)."""
        return float(self._msg("angX")) / 16

    def get_roll(self) -> float:
        """Read current roll angle (~degrees, not exact)."""
        return float(self._msg("angY")) / 16

    def get_gyro_pitch(self) -> float:
        """Read pitch rate in deg/sec."""
        return float(self._msg("gyroX"))

    def get_gyro_roll(self) -> float:
        """Read roll rate in deg/sec."""
        return float(self._msg("gyroY"))

    # ── Onboard PID Tuning ─────────────────────────────────────────

    def set_p_gain(self, p: float):
        """Set onboard PID P gain (approx 0-0.5)."""
        self._msg(f"gainP{p}")

    def set_i_gain(self, i: float):
        """Set onboard PID I gain (below 0.00003)."""
        self._msg(f"gainI{i}")

    def set_d_gain(self, d: float):
        """Set onboard PID D gain (approx 0-10)."""
        self._msg(f"gainD{d}")

    def reset_integral(self):
        """Reset the onboard PID integral accumulators to zero."""
        self._msg("irst")

    def get_i_values(self) -> list:
        """Get onboard PID integral values [I_pitch, I_roll]."""
        resp = self._msg("geti").split(",")
        return [float(resp[0]), float(resp[1])]

    def configure_onboard_pid(self, p: float, i: float, d: float):
        """Convenience: set all three onboard PID gains at once."""
        self.set_p_gain(p)
        self.set_i_gain(i)
        self.set_d_gain(d)
        self.reset_integral()
        print(f"[DroneClient] Onboard PID set: P={p}, I={i}, D={d}")
