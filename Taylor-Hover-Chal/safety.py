# safety.py
# E-stop and bounds enforcement.
# All stop triggers funnel through trigger_stop() — single entry point, idempotent.
# The _stop_event flag is checked at the top of every control loop iteration.

import signal
import threading

from drone_rc import emergency_stop, get_pitch, get_roll
from config import MAX_PITCH_DEG, MAX_ROLL_DEG

# ---------------------------------------------------------------------------
# Global stop flag
# ---------------------------------------------------------------------------
# Using an Event rather than a plain bool so it's thread-safe without a lock.
# Any thread (keyboard listener, watchdog, main loop) can call trigger_stop().
_stop_event = threading.Event()


def is_stopped() -> bool:
    """Returns True if an e-stop has been triggered."""
    return _stop_event.is_set()


def trigger_stop(reason: str = "manual"):
    """
    Central e-stop handler. Idempotent — safe to call multiple times.
    Sets the stop flag then sends mode0 to the drone.
    The try/except on emergency_stop() guards against the socket already being
    dead (e.g. Wi-Fi dropped) — we still want the flag set even if the command fails.
    """
    if not _stop_event.is_set():
        print(f"\n[E-STOP] Triggered — reason: {reason}")
        _stop_event.set()
        try:
            emergency_stop()  # sends "mode0" over TCP
        except Exception as ex:
            print(f"[E-STOP] Warning: failed to send stop command to drone: {ex}")


def register_signal_handlers():
    """
    Route SIGINT (Ctrl+C) and SIGTERM through trigger_stop so both kill paths
    result in a clean motor-off state rather than an abrupt process exit.
    """
    signal.signal(signal.SIGINT,  lambda s, f: trigger_stop("SIGINT"))
    signal.signal(signal.SIGTERM, lambda s, f: trigger_stop("SIGTERM"))


def check_attitude_bounds():
    """
    Read current pitch/roll from the drone and trigger e-stop if outside the
    safe envelope defined in config.py.

    Cost: 2 round-trip TCP calls per invocation. At CONTROL_HZ=20 this is fine,
    but don't call this more frequently than the control loop itself.
    """
    pitch = get_pitch()
    roll  = get_roll()

    if abs(pitch) > MAX_PITCH_DEG:
        trigger_stop(f"pitch limit exceeded: {pitch:.1f}° (max ±{MAX_PITCH_DEG}°)")

    if abs(roll) > MAX_ROLL_DEG:
        trigger_stop(f"roll limit exceeded: {roll:.1f}° (max ±{MAX_ROLL_DEG}°)")
