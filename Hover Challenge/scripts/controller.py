# controller.py
# Altitude PID loop + drone command abstraction.
#
# Responsibility split:
#   Drone (Mode 2): handles pitch/roll stabilization internally
#   Us:             altitude hold via external PID, yaw held at 0
#
# The run_hover() function accepts an altitude source as an injectable
# callable — pass vision.estimator.get_altitude for Test 2, or
# vision.get_altitude_stub for Test 1 (no CV). No other changes needed.

import time

from config import (
    BASELINE_THRUST, TARGET_ALTITUDE, CONTROL_HZ,
    PITCH_GAIN_P, PITCH_GAIN_I, PITCH_GAIN_D,
    ALT_P, ALT_I, ALT_D, DRY_RUN
)
from safety import is_stopped, check_attitude_bounds, trigger_stop

if DRY_RUN:
    from test_drone_rc import (
        set_mode, set_pitch, set_roll, set_yaw,
        manual_thrusts, set_p_gain, set_i_gain, set_d_gain,
        reset_integral
    )
else:
    from drone_rc import (
        set_mode, set_pitch, set_roll, set_yaw,
        manual_thrusts, set_p_gain, set_i_gain, set_d_gain,
        reset_integral
    )


# ---------------------------------------------------------------------------
# Altitude PID
# ---------------------------------------------------------------------------

class AltitudePID:
    """
    Discrete PID controller for altitude hold.

    Derivative-on-measurement rather than derivative-on-error:
      - Avoids the derivative kick that occurs when the setpoint changes
      - More stable when the altitude estimate is noisy
    Integral windup is not explicitly clamped here — keep ALT_I small (see config).
    """

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral  = 0.0
        self._prev_meas = None

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement
        self._integral += error * dt

        derivative = 0.0
        if self._prev_meas is not None:
            # Negative sign: rising measurement should reduce thrust (downward correction)
            derivative = -(measurement - self._prev_meas) / dt
        self._prev_meas = measurement

        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral  = 0.0
        self._prev_meas = None


# ---------------------------------------------------------------------------
# Drone setup helpers
# ---------------------------------------------------------------------------

def apply_drone_pid_gains():
    """
    Push PID gains from config to the drone's internal pitch/roll loops.
    The drone exposes a single P/I/D set (shared by pitch and roll).
    If asymmetric tuning is needed later, alternate set_*_gain + set_pitch/roll calls.
    """
    set_p_gain(PITCH_GAIN_P)
    set_i_gain(PITCH_GAIN_I)
    set_d_gain(PITCH_GAIN_D)
    print(f"[CTRL] PID gains set — P:{PITCH_GAIN_P} I:{PITCH_GAIN_I} D:{PITCH_GAIN_D}")


def arm_drone():
    """
    Set Mode 2 and zero attitude targets.
    Mode 2: drone stabilizes pitch/roll to our targets; we control thrust.
    """
    set_mode(2)
    set_pitch(0)
    set_roll(0)
    set_yaw(0)
    print(f"[CTRL] Mode 2 armed — pitch/roll targets zeroed.")


def disarm_drone():
    """Set mode 0 (motors off). Called on clean exit and e-stop."""
    set_mode(0)
    print("[CTRL] Mode 0 — motors off.")


# ---------------------------------------------------------------------------
# Main hover loop
# ---------------------------------------------------------------------------

def run_hover(get_altitude=None):
    """
    Hover loop. Runs at CONTROL_HZ until e-stop is triggered.

    Args:
        get_altitude: callable () -> float, returns current altitude in metres.
                      Defaults to the stub (zero PID error, hover at BASELINE_THRUST).
                      Pass vision.estimator.get_altitude for Test 2.
    """
    # Import here to avoid circular import; stub lives in vision.py
    from vision import get_altitude_stub
    if get_altitude is None:
        get_altitude = get_altitude_stub
        print("[CTRL] No altitude source provided — using stub (Test 1 mode).")
    else:
        print("[CTRL] External altitude source active (Test 2 mode).")

    print("[CTRL] Applying PID gains and arming...")
    apply_drone_pid_gains()
    reset_integral()
    arm_drone()

    alt_pid = AltitudePID(ALT_P, ALT_I, ALT_D)
    period  = 1.0 / CONTROL_HZ

    print(f"[CTRL] Hover loop running at {CONTROL_HZ}Hz. Press SPACE to e-stop.")

    prev_time = time.time()

    try:
        while not is_stopped():
            loop_start = time.time()
            dt = max(loop_start - prev_time, 1e-6)  # guard against zero dt
            prev_time = loop_start

            # --- Safety check before every command ---
            check_attitude_bounds()
            if is_stopped():
                break

            # --- Altitude PID ---
            altitude   = get_altitude()
            correction = alt_pid.update(TARGET_ALTITUDE, altitude, dt)

            thrust = int(BASELINE_THRUST + correction)
            thrust = max(0, min(250, thrust))  # clamp to valid motor range

            # Mode 2: manual_thrusts sets the baseline all four motors share.
            # The drone's internal PID then adds pitch/roll corrections on top.
            manual_thrusts(thrust, thrust, thrust, thrust)

            # --- Loop timing ---
            elapsed    = time.time() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Log if we're consistently overrunning — may need to reduce CONTROL_HZ
                print(f"[CTRL] Warning: loop overrun by {-sleep_time*1000:.1f}ms")

    except Exception as ex:
        trigger_stop(f"exception in hover loop: {ex}")
        raise

    finally:
        disarm_drone()
        print("[CTRL] Hover loop exited cleanly.")
