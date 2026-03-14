# drone_rc_mock.py
# Drop-in mock for drone_rc.py — mirrors the full API but prints calls instead
# of sending TCP commands. Use for dry-run testing without a connected drone.
#
# To use: in safety.py, controller.py — replace
#   from drone_rc import ...
# with
#   from drone_rc_mock import ...
# Or use the DRY_RUN flag in config.py to switch automatically (see below).

import time

# Simulated drone state — mutated by mock calls, read back by mock getters
_state = {
    "mode":    0,
    "pitch":   0.0,
    "roll":    0.0,
    "thrust":  [0, 0, 0, 0],
    "yaw":     0.0,
    "gain_p":  0.3,
    "gain_i":  0.00001,
    "gain_d":  5.0,
    "led_r":   0,
    "led_g":   0,
    "led_b":   0,
    "i_x":     0.0,
    "i_y":     0.0,
}

def _log(fn: str, *args):
    ts = time.strftime("%H:%M:%S")
    args_str = "  ".join(str(a) for a in args) if args else ""
    print(f"[MOCK {ts}] {fn}({args_str})")


# ---------------------------------------------------------------------------
# E-stop
# ---------------------------------------------------------------------------

def emergency_stop():
    _state["mode"] = 0
    _log("emergency_stop", "→ mode=0")

def e():
    emergency_stop()


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------

def set_mode(m: int):
    _state["mode"] = m
    _log("set_mode", m)

def get_mode() -> int:
    _log("get_mode", f"→ {_state['mode']}")
    return _state["mode"]


# ---------------------------------------------------------------------------
# Thrust
# ---------------------------------------------------------------------------

def manual_thrusts(A, B, C, D):
    _state["thrust"] = [A, B, C, D]
    _log("manual_thrusts", f"A={A} B={B} C={C} D={D}")

def increment_thrusts(A, B, C, D):
    _state["thrust"] = [
        _state["thrust"][i] + v for i, v in enumerate([A, B, C, D])
    ]
    _log("increment_thrusts", f"A={A} B={B} C={C} D={D}  → {_state['thrust']}")


# ---------------------------------------------------------------------------
# Attitude getters — simulate slow drift so PID has something to react to
# ---------------------------------------------------------------------------

_start_time = time.time()

def get_pitch() -> float:
    # Simulates a small sinusoidal drift so the safety bounds check is exercised
    drift = 2.0 * (time.time() - _start_time) % 5.0 - 2.5
    val = _state["pitch"] + drift * 0.1
    _log("get_pitch", f"→ {val:.3f}")
    return val

def get_roll() -> float:
    drift = 1.5 * (time.time() - _start_time) % 4.0 - 2.0
    val = _state["roll"] + drift * 0.1
    _log("get_roll", f"→ {val:.3f}")
    return val

def get_gyro_pitch() -> float:
    val = 0.05
    _log("get_gyro_pitch", f"→ {val}")
    return val

def get_gyro_roll() -> float:
    val = 0.03
    _log("get_gyro_roll", f"→ {val}")
    return val


# ---------------------------------------------------------------------------
# Attitude setters
# ---------------------------------------------------------------------------

def set_pitch(r):
    _state["pitch"] = r
    _log("set_pitch", r)

def set_roll(r):
    _state["roll"] = r
    _log("set_roll", r)

def set_yaw(y):
    _state["yaw"] = y
    _log("set_yaw", y)


# ---------------------------------------------------------------------------
# PID gains
# ---------------------------------------------------------------------------

def set_p_gain(p):
    _state["gain_p"] = p
    _log("set_p_gain", p)

def set_i_gain(i):
    _state["gain_i"] = i
    _log("set_i_gain", i)

def set_d_gain(d):
    _state["gain_d"] = d
    _log("set_d_gain", d)

def reset_integral():
    _state["i_x"] = 0.0
    _state["i_y"] = 0.0
    _log("reset_integral")

def get_i_values() -> list[float]:
    val = [_state["i_x"], _state["i_y"]]
    _log("get_i_values", f"→ {val}")
    return val


# ---------------------------------------------------------------------------
# LED
# ---------------------------------------------------------------------------

def red_LED(val):
    _state["led_r"] = val
    _log("red_LED", val)

def blue_LED(val):
    _state["led_b"] = val
    _log("blue_LED", val)

def green_LED(val):
    _state["led_g"] = val
    _log("green_LED", val)
