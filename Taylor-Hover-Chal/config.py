# config.py
# All tunable constants live here — change these during testing without touching logic

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
DRONE_IP   = "192.168.4.1"
DRONE_PORT = 8080

# ---------------------------------------------------------------------------
# Thrust
# ---------------------------------------------------------------------------
# Baseline thrust — the value at which the drone approximately hovers.
# Start conservative (~100), tune upward. Expect 110-130 range for a ~250g drone.
BASELINE_THRUST = 110

# ---------------------------------------------------------------------------
# Drone internal PID gains (Mode 2 — drone handles pitch/roll stabilization)
# ---------------------------------------------------------------------------
# Start with these defaults, offset if oscillating or drifting.
PITCH_GAIN_P = 0.3
PITCH_GAIN_I = 0.00001
PITCH_GAIN_D = 5.0

ROLL_GAIN_P  = 0.3
ROLL_GAIN_I  = 0.00001
ROLL_GAIN_D  = 5.0

# ---------------------------------------------------------------------------
# Altitude PID gains (our external loop — not the drone's internal PID)
# ---------------------------------------------------------------------------
ALT_P = 10.0
ALT_I = 0.001
ALT_D = 2.0

# ---------------------------------------------------------------------------
# Mission parameters
# ---------------------------------------------------------------------------
# Target hover altitude in meters (0.5m per challenge spec)
TARGET_ALTITUDE = 0.5

# ---------------------------------------------------------------------------
# Safety bounds — automatic e-stop triggers if any of these are exceeded
# ---------------------------------------------------------------------------
MAX_PITCH_DEG = 30.0
MAX_ROLL_DEG  = 30.0
MAX_ALTITUDE  = 1.2   # top of 1m cube + small margin

# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------
# Loop frequency in Hz — keep low to respect the drone's gyro bandwidth.
# The drone_rc.py advisory warns against high-bandwidth comms; 20Hz is safe.
CONTROL_HZ = 20
