import drone_rc as rc
import signal
import sys
import time

mode  = rc.get_mode()
pitch = rc.get_pitch()
roll  = rc.get_roll()
print(f"Comms OK — mode:{mode}  pitch:{pitch:.2f}°  roll:{roll:.2f}°")

# e-stop
def stop(sig=None, frame=None):
    rc.emergency_stop()
    print("e-stop")
    sys.exit(0)

signal.signal(signal.SIGINT, stop)


def get_altitude():
    return 0.2  # stub — replace with teammate's function

HOV_BASELINE  = 150   # your found hover thrust
TARGET    = 0.5   # meters — center of 1m cube

# Altitude PID gains — start with P only, tune from there
KP = 20.0
KI = 0.0
KD = 0.0

# PID state
integral   = 0.0
prev_error = 0.0
prev_time  = time.time()

rc.set_mode(2)
print("Running altitude hold. Ctrl+C to stop.")
while True:
    now = time.time()
    dt  = now - prev_time
    prev_time = now

    altitude = get_altitude()
    error    = TARGET - altitude

    integral  += error * dt
    derivative = (error - prev_error) / dt if dt > 0 else 0.0
    prev_error = error

    correction = KP * error + KI * integral + KD * derivative
    thrust     = int(HOV_BASELINE + correction)
    thrust     = max(0, min(250, thrust))

    rc.manual_thrusts(thrust, thrust, thrust, thrust)
    print(f"alt:{altitude:.2f}m  err:{error:.2f}  thrust:{thrust}", flush=True)
