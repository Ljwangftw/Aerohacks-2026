import drone_rc as rc
import signal
import sys
import time

STEP = 1
DELAY = 0.5
thrust = 0

mode  = rc.get_mode()
pitch = rc.get_pitch()
roll  = rc.get_roll()
print(f"Comms OK — mode:{mode}  pitch:{pitch:.2f}°  roll:{roll:.2f}°")

def stop(sig=None, frame=None):
    rc.emergency_stop()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)

rc.set_mode(2)

while True:
    rc.increment_thrusts(STEP, STEP, STEP, STEP)
    thrust += STEP
    print(f"Thrust: {thrust}", flush=True)
    if thrust >= 250:
        print("Max thrust reached")
        stop()
    time.sleep(DELAY)
