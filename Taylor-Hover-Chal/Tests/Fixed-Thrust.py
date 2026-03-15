from drone_rc import set_mode, manual_thrusts, emergency_stop
import signal
import sys

THRUST = 150  # increase until props spin

def stop(sig=None, frame=None):
    emergency_stop()
    sys.exit(0)

signal.signal(signal.SIGINT, stop)

set_mode(1)
while True:
    manual_thrusts(THRUST, THRUST, THRUST, THRUST)
