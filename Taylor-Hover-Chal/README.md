# Challenge 2 — Drone Hover & Turbulence Resistance

External vision-based stabilization controller for the ESP32-S2-Drone V1.2.

## Architecture

```
Camera Feeds → vision.py → position estimate
                               ↓
drone_rc.py ←── controller.py (altitude PID)
                               ↑
                           config.py (all tunable values)
                           safety.py (e-stop, bounds checking)
```

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | Entry point — wires everything together |
| `controller.py` | Altitude PID loop, drone arming/disarming |
| `vision.py` | LED detection, position estimation, Kalman filter |
| `safety.py` | E-stop, signal handlers, attitude bounds checking |
| `config.py` | All tunable constants (gains, thresholds, targets) |
| `drone_rc.py` | Provided comms library — do not modify |

## Setup

```bash
pip install -r requirements.txt
```

Connect to the drone's Wi-Fi network (`192.168.4.1`) before running.

## Usage

**Test 1 — Comms + basic hover (no CV):**
```bash
python main.py
```

**Test 2 — Full CV hover:**
```bash
python main.py --cv
# optionally specify camera indices if defaults (0, 1) are wrong:
python main.py --cv --cam0 0 --cam1 2
```

**LED calibration (run before Test 2):**
```bash
python -c "from vision import led_calibration; led_calibration()"
```
Adjust the HSV trackbars until only the drone LED is visible in the mask window,
then update `LED_HSV_LOWER` / `LED_HSV_UPPER` in `vision.py`.

## E-Stop

- **Spacebar** — primary e-stop
- **Ctrl+C** — also routed through e-stop (won't leave motors running)

E-stop sends `mode0` to the drone and exits the control loop cleanly.

## Tuning

All tunables are in `config.py`. Key values to adjust during testing:

| Parameter | What to do |
|---|---|
| `BASELINE_THRUST` | Increase from 100 until drone lifts, note hover value |
| `PITCH_GAIN_P/I/D` | Offset drone's internal PID if oscillating or drifting |
| `ALT_P` | Increase if altitude response is sluggish |
| `ALT_D` | Increase to dampen altitude oscillation / fight turbulence |
| `CONTROL_HZ` | Keep at 20 — higher values stress drone's gyro update budget |

## Test 1 Checklist

- [ ] Connected to drone Wi-Fi
- [ ] `python main.py` runs without hanging on comms check
- [ ] Spacebar triggers e-stop before any motors spin
- [ ] Mode 2 sets correctly (`get_mode()` returns expected value)
- [ ] `BASELINE_THRUST` tuned to approximate hover point
