# main.py
# Entry point for Challenge 2 — Hover & Turbulence Resistance.
#
# Test 1 (comms + basic hover):
#   python main.py
#   No arguments needed — runs with altitude stub, no CV.
#
# Test 2 (full CV hover):
#   python main.py --cv
#   Opens cameras, runs position estimator, feeds altitude into control loop.
#
# SPACE bar triggers emergency stop at any point.
# Ctrl+C is also routed through e-stop (SIGINT handler in safety.py).

import sys
import argparse
from pynput import keyboard as kb

from safety import trigger_stop, register_signal_handlers
from controller import run_hover


# ---------------------------------------------------------------------------
# Comms sanity check
# ---------------------------------------------------------------------------

def verify_comms() -> bool:
#    Quick pre-flight comms check — read mode, pitch, and roll before arming.
#    In DRY_RUN mode skips the real connection entirely (drone_rc.py connects at
#    import time with no timeout, so importing it without a drone = indefinite block).

    from config import DRY_RUN
 
    if DRY_RUN:
        print("[MAIN] DRY_RUN=True — skipping real comms check, using mock.")
        from test_drone_rc import get_mode, get_pitch, get_roll
        mode  = get_mode()
        pitch = get_pitch()
        roll  = get_roll()
        print(f"[MAIN] Mock comms OK — mode:{mode}  pitch:{pitch:.2f}°  roll:{roll:.2f}°")
        return True
 
    print("[MAIN] Verifying drone comms...")
    try:
        from drone_rc import get_mode, get_pitch, get_roll
        mode  = get_mode()
        pitch = get_pitch()
        roll  = get_roll()
        print(f"[MAIN] Comms OK — mode:{mode}  pitch:{pitch:.2f}°  roll:{roll:.2f}°")
        return True
    except Exception as ex:
        print(f"[MAIN] Comms FAILED: {ex}")
        print("[MAIN] Check: are you connected to the drone's Wi-Fi network?")
        return False

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Challenge 2 — Drone Hover Controller")
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable computer vision altitude estimation (Test 2)"
    )
    parser.add_argument(
        "--cam0", type=int, default=0,
        help="Camera index for camera 0 (default: 0)"
    )
    parser.add_argument(
        "--cam1", type=int, default=1,
        help="Camera index for camera 1 (default: 1)"
    )
    args = parser.parse_args()

    # --- Signal handlers (Ctrl+C, SIGTERM) ---
    register_signal_handlers()

    # --- Spacebar e-stop ---
    def on_key_press(key):
        if key == kb.Key.space:
            trigger_stop("spacebar")

    listener = kb.Listener(on_press=on_key_press)
    listener.start()
    print("[MAIN] Spacebar e-stop registered.")

    # --- Comms check ---
    if not verify_comms():
        listener.stop()
        sys.exit(1)

    # --- Altitude source ---
    altitude_fn = None  # defaults to stub inside run_hover()

    if args.cv:
        print("[MAIN] CV mode enabled — initialising cameras...")
        from vision import estimator
        try:
            estimator.open_cameras(args.cam0, args.cam1)
            # Calibrate homographies here before flight if cage geometry is known.
            # estimator.calibrate(cam0_pixels, cam1_pixels, world_xz, world_yz)
            # TODO: add calibration call once cage layout is confirmed at the venue
            altitude_fn = estimator.get_altitude
            print("[MAIN] Cameras ready.")
        except Exception as ex:
            print(f"[MAIN] Failed to open cameras: {ex}")
            listener.stop()
            sys.exit(1)

    # --- Hover ---
    print("[MAIN] Starting hover. Press SPACE to emergency stop.\n")
    try:
        run_hover(get_altitude=altitude_fn)
    except Exception as ex:
        # trigger_stop already called inside run_hover on exception,
        # but call again here in case it bubbled from elsewhere
        trigger_stop(f"unhandled exception in main: {ex}")
        raise
    finally:
        if args.cv:
            from vision import estimator
            estimator.release()
        listener.stop()
        print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    main()
