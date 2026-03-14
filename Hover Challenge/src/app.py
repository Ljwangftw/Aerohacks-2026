"""
app.py — Main application: ties together vision, estimation, control, and comms.

This is the heart of the AeroHover system. It runs the full pipeline:
  cameras → detect LED → triangulate 3D → estimate state → PID control → send commands

The control loop runs at ~20 Hz. Safety checks happen every tick.
Emergency stop is always available via keyboard (ESC or Space).
"""

import time
import cv2

from src.utils.config import load_config
from src.vision.camera import Camera
from src.vision.detection import LEDDetector
from src.vision.triangulation import OrthogonalTriangulator
from src.vision.visualization import draw_debug_overlay
from src.state.estimator import PositionEstimator
from src.control.hover_controller import HoverController
from src.control.safety import SafetyMonitor, SafetyZone
from src.comms.drone_client import DroneClient
from src.utils.logger import FlightLogger


class AeroHoverApp:
    """Main application for the autonomous hover system."""

    def __init__(self, config_name: str = "default"):
        self.cfg = load_config(config_name)
        self._running = False
        self._flying = False

    def run(self):
        """Full application lifecycle: init → takeoff → hover → shutdown."""
        print("=" * 60)
        print("  AeroHover — Vision-Based Autonomous Hover System")
        print("=" * 60)

        try:
            # ── Step 1: Initialize all subsystems ──
            cam_left, cam_right = self._init_cameras()
            drone = self._init_drone()
            detector = self._init_detector()
            triangulator = self._init_triangulator()
            estimator = self._init_estimator()
            controller = self._init_controller()
            safety = self._init_safety()
            logger = self._init_logger()

            # ── Step 2: Wait for operator ──
            print("\n[APP] All systems initialized.")
            print("[APP] Press ENTER to begin takeoff sequence...")
            print("[APP] (ESC or Space during flight = Emergency Stop)")
            self._show_live_preview(cam_left, cam_right, detector)

            # ── Step 3: Takeoff ──
            self._takeoff(drone, controller)

            # ── Step 4: Main control loop ──
            self._running = True
            self._flying = True
            self._control_loop(cam_left, cam_right, detector, triangulator,
                               estimator, controller, safety, drone, logger)

        except KeyboardInterrupt:
            print("\n[APP] KeyboardInterrupt — shutting down...")
        except Exception as e:
            print(f"\n[APP] EXCEPTION: {e}")
        finally:
            # ── Step 5: Emergency stop & cleanup ──
            self._shutdown(drone if 'drone' in dir() else None,
                          cam_left if 'cam_left' in dir() else None,
                          cam_right if 'cam_right' in dir() else None,
                          logger if 'logger' in dir() else None)

    # ── Initialization Helpers ─────────────────────────────────────

    def _init_cameras(self):
        """Open and warm up both cameras."""
        print("\n[APP] Initializing cameras...")
        cfg_cams = self.cfg["cameras"]

        cam_left = Camera(
            device_id=cfg_cams["left"]["id"],
            width=cfg_cams["left"]["width"],
            height=cfg_cams["left"]["height"],
            fps=cfg_cams["left"]["fps"],
            name="left",
        )
        cam_right = Camera(
            device_id=cfg_cams["right"]["id"],
            width=cfg_cams["right"]["width"],
            height=cfg_cams["right"]["height"],
            fps=cfg_cams["right"]["fps"],
            name="right",
        )

        if not cam_left.start():
            raise RuntimeError("Failed to open left camera")
        if not cam_right.start():
            raise RuntimeError("Failed to open right camera")

        # Warmup: let auto-exposure stabilize
        warmup = cfg_cams.get("warmup_frames", 30)
        cam_left.warmup(warmup)
        cam_right.warmup(warmup)

        return cam_left, cam_right

    def _init_drone(self) -> DroneClient:
        """Connect to the drone."""
        print("\n[APP] Connecting to drone...")
        cfg_comms = self.cfg["comms"]
        drone = DroneClient(
            host=cfg_comms["host"],
            port=cfg_comms["port"],
            connect_timeout=cfg_comms["connect_timeout_s"],
            recv_timeout=cfg_comms["recv_timeout_s"],
        )
        if not drone.connect():
            raise RuntimeError("Failed to connect to drone")

        # Set mode 0 (motors off) and configure onboard PID
        drone.set_mode(0)
        onboard = self.cfg.get("onboard_pid", {})
        if onboard:
            drone.configure_onboard_pid(
                onboard.get("p_gain", 0.3),
                onboard.get("i_gain", 0.00002),
                onboard.get("d_gain", 5.0),
            )
        return drone

    def _init_detector(self) -> LEDDetector:
        cfg_det = self.cfg["detection"]
        return LEDDetector(
            hsv_lower=cfg_det["hsv_lower"],
            hsv_upper=cfg_det["hsv_upper"],
            blur_kernel=cfg_det["blur_kernel"],
            min_area=cfg_det["min_contour_area"],
            max_area=cfg_det["max_contour_area"],
            morph_kernel=cfg_det["morph_kernel"],
        )

    def _init_triangulator(self) -> OrthogonalTriangulator:
        cfg_tri = self.cfg["triangulation"]
        return OrthogonalTriangulator(
            cam_left_cfg=cfg_tri["cam_left"],
            cam_right_cfg=cfg_tri["cam_right"],
        )

    def _init_estimator(self) -> PositionEstimator:
        cfg_state = self.cfg["state"]
        return PositionEstimator(
            alpha=cfg_state["ema_alpha"],
            max_jump_m=cfg_state["max_jump_m"],
        )

    def _init_controller(self) -> HoverController:
        return HoverController(self.cfg["control"])

    def _init_safety(self) -> SafetyMonitor:
        return SafetyMonitor(self.cfg["safety"])

    def _init_logger(self) -> FlightLogger:
        cfg_log = self.cfg.get("logging", {})
        return FlightLogger(
            log_dir=cfg_log.get("log_dir", "logs"),
            enabled=cfg_log.get("enabled", True),
            log_to_console=cfg_log.get("log_to_console", True),
            log_to_csv=cfg_log.get("log_to_csv", True),
        )

    # ── Live Preview (before takeoff) ──────────────────────────────

    def _show_live_preview(self, cam_left, cam_right, detector):
        """Show camera feeds with LED detection overlay until ENTER is pressed."""
        print("[APP] Showing live preview. Press ENTER in console to start takeoff.")
        while True:
            frame_l = cam_left.get_frame()
            frame_r = cam_right.get_frame()

            if frame_l is not None:
                result_l = detector.detect(frame_l)
                draw_debug_overlay(frame_l, result_l, "Left")
                cv2.imshow("Left Camera", frame_l)

            if frame_r is not None:
                result_r = detector.detect(frame_r)
                draw_debug_overlay(frame_r, result_r, "Right")
                cv2.imshow("Right Camera", frame_r)

            key = cv2.waitKey(30) & 0xFF
            if key == 13:  # Enter key
                break
            if key == 27:  # ESC
                raise KeyboardInterrupt("Aborted during preview")

        cv2.destroyAllWindows()

    # ── Takeoff Sequence ───────────────────────────────────────────

    def _takeoff(self, drone: DroneClient, controller: HoverController):
        """Ramp thrust from 0 to hover baseline over configured time."""
        print("\n[APP] === TAKEOFF SEQUENCE ===")
        cfg_safety = self.cfg["safety"]
        ramp_time = cfg_safety["takeoff_ramp_seconds"]
        base_thrust = self.cfg["control"]["base_thrust"]

        # Enable mode 2 (PID pitch/roll)
        drone.set_mode(2)
        drone.set_pitch(0)
        drone.set_roll(0)
        drone.set_yaw(0)

        # Ramp thrust up linearly
        start = time.time()
        while True:
            elapsed = time.time() - start
            progress = min(elapsed / ramp_time, 1.0)  # 0 → 1
            thrust = int(progress * base_thrust)
            drone.set_thrust_uniform(thrust)
            print(f"  Takeoff: {progress*100:.0f}% thrust={thrust}")

            if progress >= 1.0:
                break
            time.sleep(0.05)  # 20 Hz ramp

        print("[APP] Takeoff complete — entering hover control")

    # ── Main Control Loop ──────────────────────────────────────────

    def _control_loop(self, cam_left, cam_right, detector, triangulator,
                      estimator, controller, safety, drone, logger):
        """The main 20 Hz control loop."""
        loop_period = 1.0 / self.cfg["control"]["loop_rate_hz"]
        target = controller.target
        safe_thrust = self.cfg["safety"]["safe_hover_thrust"]

        print("\n[APP] === HOVER CONTROL ACTIVE ===")
        print("[APP] Press ESC or Space to Emergency Stop\n")

        while self._running:
            tick_start = time.time()

            # ── a. Capture frames ──
            frame_l = cam_left.get_frame()
            frame_r = cam_right.get_frame()

            # ── b. Detect LED in each frame ──
            result_l = detector.detect(frame_l) if frame_l is not None else None
            result_r = detector.detect(frame_r) if frame_r is not None else None

            left_centroid = result_l.centroid if (result_l and result_l.detected) else None
            right_centroid = result_r.centroid if (result_r and result_r.detected) else None

            # ── c. Triangulate 3D position ──
            tri_result = triangulator.triangulate(left_centroid, right_centroid)
            raw_position = tri_result["position"]
            confidence = tri_result["confidence"]
            vision_source = tri_result["source"]

            # ── d. Update state estimate ──
            est_result = estimator.update(raw_position, confidence)
            position = est_result["position"]
            velocity = est_result["velocity"]

            # ── e. Vision age (for safety) ──
            vision_age = max(cam_left.get_frame_age_ms(),
                             cam_right.get_frame_age_ms())

            # ── f. Read IMU (try, but don't crash on failure) ──
            imu_pitch, imu_roll = 0.0, 0.0
            try:
                imu_pitch = drone.get_pitch()
                imu_roll = drone.get_roll()
            except Exception:
                pass  # IMU read failure is not critical

            # ── g. Safety check ──
            safety_result = safety.check(
                position=position,
                vision_age_ms=vision_age,
                pitch_deg=imu_pitch,
                roll_deg=imu_roll,
                vision_source=vision_source,
            )

            action = safety_result["action"]

            # ── h. Compute control or respond to safety ──
            if action == "estop":
                drone.emergency_stop()
                self._running = False
                commands = {"pitch": 0, "roll": 0, "thrust": 0, "yaw": 0}
                print(f"[APP] E-STOP: {safety_result['reason']}")

            elif action == "neutralize":
                # Neutralize pitch/roll, hold safe thrust
                commands = controller.neutralize()
                commands["thrust"] = safe_thrust
                drone.set_pitch(0)
                drone.set_roll(0)
                drone.set_yaw(0)
                drone.set_thrust_uniform(commands["thrust"])

            elif action == "warning" and position is not None:
                # Compute PID but with tighter clamps
                commands = controller.compute(position, velocity)
                wc = safety_result["warning_clamps"]
                if wc:
                    commands = controller.apply_warning_clamps(
                        commands,
                        warning_pitch=wc["pitch_clamp"],
                        warning_roll=wc["roll_clamp"],
                        warning_rate=wc["rate_limit"],
                    )
                # Send commands
                drone.set_pitch(commands["pitch"])
                drone.set_roll(commands["roll"])
                drone.set_thrust_uniform(commands["thrust"])
                drone.set_yaw(commands["yaw"])

            elif position is not None:
                # Normal control
                commands = controller.compute(position, velocity)
                drone.set_pitch(commands["pitch"])
                drone.set_roll(commands["roll"])
                drone.set_thrust_uniform(commands["thrust"])
                drone.set_yaw(commands["yaw"])

            else:
                # No position estimate yet — hold current
                commands = controller.neutralize()
                commands["thrust"] = safe_thrust
                drone.set_pitch(0)
                drone.set_roll(0)
                drone.set_thrust_uniform(commands["thrust"])

            # ── i. Log everything ──
            loop_dt = (time.time() - tick_start) * 1000
            logger.log_tick(
                position=position,
                velocity=velocity,
                target=target,
                commands=commands,
                imu_pitch=imu_pitch,
                imu_roll=imu_roll,
                vision_source=vision_source,
                vision_confidence=confidence,
                safety_zone=safety_result["zone"],
                safety_action=action,
                loop_dt_ms=loop_dt,
            )

            # ── j. Debug display (optional, shows in background) ──
            if frame_l is not None and result_l:
                draw_debug_overlay(frame_l, result_l, "Left", position)
                cv2.imshow("Left", frame_l)
            if frame_r is not None and result_r:
                draw_debug_overlay(frame_r, result_r, "Right", position)
                cv2.imshow("Right", frame_r)

            # ── k. Keyboard check ──
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord(" "):  # ESC or Space
                print("[APP] MANUAL E-STOP triggered by keyboard!")
                drone.emergency_stop()
                self._running = False

            # ── l. Sleep to maintain loop rate ──
            elapsed = time.time() - tick_start
            sleep_time = loop_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── Shutdown ───────────────────────────────────────────────────

    def _shutdown(self, drone, cam_left, cam_right, logger):
        """Clean shutdown: E-stop, release cameras, close logs."""
        print("\n[APP] === SHUTDOWN ===")

        # Always try E-stop first
        if drone is not None:
            try:
                drone.emergency_stop()
                drone.set_mode(0)
                drone.disconnect()
            except Exception:
                pass

        # Release cameras
        if cam_left is not None:
            cam_left.stop()
        if cam_right is not None:
            cam_right.stop()

        # Close logger
        if logger is not None:
            logger.close()

        cv2.destroyAllWindows()
        print("[APP] Shutdown complete.")
