"""
safety.py — Safety monitor that enforces boundaries and triggers E-stop.

Checks every control tick:
  - Is the drone within the virtual safe cage?
  - Is vision data fresh enough?
  - Are tilt angles within limits?
  - Should we E-stop?

The drone NEVER voluntarily descends. It fights till E-stop.
"""

import time
from src.utils.math_helpers import clamp


class SafetyZone:
    """Classify drone position into graduated safety zones.

    Zones are defined by position boundaries on each axis (cage is 0 to 1m).
    The drone's target is at cage center [0.5, 0.5, 0.5].

    Zone levels (by worst axis):
      SAFE (0):  all axes within [0.45, 0.55]  — ≥45cm from wall
      LEVEL1 (1): any axis in [0.35, 0.45) or (0.55, 0.65]  — ≥35cm
      LEVEL2 (2): any axis in [0.25, 0.35) or (0.65, 0.75]  — ≥25cm
      LEVEL3 (3): any axis in [0.15, 0.25) or (0.75, 0.85]  — ≥15cm
      LEVEL4 (4): any axis in [0.10, 0.15) or (0.85, 0.90]  — ≥10cm
      ESTOP (5):  any axis outside [0.10, 0.90]              — <10cm from wall
    """

    # Zone constants — higher number = more dangerous
    SAFE   = 0   # center, full authority
    LEVEL1 = 1   # light dampening
    LEVEL2 = 2   # moderate dampening + centering bias
    LEVEL3 = 3   # strong dampening + strong bias
    LEVEL4 = 4   # maximum restriction before E-stop
    ESTOP  = 5   # kill motors

    # Zone boundaries: (inner_lo, inner_hi) — if position is within this
    # range on ALL axes, the drone is at least at this safety level.
    # Ordered from safest to most dangerous.
    ZONE_BOUNDS = [
        (0.45, 0.55),  # SAFE
        (0.35, 0.65),  # LEVEL1
        (0.25, 0.75),  # LEVEL2
        (0.15, 0.85),  # LEVEL3
        (0.10, 0.90),  # LEVEL4
    ]

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: the "safety" section of default.yaml
        """
        self.cage_size = cfg["cage_size"]

    def classify(self, position: list) -> int:
        """Classify [x, y, z] into a graduated safety zone.

        Returns the WORST (highest number) zone across all axes.
        """
        if position is None:
            return self.LEVEL3  # unknown position is dangerous

        worst_zone = self.SAFE

        for val in position:
            # Check each axis against zone boundaries (inner → outer)
            axis_zone = self.ESTOP  # assume worst until proven otherwise
            for level, (lo, hi) in enumerate(self.ZONE_BOUNDS):
                if lo <= val <= hi:
                    axis_zone = level
                    break  # found the tightest zone this axis fits in

            # Track the worst zone across all axes
            worst_zone = max(worst_zone, axis_zone)

        return worst_zone


class SafetyMonitor:
    """Monitors all safety conditions and triggers appropriate responses.

    Usage:
        safety = SafetyMonitor(cfg["safety"])
        action = safety.check(position, vision_age_ms, pitch_deg, roll_deg)
        if action == "estop":
            drone.emergency_stop()
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = cfg.get("enabled", True)
        self.zone_checker = SafetyZone(cfg)

        # Thresholds from config
        self.vision_stale_ms = cfg["vision_stale_ms"]
        self.vision_lost_ms = cfg["vision_lost_ms"]
        self.max_pitch_deg = cfg["max_pitch_deg"]
        self.max_roll_deg = cfg["max_roll_deg"]
        self.single_cam_fallback_ms = cfg["single_cam_fallback_ms"]
        self.safe_hover_thrust = cfg["safe_hover_thrust"]

        # Warning zone limits
        self.warning_pitch_clamp = cfg.get("warning_pitch_clamp", 1.5)
        self.warning_roll_clamp = cfg.get("warning_roll_clamp", 1.5)
        self.warning_rate_limit = cfg.get("warning_rate_limit", 0.5)

        # Internal state
        self._estop_triggered = False
        self._estop_reason = ""
        self._vision_lost_since = None
        self._single_cam_since = None

    def check(self, position: list, vision_age_ms: float,
              pitch_deg: float = 0.0, roll_deg: float = 0.0,
              vision_source: str = "both") -> dict:
        """Run all safety checks.

        Args:
            position: smoothed [x, y, z] in meters, or None
            vision_age_ms: time since last valid vision update
            pitch_deg: current drone pitch from IMU
            roll_deg: current drone roll from IMU
            vision_source: "both", "left_only", "right_only", or "none"

        Returns:
            dict with:
              - "action": "normal", "warning", "neutralize", or "estop"
              - "zone": SafetyZone constant
              - "reason": human-readable explanation
              - "warning_clamps": dict if in warning zone, else None
        """
        if not self.enabled:
            return {"action": "normal", "zone": SafetyZone.SAFE,
                    "reason": "safety disabled", "warning_clamps": None}

        if self._estop_triggered:
            return {"action": "estop", "zone": SafetyZone.ESTOP,
                    "reason": f"E-stop already triggered: {self._estop_reason}",
                    "warning_clamps": None}

        # ── Check 1: Excessive tilt ──
        if abs(pitch_deg) > self.max_pitch_deg or abs(roll_deg) > self.max_roll_deg:
            return self._trigger_estop(
                f"Excessive tilt: pitch={pitch_deg:.1f}°, roll={roll_deg:.1f}°")

        # ── Check 2: Vision loss ──
        if vision_source == "none":
            if self._vision_lost_since is None:
                self._vision_lost_since = time.time()
            elapsed_ms = (time.time() - self._vision_lost_since) * 1000

            if elapsed_ms > self.vision_lost_ms:
                return self._trigger_estop(
                    f"Both cameras lost for {elapsed_ms:.0f} ms")
            else:
                # Brief loss: neutralize pitch/roll, hold safe thrust
                return {"action": "neutralize", "zone": SafetyZone.LEVEL3,
                        "reason": f"Vision lost {elapsed_ms:.0f}ms, holding",
                        "warning_clamps": None}
        else:
            self._vision_lost_since = None  # reset on recovery

        # ── Check 3: Single camera fallback timeout ──
        if vision_source in ("left_only", "right_only"):
            if self._single_cam_since is None:
                self._single_cam_since = time.time()
            elapsed_ms = (time.time() - self._single_cam_since) * 1000

            if elapsed_ms > self.single_cam_fallback_ms:
                # Too long on one camera → neutralize with hover hold
                return {"action": "neutralize", "zone": SafetyZone.LEVEL2,
                        "reason": f"Single camera for {elapsed_ms:.0f}ms, neutralizing",
                        "warning_clamps": None}
        else:
            self._single_cam_since = None

        # ── Check 4: Stale vision data ──
        if vision_age_ms > self.vision_stale_ms:
            return {"action": "neutralize", "zone": SafetyZone.LEVEL2,
                    "reason": f"Vision stale: {vision_age_ms:.0f}ms",
                    "warning_clamps": None}

        # ── Check 5: Position boundaries (graduated zones) ──
        zone = self.zone_checker.classify(position)

        if zone == SafetyZone.ESTOP:
            return self._trigger_estop(
                f"Position outside [0.10, 0.90] — too close to cage: {position}")

        if zone == SafetyZone.LEVEL4:
            # 10–15cm from wall — maximum restriction before E-stop
            return {
                "action": "warning",
                "zone": zone,
                "reason": f"LEVEL4 — 10-15cm from wall: {position}",
                "warning_clamps": {
                    "pitch_clamp": self.warning_pitch_clamp * 0.25,
                    "roll_clamp": self.warning_roll_clamp * 0.25,
                    "rate_limit": self.warning_rate_limit * 0.25,
                },
            }

        if zone == SafetyZone.LEVEL3:
            # 15–25cm from wall — strong dampening + strong centering bias
            return {
                "action": "warning",
                "zone": zone,
                "reason": f"LEVEL3 — 15-25cm from wall: {position}",
                "warning_clamps": {
                    "pitch_clamp": self.warning_pitch_clamp * 0.4,
                    "roll_clamp": self.warning_roll_clamp * 0.4,
                    "rate_limit": self.warning_rate_limit * 0.4,
                },
            }

        if zone == SafetyZone.LEVEL2:
            # 25–35cm from wall — moderate dampening + centering bias
            return {
                "action": "warning",
                "zone": zone,
                "reason": f"LEVEL2 — 25-35cm from wall: {position}",
                "warning_clamps": {
                    "pitch_clamp": self.warning_pitch_clamp * 0.6,
                    "roll_clamp": self.warning_roll_clamp * 0.6,
                    "rate_limit": self.warning_rate_limit * 0.6,
                },
            }

        if zone == SafetyZone.LEVEL1:
            # 35–45cm from wall — light dampening
            return {
                "action": "warning",
                "zone": zone,
                "reason": f"LEVEL1 — 35-45cm from wall: {position}",
                "warning_clamps": {
                    "pitch_clamp": self.warning_pitch_clamp * 0.8,
                    "roll_clamp": self.warning_roll_clamp * 0.8,
                    "rate_limit": self.warning_rate_limit * 0.8,
                },
            }

        # SAFE — within [0.45, 0.55] on all axes, full authority
        return {"action": "normal", "zone": SafetyZone.SAFE,
                "reason": "OK", "warning_clamps": None}

    def _trigger_estop(self, reason: str) -> dict:
        """Record E-stop and return estop action."""
        self._estop_triggered = True
        self._estop_reason = reason
        print(f"[SAFETY] *** E-STOP: {reason} ***")
        return {"action": "estop", "zone": SafetyZone.ESTOP,
                "reason": reason, "warning_clamps": None}

    def reset(self):
        """Reset E-stop state (for re-arm after resolving issue)."""
        self._estop_triggered = False
        self._estop_reason = ""
        self._vision_lost_since = None
        self._single_cam_since = None

    @property
    def is_estopped(self) -> bool:
        return self._estop_triggered
