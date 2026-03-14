"""
test_controller.py — Unit tests for PID hover controller and safety monitor.

Tests PID math, output clamping, rate limiting, safety zones, and E-stop logic.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control.hover_controller import HoverController, PIDAxis
from src.control.safety import SafetyMonitor, SafetyZone
from src.utils.config import load_config
from src.utils.math_helpers import clamp, rate_limit, deadband, ema_update


# ── PIDAxis Tests ─────────────────────────────────────────────────

class TestPIDAxis:
    """Tests for single-axis PID controller."""

    def test_proportional_only(self):
        """P-only controller should output kp * error."""
        pid = PIDAxis(kp=2.0, ki=0.0, kd=0.0)
        output = pid.compute(1.0, dt=0.05)
        assert abs(output - 2.0) < 0.01  # kp * error = 2 * 1

    def test_zero_error_zero_output(self):
        """Zero error should produce zero output (ignoring integral)."""
        pid = PIDAxis(kp=1.0, ki=0.0, kd=0.0)
        output = pid.compute(0.0, dt=0.05)
        assert abs(output) < 0.01

    def test_negative_error(self):
        """Negative error should produce negative output."""
        pid = PIDAxis(kp=1.5, ki=0.0, kd=0.0)
        output = pid.compute(-2.0, dt=0.05)
        assert output < 0

    def test_integral_accumulates(self):
        """Integral term should accumulate over repeated calls."""
        pid = PIDAxis(kp=0.0, ki=1.0, kd=0.0, integral_limit=100)
        # Constant error of 1 for 10 ticks at dt=0.1 → integral = 1.0
        for _ in range(10):
            output = pid.compute(1.0, dt=0.1)
        # ki * integral = 1.0 * (10 * 1.0 * 0.1) = 1.0
        assert abs(output - 1.0) < 0.05

    def test_integral_windup_protection(self):
        """Integral should be clamped to integral_limit."""
        pid = PIDAxis(kp=0.0, ki=1.0, kd=0.0, integral_limit=2.0)
        # Large error for many ticks → integral should max at 2.0
        for _ in range(100):
            output = pid.compute(10.0, dt=0.1)
        assert abs(output - 2.0) < 0.1

    def test_derivative_responds_to_change(self):
        """D term should be large when error changes rapidly."""
        pid = PIDAxis(kp=0.0, ki=0.0, kd=1.0)
        pid.compute(0.0, dt=0.05)  # first call sets prev_error
        output = pid.compute(1.0, dt=0.05)  # error jumped 0→1
        # kd * (1.0 - 0.0) / 0.05 = 20.0
        assert output > 10.0  # should be large

    def test_reset_clears_state(self):
        """Reset should zero integral and prev_error."""
        pid = PIDAxis(kp=0.0, ki=1.0, kd=0.0, integral_limit=100)
        for _ in range(10):
            pid.compute(5.0, dt=0.1)
        pid.reset()
        # After reset, integral is 0 so output should be small
        output = pid.compute(0.0, dt=0.1)
        assert abs(output) < 0.01


# ── HoverController Tests ────────────────────────────────────────

class TestHoverController:
    """Tests for the full 3-axis hover controller."""

    @pytest.fixture
    def controller(self):
        """Create a controller using the default config."""
        cfg = load_config("default")
        return HoverController(cfg["control"])

    def test_at_target_zero_output(self, controller):
        """When at target [0.5, 0.5, 0.5], output should be near zero."""
        cmd = controller.compute([0.5, 0.5, 0.5])
        # Pitch and roll should be near 0
        assert abs(cmd["pitch"]) < 0.1
        assert abs(cmd["roll"]) < 0.1
        # Thrust should be near base_thrust
        assert abs(cmd["thrust"] - controller.base_thrust) <= controller.max_thrust_delta

    def test_below_target_increases_thrust(self, controller):
        """Drone below target Z should get increased thrust."""
        cmd = controller.compute([0.5, 0.5, 0.3])  # 0.2m below target
        assert cmd["thrust"] >= controller.base_thrust

    def test_pitch_roll_clamped(self, controller):
        """Pitch and roll should never exceed max values."""
        # Extreme position far from target
        controller.pid_x.reset()
        controller.pid_y.reset()
        controller._prev_roll = 0
        controller._prev_pitch = 0
        # Multiple ticks to ramp up
        for _ in range(50):
            cmd = controller.compute([0.0, 0.0, 0.5])
        assert abs(cmd["pitch"]) <= controller.max_pitch + 0.01
        assert abs(cmd["roll"]) <= controller.max_roll + 0.01

    def test_thrust_clamped(self, controller):
        """Thrust should stay within [min_thrust, max_thrust]."""
        # Way below target → max thrust request
        for _ in range(100):
            cmd = controller.compute([0.5, 0.5, 0.0])
        assert cmd["thrust"] <= controller.max_thrust
        assert cmd["thrust"] >= controller.min_thrust

    def test_rate_limiting(self, controller):
        """Consecutive commands should not change by more than max_delta."""
        cmd1 = controller.compute([0.5, 0.5, 0.5])
        # Sudden large error
        cmd2 = controller.compute([0.0, 0.0, 0.0])

        pitch_delta = abs(cmd2["pitch"] - cmd1["pitch"])
        roll_delta = abs(cmd2["roll"] - cmd1["roll"])
        thrust_delta = abs(cmd2["thrust"] - cmd1["thrust"])

        assert pitch_delta <= controller.max_angle_delta + 0.01
        assert roll_delta <= controller.max_angle_delta + 0.01
        assert thrust_delta <= controller.max_thrust_delta + 1

    def test_neutralize_zeros_angles(self, controller):
        """Neutralize should return zero pitch and roll."""
        cmd = controller.neutralize()
        assert cmd["pitch"] == 0.0
        assert cmd["roll"] == 0.0
        assert cmd["yaw"] == 0

    def test_yaw_is_zero(self, controller):
        """Normal control should output yaw=0."""
        cmd = controller.compute([0.5, 0.5, 0.5])
        assert cmd["yaw"] == 0


# ── Safety Zone Tests ─────────────────────────────────────────────

class TestSafetyZones:
    """Tests for graduated safety zone classification."""

    @pytest.fixture
    def safety(self):
        cfg = load_config("default")
        return SafetyMonitor(cfg["safety"])

    def test_center_is_safe(self, safety):
        """Dead center should be SAFE zone."""
        result = safety.check([0.5, 0.5, 0.5], vision_age_ms=10)
        assert result["zone"] == SafetyZone.SAFE
        assert result["action"] == "normal"

    def test_level1_zone(self, safety):
        """Position at 0.40 on one axis should be LEVEL1."""
        result = safety.check([0.40, 0.5, 0.5], vision_age_ms=10)
        assert result["zone"] == SafetyZone.LEVEL1
        assert result["action"] == "warning"

    def test_level2_zone(self, safety):
        """Position at 0.30 on one axis should be LEVEL2."""
        result = safety.check([0.30, 0.5, 0.5], vision_age_ms=10)
        assert result["zone"] == SafetyZone.LEVEL2

    def test_level3_zone(self, safety):
        """Position at 0.20 on one axis should be LEVEL3."""
        result = safety.check([0.20, 0.5, 0.5], vision_age_ms=10)
        assert result["zone"] == SafetyZone.LEVEL3

    def test_level4_zone(self, safety):
        """Position at 0.12 should be LEVEL4 (10-15cm from wall)."""
        result = safety.check([0.12, 0.5, 0.5], vision_age_ms=10)
        assert result["zone"] == SafetyZone.LEVEL4

    def test_estop_near_wall(self, safety):
        """Position at 0.09 (<10cm from wall) should trigger E-STOP."""
        result = safety.check([0.09, 0.5, 0.5], vision_age_ms=10)
        assert result["zone"] == SafetyZone.ESTOP
        assert result["action"] == "estop"

    def test_estop_on_other_side(self, safety):
        """Position at 0.91 should also trigger E-STOP."""
        safety.reset()
        result = safety.check([0.5, 0.5, 0.91], vision_age_ms=10)
        assert result["action"] == "estop"

    def test_worst_axis_governs(self, safety):
        """Zone should be determined by the worst axis."""
        safety.reset()
        # x is in LEVEL3 (0.20), y and z are safe
        result = safety.check([0.20, 0.50, 0.50], vision_age_ms=10)
        assert result["zone"] == SafetyZone.LEVEL3

    def test_excessive_tilt_estop(self, safety):
        """Pitch/roll exceeding max should trigger E-STOP."""
        safety.reset()
        result = safety.check([0.5, 0.5, 0.5], vision_age_ms=10,
                              pitch_deg=30.0, roll_deg=0.0)
        assert result["action"] == "estop"

    def test_stale_vision_neutralize(self, safety):
        """Stale vision data should trigger neutralize."""
        safety.reset()
        result = safety.check([0.5, 0.5, 0.5], vision_age_ms=300)
        assert result["action"] == "neutralize"

    def test_vision_lost_neutralize_then_estop(self, safety):
        """Both cameras lost should first neutralize, then E-stop."""
        safety.reset()
        # First check: just lost
        result = safety.check([0.5, 0.5, 0.5], vision_age_ms=10,
                              vision_source="none")
        assert result["action"] == "neutralize"

    def test_progressive_authority_reduction(self, safety):
        """Clamp values should decrease as zone level increases."""
        safety.reset()
        r1 = safety.check([0.40, 0.5, 0.5], vision_age_ms=10)  # LEVEL1
        safety.reset()
        r2 = safety.check([0.20, 0.5, 0.5], vision_age_ms=10)  # LEVEL3

        # LEVEL3 should have tighter clamps than LEVEL1
        c1 = r1["warning_clamps"]["pitch_clamp"]
        c2 = r2["warning_clamps"]["pitch_clamp"]
        assert c2 < c1  # tighter (smaller) at higher danger level


# ── Math Helpers Tests ────────────────────────────────────────────

class TestMathHelpers:
    """Tests for utility math functions."""

    def test_clamp_within_range(self):
        assert clamp(5, 0, 10) == 5

    def test_clamp_below(self):
        assert clamp(-5, 0, 10) == 0

    def test_clamp_above(self):
        assert clamp(15, 0, 10) == 10

    def test_rate_limit_no_change(self):
        assert rate_limit(5.0, 5.0, 1.0) == 5.0

    def test_rate_limit_caps_increase(self):
        result = rate_limit(10.0, 5.0, 2.0)
        assert result == 7.0  # 5 + 2

    def test_rate_limit_caps_decrease(self):
        result = rate_limit(0.0, 5.0, 2.0)
        assert result == 3.0  # 5 - 2

    def test_deadband_zeros_small(self):
        assert deadband(0.01, 0.05) == 0.0

    def test_deadband_passes_large(self):
        assert deadband(0.1, 0.05) == 0.1

    def test_ema_update(self):
        # alpha=1 → raw value
        assert ema_update(0.0, 1.0, 1.0) == 1.0
        # alpha=0 → old value
        assert ema_update(0.0, 1.0, 0.0) == 0.0
        # alpha=0.5 → average
        assert abs(ema_update(0.0, 1.0, 0.5) - 0.5) < 0.01
