"""Quick test to verify all graduated safety zones work correctly."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.control.safety import SafetyMonitor, SafetyZone
from src.utils.config import load_config

cfg = load_config("default")
s = SafetyMonitor(cfg["safety"])

# Test positions at various distances from wall (x-axis, others centered)
tests = [
    ([0.50, 0.50, 0.50], "dead center"),
    ([0.46, 0.50, 0.50], "1cm from safe edge"),
    ([0.40, 0.50, 0.50], "in LEVEL1 band"),
    ([0.30, 0.50, 0.50], "in LEVEL2 band"),
    ([0.20, 0.50, 0.50], "in LEVEL3 band"),
    ([0.12, 0.50, 0.50], "in LEVEL4 band"),
    ([0.09, 0.50, 0.50], "outside 0.10 -> ESTOP"),
    ([0.05, 0.50, 0.50], "5cm from wall -> ESTOP"),
    ([0.50, 0.50, 0.91], "z near ceiling -> ESTOP"),
    ([0.35, 0.35, 0.35], "all axes in LEVEL1"),
    ([0.15, 0.85, 0.50], "multi-axis danger"),
]

print("Graduated Safety Zone Test")
print("=" * 80)
for pos, desc in tests:
    result = s.check(pos, vision_age_ms=10)
    zone = result["zone"]
    action = result["action"]
    reason = result["reason"]
    print(f"  pos={pos}  zone={zone}  action={action:10s}  | {desc}")
    # Reset after E-stop for next test
    if action == "estop":
        s.reset()

print("=" * 80)
print("DONE")
