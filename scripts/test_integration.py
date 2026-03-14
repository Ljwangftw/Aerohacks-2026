"""Quick integration test — verifies all components instantiate and work together."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.control.hover_controller import HoverController
from src.control.safety import SafetyMonitor
from src.state.estimator import PositionEstimator
from src.vision.detection import LEDDetector
from src.vision.triangulation import OrthogonalTriangulator

cfg = load_config("default")
print("Config loaded OK")

det = LEDDetector(cfg["detection"]["hsv_lower"], cfg["detection"]["hsv_upper"])
print("Detector OK")

tri = OrthogonalTriangulator(cfg["triangulation"]["cam_left"], cfg["triangulation"]["cam_right"])
print("Triangulator OK")

est = PositionEstimator(alpha=cfg["state"]["ema_alpha"])
print("Estimator OK")

ctrl = HoverController(cfg["control"])
print(f"Controller OK, target={ctrl.target}")

safety = SafetyMonitor(cfg["safety"])
print("Safety OK")

# Test PID computation with synthetic data
cmd = ctrl.compute([0.5, 0.5, 0.4])
print(f"PID test: pos=[0.5,0.5,0.4] -> pitch={cmd['pitch']}, roll={cmd['roll']}, thrust={cmd['thrust']}")

# Test safety zone
result = safety.check([0.5, 0.5, 0.5], vision_age_ms=10)
print(f"Safety test: center -> {result['action']}")

result2 = safety.check([0.04, 0.5, 0.5], vision_age_ms=10)
print(f"Safety test: near wall -> {result2['action']}")

# Test estimator
est_result = est.update([0.5, 0.5, 0.5], confidence=1.0)
print(f"Estimator test: {est_result['position']} reliable={est_result['reliable']}")

# Test triangulation
tri_result = tri.triangulate((320, 240), (320, 240))
print(f"Triangulation test: {tri_result['position']} source={tri_result['source']}")

print("\nALL INTEGRATION TESTS PASS")
