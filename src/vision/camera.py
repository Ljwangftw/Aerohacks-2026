"""
camera.py — Threaded camera capture for low-latency frame grabbing.

Each Camera instance runs a background thread that continuously reads frames
so the main loop always gets the latest frame without blocking on I/O.
"""

import threading
import time
import cv2


class Camera:
    """Threaded wrapper around OpenCV VideoCapture.

    Usage:
        cam = Camera(device_id=0, width=640, height=480, fps=30, name="left")
        cam.start()
        frame = cam.get_frame()   # returns latest frame (numpy array) or None
        cam.stop()
    """

    def __init__(self, device_id: int, width: int = 640, height: int = 480,
                 fps: int = 30, name: str = "camera"):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name

        self._cap = None           # OpenCV VideoCapture object
        self._frame = None         # latest captured frame
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frame_count = 0
        self._last_frame_time = 0.0

    def start(self) -> bool:
        """Open the camera and start the capture thread.

        Returns True if camera opened successfully, False otherwise.
        """
        self._cap = cv2.VideoCapture(self.device_id)

        if not self._cap.isOpened():
            print(f"[Camera:{self.name}] ERROR: could not open device {self.device_id}")
            return False

        # Set resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Reduce internal buffer to 1 frame for lowest latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print(f"[Camera:{self.name}] started (device {self.device_id}, "
              f"{self.width}x{self.height} @ {self.fps} fps)")
        return True

    def _capture_loop(self):
        """Background thread: continuously grabs frames."""
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._frame_count += 1
                    self._last_frame_time = time.time()
            else:
                # Brief sleep on failure to avoid busy-spin
                time.sleep(0.001)

    def get_frame(self):
        """Return the latest frame (numpy BGR array) or None if no frame yet."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def get_frame_age_ms(self) -> float:
        """How many milliseconds since the last successful frame capture."""
        with self._lock:
            if self._last_frame_time == 0:
                return float("inf")
            return (time.time() - self._last_frame_time) * 1000.0

    @property
    def frame_count(self) -> int:
        """Total frames captured so far."""
        return self._frame_count

    def warmup(self, num_frames: int = 30):
        """Discard initial frames to let auto-exposure stabilize."""
        print(f"[Camera:{self.name}] warming up ({num_frames} frames)...")
        while self._frame_count < num_frames:
            time.sleep(0.05)
        print(f"[Camera:{self.name}] warmup complete")

    def stop(self):
        """Stop capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        print(f"[Camera:{self.name}] stopped")

    @property
    def is_running(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()
