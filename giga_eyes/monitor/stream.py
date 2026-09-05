"""
Continuous stream module — capture frames from a window in real-time.

Provides a streaming interface for real-time monitoring with callbacks.
"""

import logging
import threading
import time
from typing import Optional, Callable, List
from queue import Queue, Empty

from PIL import Image

from giga_eyes.capture.window import capture_window_by_title, find_window_by_title
from giga_eyes.capture.screen import capture_full_screen

logger = logging.getLogger(__name__)


class StreamBuffer:
    """Circular buffer for recent frames."""

    def __init__(self, size: int = 30):
        """
        Args:
            size: Maximum number of frames to keep.
        """
        self.size = size
        self._buffer: List[dict] = []
        self._lock = threading.Lock()

    def add(self, frame: Image.Image, timestamp: float, metadata: Optional[dict] = None):
        """Add a frame to the buffer."""
        with self._lock:
            self._buffer.append({
                "frame": frame,
                "timestamp": timestamp,
                "metadata": metadata or {},
            })
            if len(self._buffer) > self.size:
                self._buffer.pop(0)

    def get_recent(self, count: int = 5) -> List[dict]:
        """Get the most recent N frames."""
        with self._lock:
            return list(self._buffer[-count:])

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def latest(self) -> Optional[dict]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None


class StreamCallback:
    """Callback handler for stream events."""

    def __init__(self):
        self._callbacks = {
            "frame": [],
            "change": [],
            "error": [],
            "start": [],
            "stop": [],
        }

    def on(self, event: str, callback: Callable):
        """
        Register a callback for an event.

        Args:
            event: Event name ('frame', 'change', 'error', 'start', 'stop').
            callback: Function to call.
        """
        if event not in self._callbacks:
            raise ValueError(f"Unknown event: {event}")
        self._callbacks[event].append(callback)

    def emit(self, event: str, **kwargs):
        """
        Emit an event to all registered callbacks.

        Args:
            event: Event name.
            **kwargs: Arguments to pass to callbacks.
        """
        for cb in self._callbacks.get(event, []):
            try:
                cb(**kwargs)
            except Exception as e:
                logger.error("Callback error on event '%s': %s", event, e)


class WindowStream:
    """
    Continuous frame stream from a window.

    Captures frames at a specified FPS and triggers callbacks on changes.
    """

    def __init__(
        self,
        window_title: Optional[str] = None,
        hwnd: Optional[int] = None,
        fps: int = 10,
        buffer_size: int = 30,
        change_threshold: float = 0.01,
    ):
        """
        Args:
            window_title: Window title to capture (or None for full screen).
            hwnd: Window handle (alternative to title).
            fps: Frames per second.
            buffer_size: StreamBuffer size.
            change_threshold: Minimum change percentage to trigger 'change' event.
        """
        self.window_title = window_title
        self.hwnd = hwnd
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.change_threshold = change_threshold

        self.buffer = StreamBuffer(buffer_size)
        self.callback = StreamCallback()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None

    def start(self):
        """Start the stream in a background thread."""
        if self._running:
            logger.warning("Stream already running")
            return

        self._running = True
        self._frame_count = 0
        self._start_time = time.time()

        self.callback.emit("start")
        logger.info(
            "Stream started: title=%s, fps=%d",
            self.window_title or "full screen",
            self.fps,
        )

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the stream."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._running = False

        elapsed = time.time() - self._start_time if self._start_time else 0
        self.callback.emit(
            "stop",
            frames=self._frame_count,
            duration=elapsed,
        )
        logger.info(
            "Stream stopped: %d frames in %.1fs",
            self._frame_count,
            elapsed,
        )

    def _capture_loop(self):
        """Main capture loop."""
        import cv2
        import numpy as np

        last_frame: Optional[np.ndarray] = None

        while self._running:
            try:
                # Capture frame
                if self.window_title:
                    img = capture_window_by_title(
                        self.window_title, output_path=None
                    )
                elif self.hwnd:
                    from giga_eyes.capture.window import capture_window_by_hwnd
                    img = capture_window_by_hwnd(self.hwnd, output_path=None)
                else:
                    img = capture_full_screen(output_path=None)

                if img is None:
                    self.callback.emit("error", error="Failed to capture frame")
                    time.sleep(0.1)
                    continue

                # Convert to numpy for diff
                frame_cv = np.array(img)

                # Detect changes
                if last_frame is not None:
                    diff = cv2.absdiff(last_frame, frame_cv)
                    if len(diff.shape) == 3:
                        diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                    change_pct = np.sum(diff > 10) / diff.size

                    if change_pct > self.change_threshold:
                        self.callback.emit(
                            "change",
                            change_percentage=change_pct * 100,
                            frame=img,
                            frame_index=self._frame_count,
                        )

                last_frame = frame_cv

                # Store in buffer
                timestamp = time.time()
                self.buffer.add(img, timestamp, {
                    "frame_index": self._frame_count,
                    "elapsed": timestamp - self._start_time if self._start_time else 0,
                })

                # Emit frame event
                self.callback.emit(
                    "frame",
                    frame=img,
                    frame_index=self._frame_count,
                    timestamp=timestamp,
                )

                self._frame_count += 1

                # Maintain FPS
                elapsed = time.time() - (self._start_time + self._frame_count * self.frame_interval)
                if elapsed < 0:
                    time.sleep(-elapsed)

            except Exception as e:
                logger.error("Stream capture error: %s", e)
                self.callback.emit("error", error=str(e))
                time.sleep(0.1)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._running

    def get_latest_frame(self) -> Optional[Image.Image]:
        """Get the most recent frame from the buffer."""
        latest = self.buffer.latest
        return latest["frame"] if latest else None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
