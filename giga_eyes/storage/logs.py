"""
Event logging module.

Logs capture events with timestamps and metadata.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EventLogger:
    """Logs capture events to JSON file."""

    def __init__(self, log_path: str = "logs/giga_eyes.log"):
        """
        Args:
            log_path: Path to log file.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._events = []

    def log_capture(
        self,
        capture_type: str,
        window_title: Optional[str] = None,
        region: Optional[tuple] = None,
        screenshot_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Log a capture event.

        Args:
            capture_type: Type of capture ('screen', 'window', 'region').
            window_title: Window title (for window captures).
            region: (x, y, w, h) tuple (for region captures).
            screenshot_path: Path to saved screenshot.
            metadata: Additional metadata.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": capture_type,
            "window_title": window_title,
            "region": list(region) if region else None,
            "screenshot_path": screenshot_path,
            "metadata": metadata or {},
        }

        self._events.append(event)

        # Also write to file immediately
        self._write_event(event)

        logger.info(
            "Event logged: type=%s, window=%s, path=%s",
            capture_type,
            window_title,
            screenshot_path,
        )

    def _write_event(self, event: dict):
        """Append event to log file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Failed to write log event: %s", e)

    def export_json(self, output_path: Optional[str] = None) -> str:
        """
        Export all logged events to a JSON file.

        Args:
            output_path: Output path (defaults to logs/events.json).

        Returns:
            Path to exported file.
        """
        if not output_path:
            output_path = "logs/events.json"

        filepath = Path(output_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total_events": len(self._events),
            "events": self._events,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Events exported to: %s", filepath)
        return str(filepath)

    @property
    def event_count(self) -> int:
        return len(self._events)
