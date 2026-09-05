"""
Screenshot storage module.

Handles saving screenshots with auto-naming and metadata.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class ScreenshotStorage:
    """Manages screenshot storage with auto-naming."""

    def __init__(self, storage_path: str = "screenshots", format: str = "png"):
        """
        Args:
            storage_path: Directory to save screenshots.
            format: Default format ('png', 'jpg', 'bmp').
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.format = format

    def save(
        self,
        image: Image.Image,
        window_title: Optional[str] = None,
        region: Optional[tuple] = None,
        custom_name: Optional[str] = None,
    ) -> Path:
        """
        Save screenshot with auto-generated filename.

        Args:
            image: PIL.Image to save.
            window_title: Optional window title for naming.
            region: Optional (x, y, w, h) tuple.
            custom_name: Optional custom filename (without extension).

        Returns:
            Path to saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if custom_name:
            filename = f"{custom_name}.{self.format}"
        elif window_title:
            # Sanitize window title for filename
            safe_title = self._sanitize_filename(window_title)
            filename = f"{timestamp}_{safe_title}.{self.format}"
        else:
            filename = f"{timestamp}_screenshot.{self.format}"

        filepath = self.storage_path / filename

        # Handle duplicates
        if filepath.exists():
            counter = 1
            while (self.storage_path / f"{filepath.stem}_{counter}.{self.format}").exists():
                counter += 1
            filepath = self.storage_path / f"{filepath.stem}_{counter}.{self.format}"

        # Convert RGBA to RGB for JPEG
        save_kwargs = {}
        if self.format == "jpg":
            if image.mode == "RGBA":
                image = image.convert("RGB")
            save_kwargs["quality"] = 95

        image.save(filepath, **save_kwargs)
        logger.info("Screenshot saved: %s", filepath)

        return filepath

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove characters invalid in filenames."""
        invalid_chars = '<>:"/\\|?* '
        for char in invalid_chars:
            name = name.replace(char, "_")
        # Collapse multiple underscores
        while "__" in name:
            name = name.replace("__", "_")
        return name[:50].strip("_")
