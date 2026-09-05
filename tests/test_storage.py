"""
Tests for storage module.
"""

import pytest
from pathlib import Path
from PIL import Image
import tempfile
import shutil


class TestScreenshotStorage:
    """Tests for screenshot storage."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create temporary storage."""
        from giga_eyes.storage.screenshots import ScreenshotStorage
        return ScreenshotStorage(str(tmp_path / "screenshots"), "png")

    def test_save_creates_file(self, storage, tmp_path):
        """Test that save creates a PNG file."""
        img = Image.new("RGB", (100, 100), color="red")
        filepath = storage.save(img)

        assert filepath.exists()
        assert filepath.suffix == ".png"

    def test_save_with_window_title(self, storage, tmp_path):
        """Test saving with window title."""
        img = Image.new("RGB", (100, 100), color="blue")
        filepath = storage.save(img, window_title="Test Window")

        assert filepath.exists()
        assert "Test_Window" in filepath.stem

    def test_save_handles_duplicates(self, storage, tmp_path):
        """Test that duplicate filenames get counter suffix."""
        img = Image.new("RGB", (100, 100), color="green")

        # Save first time
        filepath1 = storage.save(img, custom_name="test")

        # Save second time with same name
        filepath2 = storage.save(img, custom_name="test")

        assert filepath1 != filepath2
        assert "_1" in filepath2.stem


class TestEventLogger:
    """Tests for event logging."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create temporary logger."""
        from giga_eyes.storage.logs import EventLogger
        return EventLogger(str(tmp_path / "test.log"))

    def test_log_capture(self, logger):
        """Test logging a capture event."""
        logger.log_capture(
            capture_type="screen",
            screenshot_path="/path/to/screenshot.png",
        )

        assert logger.event_count == 1

    def test_export_json(self, logger, tmp_path):
        """Test exporting events to JSON."""
        logger.log_capture(
            capture_type="window",
            window_title="Test",
            screenshot_path="/path/to/img.png",
        )

        output_path = str(tmp_path / "events.json")
        result_path = logger.export_json(output_path)

        assert Path(result_path).exists()
