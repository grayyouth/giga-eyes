"""
Tests for capture module.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestCaptureScreen:
    """Tests for screen capture."""

    @patch("giga_eyes.capture.screen.mss.mss")
    def test_capture_full_screen_returns_image(self, mock_mss):
        """Test that capture_full_screen returns a PIL Image."""
        from giga_eyes.capture.screen import capture_full_screen

        # Mock mss
        mock_sct = MagicMock()
        mock_screenshot = MagicMock()
        mock_screenshot.rgb = b"\x00" * (1920 * 1080 * 3)
        mock_screenshot.width = 1920
        mock_screenshot.height = 1080
        mock_screenshot.size = (1920, 1080)
        mock_sct.grab.return_value = mock_screenshot
        mock_mss.return_value.__enter__ = MagicMock(return_value=mock_sct)
        mock_mss.return_value.__exit__ = MagicMock(return_value=False)

        # Mock monitors — index 0 is primary
        mock_sct.monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
        ]

        result = capture_full_screen(monitor=0, output_path=None)

        assert result is not None
        assert result.size == (1920, 1080)


class TestCaptureWindow:
    """Tests for window capture."""

    @patch("giga_eyes.capture.window.win32gui.EnumWindows")
    @patch("giga_eyes.capture.window.win32gui.GetWindowText")
    @patch("giga_eyes.capture.window.win32gui.IsWindowVisible")
    def test_find_window_returns_hwnd(self, mock_visible, mock_title, mock_enum):
        """Test that find_window returns a valid HWND."""
        from giga_eyes.capture.window import find_window_by_title

        mock_visible.return_value = True
        mock_title.side_effect = lambda hwnd: "Test Window Title" if hwnd == 123 else "Other"

        def mock_enum_windows(callback, collected):
            callback(123, collected)

        mock_enum.side_effect = mock_enum_windows

        result = find_window_by_title("Test", partial=True)

        assert result == 123

    @patch("giga_eyes.capture.window.gw.getWindowsWithTitle")
    def test_find_window_by_title_partial(self, mock_get_windows):
        """Test partial title matching via pygetwindow."""
        from giga_eyes.capture.window import find_window_by_title

        mock_window = MagicMock()
        mock_window._hWnd = 999999
        mock_window.title = "Test Window Title"
        mock_get_windows.return_value = [mock_window]

        result = find_window_by_title("Test", partial=True)

        assert result == 999999


class TestCaptureRegion:
    """Tests for region capture."""

    @patch("giga_eyes.capture.region.mss.mss")
    def test_capture_region_returns_image(self, mock_mss):
        """Test that capture_region returns a PIL Image."""
        from giga_eyes.capture.region import capture_region

        mock_sct = MagicMock()
        mock_screenshot = MagicMock()
        mock_screenshot.rgb = b"\x00" * (800 * 600 * 3)
        mock_screenshot.width = 800
        mock_screenshot.height = 600
        mock_screenshot.size = (800, 600)
        mock_sct.grab.return_value = mock_screenshot
        mock_mss.return_value.__enter__ = MagicMock(return_value=mock_sct)
        mock_mss.return_value.__exit__ = MagicMock(return_value=False)

        result = capture_region(0, 0, 800, 600, output_path=None)

        assert result is not None
        assert result.size == (800, 600)
