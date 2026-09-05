"""
Window capture module — captures a specific window by title or HWND.

Uses `win32gui` + `mss` for window detection and `win32ui` for clean
capture of background/obscured windows.
"""

import logging
from typing import Optional

import pygetwindow as gw
import win32gui
import win32ui
import win32con
from PIL import Image

logger = logging.getLogger(__name__)


def find_window_by_title(title: str, partial: bool = True) -> Optional[int]:
    """
    Find window HWND by title.

    Args:
        title: Window title (exact or partial).
        partial: If True, search for partial match.

    Returns:
        HWND (int) or None if not found.
    """
    if partial:
        try:
            windows = gw.getWindowsWithTitle(title)
            if windows:
                # Return the first (most recent) match
                logger.debug("Found window '%s': %s", title, windows[0].title)
                return windows[0]._hWnd
        except Exception as e:
            logger.warning("pygetwindow search failed: %s", e)

    # Fallback: win32gui EnumWindows
    hwnds = []

    def callback(hwnd, collection):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if title.lower() in window_title.lower():
                collection.append(hwnd)

    collected = []
    win32gui.EnumWindows(callback, collected)

    if collected:
        logger.debug("Found window '%s' via EnumWindows: %s", title, collected[0])
        return collected[0]

    logger.debug("Window '%s' not found", title)
    return None


def capture_window_by_title(
    title: str,
    partial: bool = True,
    output_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Capture a window by its title.

    Args:
        title: Window title.
        partial: Use partial match.
        output_path: If provided, save to file.

    Returns:
        PIL.Image or None.
    """
    hwnd = find_window_by_title(title, partial)
    if not hwnd:
        logger.error("Window '%s' not found", title)
        return None

    return _capture_window(hwnd, output_path)


def capture_window_by_hwnd(
    hwnd: int,
    output_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Capture a window by its HWND.

    Args:
        hwnd: Window handle.
        output_path: If provided, save to file.

    Returns:
        PIL.Image or None.
    """
    return _capture_window(hwnd, output_path)


def _capture_window(hwnd: int, output_path: Optional[str] = None) -> Optional[Image.Image]:
    """
    Internal: capture window using win32ui BitBlt (works for background windows).
    Falls back to mss if BitBlt returns empty/red image.

    Args:
        hwnd: Window handle.
        output_path: If provided, save to file.

    Returns:
        PIL.Image or None.
    """
    # Try win32ui BitBlt first
    image = _capture_with_bitblt(hwnd, output_path)
    if image is not None:
        # Check if image is valid (not all red/empty)
        import numpy as np
        arr = np.array(image)
        # Check if image is mostly solid red (common BitBlt failure mode)
        if arr.shape == (0,):
            logger.warning("BitBlt returned empty image, falling back to mss")
        else:
            mean_color = arr.mean(axis=(0, 1))
            # If mostly solid red (error screen), use mss fallback
            if mean_color[0] > 200 and mean_color[1] < 50 and mean_color[2] < 50:
                logger.warning("BitBlt returned solid red image, falling back to mss")
            else:
                return image

    # Fallback: mss by coordinates
    logger.info("Falling back to mss capture by coordinates")
    return _capture_with_mss(hwnd, output_path)


def _capture_with_bitblt(hwnd: int, output_path: Optional[str] = None) -> Optional[Image.Image]:
    """Capture using win32ui BitBlt."""
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            logger.warning("Window is minimized or has zero size: %dx%d", width, height)
            return None

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        image = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        if output_path:
            image.save(output_path)
            logger.info("Window screenshot saved to: %s", output_path)
        else:
            logger.debug("Window captured via BitBlt: %dx%d", width, height)
            return image

    except Exception as e:
        logger.error("BitBlt capture failed: %s", e)
        return None


def _capture_with_mss(hwnd: int, output_path: Optional[str] = None) -> Optional[Image.Image]:
    """Capture using mss by window coordinates (fallback)."""
    try:
        import mss
        import mss.tools

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            logger.warning("Window is minimized: %dx%d", width, height)
            return None

        with mss.mss() as sct:
            monitor = {'left': left, 'top': top, 'width': width, 'height': height}
            screenshot = sct.grab(monitor)

            image = Image.frombytes("RGB", (width, height), screenshot.rgb)

        if output_path:
            image.save(output_path)
            logger.info("Window screenshot saved via mss to: %s", output_path)
        else:
            logger.debug("Window captured via mss: %dx%d", width, height)
            return image

    except Exception as e:
        logger.error("mss fallback capture failed: %s", e)
        return None
