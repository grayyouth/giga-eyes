"""
Screen capture module — captures the entire screen.

Uses `mss` for high-performance screen capture on Windows.
"""

import logging
from typing import Optional

import mss
import mss.tools
from PIL import Image

logger = logging.getLogger(__name__)


def capture_full_screen(
    monitor: int = 1,
    output_path: Optional[str] = None,
    format: str = "png",
) -> Optional[Image.Image]:
    """
    Capture the entire screen (or specific monitor).

    Args:
        monitor: Monitor index (1 = primary, 2+ = secondary).
        output_path: If provided, save screenshot to file.
        format: Output format ('png', 'jpg', 'bmp').

    Returns:
        PIL.Image if output_path is None, else None.
    """
    with mss.mss() as sct:
        # Capture full monitor
        monitor_data = sct.monitors[monitor]
        screenshot = sct.grab({
            "left": monitor_data["left"],
            "top": monitor_data["top"],
            "width": monitor_data["width"],
            "height": monitor_data["height"],
        })

        if output_path:
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_path)
            logger.info("Screenshot saved to: %s", output_path)
            return None
        else:
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            logger.debug("Screen captured: %dx%d", screenshot.width, screenshot.height)
            return img
