"""
Region capture module — captures an arbitrary screen region.

Uses `mss` for fast region capture.
"""

import logging
from typing import Optional

import mss.tools
from PIL import Image

logger = logging.getLogger(__name__)


def capture_region(
    x: int,
    y: int,
    width: int,
    height: int,
    output_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Capture an arbitrary region of the screen.

    Args:
        x: Left coordinate.
        y: Top coordinate.
        width: Region width.
        height: Region height.
        output_path: If provided, save to file.

    Returns:
        PIL.Image or None.
    """
    if width <= 0 or height <= 0:
        logger.error("Invalid region size: %dx%d", width, height)
        return None

    try:
        with mss.mss() as sct:
            screenshot = sct.grab({
                "left": x,
                "top": y,
                "width": width,
                "height": height,
            })

            if output_path:
                mss.tools.to_png(
                    screenshot.rgb,
                    screenshot.width,
                    screenshot.height,
                    output=output_path,
                )
                logger.info("Region screenshot saved to: %s", output_path)
            else:
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                logger.debug("Region captured: %dx%d at (%d, %d)", width, height, x, y)
                return img

    except Exception as e:
        logger.error("Failed to capture region: %s", e)
        return None
