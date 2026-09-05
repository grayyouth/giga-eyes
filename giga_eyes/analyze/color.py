"""
Color detection module — find colors in images.

Useful for detecting error states (red screens), UI highlights, status indicators.
"""

import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def find_color(
    image: Image.Image,
    color_rgb: Tuple[int, int, int],
    tolerance: int = 20,
) -> List[Tuple[int, int, int, int]]:
    """
    Find all occurrences of a specific color.

    Args:
        image: Input image (PIL Image).
        color_rgb: Target color (R, G, B).
        tolerance: Allowed color variation (0-255).

    Returns:
        List of (x, y, w, h) bounding boxes for color regions.
    """
    # Convert to OpenCV format (BGR)
    img_cv = _pil_to_cv2(image)
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

    # Create color range
    lower = np.array(
        [max(0, c - tolerance) for c in color_bgr], dtype=np.uint8
    )
    upper = np.array(
        [min(255, c + tolerance) for c in color_bgr], dtype=np.uint8
    )

    # Create mask
    mask = cv2.inRange(img_cv, lower, upper)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # Minimum area filter
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    logger.info(
        "Found %d regions of color RGB%s (tolerance=%d)",
        len(boxes),
        color_rgb,
        tolerance,
    )

    return boxes


def find_color_range(
    image: Image.Image,
    color_lower_rgb: Tuple[int, int, int],
    color_upper_rgb: Tuple[int, int, int],
) -> List[Tuple[int, int, int, int]]:
    """
    Find all occurrences of a color range.

    Args:
        image: Input image (PIL Image).
        color_lower_rgb: Lower bound (R, G, B).
        color_upper_rgb: Upper bound (R, G, B).

    Returns:
        List of (x, y, w, h) bounding boxes.
    """
    img_cv = _pil_to_cv2(image)
    lower_bgr = (color_lower_rgb[2], color_lower_rgb[1], color_lower_rgb[0])
    upper_bgr = (color_upper_rgb[2], color_upper_rgb[1], color_upper_rgb[0])

    lower = np.array(lower_bgr, dtype=np.uint8)
    upper = np.array(upper_bgr, dtype=np.uint8)

    mask = cv2.inRange(img_cv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    logger.info(
        "Found %d regions in color range RGB%s-RGB%s",
        len(boxes),
        color_lower_rgb,
        color_upper_rgb,
    )

    return boxes


def is_pixel_error(
    image: Image.Image,
    error_color: Tuple[int, int, int] = (255, 0, 0),
    tolerance: int = 30,
    threshold: float = 0.5,
) -> bool:
    """
    Check if image contains error state (e.g., red screen).

    Args:
        image: Input image.
        error_color: Error color (default: red).
        tolerance: Color tolerance.
        threshold: Minimum percentage of error pixels to trigger.

    Returns:
        True if error state detected.
    """
    boxes = find_color(image, error_color, tolerance)

    # Calculate total error area
    total_area = 0
    for x, y, w, h in boxes:
        total_area += w * h

    image_area = image.width * image.height
    error_percentage = total_area / image_area

    is_error = error_percentage > threshold

    logger.info(
        "Error detection: %.1f%% error pixels (threshold=%.1f%%) -> %s",
        error_percentage * 100,
        threshold * 100,
        "ERROR" if is_error else "OK",
    )

    return is_error


def get_dominant_colors(
    image: Image.Image,
    num_colors: int = 5,
) -> List[Tuple[int, int, int, float]]:
    """
    Get dominant colors in an image.

    Args:
        image: Input image.
        num_colors: Number of colors to return.

    Returns:
        List of (R, G, B, percentage) tuples, sorted by percentage.
    """
    img_cv = _pil_to_cv2(image)
    img_flat = img_cv.reshape(-1, 3)

    # Calculate color frequencies
    color_counts = {}
    for pixel in img_flat:
        key = tuple(pixel)
        color_counts[key] = color_counts.get(key, 0) + 1

    # Sort by frequency
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

    # Convert to (R, G, B, percentage)
    total_pixels = image.width * image.height
    dominant = []
    for (b, g, r), count in sorted_colors[:num_colors]:
        percentage = count / total_pixels
        dominant.append((r, g, b, percentage))

    return dominant


def _pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    img_rgb = np.array(image)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
