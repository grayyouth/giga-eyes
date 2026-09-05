"""
Frame diff detection module — compare two images and find changes.

Useful for detecting UI changes, animation frames, error states.
"""

import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def find_changes(
    frame_before: Image.Image,
    frame_after: Image.Image,
    threshold: int = 10,
) -> np.ndarray:
    """
    Find changed pixels between two frames.

    Args:
        frame_before: Initial frame (PIL Image).
        frame_after: Later frame (PIL Image).
        threshold: Pixel difference threshold (0-255).

    Returns:
        Binary mask where changed pixels are white (255).
    """
    # Convert to OpenCV format
    img_before = _pil_to_cv2(frame_before)
    img_after = _pil_to_cv2(frame_after)

    # Ensure same size
    if img_before.shape != img_after.shape:
        img_after = cv2.resize(img_after, (img_before.shape[1], img_before.shape[0]))
        logger.warning(
            "Frame sizes differ. Resized after to match before: %s",
            img_after.shape,
        )

    # Convert to grayscale
    if len(img_before.shape) == 3:
        gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    else:
        gray_before = img_before
        gray_after = img_after

    # Calculate absolute difference
    diff = cv2.absdiff(gray_before, gray_after)

    # Apply threshold
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    logger.info(
        "Diff detection: %d changed pixels out of %d (%.1f%%)",
        int(np.sum(mask > 0)),
        mask.size,
        100.0 * np.sum(mask > 0) / mask.size,
    )

    return mask


def count_changes(mask: np.ndarray) -> int:
    """
    Count changed pixels in a diff mask.

    Args:
        mask: Binary diff mask.

    Returns:
        Number of changed pixels.
    """
    return int(np.sum(mask > 0))


def get_change_percentage(mask: np.ndarray) -> float:
    """
    Get percentage of changed pixels.

    Args:
        mask: Binary diff mask.

    Returns:
        Percentage (0.0-100.0).
    """
    return 100.0 * np.sum(mask > 0) / mask.size


def get_bounding_boxes(mask: np.ndarray, min_area: int = 100) -> List[Tuple[int, int, int, int]]:
    """
    Get bounding boxes for changed regions.

    Args:
        mask: Binary diff mask.
        min_area: Minimum area for a bounding box (in pixels).

    Returns:
        List of (x, y, w, h) tuples.
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    logger.info("Found %d changed regions (min_area=%d)", len(boxes), min_area)
    return boxes


def highlight_changes(
    original: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> Image.Image:
    """
    Highlight changed regions on the original image.

    Args:
        original: Original frame (PIL Image).
        mask: Binary diff mask.
        color: Highlight color (BGR).
        thickness: Line thickness for bounding boxes.

    Returns:
        PIL Image with highlighted changes.
    """
    img_cv = _pil_to_cv2(original)

    # Draw bounding boxes
    boxes = get_bounding_boxes(mask, min_area=50)
    for x, y, w, h in boxes:
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, thickness)

    # Convert back to PIL
    return _cv2_to_pil(img_cv)


def compare_frames(
    frame_before: Image.Image,
    frame_after: Image.Image,
    threshold: int = 10,
) -> dict:
    """
    Comprehensive frame comparison.

    Args:
        frame_before: Initial frame.
        frame_after: Later frame.
        threshold: Pixel difference threshold.

    Returns:
        Dict with change statistics and bounding boxes.
    """
    mask = find_changes(frame_before, frame_after, threshold)
    boxes = get_bounding_boxes(mask)
    change_pct = get_change_percentage(mask)

    return {
        "changed_pixels": count_changes(mask),
        "change_percentage": change_pct,
        "bounding_boxes": boxes,
        "mask": mask,
        "is_changed": change_pct > 0.1,  # More than 0.1% = changed
    }


def _pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    img_rgb = np.array(image)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _cv2_to_pil(image_cv: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR) to PIL Image."""
    if len(image_cv.shape) == 2:
        img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
