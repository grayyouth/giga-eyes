"""
Template matching module — find UI elements by template.

Uses OpenCV matchTemplate for fast template-based detection.
Works best for known UI elements (buttons, icons, fields).
"""

import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Method constants
MATCH_TEMPL = cv2.TM_CCOEFF_NORMED
MATCH_CCORR = cv2.TM_CCORR_NORMED
MATCH_SQDIFF = cv2.TM_SQDIFF_NORMED

METHODS = {
    "ccoeff": cv2.TM_CCOEFF_NORMED,
    "ccoeff_n": cv2.TM_CCOEFF_NORMED,
    "ccorr": cv2.TM_CCORR_NORMED,
    "ccorr_n": cv2.TM_CCORR_NORMED,
    "sqdiff": cv2.TM_SQDIFF_NORMED,
    "sqdiff_n": cv2.TM_SQDIFF_NORMED,
}


def pil_to_opencv(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format (BGR numpy array).

    Args:
        image: PIL.Image object.

    Returns:
        numpy array in BGR format.
    """
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def opencv_to_pil(image_cv: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image (BGR) to PIL Image.

    Args:
        image_cv: numpy array in BGR format.

    Returns:
        PIL.Image object.
    """
    if len(image_cv.shape) == 2:
        img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def find_template(
    image: Image.Image,
    template: Image.Image,
    threshold: float = 0.8,
    method: str = "ccoeff",
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find a single occurrence of template in image.

    Note: Templates should have distinctive patterns (edges, text, gradients).
    Solid color templates may match in uniform regions and produce false positives.

    Args:
        image: Full screenshot (PIL Image).
        template: Template to find (PIL Image).
        threshold: Match threshold (0.0-1.0).
        method: Matching method ('ccoeff', 'ccorr', 'sqdiff').

    Returns:
        (left, top, right, bottom) bounding box or None if not found.
    """
    matches = find_all_matches(image, template, threshold, method, max_matches=1)

    if matches:
        return matches[0]

    logger.debug("Template not found (threshold=%.2f)", threshold)
    return None


def find_all_matches(
    image: Image.Image,
    template: Image.Image,
    threshold: float = 0.8,
    method: str = "ccoeff",
    max_matches: int = 10,
) -> List[Tuple[int, int, int, int]]:
    """
    Find all occurrences of template in image.

    Uses non-maximum suppression to avoid duplicate detections.

    Args:
        image: Full screenshot (PIL Image).
        template: Template to find (PIL Image).
        threshold: Match threshold (0.0-1.0).
        method: Matching method.
        max_matches: Maximum number of matches to return.

    Returns:
        List of (left, top, right, bottom) bounding boxes.
    """
    img_cv = pil_to_opencv(image)
    templ_cv = pil_to_opencv(template)

    img_h, img_w = img_cv.shape[:2]
    templ_h, templ_w = templ_cv.shape[:2]

    if templ_h > img_h or templ_w > img_w:
        logger.warning("Template larger than image")
        return []

    # Select matching method
    method_code = METHODS.get(method.lower(), MATCH_TEMPL)

    # Perform template matching
    result = cv2.matchTemplate(img_cv, templ_cv, method_code)

    # Get all matches above threshold
    if method_code in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        # For SQDIFF, lower values are better (0 = perfect match)
        # threshold here means max allowed error
        locations = np.where(result <= threshold)
    else:
        # For CCOEFF and CCORR, higher values are better (1 = perfect match)
        locations = np.where(result >= threshold)

    if len(locations[0]) == 0:
        logger.debug("No matches found (threshold=%.2f, method=%s)", threshold, method)
        return []

    # Convert to (y, x) pairs
    points = list(zip(locations[0], locations[1]))

    # Non-maximum suppression — filter overlapping detections
    boxes = _non_max_suppression(points, templ_w, templ_h, overlap_threshold=0.3)

    # Convert to (left, top, right, bottom)
    # NMS returns (x, y) pairs
    bounding_boxes = []
    for x, y in boxes[:max_matches]:
        bounding_boxes.append((x, y, x + templ_w, y + templ_h))

    logger.info(
        "Found %d matches (threshold=%.2f, method=%s)",
        len(bounding_boxes),
        threshold,
        method,
    )

    return bounding_boxes


def get_match_quality(
    image: Image.Image,
    template: Image.Image,
    x: int,
    y: int,
    method: str = "ccoeff",
) -> float:
    """
    Get the match quality score at a specific location.

    Args:
        image: Full screenshot.
        template: Template to find.
        x: Left coordinate.
        y: Top coordinate.
        method: Matching method.

    Returns:
        Match quality score (0.0-1.0).
    """
    img_cv = pil_to_opencv(image)
    templ_cv = pil_to_opencv(template)

    method_code = METHODS.get(method.lower(), MATCH_TEMPL)
    result = cv2.matchTemplate(img_cv, templ_cv, method_code)

    # Get value at the specified location
    y_int = int(y)
    x_int = int(x)

    if method_code in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        quality = 1.0 - result[y_int, x_int]
    else:
        quality = result[y_int, x_int]

    return float(np.clip(quality, 0.0, 1.0))


def resize_template(
    template: Image.Image,
    scale: float,
) -> Image.Image:
    """
    Resize template by scale factor.

    Useful for finding templates at different sizes.

    Args:
        template: Template image.
        scale: Scale factor (e.g., 1.5 for 150%).

    Returns:
        Resized PIL Image.
    """
    if scale <= 0:
        raise ValueError("Scale must be positive")

    new_size = (
        int(template.width * scale),
        int(template.height * scale),
    )
    return template.resize(new_size, Image.LANCZOS)


def multi_scale_match(
    image: Image.Image,
    template: Image.Image,
    thresholds: Optional[List[float]] = None,
    scales: Optional[List[float]] = None,
    method: str = "ccoeff",
) -> List[dict]:
    """
    Find template at multiple scales.

    Args:
        image: Full screenshot.
        template: Template to find.
        thresholds: List of thresholds for each scale.
        scales: Scale factors to try (default: 0.8, 0.9, 1.0, 1.1, 1.2).
        method: Matching method.

    Returns:
        List of dicts with 'box', 'scale', 'quality'.
    """
    if scales is None:
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    if thresholds is None:
        thresholds = [0.8] * len(scales)

    results = []

    for i, scale in enumerate(scales):
        scaled_template = resize_template(template, scale)
        threshold = thresholds[min(i, len(thresholds) - 1)]

        boxes = find_all_matches(image, scaled_template, threshold, method)

        for box in boxes:
            results.append({
                "box": box,
                "scale": scale,
                "quality": threshold,
            })

    # Sort by scale (prefer 1.0 — original size)
    results.sort(key=lambda r: abs(r["scale"] - 1.0))

    return results


def _non_max_suppression(
    points: List[Tuple[int, int]],
    templ_w: int,
    templ_h: int,
    overlap_threshold: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Non-maximum suppression — filter overlapping detections.

    Args:
        points: List of (y, x) detection points.
        templ_w: Template width.
        templ_h: Template height.
        overlap_threshold: Max allowed overlap ratio.

    Returns:
        Filtered list of (y, x) points.
    """
    if not points:
        return []

    # Convert to (x, y, x2, y2) boxes
    boxes = []
    for y, x in points:
        boxes.append((x, y, x + templ_w, y + templ_h))

    # Sort by position (top-to-bottom, left-to-right)
    boxes.sort(key=lambda b: b[1] * 10000 + b[0])

    suppressed = [False] * len(boxes)
    areas = []
    for x1, y1, x2, y2 in boxes:
        areas.append((x2 - x1) * (y2 - y1))

    for i in range(len(boxes)):
        if suppressed[i]:
            continue

        x1, y1, x2, y2 = boxes[i]
        area_i = areas[i]

        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue

            x1n, y1n, x2n, y2n = boxes[j]

            # Calculate intersection
            xi1 = max(x1, x1n)
            yi1 = max(y1, y1n)
            xi2 = min(x2, x2n)
            yi2 = min(y2, y2n)

            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            inter_area = inter_w * inter_h

            # Overlap ratio (intersection / union)
            union_area = area_i + areas[j] - inter_area
            overlap = inter_area / union_area if union_area > 0 else 0

            if overlap > overlap_threshold:
                suppressed[j] = True

    return [(boxes[i][0], boxes[i][1]) for i in range(len(boxes)) if not suppressed[i]]
