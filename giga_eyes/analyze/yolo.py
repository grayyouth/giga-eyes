"""
UI Element Detection module - detects UI elements in screenshots.

Two backends supported:
1. GGUF (primary) - Asher-1/yolo-gguf via yolo-cli + Vulkan GPU
   Model: models/YOLO/yolo26n-q8_0.gguf (~27 MB)
2. Ultralytics (fallback) - ultralytics YOLO + uitag yolo-ui.pt
   Model: uitag yolo-ui.pt (18 MB, GroundCUA, 9 UI classes)

Configuration: config.json + config/llama.json
Usage:
    detector = UIDetector()
    detections = detector.detect(screenshot_image)
    annotated = detector.detect_and_draw(screenshot_image)
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw

from giga_eyes.config import (
    get_yolo_model_path,
    get_llama_yolo_cli_path,
    llama_config,
    config,
)

logger = logging.getLogger(__name__)

# UI element classes (GroundCUA taxonomy)
UI_CLASSES = [
    "Button",
    "Menu",
    "Input_Elements",
    "Navigation",
    "Information_Display",
    "Sidebar",
    "Visual_Elements",
    "Others",
    "Unknown",
]

# Tile parameters matching training
TILE_SIZE = 640
OVERLAP_RATIO = 0.2
NMS_IOU_THRESHOLD = 0.5


def _compute_tiles(img_width: int, img_height: int) -> List[Tuple[int, int, int, int]]:
    """Compute tile positions with overlap."""
    step = int(TILE_SIZE * (1 - OVERLAP_RATIO))
    tiles = []

    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            x2 = min(x + TILE_SIZE, img_width)
            y2 = min(y + TILE_SIZE, img_height)
            x1 = max(0, x2 - TILE_SIZE)
            y1 = max(0, y2 - TILE_SIZE)
            tiles.append((x1, y1, x2, y2))
            if x2 >= img_width:
                break
            x += step
        if y2 >= img_height:
            break
        y += step

    return tiles


def _nms_boxes(
    boxes: np.ndarray,
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> np.ndarray:
    """Cross-tile NMS. Input: Nx6 [x1, y1, x2, y2, conf, cls]."""
    if len(boxes) == 0:
        return boxes

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return boxes[keep]


class UIDetector:
    """UI element detector using YOLO model.

    Supports two backends:
    - GGUF via yolo-cli (GPU-accelerated, requires yolo-cli.exe)
    - Ultralytics (fallback, uses Python + PyTorch)

    All paths and settings are loaded from config.json and config/llama.json.

    Usage:
        detector = UIDetector()
        detections = detector.detect(screenshot_image)
    """

    def __init__(
        self,
        confidence: Optional[float] = None,
        backend: Optional[str] = None,
    ):
        """
        Initialize UI detector.

        Args:
            confidence: Override confidence threshold from config.
            backend: 'gguf', 'ultralytics', or 'auto' (default).
        """
        yolo_cfg = llama_config.get("yolo", {})

        self.confidence = confidence if confidence is not None else yolo_cfg.get("confidence", 0.25)
        self.backend = backend or yolo_cfg.get("backend", "auto")
        self.input_size = yolo_cfg.get("input_size", TILE_SIZE)

        # GGUF backend paths
        self.yolo_cli = get_llama_yolo_cli_path()
        self.gguf_model = get_yolo_model_path()

        # Ultralytics backend
        self._ul_model = None

        # Determine effective backend
        if self.backend == "auto":
            self.backend = self._resolve_backend()

        self._initialized = self._check_backend()

        if not self._initialized:
            logger.warning(
                "UIDetector not initialized. "
                "Install yolo-cli for GGUF backend or uitag for ultralytics fallback."
            )

    def _resolve_backend(self) -> str:
        """Auto-detect available backend."""
        if self.yolo_cli.exists() and self.gguf_model.exists():
            return "gguf"
        logger.debug("GGUF backend unavailable, falling back to ultralytics")
        return "ultralytics"

    def _check_backend(self) -> bool:
        """Check if the selected backend is available."""
        if self.backend == "gguf":
            return self.yolo_cli.exists() and self.gguf_model.exists()
        elif self.backend == "ultralytics":
            return self._init_ultralytics() is not None
        return False

    def _init_ultralytics(self):
        """Lazy-init ultralytics model."""
        if self._ul_model is not None:
            return self._ul_model

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed")
            return None

        model_path = None
        try:
            import uitag
            candidate = Path(uitag.__file__).parent / "models" / "yolo-ui.pt"
            if candidate.exists():
                model_path = candidate
        except ImportError:
            pass

        if not model_path:
            fallback = Path("yolo-ui.pt")
            if fallback.exists():
                model_path = fallback

        if not model_path:
            logger.error(
                "YOLO model not found for ultralytics backend.\n"
                "Install uitag: pip install uitag[yolo]\n"
                "Or place yolo-ui.pt in project root."
            )
            return None

        try:
            self._ul_model = YOLO(str(model_path))
            logger.info("Ultralytics YOLO model loaded from: %s", model_path)
            return self._ul_model
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            return None

    def _detect_gguf(self, image: Image.Image) -> List[Dict]:
        """Detect UI elements using yolo-cli GGUF backend."""
        if not self.yolo_cli.exists() or not self.gguf_model.exists():
            return []

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, "PNG")
            img_path = tmp.name

        try:
            cmd = [
                str(self.yolo_cli),
                "detect",
                "--model", str(self.gguf_model),
                "--source", img_path,
                "--conf", str(self.confidence),
                "--imgsz", str(self.input_size),
                "--device", "vulkan",
                "--save-txt",
                "--no-save",
                "--verbose",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )

            return self._parse_gguf_output(result.stdout, result.stderr)

        except subprocess.TimeoutExpired:
            logger.error("yolo-cli GGUF detection timed out")
            return []
        except Exception as e:
            logger.error("yolo-cli GGUF error: %s", e)
            return []
        finally:
            try:
                Path(img_path).unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _parse_gguf_output(stdout: str, stderr: str) -> List[Dict]:
        """Parse yolo-cli output into detection dicts."""
        detections = []
        output = stdout + stderr

        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 6:
                try:
                    cls_name = parts[0]
                    conf = float(parts[1])
                    x1, y1, x2, y2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

                    if cls_name in UI_CLASSES:
                        cls_id = UI_CLASSES.index(cls_name)
                        w = x2 - x1
                        h = y2 - y1
                        detections.append({
                            "class": cls_id,
                            "name": cls_name,
                            "confidence": conf,
                            "bbox": (int(x1), int(y1), int(w), int(h)),
                        })
                except (ValueError, IndexError):
                    continue

        return detections

    def _detect_ultralytics(self, image: Image.Image) -> List[Dict]:
        """Detect UI elements using ultralytics YOLO backend."""
        model = self._init_ultralytics()
        if model is None:
            return []

        img_width, img_height = image.size
        tiles = _compute_tiles(img_width, img_height)
        logger.info("UI detect: %dx%d image -> %d tiles", img_width, img_height, len(tiles))

        all_boxes = []
        for tx1, ty1, tx2, ty2 in tiles:
            tile_img = image.crop((tx1, ty1, tx2, ty2))
            results = model(tile_img, imgsz=TILE_SIZE, conf=self.confidence, verbose=False)

            r = results[0]
            if len(r.boxes) == 0:
                continue

            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                box_conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())

                fx1 = max(0, min(xyxy[0] + tx1, img_width))
                fy1 = max(0, min(xyxy[1] + ty1, img_height))
                fx2 = max(0, min(xyxy[2] + tx1, img_width))
                fy2 = max(0, min(xyxy[3] + ty1, img_height))

                all_boxes.append([fx1, fy1, fx2, fy2, box_conf, cls_id])

        if not all_boxes:
            return []

        boxes_array = np.array(all_boxes, dtype=np.float32)
        kept = _nms_boxes(boxes_array)

        detections = []
        for box in kept:
            x1, y1, x2, y2, box_conf, cls_id = box
            cls_id = int(cls_id)
            class_name = UI_CLASSES[min(cls_id, len(UI_CLASSES) - 1)]

            w = x2 - x1
            h = y2 - y1

            detections.append({
                "class": cls_id,
                "name": class_name,
                "confidence": float(box_conf),
                "bbox": (int(x1), int(y1), int(w), int(h)),
            })

        logger.info("Detected %d UI elements (conf>=%.2f)", len(detections), self.confidence)
        return detections

    def detect(
        self,
        image: Image.Image,
        confidence: Optional[float] = None,
        classes: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Detect UI elements in an image."""
        if not self._initialized:
            return []

        conf = confidence if confidence is not None else self.confidence
        detections = self._do_detect(image, conf)

        if classes:
            detections = [d for d in detections if d["name"] in classes]

        return detections

    def _do_detect(self, image: Image.Image, confidence: float) -> List[Dict]:
        """Run detection with the selected backend."""
        if self.backend == "gguf":
            return self._detect_gguf(image)
        return self._detect_ultralytics(image)

    def detect_and_draw(
        self,
        image: Image.Image,
        confidence: Optional[float] = None,
        classes: Optional[List[str]] = None,
        color: Tuple[int, int, int] = (232, 116, 12),
        thickness: int = 2,
    ) -> Image.Image:
        """Detect UI elements and draw bounding boxes on the image."""
        detections = self.detect(image, confidence, classes)
        draw = ImageDraw.Draw(image)

        for det in detections:
            x, y, w, h = det["bbox"]
            label = f"{det['name']} {det['confidence']:.2f}"

            draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)
            text_bbox = draw.textbbox((x, y - 15), label)
            draw.rectangle(
                [x, y - 15, text_bbox[2] + 5, text_bbox[3] + 2],
                fill=(0, 0, 0),
            )
            draw.text((x, y - 15), label, fill=(255, 255, 255))

        return image

    def detect_classes(
        self,
        image: Image.Image,
        class_names: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> List[Dict]:
        """Detect UI elements by class name."""
        return self.detect(image, confidence, class_names)

    def __repr__(self) -> str:
        return f"UIDetector(backend={self.backend}, conf={self.confidence}, init={self._initialized})"


def detect_ui_elements(
    image: Image.Image,
    confidence: float = 0.25,
    classes: Optional[List[str]] = None,
) -> List[Dict]:
    """Standalone function: detect UI elements in a screenshot."""
    detector = UIDetector(confidence=confidence)
    return detector.detect(image, classes=classes)


def draw_detections(
    image: Image.Image,
    detections: List[Dict],
    color: Tuple[int, int, int] = (232, 116, 12),
    thickness: int = 2,
) -> Image.Image:
    """Draw bounding boxes on an image from detections."""
    draw = ImageDraw.Draw(image)
    for det in detections:
        x, y, w, h = det["bbox"]
        label = f"{det['name']} {det['confidence']:.2f}"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)
        text_bbox = draw.textbbox((x, y - 15), label)
        draw.rectangle(
            [x, y - 15, text_bbox[2] + 5, text_bbox[3] + 2],
            fill=(0, 0, 0),
        )
        draw.text((x, y - 15), label, fill=(255, 255, 255))
    return image
