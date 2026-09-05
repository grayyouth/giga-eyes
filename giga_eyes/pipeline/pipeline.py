"""
Universal image analysis pipeline.

Orchestrates multiple analysis stages and produces structured JSON output.
"""

import json
import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from giga_eyes.pipeline.schemas import STAGE_DESCRIPTIONS, VALID_STAGES

logger = logging.getLogger(__name__)


class AnalysisResult:
    """Container for analysis results with JSON serialization."""

    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.stages: Dict[str, Any] = {}
        self.errors: List[Dict[str, str]] = []
        self.timing: Dict[str, float] = {}
        self._start_time: Optional[float] = None

    def set_stage(self, stage_name: str, data: Any):
        """Set result for a stage."""
        self.stages[stage_name] = data

    def add_error(self, stage: str, message: str):
        """Record an error from a stage."""
        self.errors.append({"stage": stage, "message": message})

    def start_timer(self):
        """Start timing."""
        self._start_time = time.time()

    def stop_timer(self):
        """Stop timing and record total."""
        if self._start_time:
            self.timing["total_seconds"] = round(time.time() - self._start_time, 3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
            "metadata": self.metadata,
            "stages": self.stages,
            "errors": self.errors,
            "timing": self.timing,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Union[str, Path]) -> Path:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        logger.info("Analysis result saved: %s", path)
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Load from dictionary."""
        result = cls()
        result.metadata = data.get("metadata", {})
        result.stages = data.get("stages", {})
        result.errors = data.get("errors", [])
        result.timing = data.get("timing", {})
        return result

    @classmethod
    def from_json(cls, json_str: str) -> "AnalysisResult":
        """Load from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AnalysisResult":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())


class ImagePipeline:
    """
    Universal image analysis pipeline.

    Stages:
        - metadata: Image dimensions, format, size
        - ocr: Text recognition (GLM-OCR)
        - ui: UI element detection (YOLO)
        - color: Dominant colors, error detection
        - template: Template matching (requires template images)
        - vision: Design/layout analysis (Vision LLM)

    Usage:
        pipeline = ImagePipeline()
        result = pipeline.analyze("screenshot.png")
        print(result.to_json())
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline.

        Args:
            config_path: Path to config.json (default: auto-detect).
        """
        from giga_eyes.config import config

        self.config = config
        self._ocr_ready = False
        self._ui_ready = False
        self._vision_ready = False

        # Check available backends
        self._check_backends()

    def _check_backends(self):
        """Check which analysis backends are available."""
        # OCR
        try:
            from giga_eyes.analyze.ocr import GLMOcrDetector
            detector = GLMOcrDetector()
            self._ocr_ready = detector._initialized
            if self._ocr_ready:
                logger.info("OCR backend: GLM-OCR ready")
            else:
                logger.warning("OCR backend: GLM-OCR not initialized")
        except ImportError:
            logger.warning("OCR backend: module not installed")

        # UI Detection
        try:
            from giga_eyes.analyze.yolo import UIDetector
            detector = UIDetector()
            self._ui_ready = detector._initialized
            if self._ui_ready:
                logger.info("UI detection backend: ready (backend=%s)", detector.backend)
            else:
                logger.warning("UI detection backend: not initialized")
        except ImportError:
            logger.warning("UI detection backend: module not installed")

        # Vision LLM
        try:
            from giga_eyes.analyze.vision import VisionAnalyzer
            analyzer = VisionAnalyzer()
            self._vision_ready = analyzer._initialized
            if self._vision_ready:
                logger.info("Vision LLM backend: ready")
            else:
                logger.warning("Vision LLM backend: not initialized")
        except ImportError:
            logger.warning("Vision LLM backend: module not installed")

    def analyze(
        self,
        source: Union[str, Image.Image, np.ndarray],
        stages: Optional[List[str]] = None,
        output: Optional[Union[str, Path]] = None,
        templates: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> AnalysisResult:
        """
        Run full analysis on an image.

        Args:
            source: Image source — file path, URL, PIL Image, numpy array, or window title.
            stages: List of stages to run. Default: all available.
            output: Path to save JSON output.
            templates: Dict of {name: template_path} for template matching.
            **kwargs: Extra arguments passed to stages.

        Returns:
            AnalysisResult with all stage outputs.
        """
        if stages is None:
            stages = self._get_available_stages()

        # Validate stages
        invalid = [s for s in stages if s not in VALID_STAGES]
        if invalid:
            raise ValueError(f"Invalid stages: {invalid}. Valid: {VALID_STAGES}")

        result = AnalysisResult()
        result.start_timer()

        # Resolve source to PIL Image
        image = self._resolve_source(source)
        if image is None:
            result.add_error("source", "Failed to resolve image source")
            if output:
                result.save(output)
            return result

        # Stage 1: Metadata (always runs)
        if "metadata" in stages:
            self._run_stage(result, "metadata", self._stage_metadata, image)

        # Stage 2: OCR
        if "ocr" in stages and self._ocr_ready:
            self._run_stage(result, "ocr", self._stage_ocr, image, **kwargs)

        # Stage 3: UI Detection
        if "ui" in stages and self._ui_ready:
            self._run_stage(result, "ui", self._stage_ui, image, **kwargs)

        # Stage 4: Color Analysis
        if "color" in stages:
            self._run_stage(result, "color", self._stage_color, image, **kwargs)

        # Stage 5: Template Matching
        if "template" in stages and templates:
            self._run_stage(result, "template", self._stage_template, image, templates, **kwargs)

        # Stage 6: Vision LLM
        if "vision" in stages and self._vision_ready:
            self._run_stage(result, "vision", self._stage_vision, image, **kwargs)

        result.stop_timer()

        if output:
            result.save(output)

        return result

    def _get_available_stages(self) -> List[str]:
        """Get list of stages to run based on available backends."""
        stages = ["metadata", "color"]
        if self._ocr_ready:
            stages.append("ocr")
        if self._ui_ready:
            stages.append("ui")
        if self._vision_ready:
            stages.append("vision")
        return stages

    def _resolve_source(
        self, source: Union[str, Image.Image, np.ndarray]
    ) -> Optional[Image.Image]:
        """
        Resolve any source to a PIL Image.

        Supported sources:
            - str: file path, URL, or window title
            - PIL.Image: returned as-is
            - np.ndarray: converted to PIL Image
        """
        # PIL Image
        if hasattr(source, "convert"):
            return source

        # NumPy array
        if isinstance(source, np.ndarray):
            return Image.fromarray(source)

        # String source
        if isinstance(source, str):
            # Check if URL
            if source.startswith(("http://", "https://")):
                return self._load_from_url(source)
            # Check if file path
            path = Path(source)
            if path.exists():
                return Image.open(path)
            # Try as window title
            return self._capture_window(source)

        return None

    def _load_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL."""
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            image = Image.open(__import__("io").BytesIO(data))
            image.load()  # Force load
            logger.info("Loaded image from URL: %s", url)
            return image
        except Exception as e:
            logger.error("Failed to load image from URL: %s (%s)", url, e)
            return None

    def _capture_window(self, title: str) -> Optional[Image.Image]:
        """Capture window by title."""
        try:
            from giga_eyes.capture.window import capture_window_by_title
            return capture_window_by_title(title, output_path=None)
        except Exception as e:
            logger.error("Failed to capture window '%s': %s", title, e)
            return None

    def _run_stage(
        self,
        result: AnalysisResult,
        stage_name: str,
        stage_func,
        *args,
        **kwargs,
    ):
        """Run a single analysis stage with timing and error handling."""
        start = time.time()
        try:
            data = stage_func(*args, **kwargs)
            result.set_stage(stage_name, data)
            elapsed = round(time.time() - start, 3)
            result.timing[f"{stage_name}_seconds"] = elapsed
            logger.info("Stage '%s' completed in %.3fs", stage_name, elapsed)
        except Exception as e:
            error_msg = f"Stage '{stage_name}' failed: {e}"
            logger.error(error_msg)
            result.add_error(stage_name, str(e))
            result.timing[f"{stage_name}_seconds"] = round(time.time() - start, 3)

    # ─── Stage implementations ──────────────────────────────────────────

    def _stage_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Extract image metadata."""
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format or "unknown",
            "aspect_ratio": round(image.width / max(image.height, 1), 2),
            "megapixels": round((image.width * image.height) / 1_000_000, 2),
        }

    def _stage_ocr(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Run OCR text recognition."""
        from giga_eyes.analyze.ocr import GLMOcrDetector

        detector = GLMOcrDetector()
        texts = detector.recognize_text(image)

        # Classify text elements
        classified = []
        for item in texts:
            classified.append({
                "text": item["text"],
                "confidence": item["confidence"],
                "type": detector._classify_text_type(item["text"]),
            })

        # Group by type
        by_type: Dict[str, List[dict]] = {}
        for item in classified:
            t = item["type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(item)

        return {
            "total_text_regions": len(classified),
            "texts": classified,
            "by_type": {k: len(v) for k, v in by_type.items()},
            "raw": texts,
        }

    def _stage_ui(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Run UI element detection."""
        from giga_eyes.analyze.yolo import UIDetector

        confidence = kwargs.get("confidence", None)
        classes = kwargs.get("classes", None)

        detector = UIDetector(confidence=confidence)
        detections = detector.detect(image, classes=classes)

        # Group by class
        by_class: Dict[str, List[dict]] = {}
        for det in detections:
            cls_name = det["name"]
            if cls_name not in by_class:
                by_class[cls_name] = []
            by_class[cls_name].append({
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            })

        return {
            "total_elements": len(detections),
            "elements": detections,
            "by_class": {k: len(v) for k, v in by_class.items()},
        }

    def _stage_color(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Analyze colors in the image."""
        from giga_eyes.analyze.color import get_dominant_colors, is_pixel_error

        num_colors = kwargs.get("num_colors", 5)
        tolerance = kwargs.get("tolerance", 20)

        dominant = get_dominant_colors(image, num_colors=num_colors)

        # Convert to serializable format
        colors = []
        for r, g, b, pct in dominant:
            colors.append({
                "rgb": [int(r), int(g), int(b)],
                "hex": "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b)),
                "percentage": round(pct * 100, 2),
            })

        # Check for error states
        error_detected = is_pixel_error(image, tolerance=tolerance)

        return {
            "dominant_colors": colors,
            "error_screen_detected": error_detected,
        }

    def _stage_template(
        self, image: Image.Image, templates: Dict[str, str], **kwargs
    ) -> Dict[str, Any]:
        """Run template matching."""
        from giga_eyes.analyze.template import find_template

        threshold = kwargs.get("threshold", 0.8)
        results = {}

        for name, template_path in templates.items():
            try:
                template_img = Image.open(template_path)
                box = find_template(image, template_img, threshold=threshold)
                results[name] = {
                    "found": box is not None,
                    "bbox": box,
                }
            except Exception as e:
                results[name] = {
                    "found": False,
                    "error": str(e),
                }

        return {
            "threshold": threshold,
            "templates": results,
        }

    def _stage_vision(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Run Vision LLM analysis."""
        from giga_eyes.analyze.vision import VisionAnalyzer

        analyzer = VisionAnalyzer()

        # Run design analysis
        design = analyzer.extract_design(image)

        # Extract text
        text = analyzer.extract_text(image)

        return {
            "design_analysis": design,
            "text_extraction": text,
        }
