"""
GLM-OCR module — text recognition using GLM-OCR via llama.cpp + Vulkan GPU.

GLM-OCR is a specialized OCR model (CogViT encoder + GLM decoder) that:
- Achieves 94.62 on OmniDocBench (#1 ranking)
- Supports 8 languages including Russian and English
- Handles text, formulas, tables, and document layouts
- Runs on GPU via llama.cpp Vulkan backend (~60 t/s)

Configuration: config.json + config/llama.json
Usage:
    ocr = GLMOcrDetector()
    texts = ocr.recognize_text("path/to/image.png")
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional, List, Dict
from pathlib import Path

from giga_eyes.config import (
    get_llama_binary_path,
    get_ocr_model_path,
    get_ocr_mmproj_path,
    llama_config,
    resolve_path,
)

logger = logging.getLogger(__name__)


class GLMOcrDetector:
    """GLM-OCR text recognition via llama.cpp with GPU acceleration.

    All paths and settings are loaded from config.json and config/llama.json.
    """

    def __init__(self, gpu_layers: Optional[int] = None, main_gpu: Optional[int] = None):
        """
        Initialize GLM-OCR detector.

        Args:
            gpu_layers: Override GPU layers from config (default: from config/llama.json).
            main_gpu: Override main GPU from config (default: from config/llama.json).
        """
        # Load settings from config
        llama_cfg = llama_config.get("llama", {})
        gpu_cfg = llama_config.get("gpu", {})
        ocr_cfg = llama_config.get("ocr", {})

        self._llama_path = get_llama_binary_path()
        self.model_path = get_ocr_model_path()
        self.mmproj_path = get_ocr_mmproj_path()

        # Use config values, override if provided
        self.gpu_layers = gpu_layers if gpu_layers is not None else gpu_cfg.get("gpu_layers", 999)
        self.main_gpu = main_gpu if main_gpu is not None else gpu_cfg.get("main_gpu", 1)
        self.prompt = ocr_cfg.get("prompt", "Text Recognition:")
        self.context_size = ocr_cfg.get("context_size", 2048)
        self.timeout = ocr_cfg.get("timeout_seconds", 120)

        self._initialized = self._llama_path.exists()

        if not self._initialized:
            logger.warning(
                "llama-cli not found at %s. GLM-OCR disabled. "
                "Check config/llama.json 'llama.binary'",
                self._llama_path,
            )

    def recognize_text(
        self,
        image,
        threshold: float = 0.0,
        detail: int = 1,
    ) -> List[Dict]:
        """
        Recognize all text in an image using GLM-OCR.

        Args:
            image: PIL Image, numpy array, file path (str/Path), or list of paths.
            threshold: Minimum confidence (not used by GLM-OCR, kept for API compat).
            detail: Not used by GLM-OCR, kept for API compat.

        Returns:
            List of dicts with 'text', 'confidence', 'bbox'.
        """
        if not self._initialized:
            logger.error("GLM-OCR not initialized (llama-cli not found)")
            return []

        if not self.model_path.exists():
            logger.error("GLM-OCR model not found: %s", self.model_path)
            return []

        image_paths = self._resolve_image_paths(image)
        if not image_paths:
            logger.error("No valid image paths provided")
            return []

        all_texts = []
        for img_path in image_paths:
            try:
                result = self._recognize_single(img_path)
                all_texts.extend(result)
            except Exception as e:
                logger.error("GLM-OCR failed for %s: %s", img_path, e)
                continue

        logger.info("GLM-OCR recognized %d text regions", len(all_texts))
        return all_texts

    def _resolve_image_paths(self, image) -> List[str]:
        """Convert various image inputs to file paths."""
        if isinstance(image, (list, tuple)):
            return [str(self._ensure_path(item)) for item in image if self._ensure_path(item)]
        return [str(self._ensure_path(image))]

    @staticmethod
    def _ensure_path(image) -> Optional[str]:
        """Convert PIL/numpy/path to valid file path."""
        if isinstance(image, str):
            return image if Path(image).exists() else None
        if isinstance(image, Path):
            return str(image) if image.exists() else None
        if hasattr(image, "save"):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(tmp.name, "PNG")
            return tmp.name
        return None

    def _recognize_single(self, image_path: str) -> List[Dict]:
        """Run GLM-OCR on a single image via llama-cli."""
        if not Path(image_path).exists():
            logger.error("Image not found: %s", image_path)
            return []

        # Build llama-cli command
        mmproj_arg = (
            str(self.mmproj_path) if self.mmproj_path.exists() else "auto"
        )

        cmd = [
            str(self._llama_path),
            "-m", str(self.model_path),
            "--mmproj", mmproj_arg,
            "--image", image_path,
            "--prompt", self.prompt,
            "--single-turn",
            "-ngl", str(self.gpu_layers),
            "-cnv", "off",
            "-c", str(self.context_size),
            "--log-disable",
        ]

        # Capture output to temp file (Windows console encoding issues)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as tmp:
            output_path = tmp.name

        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=self.timeout,
            )

            # llama-cli outputs to stderr by default
            output = proc.stderr + proc.stdout

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)

            return self._parse_output(output)

        except subprocess.TimeoutExpired:
            logger.error("GLM-OCR timed out for %s (%ds)", image_path, self.timeout)
            return []
        except Exception as e:
            logger.error("GLM-OCR subprocess failed: %s", e)
            return []
        finally:
            try:
                os.unlink(output_path)
            except OSError:
                pass

    @staticmethod
    def _parse_output(output: str) -> List[Dict]:
        """Parse GLM-OCR text output into structured results."""
        texts = []
        lines = output.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("llama.cpp:") or line.startswith("loading model"):
                continue
            if "token" in line.lower() and "speed" in line.lower():
                continue
            if "eval time" in line.lower():
                continue
            if len(line) < 2:
                continue

            confidence = GLMOcrDetector._estimate_confidence(line)
            texts.append({
                "text": line,
                "confidence": confidence,
                "bbox": None,
            })

        return texts

    @staticmethod
    def _estimate_confidence(text: str) -> float:
        """Estimate OCR confidence based on text quality."""
        if not text:
            return 0.0

        clean_ratio = sum(
            1 for c in text
            if c.isalnum() or c in ' \n\r\t.,;:!?-()[]{}"/\'@#%&*+=<>~`^'
        ) / max(len(text), 1)

        if clean_ratio > 0.9 and len(text) > 3:
            return 0.95
        elif clean_ratio > 0.7:
            return 0.75
        return 0.5

    def recognize_with_bbox(
        self,
        image,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """Alias for recognize_text. GLM-OCR doesn't return bounding boxes."""
        return self.recognize_text(image, threshold)

    def extract_ui_text(
        self,
        image,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """Extract text from UI elements, filtering noise and classifying types."""
        texts = self.recognize_text(image, threshold)

        ui_texts = []
        for item in texts:
            text = item["text"].strip()
            if len(text) < 2:
                continue
            if text.replace(".", "").replace(",", "").isdigit() and len(text) <= 6:
                continue

            ui_texts.append({
                "text": text,
                "confidence": item["confidence"],
                "bbox": item["bbox"],
                "type": self._classify_text_type(text),
            })

        return ui_texts

    @staticmethod
    def _classify_text_type(text: str) -> str:
        """Classify text type: 'button', 'label', 'input', 'error', 'heading', 'other'."""
        text_lower = text.lower().strip()

        button_keywords = [
            "ok", "cancel", "save", "delete", "edit", "send", "submit",
            "apply", "login", "register", "signup", "sign up",
            "hire me", "see my resume", "read more", "learn more",
            "применить", "отмена", "сохранить", "удалить", "изменить",
            "отправить", "подтвердить", "закрыть", "выход", "старт",
            "stop", "pause", "resume", "узнать", "заказать", "позвони",
        ]
        error_keywords = [
            "error", "fail", "warning", "ошибка", "внимание", "не удалось",
        ]
        heading_keywords = [
            "about me", "who am i", "read me", "hello", "welcome",
            "about", "contact", "work", "blog",
        ]

        for kw in button_keywords:
            if kw in text_lower:
                return "button"
        for kw in error_keywords:
            if kw in text_lower:
                return "error"
        for kw in heading_keywords:
            if kw in text_lower:
                return "heading"

        return "label" if len(text) < 30 else "other"

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not found"
        return f"GLMOcrDetector(status={status})"


# ─── Backward compatibility alias ───────────────────────────────────────────

class OCRDetector(GLMOcrDetector):
    """Backward-compatible alias for GLMOcrDetector.

    Old code: OCRDetector(lang=["en", "ru"], gpu=False)
    New code: GLMOcrDetector()

    The `lang`, `langs` and `gpu` params are ignored.
    """

    def __init__(self, langs=None, lang=None, gpu=False):
        super().__init__()
        if langs or lang:
            logger.debug("GLM-OCR ignores `lang`/`langs` param (auto-detects)")
        if gpu:
            logger.debug("GLM-OCR uses GPU by default (llama.cpp Vulkan)")
