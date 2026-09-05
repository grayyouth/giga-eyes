"""
Vision Language Model module — SmolVLM2-2.2B via llama.cpp + Vulkan GPU.

Uses models/VISION/smolvlm2-2.2b-q4_k_m.gguf (1.11 GB) with llama.cpp
for fast vision analysis on AMD GPUs via Vulkan backend.

Configuration: config.json + config/llama.json
Usage:
    analyzer = VisionAnalyzer()
    result = analyzer.describe(screenshot)
    result = analyzer.extract_design(screenshot)
    result = analyzer.compare(before, after)
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional
from pathlib import Path

from giga_eyes.config import (
    get_llama_binary_path,
    get_vision_model_path,
    get_vision_mmproj_path,
    llama_config,
)

logger = logging.getLogger(__name__)


class VisionAnalyzer:
    """Vision Language Model analyzer using SmolVLM2-2.2B via llama.cpp.

    All paths and settings are loaded from config.json and config/llama.json.
    Supports image description, design analysis, comparison, and text extraction.

    Usage:
        analyzer = VisionAnalyzer()
        result = analyzer.describe(screenshot)
    """

    def __init__(self):
        """Initialize the vision analyzer from config."""
        llama_cfg = llama_config.get("llama", {})
        gpu_cfg = llama_config.get("gpu", {})
        vision_cfg = llama_config.get("vision", {})

        self.llama_cli = str(get_llama_binary_path())
        self.model_path = get_vision_model_path()
        self.mmproj_path = get_vision_mmproj_path()

        self.gpu_layers = gpu_cfg.get("gpu_layers", 999)
        self.devices = gpu_cfg.get("devices", ["Vulkan0", "Vulkan1"])
        self.temperature = vision_cfg.get("temperature", 0.3)
        self.timeout = vision_cfg.get("timeout_seconds", 300)

        self._initialized = Path(self.llama_cli).exists() and self.model_path.exists()

        if self._initialized:
            logger.info("VisionAnalyzer initialized with SmolVLM2-2.2B (GPU layers=%d)", self.gpu_layers)
        else:
            logger.warning("VisionAnalyzer: llama-cli or model not found")

    def _run_inference(
        self,
        image_path: str,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Run VLM inference via llama.cpp."""
        if not self._initialized:
            logger.error("llama-cli or model not available")
            return ""

        image_path = self._ensure_png(image_path)
        if not image_path:
            return ""

        temp = temperature if temperature is not None else self.temperature

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False, dir=os.environ.get("TEMP", ".")
        ) as tmp:
            out_file = tmp.name

        try:
            cmd = [
                self.llama_cli,
                "-m", str(self.model_path),
                "--mmproj", str(self.mmproj_path),
                "--image", image_path,
                "-p", prompt,
                "-st",
                "--temp", str(temp),
                "--no-display-prompt",
                "--gpu-layers", str(self.gpu_layers),
                "--device", ",".join(self.devices),
                "--split-mode", "layer",
                "--flash-attn", "auto",
            ]

            with open(out_file, "wb") as out_f:
                result = subprocess.run(
                    cmd,
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout,
                )

            with open(out_file, "r", encoding="utf-8", errors="replace") as f:
                full_output = f.read()

            # Extract response after prompt
            if prompt in full_output:
                response = full_output.split(prompt, 1)[-1].strip()
            else:
                response = full_output.strip()

            # Clean up markdown artifacts
            response = response.replace("```markdown", "").replace("```", "").strip()

            if not response:
                if result.stderr:
                    stderr = result.stderr.decode("utf-8", errors="replace")
                    logger.error("VLM error: %s", stderr[:200])
                return ""

            logger.info("VLM response: %d chars", len(response))
            return response

        except subprocess.TimeoutExpired:
            logger.error("VLM inference timeout (%ds)", self.timeout)
            return ""
        except FileNotFoundError:
            logger.error("llama-cli not found at: %s", self.llama_cli)
            return ""
        except Exception as e:
            logger.error("VLM inference error: %s", e)
            return ""
        finally:
            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                except OSError:
                    pass

    def _ensure_png(self, image_path: str) -> str:
        """Convert image to PNG if needed (Vulkan supports PNG/JPG/BMP)."""
        if not os.path.exists(image_path):
            logger.error("Image not found: %s", image_path)
            return ""

        ext = Path(image_path).suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".bmp"):
            return image_path

        if ext == ".webp":
            try:
                from PIL import Image
                img = Image.open(image_path)
                png_path = image_path.rsplit(".", 1)[0] + ".png"
                img.save(png_path, "PNG")
                logger.info("Converted WEBP -> PNG: %s", png_path)
                return png_path
            except Exception as e:
                logger.error("Failed to convert WEBP: %s", e)
                return ""

        return image_path

    def _save_image(self, image) -> str:
        """Save image to temp file and return path."""
        if isinstance(image, str):
            if os.path.exists(image):
                return image
            logger.error("Image file not found: %s", image)
            return ""

        if hasattr(image, "save"):
            tmp_path = os.path.join(os.environ.get("TEMP", "."), f"_gigaeyes_{id(image)}.png")
            try:
                image.save(tmp_path, "PNG")
                return tmp_path
            except Exception as e:
                logger.error("Failed to save image: %s", e)
                return ""

        if hasattr(image, "shape"):
            try:
                from PIL import Image
                import numpy as np
                img = Image.fromarray(image)
                tmp_path = os.path.join(os.environ.get("TEMP", "."), f"_gigaeyes_{id(image)}.png")
                img.save(tmp_path, "PNG")
                return tmp_path
            except Exception as e:
                logger.error("Failed to save numpy array as image: %s", e)
                return ""

        logger.error("Unsupported image type: %s", type(image))
        return ""

    # ─── Public API ──────────────────────────────────────────────────────

    def describe(self, image, prompt: Optional[str] = None) -> str:
        """Get a detailed description of an image."""
        image_path = self._save_image(image)
        if not image_path:
            return ""

        if prompt is None:
            prompt = (
                "Describe this UI screenshot in detail. Include:\n"
                "- What type of page/website is this?\n"
                "- Layout structure (header, hero, sections, grid)\n"
                "- Colors and visual style\n"
                "- Key UI elements (buttons, cards, forms, navigation)\n"
                "- Text content (headings, CTAs)\n"
                "- Design quality\n\n"
                "Be specific and concise."
            )

        return self._run_inference(image_path, prompt)

    def extract_design(self, image) -> str:
        """Extract design information: layout, colors, components, style."""
        image_path = self._save_image(image)
        if not image_path:
            return ""

        prompt = (
            "Analyze this UI design. Return a structured analysis with:\n"
            "1. Layout: overall structure\n"
            "2. Color scheme: dominant colors, accent colors, light/dark mode\n"
            "3. Components: list of UI elements (buttons, cards, inputs, nav, etc.)\n"
            "4. Typography: font sizes, weights, hierarchy\n"
            "5. Spacing: padding/margins observations\n"
            "6. Style: design language (material, flat, glassmorphism, etc.)\n"
            "7. Purpose: what is this page trying to achieve?\n\n"
            "Be specific and concrete."
        )

        return self._run_inference(image_path, prompt)

    def compare(self, before, after, prompt: Optional[str] = None) -> str:
        """Compare two images and describe differences."""
        if prompt is None:
            prompt = "Compare these two screenshots. What changed? Describe the differences in detail."

        before_desc = self.describe(before, "Describe this image in detail:")
        after_desc = self.describe(after, "Describe this image in detail:")

        if before_desc and after_desc:
            combined_prompt = (
                f"Image 1: {before_desc}\n\n"
                f"Image 2: {after_desc}\n\n"
                f"{prompt}"
            )
            return self._run_inference(self._save_image(before), combined_prompt)

        return ""

    def answer_question(self, image, question: str) -> str:
        """Answer a specific question about an image."""
        image_path = self._save_image(image)
        if not image_path:
            return ""

        prompt = f"""You are a helpful assistant that answers questions about UI screenshots.

Question: {question}

Answer based on what you see in the image."""

        return self._run_inference(image_path, prompt)

    def extract_text(self, image) -> str:
        """Extract and organize all text from an image."""
        image_path = self._save_image(image)
        if not image_path:
            return ""

        prompt = (
            "Extract all text visible in this screenshot. For each text element, provide:\n"
            "- The text content\n"
            "- Its approximate position (top-left, center, bottom-right, etc.)\n"
            "- Its role (label, button, heading, body text, etc.)\n\n"
            "Format as a structured list."
        )

        return self._run_inference(image_path, prompt)


def analyze_screenshot(image) -> str:
    """Standalone function: get a full analysis of a screenshot."""
    analyzer = VisionAnalyzer()
    return analyzer.extract_design(image)
