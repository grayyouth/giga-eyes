"""
Download all GigaEyes models from HuggingFace.

Models:
  1. GLM-OCR Q8_0 (~950 MB) → models/OCR/
  2. SmolVLM2-2.2B Q4_K_M (~1.11 GB) → models/VISION/
  3. YOLO26n Q8_0 GGUF (~27 MB) → models/YOLO/

Usage:
    python scripts/download_models.py
"""

import logging
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# ─── Model definitions ───────────────────────────────────────────────────────

MODELS = [
    {
        "name": "GLM-OCR Q8_0 (model)",
        "repo": "ggml-org/GLM-OCR-GGUF",
        "filename": "GLM-OCR-Q8_0.gguf",
        "dest": MODELS_DIR / "OCR" / "glm-ocr-q8_0.gguf",
        "size_approx": "~950 MB",
    },
    {
        "name": "GLM-OCR Q8_0 (mmproj)",
        "repo": "ggml-org/GLM-OCR-GGUF",
        "filename": "mmproj-GLM-OCR-Q8_0.gguf",
        "dest": MODELS_DIR / "OCR" / "mmproj-glm-ocr-q8_0.gguf",
        "size_approx": "~484 MB",
    },
    {
        "name": "SmolVLM2-2.2B Q4_K_M (model)",
        "repo": "ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        "filename": "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        "dest": MODELS_DIR / "VISION" / "smolvlm2-2.2b-q4_k_m.gguf",
        "size_approx": "~1.11 GB",
    },
    {
        "name": "SmolVLM2-2.2B Q8_0 (mmproj)",
        "repo": "ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        "filename": "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
        "dest": MODELS_DIR / "VISION" / "mmproj-smolvlm2-2.2b-q8_0.gguf",
        "size_approx": "~593 MB",
    },
    {
        "name": "YOLO26n Q8_0 GGUF",
        "repo": "Asher-1/yolo-gguf",
        "filename": "yolo26n-q8_0.gguf",
        "dest": MODELS_DIR / "YOLO" / "yolo26n-q8_0.gguf",
        "size_approx": "~27 MB",
    },
]


def download_model(model_def: dict) -> bool:
    """Download a single model file."""
    name = model_def["name"]
    repo = model_def["repo"]
    filename = model_def["filename"]
    dest = model_def["dest"]
    size = model_def["size_approx"]

    if dest.exists():
        logger.info("SKIP %s — already exists: %s", name, dest)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("DOWNLOAD %s (%s) from %s", name, size, repo)
    logger.info("  → %s", dest)

    try:
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download with local_dir saves as local_dir/filename
        actual = dest.parent / filename
        if actual != dest and actual.exists():
            actual.rename(dest)
        size_gb = dest.stat().st_size / 1024 / 1024 / 1024
        logger.info("  Done (%.2f GB)", size_gb)
        return True
    except Exception as e:
        logger.error("  ✗ Failed: %s", e)
        return False


def main():
    logger.info("=" * 60)
    logger.info("GigaEyes Model Downloader")
    logger.info("=" * 60)

    # Ensure model directories exist
    for d in ["OCR", "VISION", "YOLO"]:
        (MODELS_DIR / d).mkdir(parents=True, exist_ok=True)

    results = []
    for model in MODELS:
        ok = download_model(model)
        results.append((model["name"], ok))

    logger.info("=" * 60)
    logger.info("Summary:")
    for name, ok in results:
        status = "✓ OK" if ok else "✗ FAILED"
        logger.info("  %s — %s", status, name)

    failed = sum(1 for _, ok in results if not ok)
    if failed:
        logger.error("%d model(s) failed to download", failed)
        sys.exit(1)
    else:
        logger.info("All models downloaded successfully!")


if __name__ == "__main__":
    main()
