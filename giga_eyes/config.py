"""
Configuration loader for GigaEyes.

Loads settings from config.json (project root) and config/llama.json.
All paths are relative to the project root (parent of giga_eyes/).

Usage:
    from giga_eyes.config import config, llama_config

    # Project config
    ocr_model = config["models"]["ocr"]["model"]
    storage_path = config["paths"]["storage_dir"]

    # Llama.cpp config
    llama_binary = llama_config["llama"]["binary"]
    gpu_layers = llama_config["gpu"]["gpu_layers"]
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Resolve project root: parent of the giga_eyes package directory
_PROJECT_ROOT = Path(__file__).parents[1]

# ─── Config file paths ───────────────────────────────────────────────────────

_CONFIG_PATH = _PROJECT_ROOT / "config.json"
_LLAMA_CONFIG_PATH = _PROJECT_ROOT / "config" / "llama.json"


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON config file."""
    if not path.exists():
        logger.warning("Config file not found: %s (using defaults)", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded config: %s", path)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load config %s: %s", path, e)
        return {}


# ─── Global config objects ───────────────────────────────────────────────────

#: Main project configuration (config.json)
config: Dict[str, Any] = _load_json(_CONFIG_PATH)

#: Llama.cpp specific configuration (config/llama.json)
llama_config: Dict[str, Any] = _load_json(_LLAMA_CONFIG_PATH)


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """Resolve a relative path from config against the project root."""
    return _PROJECT_ROOT / relative_path


def get_ocr_model_path() -> Path:
    """Get the path to the GLM-OCR model file."""
    rel = config.get("models", {}).get("ocr", {}).get("model", "")
    return resolve_path(rel) if rel else Path()


def get_ocr_mmproj_path() -> Path:
    """Get the path to the GLM-OCR mmproj file."""
    rel = config.get("models", {}).get("ocr", {}).get("mmproj", "")
    return resolve_path(rel) if rel else Path()


def get_vision_model_path() -> Path:
    """Get the path to the SmolVLM2 model file."""
    rel = config.get("models", {}).get("vision", {}).get("model", "")
    return resolve_path(rel) if rel else Path()


def get_vision_mmproj_path() -> Path:
    """Get the path to the SmolVLM2 mmproj file."""
    rel = config.get("models", {}).get("vision", {}).get("mmproj", "")
    return resolve_path(rel) if rel else Path()


def get_yolo_model_path() -> Path:
    """Get the path to the YOLO GGUF model file."""
    rel = config.get("models", {}).get("yolo", {}).get("model", "")
    return resolve_path(rel) if rel else Path()


def get_llama_binary_path() -> Path:
    """Get the path to the llama-cli binary."""
    rel = llama_config.get("llama", {}).get("binary", "")
    return resolve_path(rel) if rel else Path()


def get_llama_yolo_cli_path() -> Path:
    """Get the path to the yolo-cli binary."""
    rel = llama_config.get("llama", {}).get("yolo_cli", "")
    return resolve_path(rel) if rel else Path()
