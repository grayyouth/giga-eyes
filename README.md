# Giga Eyes 👁️

> **Visual observation and AI-powered UI analysis toolkit**

Giga Eyes gives your applications "eyes" — the ability to capture, analyze, and understand screen content. From simple screenshots to AI-powered UI element detection, text recognition, and design analysis.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/gigacode/giga-eyes/actions/workflows/ci.yml/badge.svg)](https://github.com/gigacode/giga-eyes/actions)
[![PyPI version](https://img.shields.io/pypi/v/giga-eyes.svg)](https://pypi.org/project/giga-eyes/)

---

## ✨ Features

### 📸 Screen Capture
- **Full screen** capture via `mss` (~30-50 FPS)
- **Window capture** by title or HWND (with BitBlt + mss fallback)
- **Region capture** — arbitrary screen areas
- **Auto-naming** — timestamped filenames with window titles

### 🔍 Analysis Pipeline
- **OCR** — GLM-OCR with GPU acceleration (94.62 OmniDocBench, 8 languages)
- **UI Detection** — YOLO-based detection of 9 UI element classes
- **Color Analysis** — dominant colors (RGB/HEX/%), error screen detection
- **Template Matching** — OpenCV `matchTemplate` with multi-scale support
- **Frame Diff** — detect UI changes, animations, error states
- **Vision LLM** — SmolVLM2 for design/layout analysis

### 🔄 Real-time Monitoring
- **Continuous stream** capture with configurable FPS
- **Change detection** — trigger on screen changes
- **Element trigger** — react to UI element appearance
- **Color trigger** — detect specific colors
- **YOLO trigger** — detect UI elements in real-time

### 🎯 Universal Pipeline
Single API for any image source — file, URL, window, PIL Image, or numpy array:

```python
from giga_eyes.pipeline import ImagePipeline

pipeline = ImagePipeline()
result = pipeline.analyze("screenshot.png")
print(result.to_json())
```

---

## 📦 Installation

### Quick Install
```bash
pip install -e .
```

### With Optional Features
```bash
# Full installation (all features)
pip install -e ".[dev,yolo,vision]"

# Development only
pip install -e ".[dev]"

# YOLO UI detection
pip install -e ".[yolo]"

# Vision LLM analysis
pip install -e ".[vision]"
```

### Manual Dependencies
```bash
pip install mss opencv-python Pillow pywin32 pygetwindow pyyaml
# Optional: ultralytics uitag torch transformers
```

---

## 🚀 Quick Start

### CLI Usage

```bash
# Capture full screen
giga-eyes screenshot -o screenshots/

# Capture specific window
giga-eyes window -t "TinyVika" -o screenshots/

# Capture region
giga-eyes region --x 0 --y 0 --w 800 --h 600 -o screenshots/

# Universal analysis (recommended!)
giga-eyes analyze -i screenshot.png --stages metadata,color -o result.json
giga-eyes analyze -i screenshot.png --stages metadata,ocr,ui,color,vision -o full.json

# UI element detection
giga-eyes ui-detect -i screenshot.png --confidence 0.3

# Template matching
giga-eyes template -i screenshot.png -t button.png --threshold 0.8

# Color analysis
giga-eyes color -i screenshot.png --color 232,116,12

# Frame comparison
giga-eyes diff -b before.png -a after.png

# Real-time monitoring
giga-eyes monitor -t "TinyVika" --fps 10 --trigger change
```

### Python API

#### Basic Capture
```python
from giga_eyes.capture.screen import capture_full_screen
from giga_eyes.capture.window import capture_window_by_title
from giga_eyes.storage.screenshots import ScreenshotStorage

# Capture full screen
img = capture_full_screen(output_path=None)

# Capture window by title
img = capture_window_by_title("TinyVika", output_path=None)

# Save with auto-naming
storage = ScreenshotStorage("screenshots", "png")
filepath = storage.save(img, window_title="TinyVika")
print(f"Saved: {filepath}")
```

#### Universal Analysis Pipeline
```python
from giga_eyes.pipeline import ImagePipeline

# Initialize pipeline
pipeline = ImagePipeline()

# Analyze any source
result = pipeline.analyze(
    source="screenshot.png",  # or URL, window title, PIL Image
    stages=["metadata", "ocr", "ui", "color"],
    output="analysis.json",
)

# Access results
print(f"Image: {result.stages['metadata']['width']}x{result.stages['metadata']['height']}")
print(f"UI elements: {result.stages['ui']['total_elements']}")
print(f"Colors: {result.stages['color']['dominant_colors']}")
print(f"Text regions: {result.stages['ocr']['total_text_regions']}")

# Save to file
result.save("output.json")
```

#### Real-time Monitoring
```python
from giga_eyes.monitor.stream import WindowStream
from giga_eyes.monitor.trigger import TriggerManager

stream = WindowStream(window_title="TinyVika", fps=10)

def on_event(event):
    print(f"Trigger: {event.trigger_name}")
    print(f"Data: {event.data}")

manager = TriggerManager(auto_save=True)
manager.add_change_trigger(threshold=5.0, cooldown=1.0)
manager.set_callback(on_event)

def on_change(**kwargs):
    change_pct = kwargs.get("change_percentage", 0)
    frame = kwargs.get("frame")
    if frame:
        manager.process_frame(frame, change_pct)

stream.callback.on("change", on_change)
stream.start()
```

---

## 📊 Analysis Pipeline Stages

The universal pipeline supports 6 analysis stages:

| Stage | Description | Requires Models |
|-------|-------------|-----------------|
| `metadata` | Image dimensions, format, aspect ratio, megapixels | ❌ No |
| `ocr` | Text recognition with classification (button, label, heading) | ✅ GLM-OCR |
| `ui` | UI element detection (9 YOLO classes) | ✅ YOLO |
| `color` | Dominant colors (RGB/HEX/%), error screen detection | ❌ No |
| `template` | Template matching for known UI elements | ❌ No |
| `vision` | Design/layout analysis via Vision LLM | ✅ SmolVLM2 |

### Output JSON Structure

```json
{
  "pipeline_version": "0.1.0",
  "timestamp": "2026-09-05T03:14:00",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "mode": "RGB",
    "format": "PNG",
    "aspect_ratio": 1.78,
    "megapixels": 2.07
  },
  "stages": {
    "ocr": {
      "total_text_regions": 12,
      "texts": [
        {"text": "Save", "confidence": 0.95, "type": "button"},
        {"text": "Username", "confidence": 0.92, "type": "label"}
      ],
      "by_type": {"button": 3, "label": 5}
    },
    "ui": {
      "total_elements": 8,
      "elements": [
        {"name": "Button", "confidence": 0.89, "bbox": [100, 200, 150, 230]}
      ],
      "by_class": {"Button": 3, "Input_Elements": 2}
    },
    "color": {
      "dominant_colors": [
        {"rgb": [30, 30, 30], "hex": "#1e1e1e", "percentage": 65.0},
        {"rgb": [232, 116, 12], "hex": "#e8740c", "percentage": 15.0}
      ],
      "error_screen_detected": false
    }
  },
  "errors": [],
  "timing": {
    "metadata_seconds": 0.001,
    "color_seconds": 0.045,
    "total_seconds": 2.340
  }
}
```

---

## 🏗️ Architecture

```
GigaEyes/
├── giga_eyes/
│   ├── capture/          # Screen capture modules
│   │   ├── screen.py     # Full screen (mss)
│   │   ├── window.py     # Window capture (win32gui + BitBlt)
│   │   └── region.py     # Region capture
│   ├── analyze/          # Image analysis modules
│   │   ├── template.py   # OpenCV matchTemplate
│   │   ├── diff.py       # Frame comparison
│   │   ├── color.py      # Color detection
│   │   ├── ocr.py        # GLM-OCR via llama.cpp
│   │   ├── yolo.py       # UI element detection (YOLO)
│   │   └── vision.py     # Vision LLM analysis (SmolVLM2)
│   ├── monitor/          # Real-time monitoring
│   │   ├── stream.py     # WindowStream, StreamBuffer
│   │   └── trigger.py    # Change/Element/Color/YOLO triggers
│   ├── pipeline/         # Universal analysis pipeline
│   │   ├── pipeline.py   # ImagePipeline, AnalysisResult
│   │   └── schemas.py    # JSON schema definitions
│   ├── storage/          # Storage modules
│   │   ├── screenshots.py # ScreenshotStorage
│   │   └── logs.py       # EventLogger
│   ├── cli.py            # CLI interface
│   └── config.py         # Configuration loader
├── examples/             # Usage examples
├── tests/                # Test suite (86 tests)
├── config.json           # Main configuration
├── config/llama.json     # LLM model configuration
├── pyproject.toml        # Project metadata
└── README.md             # This file
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=giga_eyes --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v
```

**Current test status:** 86/86 passing ✅

---

## 📚 Examples

See `examples/` directory:

| Example | Description |
|---------|-------------|
| `capture_window.py` | Capture screen and window |
| `capture_region.py` | Capture arbitrary region |
| `pipeline_example.py` | Universal analysis pipeline (5 examples) |

### Running Examples

```bash
python examples/capture_window.py
python examples/capture_region.py
python examples/pipeline_example.py
```

---

## 🎯 Use Cases

### 1. Design Interface Generation from Screenshot

```python
from giga_eyes.pipeline import ImagePipeline

result = pipeline.analyze(
    source="design_reference.png",
    stages=["metadata", "color", "ui", "ocr"],
)

# Extract design tokens
colors = result.stages["color"]["dominant_colors"]
ui_elements = result.stages["ui"]["elements"]
text_elements = result.stages["ocr"]["texts"]

# Use with LLM to generate CSS/HTML
# "Generate CSS using these colors and UI layout..."
```

### 2. GUI Testing Automation

```python
# Test UI consistency across versions
result_before = pipeline.analyze("v1_screenshot.png", stages=["ui", "ocr"])
result_after = pipeline.analyze("v2_screenshot.png", stages=["ui", "ocr"])

# Compare UI elements
before_buttons = result_before.stages["ui"]["by_class"].get("Button", 0)
after_buttons = result_after.stages["ui"]["by_class"].get("Button", 0)

if before_buttons != after_buttons:
    print("⚠️ Button count changed!")
```

### 3. Real-time UI Monitoring

```python
# Monitor for error states
stream = WindowStream(window_title="App", fps=5)
stream.callback.on("change", lambda **kw: detect_errors(kw["frame"]))
stream.start()
```

---

## ⚙️ Configuration

### config.json
Main configuration for paths, models, and analysis parameters:

```json
{
  "models": {
    "ocr": {
      "model": "models/OCR/glm-ocr-q8_0.gguf",
      "mmproj": "models/OCR/mmproj-glm-ocr-q8_0.gguf"
    },
    "vision": {
      "model": "models/VISION/smolvlm2-2.2b-q4_k_m.gguf",
      "mmproj": "models/VISION/mmproj-smolvlm2-2.2b-q8_0.gguf"
    },
    "yolo": {
      "model": "models/YOLO/yolo26n-q8_0.gguf"
    }
  },
  "analysis": {
    "ocr": { "confidence_threshold": 0.3 },
    "color": { "default_tolerance": 20 },
    "template": { "default_threshold": 0.8 },
    "yolo": { "confidence": 0.25 }
  }
}
```

### config/llama.json
LLM model configuration (GPU settings, timeouts):

```json
{
  "llama": {
    "binary": "bin/llama-cli.exe",
    "yolo_cli": "bin/yolo-cli.exe"
  },
  "gpu": {
    "backend": "vulkan",
    "gpu_layers": 999,
    "split_mode": "layer"
  }
}
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/gigacode/giga-eyes.git
cd giga-eyes
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [mss](https://github.com/BoboPi/PyAutoGUI) - Screen capture
- [OpenCV](https://opencv.org/) - Image analysis
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference
- [GLM-OCR](https://github.com/ggml-org/GLM-OCR) - Text recognition
- [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B) - Vision language model

---

**Built with ❤️ by Giga**
