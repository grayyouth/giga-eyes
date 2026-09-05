"""
Pipeline stage definitions and JSON schema for analysis output.

Defines:
- VALID_STAGES: List of all available analysis stages
- STAGE_DESCRIPTIONS: Human-readable descriptions for each stage
- JSON_SCHEMA: Draft-07 schema for validation
"""

from typing import Any, Dict

# ─── Stage definitions ────────────────────────────────────────────────────

VALID_STAGES = [
    "metadata",
    "ocr",
    "ui",
    "color",
    "template",
    "vision",
]

STAGE_DESCRIPTIONS: Dict[str, str] = {
    "metadata": "Image dimensions, format, aspect ratio, megapixels",
    "ocr": "Text recognition using GLM-OCR with classification (button, label, heading, etc.)",
    "ui": "UI element detection using YOLO (buttons, menus, inputs, navigation, etc.)",
    "color": "Dominant colors with hex codes and percentages, error screen detection",
    "template": "Template matching for known UI elements",
    "vision": "Vision LLM analysis: design layout, color scheme, components, typography",
}

# ─── JSON Schema (Draft-07) ──────────────────────────────────────────────

JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "GigaEyes Analysis Result",
    "description": "Structured output from universal image analysis pipeline",
    "type": "object",
    "required": ["pipeline_version", "timestamp", "metadata", "stages"],
    "properties": {
        "pipeline_version": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {
            "type": "object",
            "properties": {
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
                "mode": {"type": "string"},
                "format": {"type": "string"},
                "aspect_ratio": {"type": "number"},
                "megapixels": {"type": "number"},
            },
        },
        "stages": {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "mode": {"type": "string"},
                        "format": {"type": "string"},
                        "aspect_ratio": {"type": "number"},
                        "megapixels": {"type": "number"},
                    },
                },
                "ocr": {
                    "type": "object",
                    "properties": {
                        "total_text_regions": {"type": "integer"},
                        "texts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "type": {"type": "string"},
                                },
                            },
                        },
                        "by_type": {"type": "object"},
                    },
                },
                "ui": {
                    "type": "object",
                    "properties": {
                        "total_elements": {"type": "integer"},
                        "elements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "class": {"type": "integer"},
                                    "name": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                },
                            },
                        },
                        "by_class": {"type": "object"},
                    },
                },
                "color": {
                    "type": "object",
                    "properties": {
                        "dominant_colors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "rgb": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                                    "hex": {"type": "string"},
                                    "percentage": {"type": "number"},
                                },
                            },
                        },
                        "error_screen_detected": {"type": "boolean"},
                    },
                },
                "template": {
                    "type": "object",
                    "properties": {
                        "threshold": {"type": "number"},
                        "templates": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "found": {"type": "boolean"},
                                    "bbox": {
                                        "anyOf": [
                                            {"type": "null"},
                                            {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                                        ],
                                    },
                                },
                            },
                        },
                    },
                },
                "vision": {
                    "type": "object",
                    "properties": {
                        "design_analysis": {"type": "string"},
                        "text_extraction": {"type": "string"},
                    },
                },
            },
        },
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string"},
                    "message": {"type": "string"},
                },
            },
        },
        "timing": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
}
