"""Tests for UIDetection module."""

import pytest
from unittest.mock import patch, MagicMock, Mock
from PIL import Image
import numpy as np

from giga_eyes.analyze.yolo import (
    UIDetector,
    UI_CLASSES,
    _compute_tiles,
    _nms_boxes,
    detect_ui_elements,
    draw_detections,
)


class TestUIConstants:
    """Tests for UI element constants."""

    def test_ui_classes_defined(self):
        """Test that UI_CLASSES list is populated."""
        assert len(UI_CLASSES) == 9
        assert "Button" in UI_CLASSES
        assert "Menu" in UI_CLASSES
        assert "Input_Elements" in UI_CLASSES


class TestTileComputation:
    """Tests for tile computation."""

    def test_small_image_single_tile(self):
        """Small image fits in one tile."""
        tiles = _compute_tiles(400, 300)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 400, 300)

    def test_large_image_multiple_tiles(self):
        """Large image produces multiple tiles."""
        tiles = _compute_tiles(1920, 1080)
        assert len(tiles) > 1

    def test_tile_coverage(self):
        """Tiles should cover the full image."""
        tiles = _compute_tiles(1280, 720)
        max_x = max(t[2] for t in tiles)
        max_y = max(t[3] for t in tiles)
        assert max_x == 1280
        assert max_y == 720


class TestNMS:
    """Tests for Non-Maximum Suppression."""

    def test_nms_empty(self):
        """Empty input returns empty."""
        boxes = np.array([]).reshape(0, 6)
        result = _nms_boxes(boxes)
        assert len(result) == 0

    def test_nms_overlapping(self):
        """Overlapping boxes should be suppressed."""
        # Two nearly identical boxes with different confidence
        boxes = np.array([
            [10.0, 10.0, 110.0, 110.0, 0.9, 0],   # higher conf
            [12.0, 12.0, 112.0, 112.0, 0.8, 0],   # overlapping, lower conf
            [200.0, 200.0, 300.0, 300.0, 0.85, 1],  # non-overlapping
        ], dtype=np.float32)
        result = _nms_boxes(boxes, iou_threshold=0.5)
        # Should keep the higher-confidence overlapping box and the non-overlapping one
        assert len(result) == 2

    def test_nms_no_overlap(self):
        """Non-overlapping boxes should all be kept."""
        boxes = np.array([
            [0.0, 0.0, 50.0, 50.0, 0.9, 0],
            [100.0, 100.0, 150.0, 150.0, 0.8, 1],
            [200.0, 200.0, 250.0, 250.0, 0.7, 2],
        ], dtype=np.float32)
        result = _nms_boxes(boxes, iou_threshold=0.5)
        assert len(result) == 3


class TestUIDetector:
    """Tests for UIDetector."""

    def test_init_defaults(self):
        """Test default initialization values."""
        with patch("giga_eyes.analyze.yolo.get_llama_yolo_cli_path") as mock_cli, \
             patch("giga_eyes.analyze.yolo.get_yolo_model_path") as mock_model, \
             patch("giga_eyes.analyze.yolo.llama_config", {}):
            mock_cli.return_value = MagicMock()
            mock_cli.return_value.exists.return_value = False
            mock_model.return_value = MagicMock()
            mock_model.return_value.exists.return_value = False

            detector = UIDetector()
            assert detector.confidence == 0.25
            assert detector.backend == "ultralytics"  # auto -> fallback

    def test_init_with_confidence(self):
        """Test custom confidence threshold."""
        with patch("giga_eyes.analyze.yolo.get_llama_yolo_cli_path") as mock_cli, \
             patch("giga_eyes.analyze.yolo.get_yolo_model_path") as mock_model, \
             patch("giga_eyes.analyze.yolo.llama_config", {}):
            mock_cli.return_value = MagicMock()
            mock_cli.return_value.exists.return_value = False
            mock_model.return_value = MagicMock()
            mock_model.return_value.exists.return_value = False

            detector = UIDetector(confidence=0.5)
            assert detector.confidence == 0.5

    def test_detect_returns_empty_when_not_initialized(self):
        """Test that detect returns empty list when not initialized."""
        with patch("giga_eyes.analyze.yolo.get_llama_yolo_cli_path") as mock_cli, \
             patch("giga_eyes.analyze.yolo.get_yolo_model_path") as mock_model, \
             patch("giga_eyes.analyze.yolo.llama_config", {}):
            mock_cli.return_value = MagicMock()
            mock_cli.return_value.exists.return_value = False
            mock_model.return_value = MagicMock()
            mock_model.return_value.exists.return_value = False

            detector = UIDetector()
            image = Image.new("RGB", (800, 600), color=(30, 30, 30))
            detections = detector.detect(image)
            assert detections == []

    def test_detect_and_draw_returns_image(self):
        """Test that detect_and_draw returns a PIL Image."""
        with patch("giga_eyes.analyze.yolo.get_llama_yolo_cli_path") as mock_cli, \
             patch("giga_eyes.analyze.yolo.get_yolo_model_path") as mock_model, \
             patch("giga_eyes.analyze.yolo.llama_config", {}):
            mock_cli.return_value = MagicMock()
            mock_cli.return_value.exists.return_value = False
            mock_model.return_value = MagicMock()
            mock_model.return_value.exists.return_value = False

            detector = UIDetector()
            image = Image.new("RGB", (800, 600), color=(30, 30, 30))
            result = detector.detect_and_draw(image)
            assert isinstance(result, Image.Image)


class TestStandaloneFunctions:
    """Tests for standalone helper functions."""

    def test_detect_ui_elements_returns_list(self):
        """Test standalone detect_ui_elements function."""
        with patch("giga_eyes.analyze.yolo.get_llama_yolo_cli_path") as mock_cli, \
             patch("giga_eyes.analyze.yolo.get_yolo_model_path") as mock_model, \
             patch("giga_eyes.analyze.yolo.llama_config", {}):
            mock_cli.return_value = MagicMock()
            mock_cli.return_value.exists.return_value = False
            mock_model.return_value = MagicMock()
            mock_model.return_value.exists.return_value = False

            image = Image.new("RGB", (800, 600), color=(30, 30, 30))
            detections = detect_ui_elements(image)
            assert isinstance(detections, list)

    def test_draw_detections_returns_image(self):
        """Test draw_detections function."""
        image = Image.new("RGB", (800, 600), color=(30, 30, 30))
        detections = [
            {
                "name": "Button",
                "confidence": 0.9,
                "bbox": (10, 10, 50, 30),
            }
        ]
        result = draw_detections(image, detections)
        assert isinstance(result, Image.Image)

    def test_draw_detections_empty(self):
        """Test draw_detections with empty detections."""
        image = Image.new("RGB", (800, 600), color=(30, 30, 30))
        result = draw_detections(image, [])
        assert isinstance(result, Image.Image)
