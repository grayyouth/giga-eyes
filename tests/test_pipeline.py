"""Tests for the universal image analysis pipeline."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from giga_eyes.pipeline import ImagePipeline, AnalysisResult
from giga_eyes.pipeline.schemas import VALID_STAGES, JSON_SCHEMA


class TestAnalysisResult:
    """Tests for AnalysisResult class."""

    def test_set_stage(self):
        """Test setting stage data."""
        result = AnalysisResult()
        result.set_stage("metadata", {"width": 1920, "height": 1080})
        assert result.stages["metadata"]["width"] == 1920

    def test_add_error(self):
        """Test recording errors."""
        result = AnalysisResult()
        result.add_error("ocr", "Model not found")
        assert len(result.errors) == 1
        assert result.errors[0]["stage"] == "ocr"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AnalysisResult()
        result.metadata = {"width": 800, "height": 600}
        result.set_stage("color", {"colors": []})
        result.timing["total_seconds"] = 1.5

        d = result.to_dict()
        assert d["metadata"]["width"] == 800
        assert "color" in d["stages"]
        assert d["timing"]["total_seconds"] == 1.5

    def test_to_json(self):
        """Test JSON serialization."""
        result = AnalysisResult()
        result.metadata = {"width": 100}
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["width"] == 100

    def test_save(self, tmp_path):
        """Test saving to file."""
        result = AnalysisResult()
        result.metadata = {"width": 100}
        output = tmp_path / "output" / "result.json"
        path = result.save(output)

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["metadata"]["width"] == 100

    def test_from_json(self):
        """Test loading from JSON string."""
        data = {"metadata": {"width": 200}, "stages": {}}
        json_str = json.dumps(data)
        result = AnalysisResult.from_json(json_str)
        assert result.metadata["width"] == 200

    def test_from_file(self, tmp_path):
        """Test loading from file."""
        result = AnalysisResult()
        result.metadata = {"width": 300}
        path = tmp_path / "test.json"
        result.save(path)

        loaded = AnalysisResult.from_file(path)
        assert loaded.metadata["width"] == 300

    def test_timer(self):
        """Test timing functionality."""
        result = AnalysisResult()
        result.start_timer()
        result.stop_timer()
        assert "total_seconds" in result.timing


class TestImagePipeline:
    """Tests for ImagePipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mocked backends."""
        with patch.object(ImagePipeline, "_check_backends"):
            pipeline = ImagePipeline()
            pipeline._ocr_ready = False
            pipeline._ui_ready = False
            pipeline._vision_ready = False
            return pipeline

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new("RGB", (800, 600), color=(30, 30, 30))

    def test_analyze_metadata_only(self, pipeline, test_image):
        """Test analyzing with metadata stage only."""
        result = pipeline.analyze(
            source=test_image,
            stages=["metadata"],
        )
        assert "metadata" in result.stages
        assert result.stages["metadata"]["width"] == 800
        assert result.stages["metadata"]["height"] == 600

    def test_analyze_color_only(self, pipeline, test_image):
        """Test analyzing with color stage only."""
        result = pipeline.analyze(
            source=test_image,
            stages=["color"],
        )
        assert "color" in result.stages
        assert "dominant_colors" in result.stages["color"]
        assert len(result.stages["color"]["dominant_colors"]) > 0

    def test_analyze_multiple_stages(self, pipeline, test_image):
        """Test running multiple stages."""
        result = pipeline.analyze(
            source=test_image,
            stages=["metadata", "color"],
        )
        assert "metadata" in result.stages
        assert "color" in result.stages

    def test_analyze_with_output(self, pipeline, test_image, tmp_path):
        """Test saving analysis to file."""
        output = tmp_path / "result.json"
        result = pipeline.analyze(
            source=test_image,
            stages=["metadata"],
            output=output,
        )
        assert output.exists()

    def test_analyze_invalid_stage(self, pipeline, test_image):
        """Test that invalid stages raise ValueError."""
        with pytest.raises(ValueError):
            pipeline.analyze(
                source=test_image,
                stages=["metadata", "invalid_stage"],
            )

    def test_analyze_with_error(self, pipeline, test_image):
        """Test that stage errors are recorded, not raised."""
        # Create a pipeline subclass that has a failing stage
        class FailingPipeline(ImagePipeline):
            def _stage_metadata(self, image):
                raise RuntimeError("Simulated failure")

        fp = FailingPipeline()
        fp._check_backends = lambda: None
        fp._ocr_ready = False
        fp._ui_ready = False
        fp._vision_ready = False

        result = fp.analyze(
            source=test_image,
            stages=["metadata"],
        )
        assert any(e["stage"] == "metadata" for e in result.errors)

    def test_resolve_pil_image(self, pipeline, test_image):
        """Test resolving PIL Image source."""
        resolved = pipeline._resolve_source(test_image)
        assert resolved is test_image

    def test_resolve_numpy_array(self, pipeline):
        """Test resolving numpy array source."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        resolved = pipeline._resolve_source(arr)
        assert resolved is not None
        assert resolved.size == (100, 100)

    def test_resolve_file_path(self, pipeline, tmp_path):
        """Test resolving file path source."""
        img_path = tmp_path / "test.png"
        test_img = Image.new("RGB", (50, 50), color="blue")
        test_img.save(img_path)

        resolved = pipeline._resolve_source(str(img_path))
        assert resolved is not None
        assert resolved.size == (50, 50)

    def test_resolve_nonexistent_path(self, pipeline):
        """Test resolving nonexistent path returns None."""
        resolved = pipeline._resolve_source("/nonexistent/path.png")
        assert resolved is None

    def test_resolve_url(self, pipeline):
        """Test URL resolution (mocked)."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.read.return_value = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            mock_urlopen.return_value = mock_response

            resolved = pipeline._resolve_source("http://example.com/test.png")
            # May fail due to minimal PNG data, but shouldn't crash
            assert resolved is None or hasattr(resolved, "size")

    def test_stage_metadata(self, pipeline, test_image):
        """Test metadata stage output."""
        meta = pipeline._stage_metadata(test_image)
        assert meta["width"] == 800
        assert meta["height"] == 600
        assert meta["mode"] == "RGB"
        assert meta["megapixels"] > 0

    def test_stage_color(self, pipeline, test_image):
        """Test color stage output."""
        color = pipeline._stage_color(test_image)
        assert "dominant_colors" in color
        assert "error_screen_detected" in color
        assert len(color["dominant_colors"]) > 0

    def test_stage_color_custom_params(self, pipeline):
        """Test color stage with custom parameters."""
        # Create multi-color image
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :50] = [255, 0, 0]   # red half
        arr[:, 50:] = [0, 0, 255]   # blue half
        img = Image.fromarray(arr)

        color = pipeline._stage_color(img, num_colors=3)
        # Image has exactly 2 colors
        assert len(color["dominant_colors"]) == 2

    def test_get_available_stages(self, pipeline):
        """Test auto-detection of available stages."""
        pipeline._ocr_ready = True
        pipeline._ui_ready = True
        pipeline._vision_ready = True

        stages = pipeline._get_available_stages()
        assert "metadata" in stages
        assert "color" in stages
        assert "ocr" in stages
        assert "ui" in stages
        assert "vision" in stages

    def test_result_json_schema_valid(self, pipeline, test_image):
        """Test that result conforms to basic structure."""
        result = pipeline.analyze(
            source=test_image,
            stages=["metadata", "color"],
        )
        d = result.to_dict()

        # Check required fields
        assert "pipeline_version" in d
        assert "timestamp" in d
        assert "metadata" in d
        assert "stages" in d
        assert "errors" in d
        assert "timing" in d


class TestStages:
    """Tests for individual pipeline stages."""

    @pytest.fixture
    def pipeline(self):
        with patch.object(ImagePipeline, "_check_backends"):
            p = ImagePipeline()
            p._ocr_ready = False
            p._ui_ready = False
            p._vision_ready = False
            return p

    @pytest.fixture
    def test_image(self):
        """Create image with known colors."""
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        arr[:, :] = [30, 30, 30]  # dark gray
        arr[50:150, 50:150] = [255, 0, 0]  # red square
        return Image.fromarray(arr)

    def test_metadata_aspect_ratio(self, pipeline, test_image):
        """Test aspect ratio calculation."""
        meta = pipeline._stage_metadata(test_image)
        assert meta["aspect_ratio"] > 0

    def test_color_dominant_matches_image(self, pipeline, test_image):
        """Test that dominant colors reflect image content."""
        color = pipeline._stage_color(test_image)
        # First color should be dark gray (background)
        first = color["dominant_colors"][0]
        assert first["rgb"][0] < 50  # dark
        assert first["percentage"] > 50  # majority of image

    def test_color_red_square_detected(self, pipeline, test_image):
        """Test that red square is detected as a color."""
        color = pipeline._stage_color(test_image)
        hex_values = [c["hex"] for c in color["dominant_colors"]]
        # Red should be in dominant colors
        assert any("ff" in h and "00" in h for h in hex_values)

    def test_template_stage_no_templates(self, pipeline, test_image):
        """Test template stage without templates."""
        # Should not crash
        result = pipeline.analyze(
            source=test_image,
            stages=["metadata", "template"],
        )
        # Template stage should have error since no templates provided
        assert "metadata" in result.stages
