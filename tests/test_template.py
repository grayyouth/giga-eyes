"""Tests for template matching module."""

import pytest
import numpy as np
from PIL import Image

from giga_eyes.analyze.template import (
    pil_to_opencv,
    opencv_to_pil,
    find_template,
    find_all_matches,
    get_match_quality,
    resize_template,
)


class TestImageConversion:
    """Tests for PIL/OpenCV conversion utilities."""

    def test_pil_to_opencv_rgb(self):
        """Test RGB image conversion."""
        img = Image.new("RGB", (100, 100), color=(255, 128, 64))
        result = pil_to_opencv(img)

        assert result.shape == (100, 100, 3)
        # BGR format: original (255, 128, 64) -> (64, 128, 255)
        assert result[0, 0].tolist() == [64, 128, 255]

    def test_pil_to_opencv_rgba(self):
        """Test RGBA image conversion (should drop alpha)."""
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        result = pil_to_opencv(img)

        assert result.shape == (50, 50, 3)

    def test_opencv_to_pil(self):
        """Test OpenCV to PIL conversion."""
        img_cv = np.zeros((100, 100, 3), dtype=np.uint8)
        img_cv[:, :] = [100, 150, 200]  # BGR

        result = opencv_to_pil(img_cv)

        assert result.size == (100, 100)
        assert result.mode == "RGB"


class TestTemplateMatching:
    """Tests for template matching."""

    @pytest.fixture
    def test_image(self):
        """Create a test image with distinctive patterns (not solid colors)."""
        img_array = np.zeros((400, 400, 3), dtype=np.uint8)
        img_array[:, :] = [30, 30, 30]  # dark gray background

        # Red vertical stripe with gradient edges
        img_array[100:300, 180:190] = [200, 30, 30]  # left edge
        img_array[100:300, 190:210] = [255, 50, 50]  # center
        img_array[100:300, 210:220] = [200, 30, 30]  # right edge

        return Image.fromarray(img_array)

    @pytest.fixture
    def test_template(self):
        """Create a template matching the red stripe (with gradient)."""
        template_array = np.zeros((200, 40, 3), dtype=np.uint8)
        template_array[:, :10] = [200, 30, 30]
        template_array[:, 10:30] = [255, 50, 50]
        template_array[:, 30:] = [200, 30, 30]
        return Image.fromarray(template_array)

    def test_find_template_exact_match(self, test_image, test_template):
        """Test finding an exact match."""
        box = find_template(test_image, test_template, threshold=0.85)

        assert box is not None
        left, top, right, bottom = box
        # Template is 40x200, placed at (180, 100) in image
        assert 170 <= left <= 190
        assert 90 <= top <= 110

    def test_find_template_no_match(self, test_image):
        """Test when template is not in image."""
        # Create a distinctive yellow pattern (not in image)
        yellow_array = np.zeros((50, 50, 3), dtype=np.uint8)
        yellow_array[:, :] = [255, 255, 0]  # yellow
        yellow_array[:10, :] = [0, 0, 0]  # black top border
        yellow_array[-10:, :] = [0, 0, 0]  # black bottom border
        yellow_template = Image.fromarray(yellow_array)

        box = find_template(test_image, yellow_template, threshold=0.85)

        assert box is None

    def test_find_all_matches_multiple(self):
        """Test finding multiple identical templates."""
        # Create image with two identical distinctive patterns
        # np.zeros((height, width, channels)) -> (200, 400, 3)
        img_array = np.zeros((200, 400, 3), dtype=np.uint8)
        img_array[:, :] = [30, 30, 30]

        # Pattern: blue square with white cross
        for loc_x in [40, 240]:
            for loc_y in [40]:
                # Blue background
                img_array[loc_y:loc_y+60, loc_x:loc_x+60] = [0, 0, 200]
                # White cross
                img_array[loc_y+25:loc_y+35, loc_x:loc_x+60] = [255, 255, 255]
                img_array[loc_y:loc_y+60, loc_x+25:loc_x+35] = [255, 255, 255]

        img = Image.fromarray(img_array)

        # Template: blue square with white cross
        template_array = np.zeros((60, 60, 3), dtype=np.uint8)
        template_array[:, :] = [0, 0, 200]
        template_array[25:35, :] = [255, 255, 255]
        template_array[:, 25:35] = [255, 255, 255]
        template = Image.fromarray(template_array)

        matches = find_all_matches(img, template, threshold=0.85)

        assert len(matches) == 2

        # Check both locations (allow ±1 pixel tolerance)
        boxes = sorted([(b[0], b[1]) for b in matches])
        assert any(39 <= x <= 41 and 39 <= y <= 41 for x, y in boxes)
        assert any(239 <= x <= 241 and 39 <= y <= 41 for x, y in boxes)

    def test_find_all_matches_threshold(self, test_image, test_template):
        """Test threshold filtering."""
        # Low threshold — should find it
        matches_low = find_all_matches(test_image, test_template, threshold=0.5)
        assert len(matches_low) >= 1

        # High threshold — should still find it (exact match)
        matches_high = find_all_matches(test_image, test_template, threshold=0.95)
        assert len(matches_high) >= 1

    def test_find_all_matches_max_matches(self):
        """Test max_matches limit."""
        # Create image with many identical patterns
        img_array = np.zeros((400, 400, 3), dtype=np.uint8)
        img_array[:, :] = [30, 30, 30]

        for i in range(5):
            for j in range(5):
                x = 20 + j * 80
                y = 20 + i * 80
                # Blue square with white cross
                img_array[y:y+40, x:x+40] = [0, 0, 200]
                img_array[y+15:y+25, x:x+40] = [255, 255, 255]
                img_array[y:y+40, x+15:x+25] = [255, 255, 255]

        img = Image.fromarray(img_array)

        # Template: blue square with white cross
        template_array = np.zeros((40, 40, 3), dtype=np.uint8)
        template_array[:, :] = [0, 0, 200]
        template_array[15:25, :] = [255, 255, 255]
        template_array[:, 15:25] = [255, 255, 255]
        template = Image.fromarray(template_array)

        matches = find_all_matches(img, template, threshold=0.85, max_matches=3)

        assert len(matches) <= 3


class TestMatchQuality:
    """Tests for match quality scoring."""

    def test_quality_exact_match(self):
        """Test quality score for exact match."""
        img = Image.new("RGB", (100, 100), color=(255, 128, 64))
        template = Image.new("RGB", (50, 50), color=(255, 128, 64))

        quality = get_match_quality(img, template, 0, 0)

        assert quality > 0.95

    def test_quality_no_match(self):
        """Test quality score for no match."""
        # Use distinctive patterns
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :] = [255, 0, 0]  # red background
        img_array[20:80, 20:80] = [200, 0, 0]  # slightly different red

        # Template with distinctive pattern (not solid color)
        template_array = np.zeros((50, 50, 3), dtype=np.uint8)
        template_array[:, :] = [0, 0, 255]  # blue background
        template_array[10:40, 10:40] = [255, 255, 0]  # yellow center

        img = Image.fromarray(img_array)
        template = Image.fromarray(template_array)

        quality = get_match_quality(img, template, 0, 0)

        # For completely different patterns, quality should be low
        assert quality < 0.5


class TestResize:
    """Tests for template resizing."""

    def test_resize_up(self):
        """Test resizing up (scale > 1)."""
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        resized = resize_template(img, 2.0)

        assert resized.size == (200, 200)

    def test_resize_down(self):
        """Test resizing down (scale < 1)."""
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        resized = resize_template(img, 0.5)

        assert resized.size == (50, 50)

    def test_resize_invalid(self):
        """Test invalid scale."""
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))

        with pytest.raises(ValueError):
            resize_template(img, -1.0)
