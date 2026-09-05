"""Tests for diff and color detection modules."""

import pytest
import numpy as np
from PIL import Image

from giga_eyes.analyze.diff import (
    find_changes,
    count_changes,
    get_change_percentage,
    get_bounding_boxes,
    highlight_changes,
    compare_frames,
)
from giga_eyes.analyze.color import (
    find_color,
    is_pixel_error,
    get_dominant_colors,
)


class TestDiffDetection:
    """Tests for frame diff detection."""

    @pytest.fixture
    def frame_a(self):
        """Create a static frame."""
        img_array = np.zeros((200, 200, 3), dtype=np.uint8)
        img_array[:, :] = [30, 30, 30]  # dark gray
        return Image.fromarray(img_array)

    @pytest.fixture
    def frame_b(self, frame_a):
        """Create a frame with a changed region."""
        img_array = np.array(frame_a)
        img_array[50:150, 50:150] = [255, 0, 0]  # red square
        return Image.fromarray(img_array)

    def test_find_changes_detects_difference(self, frame_a, frame_b):
        """Test that changes are detected."""
        mask = find_changes(frame_a, frame_b, threshold=10)

        # Should have changed pixels
        assert count_changes(mask) > 0

    def test_find_changes_no_difference(self, frame_a):
        """Test when frames are identical."""
        mask = find_changes(frame_a, frame_a, threshold=10)

        # Should have no changed pixels
        assert count_changes(mask) == 0

    def test_count_changes(self, frame_a, frame_b):
        """Test change counting."""
        mask = find_changes(frame_a, frame_b, threshold=10)

        # Should have around 100x100 = 10000 changed pixels
        count = count_changes(mask)
        assert 9000 <= count <= 11000

    def test_get_change_percentage(self, frame_a, frame_b):
        """Test percentage calculation."""
        mask = find_changes(frame_a, frame_b, threshold=10)
        pct = get_change_percentage(mask)

        # 100x100 out of 200x200 = 25%
        assert 20 <= pct <= 30

    def test_get_bounding_boxes(self, frame_a, frame_b):
        """Test bounding box detection."""
        mask = find_changes(frame_a, frame_b, threshold=10)
        boxes = get_bounding_boxes(mask, min_area=1000)

        # Should find one large box
        assert len(boxes) >= 1

        # Check box is near expected location
        x, y, w, h = boxes[0]
        assert 40 <= x <= 60
        assert 40 <= y <= 60
        assert 90 <= w <= 110
        assert 90 <= h <= 110

    def test_compare_frames(self, frame_a, frame_b):
        """Test comprehensive comparison."""
        result = compare_frames(frame_a, frame_b, threshold=10)

        assert bool(result["is_changed"]) is True
        assert result["changed_pixels"] > 0
        assert len(result["bounding_boxes"]) >= 1
        assert "mask" in result

    def test_highlight_changes(self, frame_a, frame_b):
        """Test highlighting changes."""
        mask = find_changes(frame_a, frame_b, threshold=10)
        highlighted = highlight_changes(frame_a, mask, color=(0, 255, 0))

        assert highlighted.size == frame_a.size
        assert highlighted.mode == "RGB"


class TestColorDetection:
    """Tests for color detection."""

    @pytest.fixture
    def test_image(self):
        """Create an image with known colors."""
        img_array = np.zeros((200, 200, 3), dtype=np.uint8)
        img_array[:, :] = [30, 30, 30]  # dark gray background
        img_array[50:150, 50:150] = [255, 0, 0]  # red square
        img_array[160:180, 160:180] = [0, 255, 0]  # green square
        return Image.fromarray(img_array)

    def test_find_color(self, test_image):
        """Test finding a specific color."""
        boxes = find_color(test_image, (255, 0, 0), tolerance=20)

        # Should find the red square
        assert len(boxes) >= 1

        # Check box is near expected location
        x, y, w, h = boxes[0]
        assert 40 <= x <= 60
        assert 40 <= y <= 60

    def test_find_color_not_present(self, test_image):
        """Test finding a color that's not in the image."""
        boxes = find_color(test_image, (0, 0, 255), tolerance=10)

        # Should not find blue (not in image)
        assert len(boxes) == 0

    def test_is_pixel_error(self, test_image):
        """Test error detection."""
        # Red square is 100x100 out of 200x200 = 25%
        is_error = is_pixel_error(test_image, (255, 0, 0), threshold=0.1)
        assert is_error is True

        # High threshold should not trigger
        is_error = is_pixel_error(test_image, (255, 0, 0), threshold=0.5)
        assert is_error is False

    def test_get_dominant_colors(self, test_image):
        """Test dominant color detection."""
        colors = get_dominant_colors(test_image, num_colors=3)

        assert len(colors) == 3

        # First color should be dark gray (background)
        r, g, b, pct = colors[0]
        assert r < 50  # dark
        assert g < 50
        assert b < 50
        assert pct > 0.5  # more than 50% of image
