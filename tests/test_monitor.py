"""Tests for monitoring module."""

import pytest
from unittest.mock import patch, MagicMock, Mock
from PIL import Image
import numpy as np


class TestStreamBuffer:
    """Tests for StreamBuffer."""

    def test_add_and_get(self):
        """Test adding frames and retrieving them."""
        from giga_eyes.monitor.stream import StreamBuffer

        buffer = StreamBuffer(size=5)
        img = Image.new("RGB", (100, 100), color="red")

        buffer.add(img, 1.0, {"index": 0})
        buffer.add(img, 2.0, {"index": 1})

        assert buffer.count == 2
        recent = buffer.get_recent(1)
        assert len(recent) == 1
        assert recent[0]["metadata"]["index"] == 1

    def test_circular_buffer(self):
        """Test that buffer wraps around."""
        from giga_eyes.monitor.stream import StreamBuffer

        buffer = StreamBuffer(size=3)
        img = Image.new("RGB", (100, 100), color="blue")

        for i in range(5):
            buffer.add(img, float(i), {"index": i})

        assert buffer.count == 3
        recent = buffer.get_recent(3)
        assert recent[0]["metadata"]["index"] == 2
        assert recent[2]["metadata"]["index"] == 4

    def test_latest(self):
        """Test getting the latest frame."""
        from giga_eyes.monitor.stream import StreamBuffer

        buffer = StreamBuffer()
        assert buffer.latest is None

        img = Image.new("RGB", (100, 100), color="green")
        buffer.add(img, 1.0)

        assert buffer.latest is not None
        assert buffer.latest["timestamp"] == 1.0


class TestTriggerManager:
    """Tests for TriggerManager."""

    def test_add_change_trigger(self):
        """Test adding a change trigger."""
        from giga_eyes.monitor.trigger import TriggerManager

        manager = TriggerManager()
        trigger = manager.add_change_trigger(threshold=5.0, cooldown=1.0)

        assert trigger.threshold == 5.0
        assert len(manager.triggers) == 1

    def test_add_element_trigger(self):
        """Test adding an element trigger."""
        from giga_eyes.monitor.trigger import TriggerManager

        manager = TriggerManager()
        trigger = manager.add_element_trigger(
            template_path="test.png",
            threshold=0.8,
        )

        assert trigger.template_path == "test.png"
        assert len(manager.triggers) == 1

    def test_add_color_trigger(self):
        """Test adding a color trigger."""
        from giga_eyes.monitor.trigger import TriggerManager

        manager = TriggerManager()
        trigger = manager.add_color_trigger(
            color_rgb=(255, 0, 0),
            threshold=1.0,
        )

        assert trigger.color_rgb == (255, 0, 0)
        assert len(manager.triggers) == 1

    def test_add_yolo_trigger(self):
        """Test adding a YOLO trigger."""
        from giga_eyes.monitor.trigger import YOLOTrigger

        trigger = YOLOTrigger(class_names=["Button", "Menu"], confidence=0.5)

        assert trigger.class_names == ["Button", "Menu"]
        assert trigger.confidence == 0.5

    def test_change_trigger_fires(self):
        """Test that change trigger fires above threshold."""
        from giga_eyes.monitor.trigger import ChangeTrigger

        trigger = ChangeTrigger(threshold=5.0, cooldown=0)
        img = Image.new("RGB", (100, 100), color="red")

        event = trigger.check(change_percentage=10.0, frame=img)

        assert event is not None
        assert event.trigger_name == "change"
        assert len(trigger._events) == 1

    def test_change_trigger_does_not_fire(self):
        """Test that change trigger doesn't fire below threshold."""
        from giga_eyes.monitor.trigger import ChangeTrigger

        trigger = ChangeTrigger(threshold=5.0, cooldown=0)
        img = Image.new("RGB", (100, 100), color="red")

        event = trigger.check(change_percentage=2.0, frame=img)

        assert event is None
        assert len(trigger._events) == 0

    def test_change_trigger_cooldown(self):
        """Test that cooldown prevents rapid firing."""
        from giga_eyes.monitor.trigger import ChangeTrigger

        trigger = ChangeTrigger(threshold=5.0, cooldown=10.0)
        img = Image.new("RGB", (100, 100), color="red")

        # First trigger
        event1 = trigger.check(change_percentage=10.0, frame=img)
        assert event1 is not None

        # Second trigger (should be blocked by cooldown)
        event2 = trigger.check(change_percentage=10.0, frame=img)
        assert event2 is None

        assert len(trigger._events) == 1

    def test_process_frame(self):
        """Test processing a frame through triggers."""
        from giga_eyes.monitor.trigger import TriggerManager

        manager = TriggerManager(auto_save=False)
        manager.add_change_trigger(threshold=5.0, cooldown=0)

        events_fired = []

        def on_event(event):
            events_fired.append(event)

        manager.set_callback(on_event)
        img = Image.new("RGB", (100, 100), color="blue")

        manager.process_frame(img, change_pct=10.0)

        assert len(events_fired) == 1
        assert events_fired[0].trigger_name == "change"
        assert manager.event_count == 1
