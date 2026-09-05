"""
Trigger module — react to changes in the screen stream.

Provides different trigger types that fire when conditions are met.
"""

import logging
import time
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass, field

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TriggerEvent:
    """A trigger firing event."""
    trigger_name: str
    timestamp: float = field(default_factory=time.time)
    data: Dict = field(default_factory=dict)
    screenshot_path: Optional[str] = None

    def __repr__(self):
        return f"TriggerEvent({self.trigger_name}, data={self.data})"


class ChangeTrigger:
    """
    Triggers when screen changes by more than a threshold.

    Args:
        threshold: Minimum change percentage to trigger (0-100).
        cooldown: Minimum seconds between triggers.
        auto_save: Whether to save a screenshot on trigger.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        cooldown: float = 1.0,
        auto_save: bool = True,
    ):
        self.threshold = threshold
        self.cooldown = cooldown
        self.auto_save = auto_save
        self._last_trigger: float = 0
        self._events: List[TriggerEvent] = []

    def check(self, change_percentage: float, frame: Optional[Image.Image] = None) -> Optional[TriggerEvent]:
        """
        Check if change exceeds threshold.

        Args:
            change_percentage: Current change percentage.
            frame: Current frame for saving.

        Returns:
            TriggerEvent if triggered, else None.
        """
        now = time.time()

        # Cooldown check
        if now - self._last_trigger < self.cooldown:
            return None

        if change_percentage >= self.threshold:
            event = TriggerEvent(
                trigger_name="change",
                data={"change_percentage": change_percentage},
            )

            if self.auto_save and frame:
                from giga_eyes.storage.screenshots import ScreenshotStorage
                storage = ScreenshotStorage()
                event.screenshot_path = str(storage.save(frame, custom_name=f"change_{int(now)}"))

            self._events.append(event)
            self._last_trigger = now

            logger.info(
                "ChangeTrigger fired: %.1f%% change (threshold=%.1f%%)",
                change_percentage,
                self.threshold,
            )

            return event

        return None


class ElementTrigger:
    """
    Triggers when a UI element (template) appears on screen.

    Args:
        template_path: Path to template image.
        threshold: Match threshold for template search.
        cooldown: Minimum seconds between triggers.
    """

    def __init__(
        self,
        template_path: str,
        threshold: float = 0.8,
        cooldown: float = 2.0,
    ):
        self.template_path = template_path
        self.threshold = threshold
        self.cooldown = cooldown
        self._last_trigger: float = 0
        self._events: List[TriggerEvent] = []

    def check(self, frame: Image.Image) -> Optional[TriggerEvent]:
        """
        Check if template is present in frame.

        Args:
            frame: Current screen frame.

        Returns:
            TriggerEvent if template found, else None.
        """
        now = time.time()

        if now - self._last_trigger < self.cooldown:
            return None

        try:
            from giga_eyes.analyze.template import find_template

            template = Image.open(self.template_path)
            box = find_template(frame, template, threshold=self.threshold)

            if box:
                event = TriggerEvent(
                    trigger_name="element",
                    data={"template": self.template_path, "bbox": box},
                )
                self._events.append(event)
                self._last_trigger = now

                logger.info(
                    "ElementTrigger fired: template=%s found at %s",
                    self.template_path,
                    box,
                )

                return event

        except Exception as e:
            logger.error("ElementTrigger check error: %s", e)

        return None


class ColorTrigger:
    """
    Triggers when a specific color appears on screen.

    Args:
        color_rgb: Target color (R, G, B).
        threshold: Minimum percentage of screen with target color.
        tolerance: Color tolerance.
        cooldown: Minimum seconds between triggers.
    """

    def __init__(
        self,
        color_rgb: tuple,
        threshold: float = 1.0,
        tolerance: int = 20,
        cooldown: float = 1.0,
    ):
        self.color_rgb = color_rgb
        self.threshold = threshold
        self.tolerance = tolerance
        self.cooldown = cooldown
        self._last_trigger: float = 0
        self._events: List[TriggerEvent] = []

    def check(self, frame: Image.Image) -> Optional[TriggerEvent]:
        """
        Check if target color is present above threshold.

        Args:
            frame: Current screen frame.

        Returns:
            TriggerEvent if color found, else None.
        """
        now = time.time()

        if now - self._last_trigger < self.cooldown:
            return None

        try:
            from giga_eyes.analyze.color import find_color

            boxes = find_color(frame, self.color_rgb, self.tolerance)

            # Calculate total area
            total_area = sum(w * h for _, _, w, h in boxes)
            frame_area = frame.width * frame.height
            color_pct = (total_area / frame_area) * 100 if frame_area > 0 else 0

            if color_pct >= self.threshold:
                event = TriggerEvent(
                    trigger_name="color",
                    data={
                        "color": self.color_rgb,
                        "percentage": color_pct,
                        "regions": len(boxes),
                    },
                )
                self._events.append(event)
                self._last_trigger = now

                logger.info(
                    "ColorTrigger fired: RGB%s %.1f%% (threshold=%.1f%%)",
                    self.color_rgb,
                    color_pct,
                    self.threshold,
                )

                return event

        except Exception as e:
            logger.error("ColorTrigger check error: %s", e)

        return None


class YOLOTrigger:
    """
    Triggers when UI element detector finds specific classes.

    Args:
        class_names: List of UI class names to detect.
        confidence: Minimum confidence threshold.
        cooldown: Minimum seconds between triggers.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        confidence: float = 0.25,
        cooldown: float = 2.0,
    ):
        self.class_names = class_names
        self.confidence = confidence
        self.cooldown = cooldown
        self._last_trigger: float = 0
        self._events: List[TriggerEvent] = []
        self._detector = None

    def check(self, frame: Image.Image) -> Optional[TriggerEvent]:
        """
        Check if UI detector finds target classes in frame.

        Args:
            frame: Current screen frame.

        Returns:
            TriggerEvent if objects detected, else None.
        """
        now = time.time()

        if now - self._last_trigger < self.cooldown:
            return None

        try:
            from giga_eyes.analyze.yolo import UIDetector

            if self._detector is None:
                self._detector = UIDetector(confidence=self.confidence)

            detections = self._detector.detect(frame, classes=self.class_names)

            if detections:
                event = TriggerEvent(
                    trigger_name="yolo",
                    data={
                        "classes": self.class_names,
                        "detections": [
                            {"name": d["name"], "confidence": d["confidence"], "bbox": d["bbox"]}
                            for d in detections
                        ],
                    },
                )
                self._events.append(event)
                self._last_trigger = now

                logger.info(
                    "YOLOTrigger fired: %d detections in classes %s",
                    len(detections),
                    self.class_names,
                )

                return event

        except Exception as e:
            logger.error("YOLOTrigger check error: %s", e)

        return None


class TriggerManager:
    """
    Manages multiple triggers and fires callbacks.

    Integrates with WindowStream to automatically check triggers.
    """

    def __init__(self, auto_save: bool = True):
        self.triggers: List[dict] = []
        self._events: List[TriggerEvent] = []
        self.auto_save = auto_save
        self._callback_fn: Optional[Callable] = None

    def add_change_trigger(self, threshold: float = 5.0, cooldown: float = 1.0) -> ChangeTrigger:
        """Add a change detection trigger."""
        trigger = ChangeTrigger(threshold=threshold, cooldown=cooldown, auto_save=self.auto_save)
        self.triggers.append({"type": "change", "trigger": trigger})
        return trigger

    def add_element_trigger(
        self, template_path: str, threshold: float = 0.8, cooldown: float = 2.0
    ) -> ElementTrigger:
        """Add a template-based trigger."""
        trigger = ElementTrigger(
            template_path=template_path,
            threshold=threshold,
            cooldown=cooldown,
        )
        self.triggers.append({"type": "element", "trigger": trigger})
        return trigger

    def add_color_trigger(
        self, color_rgb: tuple, threshold: float = 1.0, tolerance: int = 20, cooldown: float = 1.0
    ) -> ColorTrigger:
        """Add a color-based trigger."""
        trigger = ColorTrigger(
            color_rgb=color_rgb,
            threshold=threshold,
            tolerance=tolerance,
            cooldown=cooldown,
        )
        self.triggers.append({"type": "color", "trigger": trigger})
        return trigger

    def add_yolo_trigger(
        self, class_names: Optional[List[str]] = None, confidence: float = 0.25, cooldown: float = 2.0
    ) -> YOLOTrigger:
        """Add a UI element-based trigger."""
        trigger = YOLOTrigger(
            class_names=class_names,
            confidence=confidence,
            cooldown=cooldown,
        )
        self.triggers.append({"type": "yolo", "trigger": trigger})
        return trigger

    def set_callback(self, callback: Callable):
        """
        Set a callback function for trigger events.

        Args:
            callback: Function(event: TriggerEvent) -> None
        """
        self._callback_fn = callback

    def process_frame(self, frame: Image.Image, change_pct: float = 0):
        """
        Process a frame through all triggers.

        Args:
            frame: Current screen frame.
            change_pct: Current change percentage (for change trigger).
        """
        for trigger_info in self.triggers:
            t_type = trigger_info["type"]
            trigger = trigger_info["trigger"]

            try:
                if t_type == "change":
                    event = trigger.check(change_pct, frame)
                elif t_type == "element":
                    event = trigger.check(frame)
                elif t_type == "color":
                    event = trigger.check(frame)
                elif t_type == "yolo":
                    event = trigger.check(frame)
                else:
                    continue

                if event:
                    self._events.append(event)
                    if self._callback_fn:
                        self._callback_fn(event)

            except Exception as e:
                logger.error("Trigger process error (%s): %s", t_type, e)

    @property
    def event_count(self) -> int:
        return len(self._events)
