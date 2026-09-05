"""
CLI interface for Giga Eyes.

Usage:
    giga-eyes screenshot [--output PATH] [--format FORMAT]
    giga-eyes window --title TITLE [--output PATH]
    giga-eyes region --x X --y Y --w W --h H [--output PATH]
    giga-eyes template --image PATH --template PATH
    giga-eyes diff --before PATH --after PATH
    giga-eyes color --image PATH --color R,G,B
    giga-eyes ocr --image PATH
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from giga_eyes.capture.screen import capture_full_screen
from giga_eyes.capture.window import capture_window_by_title
from giga_eyes.capture.region import capture_region
from giga_eyes.storage.screenshots import ScreenshotStorage
from giga_eyes.storage.logs import EventLogger


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_screenshot(args):
    """Handle 'screenshot' command."""
    storage = ScreenshotStorage(
        storage_path=args.output or "screenshots",
        format=args.format or "png",
    )

    # Capture full screen
    img = capture_full_screen(output_path=None)

    if img is None:
        logging.error("Failed to capture screen")
        sys.exit(1)

    # Save
    filepath = storage.save(img)
    print(f"Screenshot saved: {filepath}")


def cmd_window(args):
    """Handle 'window' command."""
    storage = ScreenshotStorage(
        storage_path=args.output or "screenshots",
        format="png",
    )

    logger = logging.getLogger(__name__)

    # Capture window
    img = capture_window_by_title(
        title=args.title,
        partial=True,
        output_path=None,
    )

    if img is None:
        logging.error("Failed to capture window '%s'", args.title)
        sys.exit(1)

    # Save
    filepath = storage.save(img, window_title=args.title)
    print(f"Window screenshot saved: {filepath}")


def cmd_region(args):
    """Handle 'region' command."""
    storage = ScreenshotStorage(
        storage_path=args.output or "screenshots",
        format="png",
    )

    # Capture region
    img = capture_region(
        x=args.x,
        y=args.y,
        width=args.w,
        height=args.h,
        output_path=None,
    )

    if img is None:
        logging.error("Failed to capture region (%d, %d, %d, %d)", args.x, args.y, args.w, args.h)
        sys.exit(1)

    # Save
    filepath = storage.save(img)
    print(f"Region screenshot saved: {filepath}")


def cmd_template(args):
    """Handle 'template' command."""
    from giga_eyes.analyze.template import find_template

    try:
        image = Image.open(args.image)
        template = Image.open(args.template)
    except Exception as e:
        logging.error("Failed to load images: %s", e)
        sys.exit(1)

    box = find_template(
        image,
        template,
        threshold=getattr(args, "threshold", 0.8),
    )

    if box:
        print(f"Template found at: {box}")
    else:
        print("Template not found")


def cmd_diff(args):
    """Handle 'diff' command."""
    from giga_eyes.analyze.diff import compare_frames

    try:
        img_before = Image.open(args.before)
        img_after = Image.open(args.after)
    except Exception as e:
        logging.error("Failed to load images: %s", e)
        sys.exit(1)

    result = compare_frames(
        img_before,
        img_after,
        threshold=getattr(args, "threshold", 10),
    )

    print(f"Changed pixels: {result['changed_pixels']}")
    print(f"Change percentage: {result['change_percentage']:.2f}%")
    print(f"Bounding boxes: {len(result['bounding_boxes'])}")
    for i, box in enumerate(result["bounding_boxes"]):
        print(f"  Box {i+1}: {box}")


def cmd_color(args):
    """Handle 'color' command."""
    from giga_eyes.analyze.color import find_color

    try:
        image = Image.open(args.image)
    except Exception as e:
        logging.error("Failed to load image: %s", e)
        sys.exit(1)

    # Parse color
    try:
        r, g, b = map(int, args.color.split(","))
    except Exception as e:
        logging.error("Invalid color format: %s (use R,G,B)", e)
        sys.exit(1)

    boxes = find_color(image, (r, g, b), tolerance=getattr(args, "tolerance", 20))

    if boxes:
        print(f"Found {len(boxes)} region(s) of color RGB({r},{g},{b}):")
        for box in boxes:
            print(f"  {box}")
    else:
        print(f"Color RGB({r},{g},{b}) not found")


def cmd_ocr(args):
    """Handle 'ocr' command."""
    from giga_eyes.analyze.ocr import OCRDetector

    try:
        image = Image.open(args.image)
    except Exception as e:
        logging.error("Failed to load image: %s", e)
        sys.exit(1)

    detector = OCRDetector(lang=getattr(args, "lang", "russian"))
    texts = detector.recognize_text(
        image,
        threshold=getattr(args, "threshold", 0.5),
    )

    if texts:
        print(f"Recognized {len(texts)} text region(s):")
        for item in texts:
            print(f"  [{item['confidence']:.2f}] {item['text']} @ {item['bbox']}")
    else:
        print("No text found")


def cmd_ui_detect(args):
    """Handle 'ui-detect' command — detect UI elements in screenshot."""
    from giga_eyes.analyze.yolo import UIDetector

    try:
        image = Image.open(args.image)
    except Exception as e:
        logging.error("Failed to load image: %s", e)
        sys.exit(1)

    detector = UIDetector(confidence=args.confidence)

    # Parse class filter
    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]

    detections = detector.detect(image, classes=classes)

    if detections:
        print(f"Detected {len(detections)} UI element(s):")
        for det in detections:
            print(
                f"  [{det['confidence']:.2f}] {det['name']} @ {det['bbox']}"
            )

        # Draw and save if output specified
        if args.output:
            annotated = detector.detect_and_draw(image, classes=classes)
            annotated.save(args.output)
            print(f"Annotated image saved: {args.output}")
    else:
        print("No UI elements detected")


def cmd_vision(args):
    """Handle 'vision' command — VLM analysis of screenshot."""
    from giga_eyes.analyze.vision import VisionAnalyzer

    try:
        image = Image.open(args.image)
    except Exception as e:
        logging.error("Failed to load image: %s", e)
        sys.exit(1)

    analyzer = VisionAnalyzer(model_name=getattr(args, "model", None))

    if args.action == "describe":
        result = analyzer.describe(image, prompt=getattr(args, "prompt", None))
    elif args.action == "design":
        result = analyzer.extract_design(image)
    elif args.action == "text":
        result = analyzer.extract_text(image)
    elif args.action == "compare":
        try:
            before = Image.open(args.before)
        except Exception as e:
            logging.error("Failed to load before image: %s", e)
            sys.exit(1)
        result = analyzer.compare(before, image)
    else:
        result = analyzer.answer_question(image, args.question)

    print(result)


def cmd_analyze(args):
    """Handle 'analyze' command — universal image analysis pipeline."""
    from giga_eyes.pipeline import ImagePipeline

    # Parse stages
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]

    # Parse templates
    templates = None
    if args.templates:
        templates = {}
        for name_path in args.templates.split(","):
            name_path = name_path.strip()
            if ":" in name_path:
                name, path = name_path.split(":", 1)
                templates[name.strip()] = path.strip()
            else:
                templates[name_path] = name_path

    # Build output path
    output = args.output
    if not output:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"analysis_{ts}.json"

    # Run pipeline
    pipeline = ImagePipeline()
    result = pipeline.analyze(
        source=args.image,
        stages=stages,
        output=output,
        templates=templates,
        confidence=args.confidence,
        num_colors=args.num_colors,
    )

    # Print summary
    print(f"\nAnalysis complete:")
    print(f"  Output: {output}")
    print(f"  Stages run: {', '.join(result.stages.keys())}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"    - [{err['stage']}] {err['message']}")
    if "metadata" in result.stages:
        m = result.stages["metadata"]
        print(f"  Image: {m['width']}x{m['height']} ({m['megapixels']} MP)")
    if "timing" in result.timing:
        print(f"  Total time: {result.timing['total_seconds']}s")


def cmd_monitor(args):
    """Handle 'monitor' command."""
    import time

    from giga_eyes.monitor.stream import WindowStream
    from giga_eyes.monitor.trigger import TriggerManager

    def signal_handler(sig, frame):
        logging.info("Interrupt received, stopping monitor...")
        stream.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Create stream
    stream = WindowStream(
        window_title=getattr(args, "title", None),
        fps=getattr(args, "fps", 10),
    )

    # Create trigger manager
    manager = TriggerManager(auto_save=True)

    trigger_type = getattr(args, "trigger", None)
    threshold = getattr(args, "threshold", 5.0)

    if trigger_type == "change":
        manager.add_change_trigger(threshold=threshold, cooldown=1.0)
    elif trigger_type == "element":
        template = getattr(args, "template", None)
        if not template:
            logging.error("--template required for element trigger")
            sys.exit(1)
        manager.add_element_trigger(template_path=template, threshold=threshold)
    elif trigger_type == "color":
        color_str = getattr(args, "color", None)
        if not color_str:
            logging.error("--color required for color trigger (R,G,B)")
            sys.exit(1)
        r, g, b = map(int, color_str.split(","))
        manager.add_color_trigger(color_rgb=(r, g, b), threshold=threshold)
    elif trigger_type == "yolo":
        classes_str = getattr(args, "yolo_classes", None)
        if not classes_str:
            logging.error("--yolo-classes required (e.g., Button,Input_Elements)")
            sys.exit(1)
        classes = [c.strip() for c in classes_str.split(",")]
        manager.add_yolo_trigger(class_names=classes, confidence=threshold)

    # Set callback
    def on_event(event):
        print(f"\n[TRIGGER] {event.trigger_name}")
        print(f"  Data: {event.data}")
        if event.screenshot_path:
            print(f"  Screenshot: {event.screenshot_path}")

    manager.set_callback(on_event)

    # Connect stream to triggers
    def on_change(**kwargs):
        change_pct = kwargs.get("change_percentage", 0)
        frame = kwargs.get("frame")
        if frame:
            manager.process_frame(frame, change_pct)

    stream.callback.on("change", on_change)

    print(f"Monitoring: {args.title or 'full screen'} @ {args.fps} FPS")
    print("Press Ctrl+C to stop")

    # Run stream
    stream.start()

    try:
        while stream.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        print(f"\nStopped. Events: {manager.event_count}, Frames: {stream.frame_count}")


def main():
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="giga-eyes",
        description="Giga Eyes — Visual observation and UI debugging tool",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Giga Eyes v0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # screenshot command
    p_screenshot = subparsers.add_parser("screenshot", help="Capture full screen")
    p_screenshot.add_argument(
        "--output", "-o",
        help="Output directory (default: screenshots)",
    )
    p_screenshot.add_argument(
        "--format", "-f",
        choices=["png", "jpg", "bmp"],
        help="Output format (default: png)",
    )
    p_screenshot.set_defaults(func=cmd_screenshot)

    # window command
    p_window = subparsers.add_parser("window", help="Capture specific window")
    p_window.add_argument(
        "--title", "-t",
        required=True,
        help="Window title (or partial match)",
    )
    p_window.add_argument(
        "--output", "-o",
        help="Output directory (default: screenshots)",
    )
    p_window.set_defaults(func=cmd_window)

    # region command
    p_region = subparsers.add_parser("region", help="Capture screen region")
    p_region.add_argument("--x", type=int, required=True, help="Left coordinate")
    p_region.add_argument("--y", type=int, required=True, help="Top coordinate")
    p_region.add_argument("--w", type=int, required=True, help="Width")
    p_region.add_argument("--h", type=int, required=True, help="Height")
    p_region.add_argument(
        "--output", "-o",
        help="Output directory (default: screenshots)",
    )
    p_region.set_defaults(func=cmd_region)

    # template command
    p_template = subparsers.add_parser("template", help="Find template in image")
    p_template.add_argument("--image", "-i", required=True, help="Screenshot image")
    p_template.add_argument("--template", "-t", required=True, help="Template image")
    p_template.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Match threshold (default: 0.8)",
    )
    p_template.set_defaults(func=cmd_template)

    # diff command
    p_diff = subparsers.add_parser("diff", help="Compare two images")
    p_diff.add_argument("--before", "-b", required=True, help="Before image")
    p_diff.add_argument("--after", "-a", required=True, help="After image")
    p_diff.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Pixel difference threshold (default: 10)",
    )
    p_diff.set_defaults(func=cmd_diff)

    # color command
    p_color = subparsers.add_parser("color", help="Find color in image")
    p_color.add_argument("--image", "-i", required=True, help="Image to search")
    p_color.add_argument(
        "--color", "-c", required=True, help="Color to find (R,G,B)"
    )
    p_color.add_argument(
        "--tolerance",
        type=int,
        default=20,
        help="Color tolerance (default: 20)",
    )
    p_color.set_defaults(func=cmd_color)

    # ocr command
    p_ocr = subparsers.add_parser("ocr", help="Recognize text in image")
    p_ocr.add_argument("--image", "-i", required=True, help="Image to process")
    p_ocr.add_argument(
        "--lang",
        default="russian",
        help="Language (default: russian)",
    )
    p_ocr.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    p_ocr.set_defaults(func=cmd_ocr)

    # ui-detect command — detect UI elements in screenshot
    p_ui_detect = subparsers.add_parser(
        "ui-detect", help="Detect UI elements (GroundCUA-trained YOLO)",
    )
    p_ui_detect.add_argument("--image", "-i", required=True, help="Screenshot image")
    p_ui_detect.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    p_ui_detect.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class names (Button,Menu,Input_Elements,Navigation,Information_Display,Sidebar,Visual_Elements,Others,Unknown)",
    )
    p_ui_detect.add_argument(
        "--output", "-o",
        help="Save annotated image to path",
    )
    p_ui_detect.set_defaults(func=cmd_ui_detect)

    # vision command — VLM analysis of screenshot
    p_vision = subparsers.add_parser(
        "vision", help="Vision LLM analysis (SmolVLM2)",
    )
    p_vision.add_argument("--image", "-i", required=True, help="Screenshot image")
    p_vision.add_argument(
        "--action", "-a",
        choices=["describe", "design", "text", "compare", "question"],
        default="design",
        help="Analysis action (default: design)",
    )
    p_vision.add_argument(
        "--before", "-b",
        help="Before image (for compare action)",
    )
    p_vision.add_argument(
        "--question", "-q",
        help="Question to ask about the screenshot (for question action)",
    )
    p_vision.add_argument(
        "--prompt", "-p",
        help="Custom prompt (for describe action)",
    )
    p_vision.add_argument(
        "--model", "-m",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="HuggingFace model path (default: SmolVLM2-500M)",
    )
    p_vision.set_defaults(func=cmd_vision)

    # monitor command
    p_monitor = subparsers.add_parser("monitor", help="Real-time window monitoring")
    p_monitor.add_argument("--title", "-t", help="Window title to monitor")
    p_monitor.add_argument(
        "--fps", type=int, default=10, help="Capture FPS (default: 10)"
    )
    p_monitor.add_argument(
        "--trigger",
        choices=["change", "element", "color", "yolo"],
        help="Trigger type",
    )
    p_monitor.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Trigger threshold (default: 5.0)",
    )
    p_monitor.add_argument(
        "--template",
        help="Template path (for element trigger)",
    )
    p_monitor.add_argument(
        "--color",
        help="Color to detect (R,G,B) (for color trigger)",
    )
    p_monitor.add_argument(
        "--yolo-classes",
        help="UI class names (for yolo trigger, e.g., Button,Input_Elements)",
    )
    p_monitor.set_defaults(func=cmd_monitor)

    # analyze command — universal image analysis pipeline
    p_analyze = subparsers.add_parser(
        "analyze", help="Universal image analysis pipeline (outputs JSON)",
    )
    p_analyze.add_argument(
        "--image", "-i",
        required=True,
        help="Image source: file path, URL, or window title",
    )
    p_analyze.add_argument(
        "--stages",
        default=None,
        help='Comma-separated stages: metadata,ocr,ui,color,template,vision (default: auto)',
    )
    p_analyze.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path (default: analysis_<timestamp>.json)",
    )
    p_analyze.add_argument(
        "--templates",
        default=None,
        help='Comma-separated template files for template matching',
    )
    p_analyze.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold for UI detection",
    )
    p_analyze.add_argument(
        "--num-colors",
        type=int,
        default=5,
        help="Number of dominant colors (default: 5)",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)
