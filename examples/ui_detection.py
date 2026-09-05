"""
Example: UI element detection with YOLO.

Demonstrates detecting UI elements in screenshots.

Usage:
    python examples/ui_detection.py
"""

from giga_eyes.analyze.yolo import UIDetector


def main():
    detector = UIDetector()

    if not detector._initialized:
        print("⚠️  UIDetector not initialized (models not found)")
        print("   Install models or use ultralytics backend")
        return

    # Detect UI elements
    print("Detecting UI elements...")
    detections = detector.detect_classes(
        image=None,  # Replace with actual image
        class_names=["Button", "Input_Elements"],
    )

    for det in detections:
        print(f"  {det['name']}: {det['confidence']:.2f} @ {det['bbox']}")


if __name__ == "__main__":
    main()
