"""
Example: Capture screen region.

Usage:
    python examples/capture_region.py
"""

from giga_eyes.capture.region import capture_region
from giga_eyes.storage.screenshots import ScreenshotStorage


def main():
    storage = ScreenshotStorage("screenshots", "png")

    # Capture top-left 800x600 region
    print("Capturing region (0, 0, 800, 600)...")
    img = capture_region(
        x=0,
        y=0,
        width=800,
        height=600,
        output_path=None,
    )

    if img:
        filepath = storage.save(img)
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    main()
