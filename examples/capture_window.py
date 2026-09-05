"""
Example: Capture full screen and window.

Usage:
    python examples/capture_window.py
"""

from giga_eyes.capture.screen import capture_full_screen
from giga_eyes.capture.window import capture_window_by_title
from giga_eyes.storage.screenshots import ScreenshotStorage
from giga_eyes.storage.logs import EventLogger


def main():
    # Setup storage and logging
    storage = ScreenshotStorage("screenshots", "png")
    logger = EventLogger("logs/giga_eyes.log")

    # 1. Capture full screen
    print("Capturing full screen...")
    full_img = capture_full_screen(output_path=None)

    if full_img:
        filepath = storage.save(full_img)
        print(f"  Saved: {filepath}")
        logger.log_capture(
            capture_type="screen",
            screenshot_path=str(filepath),
        )

    # 2. Capture specific window (change title to match your window)
    print("\nCapturing window 'Untitled' (Notepad example)...")
    window_img = capture_window_by_title(
        title="Untitled",  # Change to your window title
        output_path=None,
    )

    if window_img:
        filepath = storage.save(window_img, window_title="Untitled")
        print(f"  Saved: {filepath}")
        logger.log_capture(
            capture_type="window",
            window_title="Untitled",
            screenshot_path=str(filepath),
        )
    else:
        print("  Window not found. Try with a different title.")


if __name__ == "__main__":
    main()
