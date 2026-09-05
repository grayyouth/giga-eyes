"""
Example: Run universal image analysis pipeline.

Shows how to analyze any image source and get structured JSON output.

Usage:
    python examples/pipeline_example.py

Or with CLI:
    python -m giga_eyes.cli analyze --image screenshot.png --output result.json
"""

from giga_eyes.pipeline import ImagePipeline


def main():
    pipeline = ImagePipeline()

    # ─── Example 1: Analyze a file ──────────────────────────────────────
    print("=" * 60)
    print("Example 1: Analyze image file")
    print("=" * 60)

    result = pipeline.analyze(
        source="test_tinyvika_mss.png",  # Replace with your image
        stages=["metadata", "color"],  # Fast stages only (no LLM needed)
    )

    print(result.to_json())

    # ─── Example 2: Full analysis (requires models) ─────────────────────
    print("\n" + "=" * 60)
    print("Example 2: Full analysis with all stages")
    print("=" * 60)

    result = pipeline.analyze(
        source="test_tinyvika_mss.png",
        stages=["metadata", "ocr", "ui", "color", "vision"],
        output="analysis_full.json",
    )

    print(f"Full analysis saved to: analysis_full.json")

    # ─── Example 3: Specific stages ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Example 3: UI + Color only")
    print("=" * 60)

    result = pipeline.analyze(
        source="test_tinyvika_mss.png",
        stages=["metadata", "ui", "color"],
    )

    # Access specific stage results
    if "ui" in result.stages:
        print(f"UI elements found: {result.stages['ui']['total_elements']}")
    if "color" in result.stages:
        print(f"Dominant colors: {len(result.stages['color']['dominant_colors'])}")

    # ─── Example 4: Using window title as source ────────────────────────
    print("\n" + "=" * 60)
    print("Example 4: Capture and analyze window")
    print("=" * 60)

    result = pipeline.analyze(
        source="TinyVika",  # Window title
        stages=["metadata", "color"],
    )

    print(result.to_json())

    # ─── Example 5: Template matching ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Example 5: Template matching")
    print("=" * 60)

    result = pipeline.analyze(
        source="test_tinyvika_mss.png",
        stages=["metadata", "template"],
        templates={
            "button": "templates/button.png",
            "logo": "templates/logo.png",
        },
    )

    if "template" in result.stages:
        for name, match in result.stages["template"]["templates"].items():
            status = "FOUND" if match["found"] else "not found"
            print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
