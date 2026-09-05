"""
Universal image analysis pipeline for Giga Eyes.

Accepts any image source (file, URL, window, screen, PIL Image, numpy array),
runs multiple analysis stages, and outputs structured JSON.

Usage:
    from giga_eyes.pipeline import ImagePipeline

    pipeline = ImagePipeline()
    result = pipeline.analyze("screenshot.png")
    result.save("output.json")

    # Or with custom stages
    result = pipeline.analyze(
        source="screenshot.png",
        stages=["metadata", "ocr", "ui", "color", "vision"],
        output="analysis.json",
    )
"""

from giga_eyes.pipeline.pipeline import ImagePipeline, AnalysisResult

__all__ = ["ImagePipeline", "AnalysisResult"]
