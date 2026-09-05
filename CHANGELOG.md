# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Universal Image Analysis Pipeline (`giga_eyes.pipeline`)
  - `ImagePipeline` class with 6 analysis stages
  - `AnalysisResult` with JSON serialization
  - CLI command `giga-eyes analyze`
  - Support for any image source (file, URL, window, PIL, numpy)
- Pipeline documentation and examples
- Pipeline tests (28 tests)

### Changed
- Updated `capture/screen.py` — explicit `return None` when saving
- Updated `capture/window.py` — fixed mss fallback RGB decoding
- Updated `cli.py` — reorganized `cmd_monitor` before `main()`

### Fixed
- `test_yolo.py` — completely rewritten for UIDetector API
- `test_monitor.py` — fixed `YOLOTrigger` parameter name
- `capture/screen.py` — missing return statement
- `capture/window.py` — incorrect mss RGB handling

## [0.2.0] - 2026-09-05

### Added
- Universal Image Analysis Pipeline
- 6 analysis stages: metadata, ocr, ui, color, template, vision
- JSON schema for pipeline output
- Example: `pipeline_example.py`

### Changed
- Project structure reorganized
- Updated README with full documentation

## [0.1.0] - 2026-09-04

### Added
- Screen capture (full, window, region)
- Image analysis (template, diff, color, OCR, YOLO)
- Real-time monitoring with triggers
- CLI interface
- Test suite (58 tests)
