# Contributing to Giga Eyes 👁️

Thank you for your interest in contributing! This document provides guidelines and instructions.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/giga-eyes.git
   cd giga-eyes
   ```
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```

## 📋 Development Guidelines

### Code Style
- **Python:** Follow PEP 8, use type hints
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes
- **Docstrings:** Google style for functions, module-level docstrings
- **Imports:** Standard library → third-party → local (grouped)

### Example
```python
"""Module description."""

from typing import Optional

import numpy as np
from PIL import Image


def analyze_image(image: Image.Image, threshold: float = 0.5) -> dict:
    """Analyze image and return results.

    Args:
        image: PIL Image to analyze.
        threshold: Confidence threshold.

    Returns:
        Dictionary with analysis results.
    """
    return {"status": "ok"}
```

### Testing
- Write tests for new features
- Run tests before committing:
  ```bash
  pytest tests/ -v
  ```
- Aim for >90% coverage on new code

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add universal image pipeline
fix: correct RGB decoding in mss fallback
docs: update README with pipeline examples
test: add pipeline integration tests
```

## 🧪 Adding Tests

1. Create test file in `tests/`
2. Use pytest fixtures for setup
3. Mock external dependencies (models, hardware)
4. Test both success and error cases

```python
def test_new_feature():
    """Test new feature description."""
    result = new_function(input_data)
    assert result["key"] == expected_value
```

## 📝 Pull Request Process

1. Update README.md with changes if needed
2. Update CHANGELOG.md with [Unreleased] changes
3. Ensure all tests pass: `pytest tests/ -v`
4. PR title should be descriptive
5. Link related issues

### PR Template
```markdown
## Summary
- What changed
- Why it matters

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Comments added where needed
- [ ] Documentation updated
```

## 🐛 Reporting Bugs

Use GitHub Issues with label `bug`. Include:
- Python version
- OS
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if visual)

## 💡 Feature Requests

Use GitHub Issues with label `enhancement`. Include:
- Problem description
- Proposed solution
- Use cases
- Alternatives considered

## 📚 Resources

- [README.md](README.md) — Full documentation
- [pyproject.toml](pyproject.toml) — Dependencies and config
- `examples/` — Usage examples
- `tests/` — Test patterns to follow

## 💬 Questions?

Open a GitHub Issue with label `question`.

Thank you for contributing! 🎉
