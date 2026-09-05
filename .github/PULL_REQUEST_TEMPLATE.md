name: Pull Request
description: Propose changes
title: "pr: "
labels: []

body:
  - type: textarea
    attributes:
      label: Summary
      description: What changed and why?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Testing
      description: How was this tested?
      value: |
        - [ ] Tests pass locally
        - [ ] Added tests for new functionality
        - [ ] Manual testing completed
    validations:
      required: true

  - type: textarea
    attributes:
      label: Checklist
      value: |
        - [ ] Code follows style guidelines
        - [ ] Self-reviewed code
        - [ ] Comments added where needed
        - [ ] Documentation updated
        - [ ] CHANGELOG.md updated
    validations:
      required: true

  - type: textarea
    attributes:
      label: Related Issues
      description: Links to related issues
