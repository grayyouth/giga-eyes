name: Bug Report
description: Report a bug
title: "bug: "
labels: ["bug"]

body:
  - type: textarea
    attributes:
      label: Description
      description: What happened?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Steps to Reproduce
      description: How can someone reproduce the bug?
      value: |
        1.
        2.
        3.
    validations:
      required: true

  - type: input
    attributes:
      label: Python Version
      description: e.g., 3.12.0
    validations:
      required: true

  - type: input
    attributes:
      label: OS
      description: e.g., Windows 11, Ubuntu 22.04
    validations:
      required: true

  - type: textarea
    attributes:
      label: Expected vs Actual
      description: What did you expect vs what happened?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Screenshots
      description: Attach screenshots if applicable
