name: Feature Request
description: Suggest a feature
title: "feat: "
labels: ["enhancement"]

body:
  - type: textarea
    attributes:
      label: Problem
      description: What problem does this feature solve?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Proposed Solution
      description: How should it work?
    validations:
      required: true

  - type: textarea
    attributes:
      label: Use Cases
      description: Example use cases
    validations:
      required: true

  - type: textarea
    attributes:
      label: Alternatives
      description: Any alternative solutions?
