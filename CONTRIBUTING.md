# Contributing

Keep changes focused on the added modules. Preserve source attribution and licensing for dependencies, and keep private credentials and experimental assets out of commits.

Before proposing a change, run:

```bash
python -m unittest discover -s tests -v
python examples/synthetic_usage.py
```

Use focused regression tests for changed behavior, including mask polarity, selected-index alignment, target assignment, reference geometry, and gradients. Include the relevant test results with proposed changes.
