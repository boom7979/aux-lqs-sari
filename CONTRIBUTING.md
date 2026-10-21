# Contributing

Keep this repository limited to the added mechanisms. Do not copy DQ-DETR source, inherited CUDA kernels, dataset images, or unlicensed third-party patches into it. Preserve attribution and licensing for any new dependency.

Run `python -m unittest discover -s tests -v` and `python examples/synthetic_usage.py` before submitting a change. Add focused tests for mask polarity, query-index alignment, target assignment, reference geometry, and gradients. Separate implementation changes from evaluation claims, and never add reconstructed research logs.
