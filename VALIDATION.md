# Validation record

Local checks performed on 2026-10-18:

| Check | Result |
| --- | --- |
| Eight standalone synthetic unit tests | passed |
| Import boundary | PyTorch and Python standard library only |
| Spatial-head output parity with historical default implementation | exact in three synthetic fixtures |
| LQS selected-index parity | exact in the same fixtures |
| SARI output parity | exact in the same fixtures |
| Binary center, size, and offset target parity | exact in the same fixtures |
| Auxiliary loss parity | absolute tolerance 1e-7 and relative tolerance 1e-6 |
| Dataset, detector source, CUDA kernels, checkpoints, and raw logs/evaluation assets | excluded from this repository |
| Paper-reported result summary | supplied in RESULTS.md, not independently rerun |

The private reference comparison used identical copied head parameters and matching default settings, not separately trained models. Comparison fixtures used random seeds only to generate synthetic tensors; they are not repeated training experiments. The historical comparison harness and research records are not redistributed.

The standalone tests were executed on Windows with Python 3.9.25 and PyTorch 2.4.0+cu124. The GitHub Actions CPU workflow targets Python 3.11; a remote CI run has not been executed by this local packaging step. The module's guards reject a query budget larger than the number of valid candidates, rather than returning padded indices.

The refactored default formula is covered, not every experimental option formerly present in private development code. Empty positive sets return graph-connected zero size/offset losses; this preserves loss values while making empty-target backward safe. Historical whole-detector checkpoint loading, decoder-integration parity, and final detection AP have not been certified by these module tests.

The historical dirty-checkout identifier and exact reference-file hashes are recorded in `docs/VERSION_COMPATIBILITY.md`. They must not be described as a validated clean official-upstream revision.

## Pre-publication recheck on 2026-10-21

All eight standalone synthetic unit tests passed again. The synthetic usage example produced valid query/reference shapes and finite losses and backward gradients. These checks used the existing Windows PyTorch environment; they are not a fresh detector training or benchmark rerun.
