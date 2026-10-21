# Aux LQS SARI

Independent PyTorch modules for spatial auxiliary supervision, location-guided query selection, and scale-aware reference initialization.

Associated manuscript: **Spatially supervised query selection and reference box initialization for tiny object detection**. See [paper information](PAPER.md), [paper-reported results](RESULTS.md), and [reference version metadata](docs/VERSION_COMPATIBILITY.md).

This repository contains the project's add-on implementation, four add-on configurations, tests, and integration documentation. **It does not redistribute DQ-DETR, its counting/feature-enhancement modules, or its CUDA operator source.** Obtain the upstream detector separately and comply with its applicable terms. This package is not a standalone detection training framework.

## Method

- **Aux:** a shared MLP with center, normalized-size, and sub-cell-offset prediction branches; binary center focal loss and positive-cell size/offset L1 losses.
- **LQS:** deterministic top-K selection using equal-weight sigmoid category and center scores, under the inherited query budget.
- **SARI:** blend a bounded predicted size prior with the selected reference widths/heights, keeping centers unchanged.

## Installation

Python 3.9 or later and PyTorch are required. Reference validation used PyTorch 2.4.0. Install the appropriate CPU or CUDA wheel using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/), then install this package:

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python examples/synthetic_usage.py
```

For a fresh CPU reference environment:

```bash
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
```

The add-on itself requires no CUDA extension, torchvision, AI-TOD evaluator, dataset, or trained weights. The detector you connect it to has its own additional requirements.

## Use with a detector

```python
from aux_lqs_sari import QueryConstructionAdapter

adapter = QueryConstructionAdapter(hidden_dim=256, variant="full")
# Supply detector-produced tensors, not arbitrary values in a training run:
result = adapter(memory, category_logits, candidate_boxes_cxcywh,
                 spatial_shapes, k=query_budget, padding_mask=padding_mask)
```

The tensor interface and required decoder/criterion wiring are described in [the integration guide](docs/INTEGRATION.md). The four files in `configs/` configure this add-on only; they are **not complete DQ-DETR training configs**, and the untouched upstream entry point will not consume them automatically. No claim of one-command upstream integration or final-paper metric reproduction is made.

| Variant | Aux | LQS | SARI |
| --- | --- | --- | --- |
| baseline | off | off | off |
| aux_only | on | off | off |
| aux_lqs | on | on | off |
| full | on | on | on |

The baseline adapter leaves the detector's existing operations in place; it does not replace the baseline detector. Use [method settings](docs/METHOD.md) and [reproduction guidance](docs/REPRODUCIBILITY.md) when implementing the hooks.

## Availability and validation

The manuscript's four-group result summary and metric definitions are included in [RESULTS.md](RESULTS.md), explicitly as **paper-reported results, not a rerun of this code release**. No checkpoints, dataset images/annotations, raw training logs, per-image predictions, or original evaluation JSON are included. Local tests validate the add-on implementation; they are not detection benchmarks or evidence that a particular final AP has been reproduced. See [validation](VALIDATION.md).

## License and attribution

The project-added code and documentation in this repository are distributed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). This license does not grant rights to a separate DQ-DETR checkout or external datasets. It is not a blanket relicense of the upstream detector.

The inherited detector and its original mechanisms should be credited to [DQ-DETR](https://github.com/hoiliu-0801/DQ-DETR). See [attribution](docs/ATTRIBUTION.md). No paper acceptance, DOI, or unavailable artifact is claimed.
