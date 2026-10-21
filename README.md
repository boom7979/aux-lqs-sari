# Aux LQS SARI

Independent PyTorch implementations of spatial auxiliary supervision (Aux), location-guided query selection (LQS), and scale-aware reference initialization (SARI), accompanying **Spatially supervised query selection and reference box initialization for tiny object detection**.

The modules extend [DQ-DETR](https://github.com/hoiliu-0801/DQ-DETR) at its query-selection and reference-initialization interfaces. This repository provides the added modules, four module configurations, integration guidance, synthetic tests, and paper-reported result summaries. The detector framework, datasets, and trained weights are obtained separately.

## Method overview

- **Aux** predicts center logits, normalized dimensions, and within-cell offsets using a shared MLP. Training adds binary center focal loss and positive-cell dimension/offset L1 losses.
- **LQS** ranks candidates using `0.5 * sigmoid(category_logit) + 0.5 * sigmoid(center_logit)` and selects top-K valid positions under the inherited query budget.
- **SARI** gathers the dimension prediction at the selected indices and blends `0.01 + 0.14 * prediction` with the original reference dimensions at equal weight, preserving centers.

See [method details](docs/METHOD.md) and the [architecture](figures/architecture.svg) and [module](figures/module_details.svg) diagrams.

## Repository structure

| Path | Contents |
| --- | --- |
| `aux_lqs_sari/` | Prediction head, query ranking, reference sizing, auxiliary losses, and adapter |
| `configs/` | Module variants and spatial loss weights |
| `docs/` | Method, detector interfaces, dataset resources, and training settings |
| `examples/synthetic_usage.py` | Executable tensor-interface example |
| `tests/test_modules.py` | Synthetic module regression tests |
| `RESULTS.md` | Paper-reported detection results |

## Installation and quick start

Use Python 3.9 or later and PyTorch. The reference module environment uses PyTorch 2.4.0; choose a CPU or CUDA installation from the [official instructions](https://pytorch.org/get-started/previous-versions/).

```bash
# CPU reference installation
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python examples/synthetic_usage.py
```

## Module integration

```python
from aux_lqs_sari import QueryConstructionAdapter

adapter = QueryConstructionAdapter(hidden_dim=256, variant="full")
result = adapter(memory, category_logits, candidate_boxes_cxcywh,
                 spatial_shapes, k=query_budget, padding_mask=padding_mask)
```

Here `memory`, `category_logits`, normalized `candidate_boxes_cxcywh`, `spatial_shapes`, `query_budget`, and `padding_mask` are tensors or values supplied by the detector. The [integration guide](docs/INTEGRATION.md) specifies their shapes, coordinate conventions, hooks, and gradient boundaries. Register the adapter before constructing the optimizer. Use the selected indices consistently for query content and references.

The adapter returns selected indices, ranking scores, references, and spatial predictions. The external detector supplies the backbone, counting, feature enhancement, decoder, matching, and training engine. The [interface reference](docs/VERSION_COMPATIBILITY.md) identifies the official source used for the documented hook locations.

## Configuration

| Variant | Configuration | Aux | LQS | SARI |
| --- | --- | --- | --- | --- |
| baseline | `configs/baseline.py` | off | off | off |
| aux_only | `configs/aux_only.py` | on | off | off |
| aux_lqs | `configs/aux_lqs.py` | on | on | off |
| full | `configs/full.py` | on | on | on |

Load `variant` and `hidden_dim` when constructing the adapter, and add `spatial_loss_weights` to the detector's weighted loss construction. Configure the detector training engine separately using the settings in the integration guide. See [training settings](docs/REPRODUCIBILITY.md).

## Validation and experimental results

Module tests and the synthetic example check formulas, target assignment, index selection, geometry, and gradients. See [test coverage and results](VALIDATION.md) and the detector-side checks in the [integration guide](docs/INTEGRATION.md).

The [result tables](RESULTS.md) present the manuscript's four-group AP and AR comparison. The same reported values are available in [paper_results.json](paper_results.json).

## Dataset availability

Use the [official AI-TOD-V2 project page](https://chasel-tsui.github.io/AI-TOD-v2/) for dataset resources and revised annotations. The paper uses trainval for training and test for evaluation. Dataset and evaluator details are listed in [reproduction settings](docs/REPRODUCIBILITY.md).

## Citation

Software citation metadata are provided in [CITATION.cff](CITATION.cff); the associated manuscript and authors are described in [PAPER.md](PAPER.md). Please also cite the external DQ-DETR foundation.

## License and attribution

The added code and documentation are distributed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). External detector code, pretrained weights, and datasets retain their providers' terms. [Attribution](docs/ATTRIBUTION.md) identifies the original detector and the added contributions.
