# Module tests

## Run the tests

```bash
python -m unittest discover -s tests -v
python examples/synthetic_usage.py
```

## Coverage

| Component | Checks |
| --- | --- |
| Spatial head | Output shapes, constructor compatibility, and backward gradients |
| LQS | Score formula, padding-mask polarity, top-K indices, and query-budget boundaries |
| SARI | Dimension formula, unchanged centers, disabled identity, and selected-scale gradients |
| Auxiliary losses | Binary center targets, collisions, empty-target backward, and half-precision focal stability |
| Adapter and configurations | Four variants, configuration loading, and detached encoder references |

The synthetic example returns query indices of shape `(1, 5)` and reference boxes of shape `(1, 5, 4)`, and checks finite spatial losses and backward gradients.

## Test results

| Suite | Environment | Result |
| --- | --- | --- |
| Current module suite and synthetic example | Windows, Python 3.9.25, PyTorch 2.4.0+cu124; CPU tensors | 13 tests passed; example passed on 2026-10-21 |
| Public CI at commit `54e57c5` | Ubuntu 22.04, Python 3.11, CPU PyTorch 2.4.0 | 8 tests passed; [workflow run](https://github.com/boom7979/aux-lqs-sari/actions/runs/39540892424) |

The [workflow definition](.github/workflows/tests.yml) runs the module suite and synthetic example. Workflow results are associated with the commit shown in each run.

These tests cover standalone module behavior. The [integration guide](docs/INTEGRATION.md) gives detector-side checks for index alignment, geometry, loss weighting, and gradients. Experimental detection results are presented in [RESULTS.md](RESULTS.md).
