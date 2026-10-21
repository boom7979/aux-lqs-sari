# Reproduction settings and scope

## Add-on settings

The four files under `configs/` are deliberately small original add-on configurations, not copies of the upstream detector's full configuration. Load their `variant`, `hidden_dim`, and loss weights in the adapter integration. They cannot be passed unchanged to an untouched DQ-DETR training entry point.

## Detector-side reference recipe

| Setting | Value |
| --- | --- |
| Dataset split | AI-TOD-V2 trainval for training, test for evaluation |
| Final benchmark image counts | 14,018 trainval and 14,018 test, matching the original paper |
| Backbone | ResNet-50 |
| Epochs | 24 |
| Microbatch size and accumulation | 1 and 3 |
| Main and backbone learning rates | 1e-4 and 1e-5 |
| Scheduler milestones | 13 and 23 |
| Weight decay | 1e-4 |
| Mixed precision | AMP |
| Inherited query budgets | 300, 500, 900, 1500 |

The recipe describes implementation settings, not a certified match to a final-result checkpoint. Preserve and report actual preprocessing, evaluator bins/maxDets, seed/data-seed handling, initialization/pretraining provenance, and runtime versions for a concrete run. The paper-reported numerical summary is available in [RESULTS.md](../RESULTS.md); it is not a fresh evaluation from this repository. Source-version scope is recorded in [version compatibility](VERSION_COMPATIBILITY.md).

Use the AI-TOD-specific evaluator in the external detector environment. A previously inspected evaluator reference is jwwangchn/cocoapi-aitod commit `44a230ae5197cb89bf9e5e62f313cac3ad30c7af`, subdirectory `aitodpycocotools`. That dependency is not bundled and is not needed for standalone module tests.

## Limitations

This is an independent add-on release with module-level testing, not a full end-to-end detector release. Historical checkpoint key names, original optimizer state, full training RNG streams, and exact final AP are not guaranteed to match after external integration. Validate these separately if making a reproduction claim.

Weights and raw experimental records are not redistributed; only the paper-reported summary is supplied. The public-code scope does not remove the need to retain genuine underlying research records privately or to meet a journal's actual data/code policy. Do not describe unavailable checkpoints or logs as downloadable or available on request unless they can really be supplied.
