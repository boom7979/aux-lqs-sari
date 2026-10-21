# Associated manuscript

## Title

**Spatially supervised query selection and reference box initialization for tiny object detection**

Chinese title: **空间监督引导的微小目标查询选择与参考框初始化**

This repository is associated with that manuscript's method description. The English title matches the manuscript prepared for JARS on 2026-10-21. No publication acceptance, journal assignment, DOI, or public manuscript URL is asserted by this metadata.

## Method to implementation mapping

| Manuscript mechanism | Public implementation | Add-on configuration |
| --- | --- | --- |
| Spatial auxiliary supervision Aux | `ScaleLocationHead` and `SpatialAuxLoss` | `configs/aux_only.py` and higher variants |
| Location-Guided Query Selection LQS | `LQSQueryGenerator` | `configs/aux_lqs.py` and `configs/full.py` |
| Scale-Aware Reference Initialization SARI | `ScaleAwareRefInit` | `configs/full.py` |
| Wiring helper for the added interfaces | `QueryConstructionAdapter` | supports all four variants |

DQ-DETR's inherited counting, feature enhancement, query budgets, backbone, encoder/decoder, matching, and CUDA operators remain external dependencies and are not claimed as new modules.

## Results and reproduction

The [result tables](RESULTS.md) contain paper-reported precision and recall for four configurations, not a fresh evaluation performed from this repository. The [validation record](VALIDATION.md) describes what was actually tested. The [integration guide](docs/INTEGRATION.md) describes the hooks still needed in a separately obtained detector.

## Citation metadata

Authors: Minhua Yi and Yongping Gao (corresponding author). Both authors are affiliated with the School of Artificial Intelligence and Information Engineering, East China University of Technology. The manuscript has no assigned publication DOI. `CITATION.cff` describes this software release, not an accepted journal article.

Please also credit the external DQ-DETR foundation as described in [attribution](docs/ATTRIBUTION.md).
