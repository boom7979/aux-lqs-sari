# Associated manuscript

**Spatially supervised query selection and reference box initialization for tiny object detection**

Chinese title: **空间监督引导的微小目标查询选择与参考框初始化**

Authors: Minhua Yi and Yongping Gao (corresponding author), School of Artificial Intelligence and Information Engineering, East China University of Technology.

## Method to implementation mapping

| Manuscript mechanism | Public implementation | Module configuration |
| --- | --- | --- |
| Spatial auxiliary supervision Aux | `ScaleLocationHead` and `SpatialAuxLoss` | `configs/aux_only.py` and higher variants |
| Location-guided query selection LQS | `LQSQueryGenerator` | `configs/aux_lqs.py` and `configs/full.py` |
| Scale-aware reference initialization SARI | `ScaleAwareRefInit` | `configs/full.py` |
| Wiring helper | `QueryConstructionAdapter` | All four variants |

The method uses DQ-DETR's counting, feature enhancement, query budgets, backbone, encoder/decoder, matching, and operators. Attribution is provided in [ATTRIBUTION.md](docs/ATTRIBUTION.md).

## Results and implementation scope

[RESULTS.md](RESULTS.md) records the paper's four-group comparison. [Validation](VALIDATION.md) describes module tests, and the [integration guide](docs/INTEGRATION.md) explains how to connect the modules to DQ-DETR.
