# Paper reported results

Associated manuscript: **Spatially supervised query selection and reference box initialization for tiny object detection**. See [paper information](PAPER.md).

The tables present the detection metrics reported in the associated manuscript. Machine-readable values are provided in [paper_results.json](paper_results.json).

## Detection precision

All metric values are percentages, displayed to the precision used in the paper.

| Method | Module configuration | AP | AP50 | AP75 | APvt | APtiny | APsmall | APmedium |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DQ-DETR baseline | `configs/baseline.py` | 29.2328 | 64.9573 | 22.5241 | 15.6701 | 29.5894 | 36.4798 | 46.6243 |
| Baseline + Aux | `configs/aux_only.py` | 29.3082 | 65.1185 | 22.5549 | 15.6827 | 29.8803 | 36.6378 | 46.7102 |
| Baseline + Aux + LQS | `configs/aux_lqs.py` | 29.5262 | 65.2482 | 22.5798 | 15.8281 | 29.8842 | 36.6779 | 46.7119 |
| Baseline + Aux + LQS + SARI | `configs/full.py` | 29.6190 | 65.3393 | 22.9121 | 15.8357 | 29.8421 | 36.9166 | 46.7248 |

## Detection recall

| Method | AR1500 | ARvt |
| --- | ---: | ---: |
| DQ-DETR baseline | 45.2475 | 25.2169 |
| Baseline + Aux | 45.5163 | 25.3125 |
| Baseline + Aux + LQS | 45.8548 | 28.8881 |
| Baseline + Aux + LQS + SARI | 46.0208 | 28.9171 |

## Metric definitions

- **AP:** detection average precision, averaged over IoU thresholds 0.50 to 0.95 in increments of 0.05, across the dataset categories and the evaluator's all-area range, with maxDets=1500.
- **AP50 / AP75:** corresponding detection AP at IoU 0.50 / 0.75.
- **APvt / APtiny / APsmall / APmedium:** AP over the evaluator's verytiny, tiny, small, and medium area ranges. The nominal area ranges in pixel squared are [0,64], [64,256], [256,1024], and [1024,10000000000]. Boundary handling follows the AI-TOD evaluator implementation.
- **AR1500:** detection average recall with evaluator maxDets=1500, averaged over the same IoU thresholds and the all-area range.
- **ARvt:** corresponding verytiny-area detection average recall with maxDets=1500.

Differences between metric values are expressed in **percentage points**.

## Reported comparison protocol

The manuscript reports AI-TOD-V2 **trainval → test**, ResNet-50, and a shared **24-epoch** training protocol for this comparison system. The stated reference recipe uses microbatch size 1, accumulation 3, AMP, main/backbone learning rates 1e-4/1e-5, MultiStepLR milestones 13 and 23, and weight decay 1e-4. See [reproduction guidance](docs/REPRODUCIBILITY.md).

The comparison uses **14,018 trainval images for training and 14,018 test images for evaluation**, following the [AI-TOD-V2 protocol described by DQ-DETR](https://arxiv.org/html/2404.03507v3#S4.SS1). Dataset resources and training settings are listed in [reproduction settings](docs/REPRODUCIBILITY.md).

The four configurations map to the methods shown above. The paper compares Aux+LQS and the full method with SARI as the added component. The reproduced baseline is **29.2328 AP**; the original DQ-DETR publication reports a separate external result.

## Differences calculated from the table

| Comparison | AP change | AR1500 change | ARvt change |
| --- | ---: | ---: | ---: |
| Aux minus baseline | +0.0754 | +0.2688 | +0.0956 |
| Aux + LQS minus Aux | +0.2180 | +0.3385 | +3.5756 |
| Full minus Aux + LQS | +0.0928 | +0.1660 | +0.0290 |
| Full minus baseline | +0.3862 | +0.7733 | +3.7002 |

Adding SARI increases AP by **0.0928 percentage points** and decreases APtiny by **0.0421 percentage points**. The tables report single-run results.
