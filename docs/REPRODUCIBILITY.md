# Dataset and training settings

## Dataset and evaluator

The [official AI-TOD-V2 project page](https://chasel-tsui.github.io/AI-TOD-v2/) provides dataset resources and revised annotations. Follow the dataset providers' access and usage terms. Use trainval for training and test for evaluation.

The benchmark description contains 28,036 images and 752,745 annotated instances: 11,214 train, 2,804 validation, and 14,018 test images. Train and validation combine into 14,018 trainval images. These figures follow the [DQ-DETR dataset description](https://arxiv.org/html/2404.03507v3#S4.SS1).

Use the AI-TOD evaluator's area bins and maxDets=1500 rather than ordinary COCO size bins. The evaluator source reference is [jwwangchn/cocoapi-aitod at 44a230ae5197cb89bf9e5e62f313cac3ad30c7af](https://github.com/jwwangchn/cocoapi-aitod/tree/44a230ae5197cb89bf9e5e62f313cac3ad30c7af), subdirectory `aitodpycocotools`.

## Module configuration

The files in `configs/` define `variant`, `hidden_dim`, and `spatial_loss_weights`. Pass the first two values to `QueryConstructionAdapter` and register the weights in the detector's loss-weight dictionary. The integration guide describes the required detector-side hooks.

## Training settings

| Setting | Value |
| --- | --- |
| Dataset split | AI-TOD-V2 trainval to test |
| Backbone | ResNet-50 |
| Feature levels / hidden dimension | 5 / 256 |
| Encoder / decoder layers | 6 / 6 |
| Training epochs / endpoint | 24 / epoch 23 with zero-based indexing |
| Microbatch / gradient accumulation | 1 / 3 |
| Optimizer | AdamW |
| Main / backbone learning rates | 1e-4 / 1e-5 |
| Scheduler milestones | 13 and 23 |
| Weight decay / gradient clipping | 1e-4 / 0.1 |
| Training resize | Short-side discrete targets from 480 to 800 pixels; long side capped at 1333 |
| Mixed precision | AMP |
| Inherited query budgets | 300, 500, 900, 1500 |
| Spatial loss weights | 0.1 each for location, dimension, offset |
| LQS category / center weights | 0.5 / 0.5 |
| SARI bounds / blend | [0.01, 0.15] / 0.5 |

This table summarizes the manuscript's settings. When adapting the modules to a detector environment, specify its initialization, preprocessing, annotation version, evaluator settings, random states, and detector commit alongside the module configuration.
