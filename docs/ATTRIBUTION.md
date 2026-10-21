# Attribution

## Added implementation

This repository provides the project's spatial supervision head and losses, LQS score fusion, SARI reference sizing, adapter, module configurations, tests, and documentation. Focal and L1 losses are established objectives implemented using PyTorch. The figures illustrate the architecture and module formulas.

## External detector

The detector foundation is **DQ-DETR: DETR with Dynamic Query for Tiny Object Detection**, by Yi-Xin Huang, Hou-I Liu, Hong-Han Shuai, and Wen-Huang Cheng, ECCV 2024 proceedings.

- [Paper](https://arxiv.org/abs/2404.03507)
- [Official implementation](https://github.com/hoiliu-0801/DQ-DETR)

Counting, feature enhancement, dynamic query budgets, backbone, encoder/decoder, matching, and deformable-attention operators belong to the external detector and its credited predecessors. Obtain that code separately under its applicable terms. The project's Apache-2.0 license covers the added distribution; it does not relicense external detector code, pretrained assets, or datasets.

See [NOTICE](../NOTICE), [LICENSE](../LICENSE), and [detector interface reference](VERSION_COMPATIBILITY.md).
