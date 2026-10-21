# Attribution and distribution boundary

This repository contains project-added spatial supervision, score fusion, reference-size initialization, integration helpers, and tests. DQ-DETR source, CUDA kernels, datasets, pretrained weights, and inherited counting/feature-enhancement code are not redistributed.

DQ-DETR is the intended external detector foundation:

Huang, Yi-Xin; Liu, Hou-I; Shuai, Hong-Han; Cheng, Wen-Huang. DQ-DETR: DETR with Dynamic Query for Tiny Object Detection. ECCV 2024 proceedings. https://arxiv.org/abs/2404.03507

Official source: https://github.com/hoiliu-0801/DQ-DETR

This separation avoids distributing an uncleared combined source tree. It does not waive the terms applicable to separately downloaded dependencies, nor does it constitute legal clearance for downstream redistribution of a combined detector. Do not copy upstream functions or patches into this repository without checking their permissions.

The dependency's counting, feature enhancement, dynamic query budgets, matching framework, and operators should be credited to their original authors.

The code implements commonly used focal and L1 objectives using PyTorch. These objectives are not claimed as newly invented losses. Method figures in `figures/` are project-created diagrams, not images of experimental results. No new manuscript DOI, author list, or publication status is asserted here.
