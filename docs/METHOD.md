# Added mechanisms

## Aux

A shared three-layer token-wise MLP feeds three branches that predict a center logit, normalized width/height, and sub-cell x/y offsets. Each branch uses a hidden linear layer and ReLU before its output layer. Size and offset outputs are sigmoid activated.

GT centers are assigned to the level where the object's maximum side is closest to two grid cells. Centers are binary target cells, not Gaussian maps. At collisions, the later target overwrites size and offset values. The center objective is binary focal loss with alpha 0.25 and gamma 2, averaged over all batch/token positions. Size and offset losses are mean L1 over assigned cells. Spatial loss weights are 0.1 each in the reference recipe. Empty positive sets yield zero size/offset loss.

## LQS

Let c be the maximum encoder class logit and l be the spatial center logit. The score is:

```text
score = 0.5 * sigmoid(c) + 0.5 * sigmoid(l)
```

Invalid padded positions are masked before deterministic top-K selection. The detector retains its existing query budget. There is no stochastic selector in this package. Top-K boundary ties may be ordered differently across runtimes; exact cross-device bitwise behavior is not guaranteed.

## SARI

Selected size predictions p map to a bounded prior:

```text
prior = 0.01 + 0.14 * p
wh_after = 0.5 * wh_before + 0.5 * prior
xy_after = xy_before
```

The input references are normalized encoder-regressed boxes. This operation changes dimensions, not centers. Its effect on initial IoU and final detection quality requires empirical evaluation; the formula does not guarantee an improvement.

## Inherited components

Backbone, encoder, dynamic counting/query budgets, feature enhancement, decoder, Hungarian matching, and deformable-attention operators belong to the separately obtained detector. Their source is not included here, and these mechanisms are not claimed as new contributions of this package.
