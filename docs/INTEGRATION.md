# Integrating with a separately obtained detector

## Responsibility boundary

Download DQ-DETR from its official repository separately. This repository supplies neither an upstream checkout nor an upstream-source patch. The following interface instructions and examples describe the added logic; they do not reproduce complete upstream functions. Verify the chosen upstream revision and comply with its terms. If its internal interface differs, adapt the hook locations and test them rather than assuming line numbers match.

## Tensors

| Input | Shape and meaning |
| --- | --- |
| memory | `[B,N,C]`, flattened multi-scale features after the inherited feature-enhancement stage |
| category logits | `[B,N]`, maximum encoder classification logit over categories, before sigmoid |
| candidate boxes | `[B,N,4]`, normalized cxcywh after encoder box regression |
| spatial shapes | `[L,2]`, integer height and width per level, with total token count N |
| padding mask | `[B,N]`, Boolean, True means invalid/padding |
| query budget | positive integer K from the detector's existing counting/query-budget mechanism |
| targets | list of dictionaries with normalized cxcywh `boxes`, in the same transformed coordinates as the features |

Do not pass class probabilities into an interface expecting logits, raw proposal logits as normalized boxes, or a mask with inverted polarity. This release does not change how DQ-DETR predicts the query budget.

## Hook 1 Create the added module

Create `QueryConstructionAdapter(hidden_dim, variant)` as a registered `nn.Module` on the transformer/detector, not a fresh local variable inside every forward. Construct it before creating the optimizer so its trainable parameters are included. Put it on the same device as the detector.

## Hook 2 Predict spatial outputs

Call `adapter.predict(memory, spatial_shapes)` once on the enhanced encoder features. Retain the returned dictionary for selection, initialization, and spatial losses. The prediction head is shared across these uses; do not instantiate a separate randomly initialized head for each use.

## Hook 3 Select indices

At the detector's encoder top-K selection point, call:

```python
query_indices, ranking_scores = adapter.select(
    category_logits, spatial_outputs, query_budget, padding_mask)
```

For the Aux-only variant this preserves category-only ranking. LQS/full use the additive fused scores. Preserve the upstream query-budget computation. Use the same indices to gather query embeddings, encoder references, and any aligned encoder outputs. Updating references but leaving query embeddings at different indices is incorrect.

## Hook 4 Initialize decoder references

Gather the encoder-regressed normalized cxcywh boxes at the selected indices. Preserve the detector's stop-gradient boundary for encoder boxes. Pass these detached boxes, the retained spatial outputs, and the same indices into `adapter.initialize`. SARI leaves centers unchanged and updates only width and height.

If the decoder expects unactivated references, apply the detector's own inverse-sigmoid helper **after** SARI. Preserve its epsilon/clamping behavior; the historical helper uses epsilon 1e-3. Do not apply SARI to raw logit-space boxes or to final decoder predictions. Do not detach the resulting SARI output again if doing so would unintentionally remove the intended size-prior gradient path.

The adapter's convenience `forward` performs spatial prediction, selection, detached encoder-box gathering, and dimension initialization. It returns normalized references, not decoder predictions. It does not gather decoder query embeddings or run the decoder; those remain detector-side work.

## Hook 5 Add spatial supervision

Expose the retained spatial-output dictionary to the criterion. In training:

```python
spatial_losses = adapter.auxiliary_losses(spatial_outputs, targets)
weights = {"loss_loc": 0.1, "loss_scale": 0.1, "loss_offset": 0.1}
# Add each named loss to the detector's normal weighted loss construction.
```

Retain the existing detector matching, classification, box losses, denoising path, and decoder auxiliary losses. The spatial Aux terms are additional terms, not a replacement for inherited decoder auxiliary supervision. Register their weights in the same loss-weight dictionary used by the training engine so they are actually optimized. Do not use test-set labels during inference or optimization.

## Hook 6 Training and evaluation

Train/evaluate with the detector's own scripts after implementing these hooks. The method reference recipe uses AI-TOD-V2 trainval → test, 24 epochs, microbatch size 1, accumulation 3, and AMP. These settings require detector-side configuration and are not provided by the add-on's variant dictionaries. Obtain any pretraining artifacts separately and record their exact source and load behavior.

## Verify the integration

1. Run the standalone tests and synthetic example first.
2. Check a detector forward with each variant: baseline, Aux-only, Aux+LQS, full.
3. Confirm all selected indices are valid and shared across gathered outputs.
4. Verify center identity and the SARI width-height formula numerically.
5. Check that spatial losses have finite values and gradients under AMP.
6. Compare a frozen, identical input/state against the intended reference implementation before making checkpoint or metric-parity claims.

This release does not include an automatic installer that edits upstream detector files, nor a validated integration against every upstream revision. Parameter names under an adapter may differ from historical transformer attributes, so old full-detector checkpoints are not promised to load without a separately validated key mapping.
