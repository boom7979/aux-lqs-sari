# Integrating with DQ-DETR

## Detector source

Obtain [DQ-DETR](https://github.com/hoiliu-0801/DQ-DETR) separately. The [interface reference](VERSION_COMPATIBILITY.md) identifies the source used for the hook map below. The hooks below describe how to connect the added module interfaces to the detector.

## DQ-DETR hook locations

The map below refers to [official commit c5246797b111d7e5da9e6c20667fe0901aae35ba](https://github.com/hoiliu-0801/DQ-DETR/tree/c5246797b111d7e5da9e6c20667fe0901aae35ba). The table identifies where each added operation connects to the detector.

| Location in the external checkout | Added operation |
| --- | --- |
| `models/dqdetr/deformable_transformer.py`, `DeformableTransformer.forward`, after CGFE at line 332 | Predict the spatial outputs once from enhanced memory and retain them |
| Same function, query budget at line 335 | Retain the inherited count-based K |
| Same function, category-only top-K at line 371 | Obtain aligned indices with `adapter.select` from encoder max class logits and spatial outputs |
| Same function, detached selected encoder references at line 375 | Apply `adapter.initialize` to sigmoid-normalized boxes, then convert back with the detector's inverse-sigmoid helper |
| Same function, content gather at line 379 | Gather query content with the same selected indices |
| Same function, decoder call at line 415 | Supply initialized references while keeping dense memory and decoder logic unchanged |
| `models/dqdetr/dqdetr.py`, `SetCriterion.forward` at line 472; loss-weight construction from line 765 | Expose spatial outputs to the criterion and add the three weighted spatial losses |

To check the source identity of an existing checkout without importing or modifying the detector:

```bash
python examples/check_dqdetr_interfaces.py /path/to/DQ-DETR
```

The command checks source identity and reports function locations. Connect the module interfaces using the steps below, then run the detector-side checks at the end of this guide.

## Tensor inputs

| Input | Shape and meaning |
| --- | --- |
| memory | `[B,N,C]`, flattened multi-scale features after the inherited feature-enhancement stage |
| category logits | `[B,N]`, maximum encoder classification logit over categories, before sigmoid |
| candidate boxes | `[B,N,4]`, normalized cxcywh after encoder box regression |
| spatial shapes | `[L,2]`, integer height and width per level, with total token count N |
| padding mask | `[B,N]`, Boolean, True means invalid/padding |
| query budget | positive integer K from the detector's existing counting/query-budget mechanism |
| targets | list of dictionaries with normalized cxcywh `boxes`, in the same transformed coordinates as the features |

Supply raw category logits, normalized candidate boxes, and a mask with True at padded positions. DQ-DETR supplies the count-based query budget.

## Query-budget and coordinate conventions

Selection caps the requested K at the total candidate count N. With a padding mask, it raises an error if fewer valid candidates remain than this capped budget. Both category-only and LQS selection follow this rule.

The auxiliary location loss averages over all flattened positions, including padding. Its inputs are separate from the selection padding mask. GT center coordinates, normalized dimensions, and feature-grid dimensions must use a consistent coordinate system. For heterogeneous padded batches, account for each image's valid extent when constructing targets. The paper uses microbatch size one.

## Hook 1 Create the added module

Register one `QueryConstructionAdapter(hidden_dim, variant)` on the transformer or detector before constructing the optimizer. Reuse this instance across forward calls and place it on the same device as the detector.

## Hook 2 Predict spatial outputs

Call `adapter.predict(memory, spatial_shapes)` once on the enhanced encoder features. Retain the returned dictionary for selection, initialization, and spatial losses. Selection, initialization, and spatial losses share this prediction head.

## Hook 3 Select indices

At the detector's encoder top-K selection point, call:

```python
query_indices, ranking_scores = adapter.select(
    category_logits, spatial_outputs, query_budget, padding_mask)
```

For the Aux-only variant this preserves category-only ranking. LQS/full use the additive fused scores. Preserve the upstream query-budget computation. Use the same indices to gather query embeddings, encoder references, and any aligned encoder outputs.

## Hook 4 Initialize decoder references

Gather the encoder-regressed normalized cxcywh boxes at the selected indices. Preserve the detector's stop-gradient boundary for encoder boxes. Pass these detached boxes, the retained spatial outputs, and the same indices into `adapter.initialize`. SARI leaves centers unchanged and updates only width and height.

Apply SARI to normalized encoder references. If the decoder expects logit-space references, apply the detector's inverse-sigmoid helper **after** SARI, preserving its epsilon/clamping behavior (epsilon 1e-3 in the referenced DQ-DETR source). Keep the gradient path through the predicted size prior.

The adapter's `forward` performs spatial prediction, selection, detached encoder-box gathering, and dimension initialization. It returns normalized references and the intermediate spatial outputs. Gather the decoder query embeddings with the returned indices and run the decoder in the detector.

## Hook 5 Add spatial supervision

Expose the retained spatial-output dictionary to the criterion. In training:

```python
spatial_losses = adapter.auxiliary_losses(spatial_outputs, targets)
weights = {"loss_loc": 0.1, "loss_scale": 0.1, "loss_offset": 0.1}
# Add each named loss to the detector's normal weighted loss construction.
```

Retain the existing detector matching, classification, box losses, denoising path, and decoder auxiliary losses. Add the spatial Aux terms alongside the existing decoder auxiliary supervision. Register their weights in the same loss-weight dictionary used by the training engine so they are actually optimized.

## Hook 6 Training and evaluation

Train/evaluate with the detector's own scripts after implementing these hooks. The method reference recipe uses AI-TOD-V2 trainval → test, 24 epochs, microbatch size 1, accumulation 3, and AMP. Apply these settings in the detector's training configuration and specify the initialization used for your run.

## Verify the integration

1. Run the standalone tests and synthetic example first.
2. Check a detector forward with each variant: baseline, Aux-only, Aux+LQS, full.
3. Confirm all selected indices are valid and shared across gathered outputs.
4. Verify center identity and the SARI width-height formula numerically.
5. Check that spatial losses have finite values and gradients under AMP.
6. When loading a checkpoint, match its parameter names and shapes to the registered modules.
