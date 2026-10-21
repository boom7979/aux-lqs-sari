# SPDX-License-Identifier: Apache-2.0
"""Detector-independent wiring helper; not a copy of the DQ-DETR transformer."""
import torch
from torch import nn
from .modules import LQSQueryGenerator,ScaleAwareRefInit,SpatialAuxLoss

class QueryConstructionAdapter(nn.Module):
    def __init__(self,hidden_dim=256,variant='full'):
        super().__init__()
        if variant not in {'baseline','aux_only','aux_lqs','full'}:
            raise ValueError('Unknown variant')
        self.variant=variant
        self.generator=LQSQueryGenerator(hidden_dim) if variant!='baseline' else None
        self.sari=ScaleAwareRefInit() if variant=='full' else None
        self.criterion=SpatialAuxLoss() if variant!='baseline' else None

    def predict(self,memory,spatial_shapes):
        if self.generator is None:return {}
        loc,scale,offset=self.generator.predict(memory)
        return {'loc_logits':loc,'scale_pred':scale,'offset_pred':offset,'spatial_shapes':spatial_shapes}

    def select(self,class_scores,spatial_outputs,k,padding_mask=None):
        if self.variant in {'aux_lqs','full'}:
            return self.generator.select_queries(class_scores,spatial_outputs['loc_logits'],k,padding_mask)
        scores=class_scores
        if padding_mask is not None:scores=scores.masked_fill(padding_mask,float('-inf'))
        k=min(int(k),scores.shape[1])
        if k<=0:raise ValueError('k must be positive')
        if padding_mask is not None and ((~padding_mask).sum(1)<k).any():
            raise ValueError('Query budget exceeds the valid candidate count')
        return torch.topk(scores,k,dim=1).indices,scores

    def initialize(self,references,spatial_outputs,indices):
        if self.sari is None:return references
        return self.sari(references,spatial_outputs['scale_pred'],indices)

    def auxiliary_losses(self,spatial_outputs,targets):
        if self.criterion is None:return {}
        return self.criterion(spatial_outputs['loc_logits'],spatial_outputs['scale_pred'],
                              spatial_outputs['offset_pred'],targets,spatial_outputs['spatial_shapes'])

    def forward(self,memory,class_scores,candidate_boxes,spatial_shapes,k,padding_mask=None):
        outputs=self.predict(memory,spatial_shapes)
        indices,scores=self.select(class_scores,outputs,k,padding_mask)
        # Preserve the reference detector's detached encoder-box input while
        # leaving the scale-prior path differentiable through SARI.
        before=torch.gather(candidate_boxes,1,indices.unsqueeze(-1).expand(-1,-1,4)).detach()
        after=self.initialize(before,outputs,indices)
        return {'query_indices':indices,'ranking_scores':scores,'references_before':before,
                'reference_boxes':after,'spatial_outputs':outputs}
