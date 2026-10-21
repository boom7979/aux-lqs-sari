# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aux LQS SARI contributors
"""Project-added spatial supervision, query ranking, and reference sizing.

This module depends only on PyTorch, not on a DQ-DETR source checkout.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ScaleLocationHead(nn.Module):
    """Predict center logits, normalized sizes, and offsets from [B,N,C]."""
    def __init__(self, hidden_dim=256, num_conv_layers=3):
        super().__init__()
        layers=[]
        for _ in range(num_conv_layers):
            layers.extend([nn.Linear(hidden_dim,hidden_dim),nn.ReLU(inplace=True)])
        self.shared=nn.Sequential(*layers)
        self.loc_head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(inplace=True),nn.Linear(hidden_dim,1))
        self.scale_head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(inplace=True),nn.Linear(hidden_dim,2))
        self.offset_head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(inplace=True),nn.Linear(hidden_dim,2))

    def forward(self,memory):
        features=self.shared(memory)
        return (self.loc_head(features).squeeze(-1),
                self.scale_head(features).sigmoid(),self.offset_head(features).sigmoid())


class LQSQueryGenerator(nn.Module):
    """Equal-weight category/center ranking; True in a mask means padding."""
    def __init__(self,hidden_dim=256,alpha=0.5,beta=0.5):
        super().__init__()
        self.sl_head=ScaleLocationHead(hidden_dim)
        self.alpha=alpha
        self.beta=beta

    def predict(self,memory):return self.sl_head(memory)

    def combine_scores(self,enc_scores,loc_logits,mask_flatten=None):
        scores=self.alpha*enc_scores.sigmoid()+self.beta*loc_logits.sigmoid()
        if mask_flatten is not None:scores=scores.masked_fill(mask_flatten,float('-inf'))
        return scores

    def select_queries(self,enc_scores,loc_logits,num_select,mask_flatten=None):
        scores=self.combine_scores(enc_scores,loc_logits,mask_flatten)
        k=min(int(num_select),enc_scores.shape[1])
        if k<=0:raise ValueError('num_select must be positive')
        if mask_flatten is not None and ((~mask_flatten).sum(1)<k).any():
            raise ValueError('Query budget exceeds the valid candidate count')
        return torch.topk(scores,k,dim=1).indices,scores

    def forward(self,memory,enc_scores,mask_flatten=None):
        loc,scale,offset=self.predict(memory)
        return self.combine_scores(enc_scores,loc,mask_flatten),loc,scale,offset


class ScaleAwareRefInit(nn.Module):
    """Blend selected size priors with normalized cxcywh; preserve centers."""
    def __init__(self,s_min=0.01,s_max=0.15,use_predicted_scale=True,blend=0.5):
        super().__init__()
        if not 0<=s_min<s_max<=1:raise ValueError('Invalid size bounds')
        if not 0<=blend<=1:raise ValueError('blend must be in [0,1]')
        self.s_min=s_min
        self.s_max=s_max
        self.use_predicted_scale=use_predicted_scale
        self.blend=blend

    def forward(self,ref_boxes,scale_pred=None,topk_indices=None):
        if not self.use_predicted_scale or scale_pred is None:return ref_boxes
        if topk_indices is None:raise ValueError('topk_indices are required')
        scales=torch.gather(scale_pred,1,topk_indices.unsqueeze(-1).expand(-1,-1,2))
        prior=self.s_min+scales*(self.s_max-self.s_min)
        result=ref_boxes.clone()
        result[:,:,2:]=(1-self.blend)*ref_boxes[:,:,2:]+self.blend*prior
        return result


class SpatialAuxLoss(nn.Module):
    """Binary center focal loss and positive-cell size/offset L1 losses.

    Targets are binary center cells, not Gaussian heatmaps. A target is assigned
    to the level where its maximum side spans closest to two cells. Later GT
    boxes overwrite size/offset targets at a colliding cell, preserving the
    reference implementation's default assignment rule.
    """
    def __init__(self,focal_alpha=0.25,focal_gamma=2.0):
        super().__init__()
        self.focal_alpha=focal_alpha
        self.focal_gamma=focal_gamma

    def compute_location_targets(self,targets,spatial_shapes,device):
        shapes=spatial_shapes.tolist()
        count=sum(int(h)*int(w) for h,w in shapes)
        center=torch.zeros(len(targets),count,device=device)
        scale=torch.zeros(len(targets),count,2,device=device)
        offset=torch.zeros_like(scale)
        valid=torch.zeros(len(targets),count,device=device,dtype=torch.bool)
        starts=[]
        start=0
        for h,w in shapes:starts.append(start);start+=h*w
        for batch,target in enumerate(targets):
            for box in target['boxes']:
                cx,cy,bw,bh=box
                level=min(range(len(shapes)),key=lambda j:abs(math.log2(max(float(bw)*shapes[j][1],float(bh)*shapes[j][0],1e-6))-1))
                h,w=shapes[level]
                fx,fy=float(cx)*w,float(cy)*h
                ix,iy=min(w-1,max(0,int(fx))),min(h-1,max(0,int(fy)))
                index=starts[level]+iy*w+ix
                center[batch,index]=1
                scale[batch,index,0]=bw;scale[batch,index,1]=bh
                offset[batch,index,0]=fx-ix;offset[batch,index,1]=fy-iy
                valid[batch,index]=True
        return center,scale,offset,valid

    def focal_loss(self,pred,target):
        logits=pred.float() if pred.dtype in (torch.float16,torch.bfloat16) else pred
        target=target.to(logits.dtype)
        probability=logits.sigmoid()
        positive=self.focal_alpha*target*(1-probability)**self.focal_gamma*F.softplus(-logits)
        negative=(1-self.focal_alpha)*(1-target)*probability**self.focal_gamma*F.softplus(logits)
        return (positive+negative).mean()

    def forward(self,loc_logits,scale_pred,offset_pred,targets,spatial_shapes):
        center,scale,offset,valid=self.compute_location_targets(targets,spatial_shapes,loc_logits.device)
        loss_loc=self.focal_loss(loc_logits,center)
        if valid.any():
            loss_scale=F.l1_loss(scale_pred[valid],scale[valid],reduction='mean')
            loss_offset=F.l1_loss(offset_pred[valid],offset[valid],reduction='mean')
        else:
            loss_scale=scale_pred.sum()*0
            loss_offset=offset_pred.sum()*0
        return {'loss_loc':loss_loc,'loss_scale':loss_scale,'loss_offset':loss_offset}


# Compatibility names for the project's historical symbols, not an upstream API.
SLSQueryGenerator=LQSQueryGenerator
SLSAuxLoss=SpatialAuxLoss
