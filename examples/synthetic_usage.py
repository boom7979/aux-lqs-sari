# SPDX-License-Identifier: Apache-2.0
"""Synthetic example of query selection, reference initialization, and spatial losses."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from aux_lqs_sari import QueryConstructionAdapter

torch.manual_seed(42)
model=QueryConstructionAdapter(hidden_dim=16,variant='full')
memory=torch.randn(1,20,16,requires_grad=True)
class_logits=torch.randn(1,20)
boxes=torch.rand(1,20,4)
shapes=torch.tensor([[4,4],[2,2]])
result=model(memory,class_logits,boxes,shapes,k=5)
targets=[{'boxes':torch.tensor([[.5,.5,.15,.20]])}]
losses=model.auxiliary_losses(result['spatial_outputs'],targets)
loss=sum(.1*value for value in losses.values())
loss.backward()
assert torch.isfinite(memory.grad).all()
print('Query indices:',tuple(result['query_indices'].shape))
print('Normalized reference boxes:',tuple(result['reference_boxes'].shape))
print('Spatial losses and backward: finite')
