# SPDX-License-Identifier: Apache-2.0
import unittest
import torch
from aux_lqs_sari import LQSQueryGenerator,ScaleAwareRefInit,SpatialAuxLoss,QueryConstructionAdapter

class ModuleTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7)

    def test_score_formula_and_padding(self):
        g=LQSQueryGenerator(8);a=torch.randn(2,9);b=torch.randn(2,9)
        mask=torch.zeros(2,9,dtype=torch.bool);mask[:,-1]=True
        actual=g.combine_scores(a,b,mask)
        torch.testing.assert_close(actual[:,:-1],(.5*a.sigmoid()+.5*b.sigmoid())[:,:-1])
        self.assertTrue(torch.isneginf(actual[:,-1]).all())

    def test_shapes_and_backward(self):
        g=LQSQueryGenerator(8);x=torch.randn(2,9,8,requires_grad=True)
        loc,scale,offset=g.predict(x)
        self.assertEqual(loc.shape,(2,9));self.assertEqual(scale.shape,(2,9,2))
        (loc.sum()+scale.sum()+offset.sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_topk_and_insufficient_valid_candidates(self):
        g=LQSQueryGenerator(8);a=torch.randn(1,4);b=torch.randn(1,4)
        idx,scores=g.select_queries(a,b,2)
        self.assertTrue(torch.equal(idx,scores.topk(2,dim=1).indices))
        with self.assertRaises(ValueError):g.select_queries(a,b,2,torch.tensor([[False,True,True,True]]))

    def test_sari_formula_and_center_identity(self):
        pre=torch.rand(2,3,4);scale=torch.rand(2,9,2);idx=torch.tensor([[0,1,2],[3,4,5]])
        out=ScaleAwareRefInit()(pre,scale,idx)
        prior=.01+.14*torch.gather(scale,1,idx[...,None].expand(-1,-1,2))
        torch.testing.assert_close(out[:,:,2:],.5*pre[:,:,2:]+.5*prior)
        self.assertTrue(torch.equal(out[:,:,:2],pre[:,:,:2]))

    def test_sari_disabled_is_identity(self):
        pre=torch.rand(1,2,4)
        self.assertIs(ScaleAwareRefInit(use_predicted_scale=False)(pre),pre)

    def test_binary_targets_collision_and_empty_targets(self):
        loss=SpatialAuxLoss();shapes=torch.tensor([[4,4],[2,2]])
        boxes=torch.tensor([[.25,.25,.30,.30],[.26,.26,.31,.31]])
        center,scale,offset,valid=loss.compute_location_targets([{'boxes':boxes}],shapes,'cpu')
        self.assertEqual(valid.sum().item(),1)
        torch.testing.assert_close(scale[valid],boxes[1,2:].unsqueeze(0))
        empty=loss(torch.randn(1,20,requires_grad=True),torch.rand(1,20,2),torch.rand(1,20,2),[{'boxes':torch.empty(0,4)}],shapes)
        self.assertEqual(empty['loss_scale'].item(),0)
        self.assertTrue(torch.isfinite(sum(empty.values())))

    def test_focal_half_precision_is_finite(self):
        pred=torch.tensor([[-100.,100.,0.]],dtype=torch.float16,requires_grad=True)
        value=SpatialAuxLoss().focal_loss(pred,torch.tensor([[1.,0.,1.]]))
        value.backward();self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(pred.grad).all())

    def test_adapter_variants(self):
        x=torch.rand(1,20,8);cat=torch.rand(1,20);boxes=torch.rand(1,20,4);shapes=torch.tensor([[4,4],[2,2]])
        for name in ['baseline','aux_only','aux_lqs','full']:
            model=QueryConstructionAdapter(8,name)
            result=model(x,cat,boxes,shapes,4)
            self.assertEqual(result['reference_boxes'].shape,(1,4,4))
            if name!='full':self.assertTrue(torch.equal(result['reference_boxes'],result['references_before']))

if __name__=='__main__':unittest.main()
