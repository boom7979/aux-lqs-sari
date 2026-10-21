# SPDX-License-Identifier: Apache-2.0
import unittest
import torch
from aux_lqs_sari import LQSQueryGenerator,ScaleAwareRefInit,SpatialAuxLoss,QueryConstructionAdapter,ScaleLocationHead

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

    def test_head_constructor_keyword_compatibility(self):
        model=ScaleLocationHead(hidden_dim=8,num_conv_layers=2)
        self.assertEqual(sum(isinstance(m,torch.nn.Linear) for m in model.shared),2)
        clone=ScaleLocationHead(8,2)
        clone.load_state_dict(model.state_dict(),strict=True)
        x=torch.randn(1,5,8)
        for actual,expected in zip(clone(x),model(x)):
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_query_budget_caps_before_padding_check(self):
        scores=torch.tensor([[1.,2.,3.,4.]])
        for variant in ['baseline','aux_only','aux_lqs','full']:
            adapter=QueryConstructionAdapter(8,variant)
            spatial={} if variant=='baseline' else {'loc_logits':torch.zeros_like(scores)}
            for mask in [None,torch.zeros_like(scores,dtype=torch.bool)]:
                indices,_=adapter.select(scores,spatial,8,mask)
                self.assertEqual(indices.shape,(1,4))
                self.assertEqual(indices.unique().numel(),4)
            with self.assertRaises(ValueError):
                adapter.select(scores,spatial,8,torch.tensor([[False,False,False,True]]))
            with self.assertRaises(ValueError):adapter.select(scores,spatial,0)

    def test_empty_targets_keep_zero_regression_gradients(self):
        loc=torch.randn(1,20,requires_grad=True)
        scale=torch.rand(1,20,2,requires_grad=True)
        offset=torch.rand(1,20,2,requires_grad=True)
        losses=SpatialAuxLoss()(loc,scale,offset,[{'boxes':torch.empty(0,4)}],torch.tensor([[4,4],[2,2]]))
        sum(losses.values()).backward()
        self.assertEqual(losses['loss_scale'].item(),0)
        self.assertEqual(losses['loss_offset'].item(),0)
        for value in [loc,scale,offset]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertEqual(scale.grad.count_nonzero().item(),0)
        self.assertEqual(offset.grad.count_nonzero().item(),0)

    def test_sari_selected_scale_gradient_and_encoder_detachment(self):
        priors=torch.rand(1,5,2,requires_grad=True)
        indices=torch.tensor([[1,3]])
        refs=torch.rand(1,2,4)
        ScaleAwareRefInit()(refs,priors,indices)[:,:,2:].sum().backward()
        expected=torch.zeros_like(priors)
        expected[:,[1,3],:]=0.07
        torch.testing.assert_close(priors.grad,expected)
        memory=torch.randn(1,20,8,requires_grad=True)
        candidate_boxes=torch.rand(1,20,4,requires_grad=True)
        model=QueryConstructionAdapter(8,'full')
        output=model(memory,torch.randn(1,20),candidate_boxes,torch.tensor([[4,4],[2,2]]),4)
        output['reference_boxes'][:,:,2:].sum().backward()
        self.assertIsNone(candidate_boxes.grad)
        self.assertIsNotNone(memory.grad)
        self.assertTrue(torch.isfinite(memory.grad).all())
        self.assertGreater(memory.grad.abs().sum().item(),0)

    def test_configs_load_adapter_and_loss_weights(self):
        from pathlib import Path
        import runpy
        for path in sorted((Path(__file__).resolve().parents[1]/'configs').glob('*.py')):
            config=runpy.run_path(str(path))
            model=QueryConstructionAdapter(config['hidden_dim'],config['variant'])
            self.assertEqual(model.variant,path.stem)
            expected={} if path.stem=='baseline' else {'loss_loc':.1,'loss_scale':.1,'loss_offset':.1}
            self.assertEqual(config['spatial_loss_weights'],expected)

    def test_adapter_variants(self):
        x=torch.rand(1,20,8);cat=torch.rand(1,20);boxes=torch.rand(1,20,4);shapes=torch.tensor([[4,4],[2,2]])
        for name in ['baseline','aux_only','aux_lqs','full']:
            model=QueryConstructionAdapter(8,name)
            result=model(x,cat,boxes,shapes,4)
            self.assertEqual(result['reference_boxes'].shape,(1,4,4))
            if name!='full':self.assertTrue(torch.equal(result['reference_boxes'],result['references_before']))

if __name__=='__main__':unittest.main()
