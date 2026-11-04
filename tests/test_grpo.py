import unittest
from types import SimpleNamespace
from ppnl.rlvr import group_advantages, grpo_objective, completion_mask


class GRPOTests(unittest.TestCase):
    def test_group_normalization_and_constant_group(self):
        import torch
        self.assertTrue(torch.equal(group_advantages([2, 2]), torch.zeros(2)))
        torch.testing.assert_close(group_advantages([0, 2]), torch.tensor([-1., 1.]))

    def test_clipping_blocks_overlarge_update_for_both_advantage_signs(self):
        import torch
        # Positive reward with ratio > 1.2, negative reward with ratio < .8.
        logp = torch.tensor([[-1.], [-3.]], requires_grad=True)
        old = torch.full_like(logp, -2.)
        loss = grpo_objective(logp, old, old, torch.ones_like(logp), torch.tensor([1., -1.]), 0.)
        loss.backward()
        torch.testing.assert_close(logp.grad, torch.zeros_like(logp))

    def test_unclipped_gradient_and_padding_mask(self):
        import torch
        logp = torch.full((2, 2), -2., requires_grad=True)
        old = logp.detach().clone()
        loss = grpo_objective(logp, old, old, torch.tensor([[1, 0], [1, 0]]), torch.tensor([1., -1.]), .02)
        loss.backward()
        self.assertLess(logp.grad[0, 0], 0)
        self.assertGreater(logp.grad[1, 0], 0)
        self.assertEqual(logp.grad[:, 1].abs().sum(), 0)

    def test_eos_included_padding_and_trailing_tokens_excluded(self):
        import torch
        labels = torch.tensor([[3, 2, 1, 1], [4, 5, 2, 7]])
        self.assertEqual(completion_mask(labels, 2, 1).tolist(), [[True, True, False, False], [True, True, True, False]])

    def test_real_bart_lora_matches_projection_names_and_freezes_base(self):
        import torch
        from transformers import BartConfig, BartForConditionalGeneration
        from ppnl.task_adapters import attach_adapter
        config = BartConfig(vocab_size=16, d_model=16, encoder_layers=1, decoder_layers=1,
                            encoder_attention_heads=2, decoder_attention_heads=2,
                            encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32)
        runner = SimpleNamespace(model=BartForConditionalGeneration(config), _collator=SimpleNamespace())
        info = attach_adapter(runner, 'repair', {'r': 2, 'alpha': 4, 'dropout': 0.})
        self.assertGreater(info['matched_modules'], 0)
        self.assertTrue(all('lora_' in n for n,p in runner.model.named_parameters() if p.requires_grad))
        loss = runner.model(input_ids=torch.tensor([[3, 4, 2]]), labels=torch.tensor([[5, 6, 2]])).loss
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in runner.model.named_parameters() if 'lora_B' in n))

    def test_bart_policy_action_labels_skip_bos_but_verifier_keeps_it(self):
        from ppnl.modeling import Seq2Seq
        class Tokenizer:
            bos_token_id = 0
            def __call__(self, inputs=None, text_target=None, **kwargs):
                return {'input_ids': [[0,3,2] for _ in (inputs or text_target)]}
        runner=Seq2Seq.__new__(Seq2Seq)
        runner.tokenizer=Tokenizer()
        runner.input_limit=runner.output_limit=16
        runner.policy_loss='optimal_actions'
        runner.model=SimpleNamespace(config=SimpleNamespace(model_type='bart'))
        rows=[{'id':'p','input':'map','target':'right','optimal_actions':['right']},
              {'id':'v','input':'verify','target':'valid'}]
        encoded=runner.encode_rows(rows)
        self.assertEqual(encoded[0]['labels'],[3,2])
        self.assertEqual(encoded[1]['labels'],[0,3,2])

    def test_bart_status_auxiliary_loss_targets_verdict_after_bos(self):
        import torch
        import torch.nn.functional as F
        from ppnl.gvr_objectives import training_loss
        logits=torch.tensor([[[4.,0.,0.],[0.,0.,4.]]],requires_grad=True)
        labels=torch.tensor([[0,1]])
        class Model:
            config=SimpleNamespace(model_type='bart')
            def __call__(self,**kwargs):return SimpleNamespace(logits=logits)
        runner=SimpleNamespace(model=Model())
        actual=training_loss(runner,{'labels':labels},{'loss':'sequence_and_status','status_weight':1.})
        ce=F.cross_entropy(logits.transpose(1,2),labels,reduction='none')
        torch.testing.assert_close(actual,ce.mean()+ce[:,1].mean())
