"""Offline, real T5/PEFT boundary tests; no pretrained download or GPU.

Run in the versions pinned by requirements.txt. These check implementation
contracts, not planning quality of the published checkpoints.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ppnl.modeling import Seq2Seq
from ppnl.optimization import accumulated_batches
from ppnl.task_adapters import GVRModel, attach_adapter, checkpoint_hash


class Seq2SeqIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import peft
            from transformers import T5Config, T5ForConditionalGeneration
        except ImportError as exc:
            raise unittest.SkipTest(f"Install requirements.txt for real-model checks: {exc}")
        cls.torch = torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.model_config = T5Config(vocab_size=16, d_model=16, d_ff=32, d_kv=8,
                                   num_layers=1, num_decoder_layers=1, num_heads=2,
                                   dropout_rate=0., decoder_start_token_id=0,
                                   eos_token_id=1, pad_token_id=0)
        cls.model_class = T5ForConditionalGeneration

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.threads)

    def tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import PreTrainedTokenizerFast
        vocab = {word: i for i, word in enumerate(
            ["<pad>", "</s>", "<unk>", "up", "down", "left", "right", "valid", "invalid"])}
        raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        raw.post_processor = processors.TemplateProcessing(single="$A </s>",
                                                           special_tokens=[("</s>", 1)])
        return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>",
                                      eos_token="</s>", unk_token="<unk>",
                                      model_input_names=["input_ids", "attention_mask"])

    def runner(self, model, tokenizer):
        from transformers import DataCollatorForSeq2Seq
        result = Seq2Seq.__new__(Seq2Seq)
        result.model, result.tokenizer, result.torch = model, tokenizer, self.torch
        result.device, result.precision, result.policy_loss = "cpu", "float32", None
        result.input_limit, result.output_limit = 128, 8
        result.config = {"training": {"eval_batch_size": 2}}
        result._collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
        return result

    def test_t5_label_shift_and_accumulated_gradients_match_full_batch(self):
        torch = self.torch
        torch.manual_seed(17)
        runner = self.runner(self.model_class(self.model_config), self.tokenizer())
        rows = [{"id": "a", "input": "up", "target": "right"},
                {"id": "b", "input": "left down", "target": "up right right down"},
                {"id": "c", "input": "right", "target": "down left"}]
        features = runner.encode_rows(rows)
        batch = runner.collate(features)
        self.assertEqual(features[0]["labels"], [6, 1])  # Original target + EOS.
        self.assertEqual(batch["labels"][0, :3].tolist(), [6, 1, -100])
        self.assertEqual(batch["decoder_input_ids"][0, :3].tolist(), [0, 6, 1])
        without_decoder_inputs = {k: v for k, v in batch.items() if k != "decoder_input_ids"}
        torch.testing.assert_close(runner.loss(batch), runner.model(**without_decoder_inputs).loss)
        runner.loss(batch).backward()
        full = {name: p.grad.clone() for name, p in runner.model.named_parameters()}
        runner.model.zero_grad(set_to_none=True)
        microbatches = [runner.collate(features[:1]), runner.collate(features[1:])]
        for _, microbatch, scale, _ in accumulated_batches(microbatches, 2):
            (runner.loss(microbatch) * scale).backward()
        for name, p in runner.model.named_parameters():
            torch.testing.assert_close(p.grad, full[name], atol=2e-6, rtol=2e-5)

    def test_real_named_adapters_freeze_base_and_survive_switching_and_reload(self):
        torch = self.torch
        torch.manual_seed(18)
        tokenizer = self.tokenizer()
        base = self.model_class(self.model_config).eval()
        inputs = tokenizer(["up right"], return_tensors="pt")
        labels = torch.tensor([[3, 6, 1]])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base.save_pretrained(root / "base")
            tokenizer.save_pretrained(root / "base")
            base_hash = checkpoint_hash(root / "base")
            paths, expected = {}, {}
            for task, target in (("verification", 7), ("repair", 4)):
                runner = self.runner(deepcopy(base), tokenizer)
                attach_adapter(runner, task, {"r": 2, "alpha": 4, "dropout": 0.})
                frozen = {n: p.detach().clone() for n, p in runner.model.named_parameters()
                          if not p.requires_grad}
                optimizer = torch.optim.AdamW([p for p in runner.model.parameters() if p.requires_grad], lr=.01)
                task_labels = labels.clone()
                task_labels[0, 0] = target
                runner.model(**inputs, labels=task_labels).loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in runner.model.named_parameters() if "lora_B" in n))
                optimizer.step()
                for name, p in runner.model.named_parameters():
                    if name in frozen:
                        torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
                runner.model.eval()
                expected[task] = runner.model(**inputs, labels=labels).logits.detach()
                runner.model.save_pretrained(root / task)
                paths[task] = root / task / task
                (paths[task] / "base_identity.json").write_text(json.dumps(
                    {"stage": task, "generation_base_sha256": base_hash}), encoding="utf-8")

            def cpu_runner(path, config):
                return self.runner(self.model_class.from_pretrained(path), tokenizer)

            with patch("ppnl.task_adapters.Seq2Seq", side_effect=cpu_runner):
                routed = GVRModel(root / "base", paths, {})
            for task in ("verification", "repair", "verification"):
                routed.runner.model.set_adapter(task)
                routed.runner.model.eval()
                actual = routed.runner.model(**inputs, labels=labels).logits
                torch.testing.assert_close(actual, expected[task])
            with routed.runner.model.disable_adapter():
                torch.testing.assert_close(routed.runner.model(**inputs, labels=labels).logits,
                                           base(**inputs, labels=labels).logits)
            self.assertFalse(torch.allclose(expected["verification"], expected["repair"]))
            from ppnl.core import Grid
            from ppnl.prompts import prompt
            grid = Grid(2, 3, (), (0, 0), (0, 2))
            # Exercise the actual GVR route after each other adapter was active.
            bare = self.runner(base, tokenizer)
            baseline = bare.predict([prompt("generation", grid)])[0]
            for task in ("verification", "repair"):
                routed.predict(task, grid, "up", "invalid")
                self.assertEqual(routed.predict("generation", grid), baseline)


if __name__ == "__main__":
    unittest.main()
