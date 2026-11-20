import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

from ppnl.core import Grid
from ppnl.modeling import action_set_loss
from . import g_probes
from .five_class import CLASS_LABELS, HEAD, VERIFIER_LOSS, FiveClassRunner, T5WithStepClassifier
from .schema import State


def tiny_runner():
    torch.manual_seed(42)
    runner = FiveClassRunner.__new__(FiveClassRunner)
    cfg = T5Config(vocab_size=16, d_model=16, d_ff=24, d_kv=8, num_layers=1, num_heads=2,
                   dropout_rate=0.0, pad_token_id=0, decoder_start_token_id=0, eos_token_id=1)
    runner.model = T5WithStepClassifier(cfg).eval()
    tokens = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "up": 2, "down": 3, "left": 4, "right": 5},
                                 unk_token="[UNK]"))
    tokens.pre_tokenizer = Whitespace()
    runner.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokens, unk_token="[UNK]", pad_token="[PAD]")
    runner.action_ids = [2, 3, 4, 5]
    runner.config = {"training": {"eval_batch_size": 4}}
    runner.policy_loss, runner.device, runner.torch = "optimal_actions", "cpu", torch
    runner.input_limit, runner.precision = 256, "float32"
    return runner


def verification_rows():
    return [dict(id=label, mode="v_step", input="down right", target=label) for label in CLASS_LABELS]


class FiveClassTest(unittest.TestCase):
    def test_each_v_row_uses_one_class_and_skips_the_decoder(self):
        runner = tiny_runner()
        self.assertEqual(CLASS_LABELS, ("valid", "goal_not_reached", "obstacle_collision",
                                        "out_of_bounds", "invalid_action_token"))
        batch = runner.collate(runner.encode_rows(verification_rows()))
        self.assertEqual(batch["labels"].tolist(), list(range(5)))
        batch["labels"][-1] = 2
        with torch.no_grad():
            runner.model.verifier_head.weight.zero_()
            runner.model.verifier_head.bias.copy_(torch.tensor([1., -1., .5, 0., .3]))
        with patch.object(runner.model.decoder, "forward", side_effect=AssertionError("V entered decoder")):
            loss = runner.batch_loss(batch)
            torch.testing.assert_close(g_probes.batch_loss(runner, batch), loss)
            torch.testing.assert_close(runner.loss(batch), loss)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            runner.loss(batch, split_verification_loss=True)
        expected = torch.nn.functional.cross_entropy(runner.model.verifier_head.bias.expand(5, -1), batch["labels"])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        prob = runner.model.verifier_head.bias.detach().softmax(-1)
        torch.testing.assert_close(runner.model.verifier_head.bias.grad, prob - torch.tensor([.2, .2, .4, .2, 0.]))

    def test_public_preparation_keeps_five_class_targets_and_metadata(self):
        runner = tiny_runner()
        with patch("step_data_v3.model_io.read_split", return_value=verification_rows()) as reader:
            features = g_probes.prepare_training(runner, "frozen")
        reader.assert_called_once_with("frozen", "train", version=g_probes.VERSION, fingerprint=g_probes.template_hash())
        self.assertEqual([f["labels"] for f in features], list(range(5)))
        self.assertEqual(runner.model.config.ppnl_verifier_labels, list(CLASS_LABELS))
        self.assertEqual(runner.model.config.ppnl_verifier_loss, VERIFIER_LOSS)

    def test_mixed_batch_keeps_original_g_objective_and_equal_example_weight(self):
        runner = tiny_runner()
        gs = [dict(id="g", mode="g_next", input="up left", target="left", optimal_actions=["left", "down"])]
        vs = verification_rows()
        g_batch = runner.collate(runner.encode_rows(gs))
        logits = runner.model(input_ids=g_batch.input_ids, attention_mask=g_batch.attention_mask,
                              decoder_input_ids=torch.zeros(1, 1, dtype=torch.long)).logits
        old_g = action_set_loss(logits, torch.tensor([[4]]), runner.action_ids, g_batch.policy_actions, "optimal_actions")
        torch.testing.assert_close(runner.batch_loss(g_batch), old_g)
        v_loss = runner.batch_loss(runner.collate(runner.encode_rows(vs)))
        mixed = runner.batch_loss(runner.collate(runner.encode_rows(gs + vs)))
        torch.testing.assert_close(mixed, (old_g + 5 * v_loss) / 6)
        mixed.backward()
        for parameter in (runner.model.verifier_head.weight, runner.model.encoder.block[0].layer[0].SelfAttention.q.weight,
                          runner.model.decoder.block[0].layer[0].SelfAttention.q.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(runner.model.verifier_head.weight.grad.abs().sum().item(), 0)

    def test_padding_and_checkpoint_roundtrip_through_the_path_verifier(self):
        runner = tiny_runner()
        with tempfile.TemporaryDirectory() as root:
            base, saved = Path(root) / "base", Path(root) / "new"
            T5ForConditionalGeneration(runner.model.config).save_pretrained(base)
            runner.model = runner.load_model(base, local_files_only=True).eval()
            self.assertTrue(torch.isfinite(runner.model.verifier_head.weight).all())
            cfg = runner.model.config
            cfg.ppnl_verifier_head, cfg.ppnl_verifier_loss = HEAD, VERIFIER_LOSS
            cfg.ppnl_verifier_labels, cfg.ppnl_prompt_sha256 = list(CLASS_LABELS), g_probes.template_hash()
            with torch.no_grad():
                runner.model.verifier_head.bias[2] = 100
            hidden = torch.randn(1, 3, cfg.d_model)
            reference = runner.model.classify(hidden[:, :2], torch.tensor([[1, 1]])).detach()
            torch.testing.assert_close(runner.model.classify(hidden, torch.tensor([[1, 1, 0]])), reference)
            runner.model.save_pretrained(saved)
            runner.model = runner.load_model(saved, local_files_only=True).eval()
            torch.testing.assert_close(runner.model.classify(hidden, torch.tensor([[1, 1, 0]])), reference)
            grid = Grid(3, 3, ((0, 1),), (0, 0), (1, 2))
            self.assertEqual(g_probes.verify_last(runner, State.from_grid(grid), "right"), "obstacle_collision")
            self.assertEqual(g_probes.verify_plans(runner, [grid], ["right"]),
                             ["invalid | step=1 | reason=obstacle_collision"])
            runner.model.config.ppnl_verifier_labels = list(reversed(CLASS_LABELS))
            runner.model.config.save_pretrained(saved)
            with self.assertRaisesRegex(ValueError, "label order"):
                runner.load_model(saved, local_files_only=True)
