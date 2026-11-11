from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from ppnl.modeling import Seq2Seq


class Batch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class DecodingTests(unittest.TestCase):
    def test_repeated_path_actions_are_not_blocked_by_bart_defaults(self):
        runner = Seq2Seq.__new__(Seq2Seq)
        runner.torch = SimpleNamespace(inference_mode=nullcontext)
        runner.precision = "float32"
        runner.device = "cpu"
        runner.config = {"training": {"eval_batch_size": 8}}
        runner.input_limit, runner.output_limit = 512, 128
        runner.tokenizer = Mock(return_value=Batch(input_ids=SimpleNamespace(shape=(1, 16))))
        runner.tokenizer.batch_decode.return_value = ["right right right right"]
        runner.model = Mock()
        result = runner.predict(["path prompt"])
        self.assertEqual(result, ["right right right right"])
        options = runner.model.generate.call_args.kwargs
        self.assertEqual(options["no_repeat_ngram_size"], 0)
        self.assertEqual(options["repetition_penalty"], 1.0)
        self.assertEqual(options["num_beams"], 1)
        self.assertFalse(options["do_sample"])

    def test_tiny_t5_action_scores_match_full_vocabulary_logits(self):
        try:
            import torch
            from transformers import T5Config, T5ForConditionalGeneration
        except ImportError:
            self.skipTest("torch and transformers are needed for the tiny model check")

        class Tokens(dict):
            def to(self, device):
                return self

            def __getattr__(self, name):
                return self[name]

        class Tokenizer:
            eos_token_id = 1
            ids = {"up": 3, "down": 4, "left": 5, "right": 6}

            def encode(self, text, add_special_tokens=False):
                return [self.ids[word] for word in text.split()]

            def __call__(self, inputs, **kwargs):
                return Tokens(input_ids=torch.tensor([[7, 8]]), attention_mask=torch.tensor([[1, 1]]))

        torch.manual_seed(42)
        runner = Seq2Seq.__new__(Seq2Seq)
        runner.torch, runner.tokenizer = torch, Tokenizer()
        runner.model = T5ForConditionalGeneration(T5Config(
            vocab_size=16, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
            num_heads=2, decoder_start_token_id=0, eos_token_id=1, pad_token_id=0))
        runner.device, runner.precision, runner.input_limit = "cpu", "float32", 32
        scorer, usage = runner.action_scorer("repair prompt")
        measured = scorer([("right",), ("up",)])
        inputs = runner.tokenizer(["repair prompt"])
        with torch.inference_mode():
            expected = runner.model(**inputs, decoder_input_ids=torch.tensor([[0, 6]])).logits[0, -1]
            expected = torch.log_softmax(expected.float(), -1)
        self.assertAlmostEqual(measured[0]["right"], expected[6].item(), places=5)
        self.assertAlmostEqual(measured[0]["<eos>"], expected[1].item(), places=5)
        self.assertEqual(usage["decoder_input_tokens_processed"], 4)


if __name__ == "__main__":
    unittest.main()
