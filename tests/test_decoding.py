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


if __name__ == "__main__":
    unittest.main()
