from pathlib import Path
import tempfile
import unittest

from ppnl.constrained_decoding import DecodeSpec
from ppnl.core import Grid
from ppnl.full_finetune_gvr import FullFinetuneGVRModel
from ppnl.gvr_harness import create_gvr_session, run_gvr_session, score_gvr_session


class FakeRunner:
    @staticmethod
    def tokenizer(text):
        return {"input_ids": text.split()}

    def predict(self, sources, check=lambda: None):
        return ["up" for _ in sources]

    def action_scorer(self, source, check=lambda: None):
        assert "Feedback: invalid | step=1 | reason=out_of_bounds" in source
        usage = {"encoder_input_tokens": 20, "decoder_input_tokens_processed": 0}

        def score(prefixes):
            usage["decoder_input_tokens_processed"] += sum(len(p) + 1 for p in prefixes)
            return [{"up": -0.01, "down": -3, "left": -0.02,
                     "right": -0.1, "<eos>": -0.01} for _ in prefixes]

        return score, usage


class GVRDecodingIntegrationTests(unittest.TestCase):
    def test_executor_repair_records_cost_and_survives_reload(self):
        grid = Grid(2, 3, (), (0, 0), (0, 2))
        identity = {"model": "fixed", "repair_decoding": {"beam_width": 4}}
        model = FullFinetuneGVRModel(FakeRunner(), DecodeSpec(max_actions=4))
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"
            create_gvr_session(session, grid, "executor_one", identity)
            state = run_gvr_session(session, model, identity)
            self.assertEqual(state["candidate"], "right right")
            self.assertEqual(state["decode_status"], "complete")
            self.assertGreater(state["forward_batches"], 0)
            self.assertGreater(state["decoder_input_tokens_processed"], 0)
            self.assertTrue(score_gvr_session(session)["success"])
            self.assertEqual(run_gvr_session(session, model, identity)["forward_batches"],
                             state["forward_batches"])

    def test_self_mode_refuses_experimental_repair(self):
        grid = Grid(2, 3, (), (0, 0), (0, 2))
        identity = {"model": "fixed"}
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session"
            create_gvr_session(session, grid, "self", identity)
            with self.assertRaisesRegex(ValueError, "Self-verification"):
                run_gvr_session(session, FullFinetuneGVRModel(FakeRunner(), DecodeSpec()), identity)


if __name__ == "__main__":
    unittest.main()
