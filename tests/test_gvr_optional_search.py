"""CPU contracts for bounded backtracking and repair-feedback ablations."""
import unittest

from ppnl.core import Grid, execute
from ppnl.constrained_decoding import DecodeSpec, decode_actions
from ppnl.full_finetune_gvr import FullFinetuneGVRModel


class OptionalSearchTests(unittest.TestCase):
    def test_backtrack_recovers_discarded_sibling(self):
        grid = Grid(2, 3, ((1, 1),), (0, 0), (0, 2))

        def scores(prefixes):
            self.assertEqual(len({len(prefix) for prefix in prefixes}), 1)
            return [{"up": -1, "down": -0.01, "left": -1, "right": -0.1,
                     "<eos>": -0.1} for _ in prefixes]

        ordinary = decode_actions(grid, scores, DecodeSpec(beam_width=1, max_score_rows=20))
        recovered = decode_actions(grid, scores, DecodeSpec(beam_width=1, backtrack=True,
                                                            max_score_rows=20))
        self.assertFalse(ordinary.completed)
        self.assertTrue(execute(grid, recovered.plan).valid)
        self.assertLessEqual(recovered.decoder_row_steps, 20)

    def test_row_cap_counts_scored_prefixes(self):
        calls = []

        def scores(prefixes):
            calls.extend(prefixes)
            return [{"up": -1, "down": -1, "left": -1, "right": -1,
                     "<eos>": 0} for _ in prefixes]

        result = decode_actions(Grid(3, 3, (), (0, 0), (2, 2)), scores,
                                DecodeSpec(backtrack=True, max_score_rows=2))
        self.assertEqual(len(calls), 2)
        self.assertFalse(result.completed)
        self.assertEqual(result.stop_reason, "scorer_budget")

    def test_backtrack_needs_explicit_budget(self):
        with self.assertRaises(ValueError):
            DecodeSpec(backtrack=True)
        with self.assertRaises(ValueError):
            DecodeSpec(max_score_rows=True)

    def test_feedback_only_changes_repair_prompt(self):
        class Runner:
            tokenizer = staticmethod(lambda text: {"input_ids": text.split()})

            def predict(self, sources, check):
                self.source = sources[0]
                return ["right"]

        runner = Runner()
        grid = Grid(1, 2, (), (0, 0), (0, 1))
        baseline = FullFinetuneGVRModel(runner)
        sources = {}
        for task in ("generation", "verification"):
            baseline.predict(task, grid, "left")
            sources[task] = runner.source

        expected = {"original": "private-diagnostic", "none": "",
                    "generic": "The candidate is invalid. Find a valid path."}
        for kind, feedback in expected.items():
            with self.subTest(feedback=kind):
                model = FullFinetuneGVRModel(runner, repair_feedback=kind)
                for task in ("generation", "verification"):
                    model.predict(task, grid, "left")
                    self.assertEqual(runner.source, sources[task])
                model.predict("repair", grid, "left", "private-diagnostic")
                self.assertIn("Candidate: left", runner.source)
                self.assertEqual(model.last_usage["repair_feedback_used"], feedback)
                if kind != "original":
                    self.assertNotIn("private-diagnostic", runner.source)
                if feedback:
                    self.assertIn(feedback, runner.source)
