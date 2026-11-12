import math
import unittest

from ppnl.constrained_decoding import DecodeSpec, decode_actions
from ppnl.core import Grid, execute, shortest_path


class ConstrainedDecodingTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(2, 3, (), (0, 0), (0, 2))

    @staticmethod
    def scores(prefixes):
        return [{"up": -0.01, "down": -3.0, "left": -0.02,
                 "right": -0.1, "<eos>": -0.01} for _ in prefixes]

    def test_constraints_rescue_path_that_grammar_beam_ends_early(self):
        bare = decode_actions(self.grid, self.scores, DecodeSpec(beam_width=4, max_actions=4,
                                                                state_constraints=False))
        constrained = decode_actions(self.grid, self.scores, DecodeSpec(beam_width=4, max_actions=4))
        self.assertTrue(bare.completed)  # EOS alone is not a task-success signal.
        self.assertFalse(execute(self.grid, bare.plan).valid)
        self.assertTrue(constrained.completed)
        self.assertTrue(execute(self.grid, constrained.plan).valid)
        self.assertEqual(constrained.plan, "right right")
        self.assertGreater(constrained.pruned_illegal, 0)
        self.assertGreater(constrained.pruned_revisit, 0)
        self.assertGreater(constrained.forward_batches, 0)

    def test_state_constraints_preserve_model_ranking_not_shortest_path(self):
        detour = ("down", "right", "right", "up")

        def scores(prefixes):
            rows = []
            for prefix in prefixes:
                row = dict.fromkeys(("up", "down", "left", "right", "<eos>"), -10.0)
                if prefix == detour[:len(prefix)]:
                    row[detour[len(prefix)] if len(prefix) < len(detour) else "<eos>"] = -0.01
                rows.append(row)
            return rows

        result = decode_actions(self.grid, scores, DecodeSpec(max_actions=4))
        self.assertTrue(result.completed)
        self.assertTrue(execute(self.grid, result.plan).valid)
        self.assertEqual(result.plan, " ".join(detour))
        self.assertGreater(result.action_count, len(shortest_path(self.grid)))

    def test_straight_repeated_actions_are_allowed(self):
        grid = Grid(1, 4, (), (0, 0), (0, 3))
        result = decode_actions(grid, self.scores, DecodeSpec(max_actions=3))
        self.assertEqual(result.plan, "right right right")
        self.assertTrue(result.completed)

    def test_dead_branches_do_not_crash_or_fake_success(self):
        grid = Grid(2, 2, ((0, 1), (1, 0)), (0, 0), (1, 1))
        result = decode_actions(grid, self.scores, DecodeSpec())
        self.assertEqual(result.stop_reason, "dead_end")
        self.assertFalse(result.completed)
        self.assertFalse(execute(grid, result.plan).valid)

    def test_budget_and_nonfinite_scores(self):
        result = decode_actions(self.grid, self.scores, DecodeSpec(max_actions=1))
        self.assertEqual(result.stop_reason, "action_budget")
        self.assertFalse(result.completed)
        self.assertEqual(result.action_count, 1)
        with self.assertRaisesRegex(ValueError, "numeric"):
            decode_actions(self.grid, lambda p: [{"right": math.nan} for _ in p])

    def test_equal_scores_follow_declared_action_order(self):
        grid = Grid(2, 3, (), (1, 1), (0, 2))
        def uniform(prefixes):
            return [{"up": -1, "down": -1, "left": -1, "right": -1,
                     "<eos>": 0} for _ in prefixes]
        result = decode_actions(grid, uniform, DecodeSpec(max_actions=3))
        self.assertEqual(result.plan, "up right")


if __name__ == "__main__":
    unittest.main()
