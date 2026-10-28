from unittest.mock import patch
import unittest

from ppnl.core import Grid, execute
from ppnl.policy import greedy_plans, neighbors, search_plans, state_prompt
from scripts.evaluate_policy import repair_prefix


class Scores:
    def __init__(self, values):
        self.values = values

    def scores(self, observations):
        return [self.values for _ in observations]


class PolicyTests(unittest.TestCase):
    def test_state_observation_tracks_current_position(self):
        g = Grid(3, 3, ((1, 1),), (0, 0), (2, 2))
        text = state_prompt(g, (0, 1))
        self.assertIn("Current row 0 column 1", text)
        self.assertIn("Goal direction: below, right", text)
        self.assertEqual(set(neighbors(g, (0, 1))), {"left", "right"})

    def test_raw_policy_can_reach_goal(self):
        grid = Grid(2, 3, (), (0, 0), (0, 2))
        plans = greedy_plans([grid], Scores([0, 0, 0, 5]), masked=False)
        self.assertEqual(plans, ["right right"])

    def test_mask_does_not_claim_success_when_stuck(self):
        grid = Grid(3, 3, ((0, 1), (1, 0)), (0, 0), (2, 2))
        plan = greedy_plans([grid], Scores([5, 0, 0, 0]), masked=True)[0]
        self.assertEqual(plan, "")
        self.assertFalse(execute(grid, plan).valid)

    def test_search_routes_around_wall_without_reference_path(self):
        grid = Grid(3, 3, ((0, 1), (1, 1)), (0, 0), (0, 2))
        with patch("ppnl.core.shortest_path", side_effect=AssertionError("No reference-path oracle at inference")):
            plans, counts = search_plans([grid], Scores([0, 0, 0, 10]))
        self.assertTrue(execute(grid, plans[0]).valid)
        self.assertLessEqual(counts[0], 9)

    def test_symbolic_ablation_is_explicit(self):
        grid = Grid(3, 3, (), (0, 0), (2, 2))
        plans, _ = search_plans([grid], scorer=None, weight=0)
        self.assertTrue(execute(grid, plans[0]).valid)
        self.assertEqual(len(plans[0].split()), 4)

    def test_tiny_search_budget_does_not_use_hidden_fallback(self):
        grid = Grid(3, 3, (), (0, 0), (2, 2))
        plans, _ = search_plans([grid], max_expansions=1)
        self.assertEqual(plans, [""])

    def test_repair_keeps_only_executed_prefix(self):
        grid = Grid(2, 3, (), (0, 0), (0, 2))
        prefix, remaining = repair_prefix(grid, "right up right")
        self.assertEqual(prefix, ["right"])
        self.assertEqual(remaining.start, (0, 1))

    def test_repair_can_stop_when_goal_was_already_reached(self):
        grid = Grid(2, 3, (), (0, 0), (0, 2))
        prefix, remaining = repair_prefix(grid, "right right up")
        self.assertIsNone(remaining)
        self.assertTrue(execute(grid, " ".join(prefix)).valid)


if __name__ == "__main__":
    unittest.main()
