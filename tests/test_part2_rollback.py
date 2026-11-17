import unittest
from ppnl.core import Grid
from ppnl.core import execute
from ppnl.repair_variants import repair
from part2.run_rollback import reported_prefix


class OriginalRollbackTest(unittest.TestCase):
    def test_same_memoryless_policy_returns_to_same_cycle_for_every_cut(self):
        class Cycle:
            def scores(self, observations):
                return [
                    [0, 4, 0, 0] if position[0] == 0 else [4, 0, 0, 0]
                    for _, position in observations
                ]

        grid = Grid(3, 3, (), (0, 0), (2, 0))
        candidate = " ".join(["down", "up"] * 18)
        for k in range(37):
            prefix, _ = reported_prefix(
                grid, candidate, "invalid | step=37 | reason=goal_not_reached", k
            )
            feedback = f"invalid | step={len(prefix.split())+1} | reason=goal_not_reached"
            result, _, _ = repair(grid, prefix, feedback, Cycle(), {"mode": "raw"})
            self.assertFalse(execute(grid, result).valid)

    def test_valid_does_not_trigger_repair(self):
        g = Grid(3, 3, (), (0, 0), (2, 0))
        self.assertEqual(reported_prefix(g, "down", "valid", 2), (None, "accepted"))

    def test_no_hidden_true_step_correction(self):
        g = Grid(3, 3, (), (0, 0), (2, 0))
        feedback = "invalid | step=3 | reason=obstacle_collision"
        self.assertEqual(reported_prefix(g, "up down down", feedback, 0), (None, "illegal_prefix"))
        self.assertEqual(reported_prefix(g, "up down down", feedback, 2), ("", "ready"))
