import unittest
from ppnl.core import Grid
from ppnl.rollback import choose_prefix, offline_diagnosis


class RollbackTest(unittest.TestCase):
    def test_multiple_shortest_routes_are_not_divergence(self):
        g = Grid(3, 3, (), (0, 0), (2, 2))
        self.assertIsNone(
            offline_diagnosis(g, "right down right down", "valid")["first_suboptimal_action"]
        )

    def test_legal_loop_diverges_before_execution_failure(self):
        g = Grid(3, 3, (), (0, 0), (2, 0))
        plan = "down up down"
        result = offline_diagnosis(g, plan, "invalid | step=4 | reason=goal_not_reached")
        self.assertEqual(result["first_suboptimal_action"], 2)
        self.assertEqual(result["oracle_backoff"], 2)
        prefix, usage = choose_prefix(
            g, plan, "invalid | step=4 | reason=goal_not_reached", rewind_cycle=True
        )
        self.assertEqual(prefix, "")
        self.assertEqual(usage["executor_calls"], 1)

    def test_wrong_late_diagnosis_is_explicitly_corrected(self):
        g = Grid(3, 3, (), (0, 0), (2, 0))
        prefix, usage = choose_prefix(
            g, "up down down", "invalid | step=3 | reason=obstacle_collision"
        )
        self.assertEqual(prefix, "")
        self.assertTrue(usage["symbolic_prefix_correction"])
