import unittest
from ppnl.core import Grid, execute
from ppnl.repair_variants import probabilities, erase_loops, repair


class VariantsTest(unittest.TestCase):
    def test_probability_and_erasure(self):
        self.assertAlmostEqual(sum(probabilities([1000, 999, -1000, 0]).values()), 1)
        g = Grid(3, 3, (), (0, 0), (2, 0))
        self.assertEqual(erase_loops(g, "down up down down"), ("down down", 2))
        self.assertEqual(erase_loops(g, "up down"), ("up down", 0))
        self.assertEqual(erase_loops(g, "down down"), ("down down", 0))

    def test_branch_budget_and_cycle_ban(self):
        class Scorer:
            def scores(self, obs):
                return [[4, 3, 2, 1] for _ in obs]

        g = Grid(3, 3, (), (0, 0), (2, 0))
        fb = "invalid | step=1 | reason=goal_not_reached"
        plan, status, info = repair(g, "", fb, Scorer(), {"mode": "beam", "score_cap": 4})
        self.assertLessEqual(info["score_rows"], 4)
        self.assertTrue(execute(g, plan).feasible)
        plan, status, info = repair(g, "", fb, Scorer(), {"mode": "ban"})
        self.assertTrue(execute(g, plan).valid)
