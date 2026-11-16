import unittest
from ppnl.core import Grid
from scripts.run_phase2_feedback import continue_plan


class FeedbackTest(unittest.TestCase):
    def test_feedback_controls_repair_without_oracle_gating(self):
        class Policy:
            def scores(self, observations):
                return [[0, 3, 0, 0]]

        g = Grid(3, 3, (), (0, 0), (2, 0))
        self.assertEqual(continue_plan(g, "down", "valid", Policy()), ("down", "accepted"))
        self.assertEqual(
            continue_plan(g, "down", "invalid | step=2 | reason=goal_not_reached", Policy()),
            ("down down", "policy_reaches_goal"),
        )
        self.assertEqual(
            continue_plan(g, "up", "invalid | step=2 | reason=goal_not_reached", Policy())[1],
            "illegal_reported_prefix",
        )

    def test_visit_guard_can_break_memoryless_cycle(self):
        class Cycle:
            def scores(self, observations):
                return [[3, 2, 0, 0] if p[0] else [0, 3, 0, 0] for g, p in observations]

        g = Grid(3, 3, (), (0, 0), (2, 0))
        fb = "invalid | step=2 | reason=goal_not_reached"
        self.assertEqual(continue_plan(g, "down", fb, Cycle(), max_steps=4)[1], "action_budget")
        self.assertEqual(
            continue_plan(g, "down", fb, Cycle(), max_steps=4, visit_guard=True),
            ("down down", "policy_reaches_goal"),
        )
