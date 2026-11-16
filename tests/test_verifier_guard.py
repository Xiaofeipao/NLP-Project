import unittest
from ppnl.core import Grid
from ppnl.verifier_guard import guard_verdict


class GuardTest(unittest.TestCase):
    def test_first_failure_precedes_later_token(self):
        g = Grid(3, 3, ((0, 1),), (0, 0), (2, 2))
        self.assertEqual(guard_verdict(g, "right jump", "valid", "vocabulary"), "valid")
        self.assertEqual(
            guard_verdict(g, "right jump", "valid"), "invalid | step=1 | reason=obstacle_collision"
        )

    def test_terminal_and_valid_detour(self):
        g = Grid(3, 3, (), (0, 0), (2, 2))
        self.assertEqual(
            guard_verdict(g, "down", "valid", "terminal"),
            "invalid | step=2 | reason=goal_not_reached",
        )
        self.assertEqual(
            guard_verdict(g, "down up down down right right", "invalid", "trace"), "valid"
        )
        self.assertEqual(guard_verdict(g, "up", "valid", "terminal"), "valid")
        self.assertEqual(
            guard_verdict(g, "jump", "valid", "vocabulary"),
            "invalid | step=1 | reason=invalid_action_token",
        )
