"""CPU regression tests, including differential checks against official source."""
import unittest

from ppnl.core import Grid, execute, parse_verdict, shortest_path
from ppnl.data import corrupt, edit_distance, nonoptimal
from pathlib import Path
from ppnl.core import OfficialOracle
import random


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(3, 3, ((1, 1),), (0, 0), (2, 2))

    def test_first_failure(self):
        result = execute(self.grid, "right down jump")
        self.assertEqual((result.step, result.reason), (2, "obstacle_collision"))
        self.assertEqual(len(result.trace), 2)

    def test_failure_modes(self):
        for plan, step, reason in (("up", 1, "out_of_bounds"), ("jump", 1, "invalid_action_token"),
                                   ("right", 2, "goal_not_reached"), ("", 1, "goal_not_reached")):
            with self.subTest(plan=plan):
                result = execute(self.grid, plan)
                self.assertEqual((result.step, result.reason), (step, reason))

    def test_feasible_does_not_require_goal(self):
        result = execute(self.grid, "right")
        self.assertTrue(result.feasible)
        self.assertFalse(result.valid)

    def test_entire_plan_must_be_executed(self):
        self.assertTrue(execute(self.grid, "right right down down").valid)
        self.assertEqual(execute(self.grid, "right right down down down").reason, "out_of_bounds")

    def test_shortest_and_nonoptimal(self):
        path = shortest_path(self.grid)
        self.assertEqual(len(path), 4)
        longer = nonoptimal(self.grid, path)
        self.assertTrue(execute(self.grid, " ".join(longer)).valid)
        self.assertEqual(len(longer), 6)

    def test_layout_grouping_ignores_start_goal(self):
        other = Grid(3, 3, ((1, 1),), (2, 0), (0, 2))
        self.assertEqual(other.layout_id, self.grid.layout_id)

    def test_strict_verification_format(self):
        self.assertEqual(parse_verdict("invalid | step=2 | reason=obstacle_collision"),
                         (False, 2, "obstacle_collision"))
        self.assertIsNone(parse_verdict("invalid | step=0 | reason=obstacle_collision"))
        self.assertIsNone(parse_verdict("This plan is valid"))

    def test_two_substitution_sites_are_distinct(self):
        tokens, edits = corrupt(["down", "down", "right", "right"], 2, random.Random(42))
        self.assertEqual(len(edits), 2)
        self.assertEqual(len({e["original_index"] for e in edits}), 2)

    def test_ood_construction_can_produce_exactly_two_token_edits(self):
        base = ["down", "down", "right", "right"]
        rng = random.Random(42)
        accepted = []
        for _ in range(100):
            tokens, edits = corrupt(base, 2, rng)
            if edit_distance(base, tokens) == 2 and not execute(self.grid, " ".join(tokens)).valid:
                accepted.append((tokens, edits))
        self.assertTrue(accepted)
        self.assertTrue(all(len(edits) == 2 for _, edits in accepted))

    @unittest.skipUnless(Path("vendor/ppnl/executor-point-sg.py").exists(), "Download official source first")
    def test_official_differential(self):
        oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
        rng = random.Random(42)
        for _ in range(200):
            plan = " ".join(rng.choices(["up", "down", "left", "right"], k=rng.randrange(12)))
            self.assertEqual(oracle.check(self.grid, plan).label(), execute(self.grid, plan).label())


if __name__ == "__main__":
    unittest.main()
