import unittest
from ppnl.core import Grid
from ppnl.method_sweep import repair_once, extract_verdict, verification_input, METHODS


class SweepTest(unittest.TestCase):
    def test_cot_requires_explicit_final(self):
        self.assertEqual(
            extract_verdict("Reason. FINAL: invalid | step=2 | reason=out_of_bounds", "cot"),
            "invalid | step=2 | reason=out_of_bounds",
        )
        self.assertEqual(extract_verdict("valid", "cot"), "missing_FINAL: valid")

    def test_constraint_and_uniform_are_explicit(self):
        class Scorer:
            def scores(self, observations):
                return [[5, 4, 0, 0]]

        class Forbidden:
            def scores(self, observations):
                raise AssertionError("Uniform must not call model")

        g = Grid(3, 3, (), (0, 0), (2, 0))
        fb = "invalid | step=1 | reason=goal_not_reached"
        self.assertEqual(repair_once(g, "", fb, Scorer(), 2, "raw")[2], "illegal_action")
        self.assertEqual(repair_once(g, "", fb, Scorer(), 2, "visit")[0], "down down")
        repair_once(g, "", fb, Forbidden(), 2, "uniform_visit")

    def test_prompt_without_oracle_label_and_unique_names(self):
        g = Grid(3, 3, (), (0, 0), (2, 0))
        s = verification_input(g, "down", "structured")
        self.assertNotIn("Executor observations", s)
        self.assertEqual(len(METHODS), len({m["name"] for m in METHODS}))
