import unittest

from ppnl.core import Grid, execute
from state_baselines.workflow import repair


class FeedbackTests(unittest.TestCase):
    grid = Grid(6, 6, (), (0, 0), (0, 2))

    def test_model_acceptance_is_preserved(self):
        self.assertEqual(repair(self.grid, "left", "valid"), ("left", "accepted"))

    def test_mislocated_error_does_not_use_hidden_diagnosis(self):
        plan = "left right"
        self.assertEqual(repair(self.grid, plan, "invalid | step=2 | reason=out_of_bounds"),
                         (plan, "illegal_reported_prefix"))

    def test_supplied_executor_feedback_can_repair(self):
        plan = "left right"
        revised, status = repair(self.grid, plan, execute(self.grid, plan).label())
        self.assertEqual((revised, status), ("right right", "repaired"))
