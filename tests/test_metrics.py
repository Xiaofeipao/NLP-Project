import unittest

from ppnl.core import Grid
from ppnl.evaluation import plan_metrics, verification_metrics


class MetricsTests(unittest.TestCase):
    def test_malformed_predictions_are_penalized(self):
        rows = [{"valid": True, "step": None, "reason": None},
                {"valid": False, "step": 1, "reason": "out_of_bounds"}]
        scores = verification_metrics(rows, ["valid", "garbage"])
        self.assertEqual(scores["macro_f1"], 0.5)
        self.assertEqual(scores["first_error_localization_accuracy"], 0.0)

    def test_optimality_uses_action_count(self):
        grid = Grid(2, 3, (), (0, 0), (0, 2))
        scores = plan_metrics([grid, grid, grid], ["right right", "down up right right", "right"])
        self.assertEqual(scores["success_rate"], 2 / 3)
        self.assertEqual(scores["feasibility"], 1)
        self.assertEqual(scores["optimality"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
