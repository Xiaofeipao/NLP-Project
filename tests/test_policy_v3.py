import unittest

from ppnl.core import Grid, shortest_path
from ppnl.policy import optimal_search_plans
from ppnl.policy_v3_data import distances_to_goal, optimal_moves, critical


class Scores:
    def __init__(self, scores):
        self.values = scores

    def scores(self, observations):
        return [self.values for _ in observations]


class PolicyV3Tests(unittest.TestCase):
    def test_all_shortest_actions_are_valid_targets(self):
        grid = Grid(3, 3, (), (0, 0), (2, 2))
        distances = distances_to_goal(grid)
        self.assertEqual(set(optimal_moves(grid, (0, 0), distances)), {"down", "right"})

    def test_obstacle_critical_state(self):
        grid = Grid(3, 3, ((1, 0),), (0, 0), (2, 2))
        self.assertTrue(critical(grid, (0, 0)))
        self.assertNotIn("down", optimal_moves(grid, (0, 0), distances_to_goal(grid)))

    def test_neural_ties_do_not_override_shortest_path_cost(self):
        grid = Grid(4, 4, ((1, 1), (1, 2), (2, 2)), (0, 0), (3, 3))
        for scores in ([50, 0, 0, -50], [-100, 50, 30, 0]):
            plans, _ = optimal_search_plans([grid], Scores(scores))
            self.assertEqual(len(plans[0].split()), len(shortest_path(grid)))

    def test_loss_allows_alternative_optimal_action_with_finite_verifier_gradient(self):
        import torch
        from ppnl.modeling import action_set_loss
        logits = torch.zeros((2, 2, 10), requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 1] = 10
        labels = torch.tensor([[0, 9], [5, -100]])
        mask = torch.tensor([[True, True, False, False], [False] * 4])
        single = action_set_loss(logits, labels, [0, 1, 2, 3], mask, "single_action")
        multiple = action_set_loss(logits, labels, [0, 1, 2, 3], mask, "optimal_actions")
        self.assertLess(multiple.item(), single.item())
        multiple.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
