import unittest
from ppnl.core import Grid
from ppnl.rlvr import plan_reward, leave_one_out, sequence_objective


class RLVRTests(unittest.TestCase):
    def test_success_beats_shaping_and_illegal(self):
        g = Grid(3, 3, (), (0, 0), (0, 2))
        self.assertGreater(plan_reward(g, "right right"), plan_reward(g, "right"))
        self.assertGreater(plan_reward(g, "right"), plan_reward(g, "right jump"))
        self.assertGreater(plan_reward(g, "right right"), plan_reward(g, "down up right right"))

    def test_constant_reward_has_no_policy_gradient_advantage(self):
        self.assertEqual(leave_one_out([1, 1, 1, 1]), [0, 0, 0, 0])
        self.assertEqual(sum(leave_one_out([0, 1, 2, 3])), 0)

    def test_gradient_rewards_better_sample_and_masks_padding(self):
        import torch
        logp = torch.tensor([[-1., -2.], [-1., -2.]], requires_grad=True)
        loss = sequence_objective(logp, logp.detach(), torch.tensor([[1, 0], [1, 0]]),
                                  torch.tensor([1., -1.]), .02)
        loss.backward()
        self.assertLess(logp.grad[0, 0], 0)
        self.assertGreater(logp.grad[1, 0], 0)
        self.assertEqual(logp.grad[:, 1].abs().sum(), 0)
