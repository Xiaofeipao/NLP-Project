"""Check effective-batch gradients, including variable-length targets."""
import unittest

from ppnl.optimization import accumulated_batches


class AccumulationTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        self.torch = torch

    def test_token_gradients_match_full_batch_and_old_averaging_does_not(self):
        torch = self.torch
        from torch.nn.functional import cross_entropy
        torch.manual_seed(7)
        initial = torch.randn(3, 4, 5, dtype=torch.float64)
        labels = torch.tensor([[1, -100, -100, -100], [2, 3, 4, 1], [3, 2, 1, -100]])
        full = initial.clone().requires_grad_()
        cross_entropy(full.transpose(1, 2), labels).backward()
        corrected = initial.clone().requires_grad_()
        old = initial.clone().requires_grad_()
        batches = [{"labels": labels[:1], "slice": slice(0, 1)},
                   {"labels": labels[1:], "slice": slice(1, 3)}]
        for _, batch, scale, _ in accumulated_batches(batches, 2):
            cross_entropy(corrected[batch["slice"]].transpose(1, 2), batch["labels"]).mul(scale).backward()
            cross_entropy(old[batch["slice"]].transpose(1, 2), batch["labels"]).div(2).backward()
        torch.testing.assert_close(corrected.grad, full.grad)
        self.assertGreater((old.grad - full.grad).abs().max().item(), .1)

    def test_example_objective_matches_full_batch_with_short_microbatch(self):
        torch = self.torch
        from ppnl.gvr_objectives import sequence_status_loss
        torch.manual_seed(8)
        labels = torch.tensor([[1, 2, 3], [2, -100, -100], [3, 1, -100]])
        logits = torch.randn(3, 3, 5, requires_grad=True)
        other = logits.detach().clone().requires_grad_()
        sequence_status_loss(logits, labels, 1.).backward()
        batches = [{"labels": labels[:2], "slice": slice(0, 2)},
                   {"labels": labels[2:], "slice": slice(2, 3)}]
        for _, batch, scale, _ in accumulated_batches(batches, 2, "examples"):
            (sequence_status_loss(other[batch["slice"]], batch["labels"], 1.) * scale).backward()
        torch.testing.assert_close(logits.grad, other.grad)

    def test_short_final_window_updates_once_and_ignores_padding(self):
        torch = self.torch
        batches = [{"labels": torch.tensor([[1, -100]])},
                   {"labels": torch.tensor([[1, 2]])},
                   {"labels": torch.tensor([[3, -100]])}]
        result = list(accumulated_batches(batches, 2))
        self.assertEqual([r[0] for r in result], [0, 1, 2])
        self.assertEqual([r[2] for r in result], [1 / 3, 2 / 3, 1.])
        self.assertEqual([r[3] for r in result], [False, True, True])
        with self.assertRaises(ValueError):
            list(accumulated_batches([{"labels": torch.tensor([[-100]])}], 2))
        with self.assertRaises(ValueError):
            list(accumulated_batches(batches, 0))


if __name__ == "__main__":
    unittest.main()
