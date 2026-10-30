"""Regression checks prepared for the next run; CPU tensors only."""
from pathlib import Path
import tempfile
import unittest

from ppnl.gvr_objectives import checkpoint_rank, stage_options
from ppnl.optimization import ACCUMULATION_NORMALIZATION
from scripts.run_gvr import evaluation_output


class GVRCandidateTests(unittest.TestCase):
    def test_changed_lora_or_accumulation_rejects_old_pilot_before_cuda(self):
        import json
        from unittest.mock import patch
        from ppnl.gvr_training import train_stage
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_text("{}")
            index = {"source_manifest_sha256": "frozen"}
            (root / "index.json").write_text(json.dumps(index))
            settings = {"epochs": 2}
            config = {"training": settings, "models": {"flan_t5_small": {
                "checkpoint": "fixed", "user_confirmed": True, "confirmation_note": "Approved"}}}
            protocol = {"dataset_root": str(root / "data"), "task_index": str(root / "index.json"),
                        "generation_base": str(base), "output_root": temp, "lora": {"r": 16},
                        "evidence": {"gpu_ledger_reconciled": True, "checkpoint_confirmed": True,
                                     "baseline_outputs_verified": True}}
            prior = {"stage": "verification", "pilot": True, "task_index_source_sha256": "frozen",
                     "training_config": settings, "generation_base_sha256": "base-hash",
                     "training_signature": {"training": settings,
                                            "stage_options": stage_options({}, "verification"),
                                            "lora": {"r": 8},
                                            "accumulation_normalization": ACCUMULATION_NORMALIZATION}}
            pilot = root / "verification/pilot/training.json"
            pilot.parent.mkdir(parents=True)
            pilot.write_text(json.dumps(prior))
            with patch("ppnl.gvr_training.task_index_manifest", return_value=index), \
                    patch("ppnl.gvr_training.checkpoint_hash", return_value="base-hash"), \
                    patch("ppnl.gvr_training.require_cuda") as cuda:
                with self.assertRaisesRegex(ValueError, "Pilot .*settings differ"):
                    train_stage(config, protocol, "verification", root / "train")
                prior["training_signature"]["lora"] = protocol["lora"]
                prior["training_signature"].pop("accumulation_normalization")
                pilot.write_text(json.dumps(prior))
                with self.assertRaisesRegex(ValueError, "Pilot .*settings differ"):
                    train_stage(config, protocol, "verification", root / "train")
                cuda.assert_not_called()

    def test_original_protocol_keeps_loss_selection(self):
        options = stage_options({}, "verification")
        self.assertEqual(options["loss"], "token_mean")
        self.assertEqual(checkpoint_rank("verification", .3, None, options), (-.3,))

    def test_task_score_beats_lower_teacher_forcing_loss(self):
        options = {"selection": "task_metrics"}
        collapsed = {"macro_f1": 1 / 3, "first_error_localization_accuracy": 0., "failure_type_accuracy": 0.}
        improved = {"macro_f1": .6, "first_error_localization_accuracy": .2, "failure_type_accuracy": .5}
        self.assertGreater(checkpoint_rank("verification", .8, improved, options),
                           checkpoint_rank("verification", .2, collapsed, options))
        self.assertGreater(checkpoint_rank("generation", 2., {"success_rate": .1, "optimality": .1}, options),
                           checkpoint_rank("generation", .1, {"success_rate": 0., "optimality": 0.}, options))

    def test_bad_stage_options_fail_before_gpu(self):
        for config in ({"verification": {"status_weight": 1}},
                       {"generation": {"loss": "sequence_and_status", "status_weight": 1}},
                       {"repair": {"learning_rate": float("nan")}},
                       {"verification": {"selection": "test_score"}},
                       {"verfication": {}}):
            with self.assertRaises(ValueError):
                stage_options({"stages": config}, next(iter(config)))

    def test_existing_output_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = {"output_root": temp}
            (Path(temp) / "verify_dev.json").write_text("old")
            with self.assertRaises(FileExistsError):
                evaluation_output(protocol, "verify")
            out = evaluation_output(protocol, "evaluate", "dev")
            out.mkdir(parents=True)
            self.assertEqual(evaluation_output(protocol, "evaluate", "dev"), out)
            (out / "metrics.json").write_text("old")
            with self.assertRaises(FileExistsError):
                evaluation_output(protocol, "evaluate", "dev")

    def test_status_loss_balances_examples_and_masks_padding(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("PyTorch is not installed")
        from ppnl.gvr_objectives import sequence_status_loss
        logits = torch.tensor([[[2., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 2., 0.], [0., 0., 2.]]], requires_grad=True)
        labels = torch.tensor([[0, -100, -100], [0, 1, 2]])
        losses = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
        expected = (losses[0, 0] + losses[1].mean()) / 2 + losses[:, 0].mean()
        actual = sequence_status_loss(logits, labels, 1.)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 1:].abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
