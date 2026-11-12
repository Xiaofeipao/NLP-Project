"""G/V/R runtime contracts with the original full checkpoint interface, without GPU."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ppnl.core import Grid, execute
from ppnl.full_finetune_gvr import FullFinetuneGVRModel, checkpoint_identity
from ppnl.gvr_harness import create_gvr_session, run_gvr_session, score_gvr_session, validate_gvr_protocol
from ppnl.harness import read_session


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
        self.calls.append((task, candidate, feedback))
        return next(self.responses)


class FakeTokenizer:
    def __call__(self, value):
        return {"input_ids": value.split()}


class FakeRunner:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.inputs = []

    def predict(self, inputs, check):
        self.inputs.extend(inputs)
        return ["right right"]


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.grid = Grid(2, 3, (), (0, 0), (0, 2))
        self.identity = {"kind": "full_finetune_fixture"}

    def session(self, name, mode, calls=8):
        path = self.root / name
        create_gvr_session(path, self.grid, mode, self.identity, model_calls=calls)
        return path

    def test_full_model_adapter_uses_original_prompts_for_all_tasks(self):
        runner = FakeRunner()
        model = FullFinetuneGVRModel(runner)
        self.assertEqual(model.predict("generation", self.grid), "right right")
        model.predict("verification", self.grid, "up")
        model.predict("repair", self.grid, "up", "invalid | step=1 | reason=out_of_bounds")
        self.assertIn("Plan a path.", runner.inputs[0])
        self.assertIn("Verify a path.", runner.inputs[1])
        self.assertIn("Candidate: up", runner.inputs[1])
        self.assertIn("Feedback: invalid | step=1", runner.inputs[2])
        self.assertGreater(model.last_usage["input_tokens"], 0)

    def test_failed_adapter_call_does_not_expose_previous_usage(self):
        runner = FakeRunner()
        model = FullFinetuneGVRModel(runner)
        model.predict("generation", self.grid)
        self.assertGreater(model.last_usage["output_tokens"], 0)

        def fail(inputs, check):
            raise RuntimeError("runner failed")

        runner.predict = fail
        with self.assertRaisesRegex(RuntimeError, "runner failed"):
            model.predict("repair", self.grid, "up", "invalid | step=1 | reason=out_of_bounds")
        self.assertEqual(model.last_usage, {})

    def test_direct_and_self_feedback_are_separate_from_scoring(self):
        direct = self.session("direct", "direct", calls=1)
        self.assertEqual(run_gvr_session(direct, ScriptedModel(["up"]), self.identity)["stop_reason"],
                         "direct_complete")
        self.assertFalse(score_gvr_session(direct)["success"])
        trial = self.session("self", "self", calls=2)
        model = ScriptedModel(["up", "valid"])
        state = run_gvr_session(trial, model, self.identity)
        self.assertEqual([call[0] for call in model.calls], ["generation", "verification"])
        self.assertEqual(state["executor_calls"], 0)
        self.assertEqual(state["stop_reason"], "feedback_valid")
        self.assertTrue(score_gvr_session(trial)["false_valid_accept"])

    def test_executor_repairs_and_resume_reject_changed_identity(self):
        trial = self.session("executor", "executor_one")
        with self.assertRaises(ValueError):
            run_gvr_session(trial, ScriptedModel([]), {"kind": "changed"})
        model = ScriptedModel(["up", "right right"])
        state = run_gvr_session(trial, model, self.identity)
        self.assertEqual(state["repairs"], 1)
        self.assertEqual(state["executor_calls"], 2)
        self.assertEqual(model.calls[1][2], execute(self.grid, "up").label())
        self.assertTrue(score_gvr_session(trial)["success"])
        self.assertEqual(run_gvr_session(trial, ScriptedModel([]), self.identity)["model_calls"], 2)

    def test_budget_and_journal_recovery(self):
        trial = self.session("budget", "self", calls=2)
        state = run_gvr_session(trial, ScriptedModel(["up", "invalid | step=1 | reason=out_of_bounds"]),
                                self.identity)
        self.assertEqual(state["status"], "budget_exhausted")
        self.assertEqual(state["model_calls"], 2)
        (trial / "state.json").write_text("{}", encoding="utf-8")
        self.assertEqual(read_session(trial)[1]["model_calls"], 2)

    def test_invalid_protocol_is_rejected_before_creating_a_trial(self):
        for mode, calls, seconds in (("unknown", 8, 300), ("self", 0, 300),
                                     ("self", True, 300), ("self", 8, float("nan"))):
            with self.assertRaises(ValueError):
                validate_gvr_protocol(mode, calls, seconds)

    def test_identity_accepts_full_weights_and_rejects_adapter(self):
        checkpoint = self.root / "model"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"full")
        identity = checkpoint_identity(checkpoint, {"seed": 42})
        self.assertEqual(identity["kind"], "full_finetune_gvr")
        self.assertEqual(len(identity["checkpoint"]["model.safetensors"]), 64)
        (checkpoint / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "LoRA adapter"):
            checkpoint_identity(checkpoint, {"seed": 42})

    def test_cli_dry_run_checks_full_checkpoint_without_gpu_or_writes(self):
        checkpoint = self.root / "model"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"full")
        dataset = self.root / "data"
        dataset.mkdir()
        executor = self.root / "executor.py"
        source = "def is_goal(world, start, plan, rows, cols):\n    return 1\n"
        executor.write_text(source, encoding="utf-8")
        (dataset / "manifest.json").write_text(json.dumps({
            "executor_path": str(executor),
            "executor_sha256": hashlib.sha256(source.encode()).hexdigest()}), encoding="utf-8")
        config = self.root / "config.json"
        config.write_text(json.dumps({"dataset_root": str(dataset)}), encoding="utf-8")
        grid = self.root / "grid.json"
        grid.write_text(json.dumps(self.grid.__dict__), encoding="utf-8")
        output = self.root / "outputs" / "full_finetune_gvr" / "trial"
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_full_finetune_gvr.py"
        process = subprocess.run([sys.executable, str(script), "run", "--checkpoint", str(checkpoint),
                                  "--grid", str(grid), "--model-config", str(config), "--mode", "direct",
                                  "--out", str(output), "--dry-run"], cwd=self.root,
                                 capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(process.stdout)["validated"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
