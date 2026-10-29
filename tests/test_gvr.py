from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from ppnl.core import Grid, execute
from ppnl.data import task_index, task_index_manifest
from ppnl.gvr_harness import create_gvr_session, run_gvr_session, score_gvr_session
from ppnl.gvr_evaluation import evaluate_gvr


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
        self.calls.append((task, candidate, feedback))
        return next(self.responses)


class GVRTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(2, 3, (), (0, 0), (0, 2))
        self.identity = {"base": "test", "verification": "v", "repair": "r"}

    def test_self_feedback_never_sees_executor_truth(self):
        # The model incorrectly accepts an invalid candidate. The workflow must stop.
        model = ScriptedModel(["up", "valid"])
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp) / "trial"
            create_gvr_session(trial, self.grid, "self", self.identity)
            state = run_gvr_session(trial, model, self.identity)
            self.assertEqual(state["stop_reason"], "feedback_valid")
            self.assertEqual(state["executor_calls"], 0)
            self.assertEqual([c[0] for c in model.calls], ["generation", "verification"])
            self.assertFalse(score_gvr_session(trial)["success"])

    def test_malformed_feedback_repair_and_repeat_stop(self):
        model = ScriptedModel(["up", "unparseable", "up", "also unparseable"])
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp) / "trial"
            create_gvr_session(trial, self.grid, "self", self.identity)
            state = run_gvr_session(trial, model, self.identity)
            self.assertEqual(state["stop_reason"], "repeated_candidate")
            self.assertEqual(model.calls[2][2], "unparseable")

    def test_executor_feedback_and_resume_identity(self):
        model = ScriptedModel(["up", "right right"])
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp) / "trial"
            create_gvr_session(trial, self.grid, "executor_one", self.identity)
            with self.assertRaises(ValueError):
                run_gvr_session(trial, model, {"base": "changed"})
            state = run_gvr_session(trial, model, self.identity)
            self.assertEqual(state["executor_calls"], 2)
            self.assertEqual(state["repairs"], 1)
            self.assertTrue(score_gvr_session(trial)["success"])
            self.assertEqual(model.calls[1][2], "invalid | step=1 | reason=out_of_bounds")

    def test_frozen_task_index(self):
        root = Path(__file__).resolve().parents[1] / "data" / "derived"
        index = task_index_manifest(root)
        self.assertEqual([index["splits"]["train"]["tasks"][t]["count"]
                          for t in ("generation", "verification", "repair")], [800, 2000, 1200])
        self.assertEqual(len(task_index(root, "dev", "verification")), 100)

    def test_candidate_and_paired_pipeline_evaluation(self):
        class Model:
            def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
                return {"generation": "up", "verification": "valid" if candidate == "right right" else
                        "invalid | step=1 | reason=out_of_bounds", "repair": "right right"}[task]

        class Oracle:
            sha256 = "test"

            def check(self, grid, plan):
                return execute(grid, plan)

        verdict = execute(self.grid, "up")
        row = {"id": "x", "source_id": "fixture", "grid": asdict(self.grid),
               "layout_id": self.grid.layout_id, "candidate": "up", "valid": False,
               "reason": verdict.reason, "step": verdict.step, "label": verdict.label()}
        with tempfile.TemporaryDirectory() as temp:
            metrics = evaluate_gvr(Model(), [row], Path(temp) / "eval", self.identity, Oracle())
            self.assertEqual(metrics["verification"]["macro_f1"], 0.5)
            self.assertEqual(metrics["repair"]["repair_success_rate"], 1.0)
            self.assertEqual(metrics["direct"]["success_rate"], 0.0)
            for mode in ("self", "executor_one", "executor_three"):
                self.assertEqual(metrics[mode]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
