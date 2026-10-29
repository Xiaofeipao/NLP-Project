from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest

from ppnl.core import Grid, execute
from ppnl.data import read_jsonl, task_index, task_index_manifest, write_jsonl
from ppnl.gvr_harness import create_gvr_session, run_gvr_session, score_gvr_session
from ppnl.gvr_evaluation import dev_model_diagnostics, evaluate_gvr, evaluation_subsets
from ppnl.gvr_audit import recompute_metrics
from ppnl.task_adapters import GVRModel
from ppnl.gvr_training import validate_stage_rows


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
            score = score_gvr_session(trial)
            self.assertFalse(score["success"])
            self.assertTrue(score["false_valid_accept"])

    def test_malformed_feedback_repair_and_repeat_stop(self):
        model = ScriptedModel(["up", "unparseable", "  up  "])
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp) / "trial"
            create_gvr_session(trial, self.grid, "self", self.identity)
            state = run_gvr_session(trial, model, self.identity)
            self.assertEqual(state["stop_reason"], "repeated_candidate")
            self.assertEqual(model.calls[2][2], "unparseable")
            self.assertEqual(len(model.calls), 3)

    def test_completed_response_is_processed_at_exact_call_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            direct = Path(temp) / "direct"
            create_gvr_session(direct, self.grid, "direct", self.identity, model_calls=1)
            self.assertEqual(run_gvr_session(direct, ScriptedModel(["up"]), self.identity)["stop_reason"],
                             "direct_complete")
            accepted = Path(temp) / "accepted"
            create_gvr_session(accepted, self.grid, "self", self.identity, model_calls=2)
            self.assertEqual(run_gvr_session(accepted, ScriptedModel(["up", "valid"]), self.identity)["stop_reason"],
                             "feedback_valid")
            rejected = Path(temp) / "rejected"
            create_gvr_session(rejected, self.grid, "self", self.identity, model_calls=2)
            self.assertEqual(run_gvr_session(rejected, ScriptedModel(["up", "invalid | step=1 | reason=out_of_bounds"]),
                                             self.identity)["stop_reason"], "budget_exhausted")

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
        self.assertEqual(tuple(len(group) for group in evaluation_subsets(read_jsonl(root / "dev.jsonl"))),
                         (100, 60, 40))
        for split in ("test_id", "test_ood"):
            self.assertEqual(tuple(len(group) for group in evaluation_subsets(read_jsonl(root / f"{split}.jsonl"))),
                             (200, 100, 200))

    def test_training_data_contract_and_bad_label_detection(self):
        root = Path(__file__).resolve().parents[1] / "data" / "derived"
        for stage in ("generation", "verification", "repair"):
            train = task_index(root, "train", stage)
            dev = task_index(root, "dev", stage)
            validate_stage_rows(stage, train, dev)
        bad = [dict(row) for row in task_index(root, "train", "verification")]
        invalid = next(i for i, row in enumerate(bad) if not row["valid"])
        bad[invalid]["target"] = "valid"
        with self.assertRaisesRegex(ValueError, "Incorrect verification supervision"):
            validate_stage_rows("verification", bad, task_index(root, "dev", "verification"))

    def test_dev_adapter_diagnostics_detect_all_valid(self):
        class Model:
            def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
                return "valid" if task == "verification" else "up up up"

        root = Path(__file__).resolve().parents[1] / "data" / "derived"
        metrics = dev_model_diagnostics(Model(), root)
        self.assertEqual(metrics["verification"]["n"], 100)
        self.assertEqual(metrics["repair"]["n"], 60)
        self.assertEqual(metrics["generation"]["n"], 40)
        self.assertTrue(metrics["verification"]["all_valid_warning"])
        self.assertEqual(metrics["generation"]["repeated_action_run_ge_3"], 40)

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

    def test_dev_tasks_are_not_mixed_in_metrics(self):
        class Model:
            def __init__(self):
                self.calls = []

            def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
                self.calls.append(task)
                return {"generation": "up", "verification": "valid", "repair": "right right"}[task]

        class Oracle:
            sha256 = "test"

            def check(self, grid, plan):
                return execute(grid, plan)

        base = {"source_id": "fixture", "grid": asdict(self.grid), "layout_id": self.grid.layout_id}
        bad = execute(self.grid, "up")
        rows = [dict(base, id="g", task="generation"),
                dict(base, id="v-good", task="verification", candidate="right right",
                     valid=True, reason=None, step=None, label="valid"),
                dict(base, id="v-bad", task="verification", candidate="up",
                     valid=False, reason=bad.reason, step=bad.step, label=bad.label()),
                dict(base, id="r", task="repair", candidate="up",
                     valid=False, reason=bad.reason, step=bad.step, label=bad.label())]
        self.assertEqual(tuple(len(group) for group in evaluation_subsets(rows)), (2, 1, 1))
        with tempfile.TemporaryDirectory() as temp:
            model = Model()
            evaluation_dir = Path(temp) / "eval"
            metrics = evaluate_gvr(model, rows, evaluation_dir, self.identity, Oracle())
            self.assertEqual(metrics["subset_counts"], {"verification": 2, "repair": 1, "generation": 1})
            self.assertTrue(metrics["verification"]["all_valid_warning"])
            self.assertEqual(model.calls.count("verification"), 3)  # 2 candidates + 1 self check
            self.assertEqual(model.calls.count("repair"), 3)  # 1 candidate + 2 executor repairs
            write_jsonl(Path(temp) / "data" / "dev.jsonl", rows)
            # Older saved traces did not contain a task field; frozen IDs still route them.
            old_traces = read_jsonl(evaluation_dir / "candidate_traces.jsonl")
            for trace in old_traces:
                trace.pop("task")
            write_jsonl(evaluation_dir / "candidate_traces.jsonl", old_traces)
            recomputed = recompute_metrics(Path(temp) / "data", "dev", evaluation_dir,
                                           Path(temp) / "metrics_recomputed.json", Oracle())
            self.assertEqual(recomputed["subset_counts"], metrics["subset_counts"])
            self.assertEqual(recomputed["verification"]["macro_f1"], metrics["verification"]["macro_f1"])
            self.assertEqual(recomputed["repair"]["repair_success_rate"], metrics["repair"]["repair_success_rate"])
            old_traces[0]["candidate"] = "tampered"
            write_jsonl(evaluation_dir / "candidate_traces.jsonl", old_traces)
            with self.assertRaisesRegex(ValueError, "Candidate or gold feedback changed"):
                recompute_metrics(Path(temp) / "data", "dev", evaluation_dir,
                                  Path(temp) / "invalid_recompute.json", Oracle())

    def test_generation_disables_lora_and_task_routes_are_exclusive(self):
        class Adapter:
            active = "verification"
            disabled = False

            @contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False

            def set_adapter(self, task):
                self.active = task

        class Tokenizer:
            def __call__(self, value):
                return {"input_ids": value.split()}

        class Runner:
            model = Adapter()
            tokenizer = Tokenizer()

            def predict(self, inputs, check):
                self.assertions.append((self.model.active, self.model.disabled))
                return ["valid"]

        routed = GVRModel.__new__(GVRModel)
        routed.runner = Runner()
        routed.runner.assertions = []
        routed.adapter_hashes = {"verification": "v", "repair": "r"}
        routed.predict("generation", self.grid)
        routed.predict("verification", self.grid, "up")
        routed.predict("repair", self.grid, "up", "invalid | step=1 | reason=out_of_bounds")
        self.assertEqual(routed.runner.assertions, [("verification", True),
                                                     ("verification", False), ("repair", False)])


if __name__ == "__main__":
    unittest.main()
