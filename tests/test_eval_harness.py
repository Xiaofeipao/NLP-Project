import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ppnl.core import Grid
from ppnl.eval_harness import (METHOD_BY_NAME, _compare, _evaluate_split, repair_prefix,
                          resolve_methods, run, split_rows)

ROOT = Path(__file__).resolve().parents[1]

WORLD_ROWS = [
    {"id": "a", "task": "generation", "valid": True, "candidate": "",
     "grid": {"rows": 3, "cols": 3, "obstacles": [], "start": [0, 0], "goal": [2, 2]}},
    {"id": "b", "task": "generation", "valid": True, "candidate": "",
     "grid": {"rows": 3, "cols": 3, "obstacles": [[1, 1]], "start": [0, 0], "goal": [2, 2]}},
]


def script_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConstantScores:
    """Deterministic stand-in for a loaded model: equal logits for all four actions."""

    def scores(self, observations):
        return [[0.0, 0.0, 0.0, 0.0] for _ in observations]


class HarnessTests(unittest.TestCase):
    def test_registry_covers_every_frozen_method(self):
        script = script_module("_evaluate_policy_names", "scripts/evaluate_policy.py")
        source = (ROOT / "scripts/evaluate_policy.py").read_text(encoding="utf-8")
        block = source.split("methods_to_run", 1)[1].split("for method in methods_to_run", 1)[0]
        for name in METHOD_BY_NAME:
            self.assertIn(f'"{name}"', block, f"{name} missing from evaluate_policy.py")
        self.assertTrue(callable(script.evaluate))

    def test_repair_prefix_is_identical_to_frozen_pipeline(self):
        script = script_module("_evaluate_policy_repair", "scripts/evaluate_policy.py")
        grid = Grid(4, 4, ((1, 1),), (0, 0), (3, 3))
        candidates = ["down down right right right",
                      "right right down down down",
                      "up right down down down right right",
                      "left down right right down down"]
        for candidate in candidates:
            ours_prefix, ours_rest = repair_prefix(grid, candidate)
            theirs_prefix, theirs_rest = script.repair_prefix(grid, candidate)
            self.assertEqual(ours_prefix, theirs_prefix)
            self.assertEqual(ours_rest, theirs_rest)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_methods("state_policy,teleport")
        symbolic = [m.name for m in resolve_methods("all", symbolic_only=True)]
        self.assertEqual(symbolic, ["symbolic_astar_no_model", "symbolic_astar_goal_tiebreak"])

    def test_split_rows_deduplicates_by_layout_start_goal(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            rows = WORLD_ROWS + [dict(WORLD_ROWS[0], id="a_duplicate")]
            (data / "test_id.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            seen, grids = split_rows({"policy_source_dataset": str(data)}, "test_id")
            self.assertEqual(len(seen), 3)
            self.assertEqual(len(grids), 2)

    def test_comparison_reports_field_level_differences(self):
        actual = {"planning": {"state_policy": {"n": 2, "success_rate": 0.5, "feasibility": 1.0,
                                                "optimality": 0.5,
                                                "path_efficiency_on_success": 1.0}},
                  "prefix_repair": {}}
        expected = {"planning": {"state_policy": {"n": 2, "success_rate": 0.85, "feasibility": 1.0,
                                                 "optimality": 0.5,
                                                 "path_efficiency_on_success": 1.0}},
                    "prefix_repair": {}}
        comparison = _compare(actual, expected, "frozen")
        self.assertEqual(comparison["status"], "differs")
        self.assertEqual([d["field"] for d in comparison["differences"]], ["success_rate"])
        self.assertAlmostEqual(comparison["differences"][0]["expected"], 0.85)
        self.assertEqual(_compare(actual, actual, "frozen")["status"], "match")
        self.assertEqual(_compare(actual, None, "missing")["status"], "no_frozen_result")

    def test_symbolic_only_run_end_to_end_on_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "test_id.jsonl").write_text(
                "\n".join(json.dumps(row) for row in WORLD_ROWS) + "\n", encoding="utf-8")
            config = {"dataset_root": str(data), "policy_source_dataset": str(data),
                      "output_root": str(root / "baseline"), "baseline_root": str(root / "baseline"),
                      "gpu_ledger": str(root / "ledger.json"), "gpu_hours_cap": 1,
                      "training": {"eval_batch_size": 8}}
            result = run(config, splits=("test_id",), methods="symbolic_astar_goal_tiebreak",
                         out=str(root / "out"), symbolic_only=True, write_html=False, require_official=False)
            metrics = result["splits"]["test_id"]["planning"]["symbolic_astar_goal_tiebreak"]
            self.assertEqual(metrics["success_rate"], 1.0)
            self.assertEqual(metrics["optimality"], 1.0)
            self.assertTrue((root / "out" / "results.json").exists())
            self.assertTrue((root / "out" / "report.md").exists())
            self.assertTrue((root / "out" / "report.md").read_text(encoding="utf-8").startswith("# Evaluation"))
            self.assertEqual(result["splits"]["test_id"]["comparison"]["status"], "no_frozen_result")
            self.assertFalse((root / "ledger.json").exists(),
                             "symbolic-only runs must not touch the GPU ledger")

    def test_refuses_to_write_inside_baseline_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"dataset_root": str(root), "policy_source_dataset": str(root),
                      "output_root": str(root / "exp"), "baseline_root": str(root / "exp"),
                      "gpu_ledger": str(root / "ledger.json"), "gpu_hours_cap": 1}
            with self.assertRaises(ValueError):
                run(config, splits=("test_id",), methods="symbolic_astar_no_model",
                    out=str(root / "exp" / "rerun"), symbolic_only=True, write_html=False)

    def test_constant_scores_follow_action_order_without_masking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "test_id.jsonl").write_text(
                "\n".join(json.dumps(row) for row in WORLD_ROWS) + "\n", encoding="utf-8")
            config = {"policy_source_dataset": str(data), "baseline_root": str(root / "missing")}
            entry, grids, _, plans, _, _, _ = _evaluate_split(
                config, "test_id", [METHOD_BY_NAME["state_policy"]], ConstantScores(),
                lambda: None, root / "baseline", False)
            # Ties resolve to the first name in ACTIONS ("up"), which leaves the grid here.
            self.assertEqual(plans["state_policy"], ["up", "up"])
            self.assertEqual(entry["n_grids"], len(grids))
            self.assertIsNone(entry["mean_expansions"].get("state_policy"))
            self.assertEqual(entry["planning"]["state_policy"]["success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
