"""Executor-prefix repair tests use scripted scores and no model or GPU."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ppnl.core import Grid, execute
from ppnl.data import write_jsonl
from ppnl.prefix_state_repair import (evaluate_generated, load_generated_candidates,
                                      repair_generated, save_result)


class Oracle:
    sha256 = "fixture"

    def check(self, grid, plan):
        return execute(grid, plan)


class Scores:
    def __init__(self, values=(0, 0, 0, 10)):
        self.values = values
        self.calls = []

    def scores(self, observations):
        self.calls.extend(observations)
        return [self.values for _ in observations]


class PrefixRepairTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(2, 3, (), (0, 0), (0, 2))
        self.oracle = Oracle()

    def test_failure_action_and_suffix_are_discarded(self):
        scorer = Scores()
        result = repair_generated(self.grid, "right up left down", scorer, self.oracle)
        self.assertEqual(result["kept_prefix"], "right")
        self.assertEqual(result["failure_position"], [0, 1])
        self.assertEqual(result["plan"], "right right")
        self.assertEqual(result["policy_calls"], 1)
        self.assertEqual(scorer.calls[0][1], (0, 1))
        self.assertEqual(result["final_feedback"], "valid")

    def test_end_of_sequence_repair_and_no_repeat_of_visited_prefix(self):
        scorer = Scores((0, 0, 100, 10))
        result = repair_generated(self.grid, "right", scorer, self.oracle)
        self.assertEqual(result["initial_feedback"], "invalid | step=2 | reason=goal_not_reached")
        self.assertEqual(result["plan"], "right right")
        self.assertNotIn("left", result["steps"][0]["least_visited_choices"])

    def test_valid_candidate_needs_no_policy_call(self):
        scorer = Scores()
        result = repair_generated(self.grid, "right right", scorer, self.oracle)
        self.assertEqual(result["plan"], "right right")
        self.assertEqual(result["stop_reason"], "already_valid")
        self.assertEqual(scorer.calls, [])

    def test_prefix_already_at_goal_drops_later_failure(self):
        scorer = Scores()
        result = repair_generated(self.grid, "right right up", scorer, self.oracle)
        self.assertEqual(result["plan"], "right right")
        self.assertEqual(result["stop_reason"], "goal_reached_in_prefix")
        self.assertEqual(scorer.calls, [])

    def test_no_legal_neighbor_stops_without_policy_call(self):
        grid = Grid(2, 2, ((0, 1), (1, 0)), (0, 0), (1, 1))
        scorer = Scores()
        result = repair_generated(grid, "up right", scorer, self.oracle)
        self.assertEqual(result["plan"], "")
        self.assertEqual(result["stop_reason"], "no_legal_action")
        self.assertEqual(scorer.calls, [])

    def test_action_budget_and_bad_scores(self):
        result = repair_generated(self.grid, "right", Scores(), self.oracle, max_actions=1)
        self.assertEqual(result["stop_reason"], "action_budget")
        self.assertEqual(result["plan"], "right")
        with self.assertRaisesRegex(ValueError, "prefix exceeds"):
            repair_generated(self.grid, "right left", Scores(), self.oracle, max_actions=1)
        with self.assertRaisesRegex(ValueError, "finite"):
            repair_generated(self.grid, "right", Scores((0, 0, 0, float("nan"))), self.oracle)

    def test_generated_trace_requires_exact_frozen_environment_and_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "data"
            dataset.mkdir()
            grid_dict = asdict(self.grid)
            write_jsonl(dataset / "dev.jsonl", [{"task": "verification", "grid": grid_dict}])
            trace = root / "pipeline_traces.jsonl"
            row = {"source_id": "one", "grid": grid_dict,
                   "direct": {"plan": "right", "verdict": "invalid | step=2 | reason=goal_not_reached"}}
            write_jsonl(trace, [row])
            paired = load_generated_candidates(dataset, "dev", trace, self.oracle)
            self.assertEqual(len(paired), 1)
            metrics, records = evaluate_generated(paired, Scores(), self.oracle)
            self.assertEqual(metrics["direct"]["success_rate"], 0)
            self.assertEqual(metrics["prefix_state_repair"]["success_rate"], 1)
            report = save_result(root / "results", dataset, "dev", trace, "weights", "executor",
                                 36, metrics, records)
            self.assertEqual(report["n"], 1)
            self.assertTrue((root / "results" / "traces.jsonl").exists())
            write_jsonl(trace, [dict(row, direct={"plan": "right", "verdict": "valid"})])
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_generated_candidates(dataset, "dev", trace, self.oracle)
            write_jsonl(trace, [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_generated_candidates(dataset, "dev", trace, self.oracle)

    def test_cli_check_only_validates_saved_paths_without_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "data"
            dataset.mkdir()
            write_jsonl(dataset / "dev.jsonl", [{"task": "verification", "grid": asdict(self.grid)}])
            executor = root / "executor.py"
            source = ("def is_goal(world, start, plan, rows, cols):\n"
                      "    return 1 if plan.strip() == 'right right' else 0\n")
            executor.write_text(source, encoding="utf-8")
            (dataset / "manifest.json").write_text(json.dumps({
                "executor_path": str(executor),
                "executor_sha256": hashlib.sha256(source.encode()).hexdigest()}), encoding="utf-8")
            trace = root / "pipeline_traces.jsonl"
            write_jsonl(trace, [{"grid": asdict(self.grid), "direct": {
                "plan": "right", "verdict": "invalid | step=2 | reason=goal_not_reached"}}])
            config = root / "config.json"
            config.write_text(json.dumps({"policy_source_dataset": str(dataset),
                                          "baseline_root": str(root)}), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "scripts" / "run_main_prefix_state_repair.py"
            completed = subprocess.run([sys.executable, str(script), "--config", str(config),
                                        "--split", "dev", "--trace", str(trace), "--check-only"],
                                       cwd=root, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(completed.stdout)["n"], 1)


if __name__ == "__main__":
    unittest.main()
