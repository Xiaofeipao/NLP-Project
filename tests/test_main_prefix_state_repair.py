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
from ppnl.full_prefix_repair import (evaluate_generated, load_generated_candidates,
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
        scorer = Scores()
        overflow = repair_generated(self.grid, "right left", scorer, self.oracle, max_actions=1)
        self.assertEqual(overflow["plan"], "right")
        self.assertEqual(overflow["kept_prefix"], "right")
        self.assertEqual(overflow["failure_position"], [0, 0])
        self.assertEqual(overflow["budget_position"], [0, 1])
        self.assertEqual(overflow["stop_reason"], "prefix_exceeds_budget")
        self.assertEqual(overflow["policy_calls"], 0)
        self.assertEqual(scorer.calls, [])
        with self.assertRaisesRegex(ValueError, "finite"):
            repair_generated(self.grid, "right", Scores((0, 0, 0, float("nan"))), self.oracle)

    def test_valid_long_candidate_obeys_repair_budget(self):
        scorer = Scores()
        long_valid = "right left right right"
        result = repair_generated(self.grid, long_valid, scorer, self.oracle, max_actions=2)
        self.assertEqual(result["initial_feedback"], "valid")
        self.assertEqual(result["plan"], "right left")
        self.assertEqual(result["final_feedback"], "invalid | step=3 | reason=goal_not_reached")
        self.assertEqual(result["stop_reason"], "prefix_exceeds_budget")
        self.assertEqual(result["failure_position"], None)
        self.assertEqual(scorer.calls, [])

    def test_valid_long_candidate_can_stop_at_earlier_goal_within_budget(self):
        result = repair_generated(self.grid, "right right left right", Scores(), self.oracle,
                                  max_actions=2)
        self.assertEqual(result["plan"], "right right")
        self.assertEqual(result["final_feedback"], "valid")
        self.assertEqual(result["stop_reason"], "already_valid_trimmed_to_goal")

    def test_one_overlong_prefix_does_not_abort_paired_evaluation(self):
        paired = [(self.grid, "right left right right", "long"),
                  (self.grid, "right", "short")]
        metrics, records = evaluate_generated(paired, Scores(), self.oracle, max_actions=2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["stop_reason"], "prefix_exceeds_budget")
        self.assertEqual(records[1]["final_feedback"], "valid")
        self.assertEqual(metrics["direct"]["over_repair_budget_count"], 1)
        self.assertEqual(metrics["prefix_state_repair"]["prefix_over_budget_count"], 1)

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

    def test_test_cli_requires_freeze_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = root / "spec.json"
            spec.write_text("{}")
            script = Path(__file__).resolve().parents[1] / "scripts/run_prefix_experiment.py"
            completed = subprocess.run([sys.executable, str(script), "evaluate", "--spec", str(spec),
                                        "--split", "test_id", "--out", str(root / "out")],
                                       capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("requires --freeze", completed.stderr)
            self.assertNotIn("CUDA", completed.stderr)


if __name__ == "__main__":
    unittest.main()
