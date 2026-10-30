"""Executor/policy boundary checks for generated-plan prefix repair."""
from pathlib import Path
import tempfile
import unittest

from ppnl.core import Grid, execute
from ppnl.data import write_jsonl
from ppnl.prefix_state_repair import (evaluate_generated, load_generated_candidates,
                                      repair_generated, save_result)


class LocalOracle:
    def check(self, grid, plan):
        return execute(grid, plan)


class Scores:
    def __init__(self, values):
        self.values = values
        self.positions = []

    def scores(self, observations):
        self.positions.extend(position for _, position in observations)
        return [self.values for _ in observations]


class PrefixStateRepairTests(unittest.TestCase):
    def setUp(self):
        self.oracle = LocalOracle()
        self.grid = Grid(2, 3, (), (0, 0), (0, 2))

    def test_keeps_executor_prefix_and_replans_from_failure_position(self):
        scorer = Scores([0, 0, 0, 9])
        result = repair_generated(self.grid, "right up right", scorer, self.oracle,
                                  mode="prefix_visit", max_steps=3)
        self.assertEqual(result["feedback"], "invalid | step=2 | reason=out_of_bounds")
        self.assertEqual(result["kept_prefix"], "right")
        self.assertEqual(result["plan"], "right right")
        self.assertEqual(scorer.positions, [(0, 1)])
        self.assertEqual(result["policy_decisions"], 1)
        self.assertEqual(result["stop_reason"], "goal_reached")

    def test_valid_candidate_is_not_replaced_or_scored_by_policy(self):
        scorer = Scores([0, 0, 0, 9])
        result = repair_generated(self.grid, "right right", scorer, self.oracle)
        self.assertEqual(result["plan"], "right right")
        self.assertEqual(result["stop_reason"], "already_valid")
        self.assertEqual(scorer.positions, [])

    def test_trailing_failure_after_goal_keeps_successful_prefix(self):
        scorer = Scores([0, 0, 0, 9])
        result = repair_generated(self.grid, "right right jump", scorer, self.oracle)
        self.assertEqual(result["plan"], "right right")
        self.assertEqual(result["stop_reason"], "goal_reached_in_prefix")
        self.assertEqual(scorer.positions, [])

    def test_constraints_and_restart_are_distinct(self):
        scorer = Scores([9, 0, 0, 5])  # up is preferred but illegal on the top row
        raw = repair_generated(self.grid, "jump", scorer, self.oracle, "prefix_raw", 3)
        legal = repair_generated(self.grid, "jump", scorer, self.oracle, "prefix_legal", 3)
        restart = repair_generated(self.grid, "right jump", scorer, self.oracle,
                                   "restart_visit", 3)
        self.assertEqual(raw["stop_reason"], "illegal_proposal")
        self.assertEqual(legal["plan"], "right right")
        self.assertEqual(restart["kept_prefix"], "right")
        self.assertEqual(restart["plan"], "right right")
        self.assertEqual(restart["policy_decisions"], 2)

    def test_visit_rule_can_backtrack_and_budget_counts_prefix(self):
        grid = Grid(3, 3, ((0, 2), (1, 1)), (0, 0), (2, 2))
        scorer = Scores([0, 0, 9, 10])
        result = repair_generated(grid, "right right", scorer, self.oracle,
                                  "prefix_visit", max_steps=2)
        self.assertEqual(result["kept_prefix"], "right")
        self.assertEqual(result["plan"], "right left")
        self.assertEqual(result["stop_reason"], "action_budget")
        self.assertTrue(all(step["next_position"] is not None for step in result["policy_trace"]))

    def test_prefix_over_budget_and_nonfinite_policy_scores_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "prefix exceeds"):
            repair_generated(self.grid, "right left jump", Scores([0, 0, 0, 9]),
                             self.oracle, max_steps=1)
        with self.assertRaisesRegex(ValueError, "finite action scores"):
            repair_generated(self.grid, "jump", Scores([float("nan"), 0, 0, 1]),
                             self.oracle)

    def test_generated_source_and_metrics_use_same_initial_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "data"
            dataset.mkdir()
            row = {"id": "test_id-0", "grid": self.grid.__dict__, "candidate": "right jump"}
            write_jsonl(dataset / "test_id.jsonl", [row])
            trace = root / "pipeline_traces.jsonl"
            source = {"grid": self.grid.__dict__, "initial_candidate": "right jump",
                      "modes": {"direct": {"candidate": "right jump", "model_calls": 1,
                                           "truth": "invalid | step=2 | reason=invalid_action_token"},
                                "executor_one": {"candidate": "up", "model_calls": 2,
                                                 "truth": "invalid | step=1 | reason=out_of_bounds"},
                                "executor_three": {"candidate": "", "model_calls": 4,
                                                   "truth": "invalid | step=1 | reason=goal_not_reached"}}}
            write_jsonl(trace, [source])
            paired = load_generated_candidates(dataset, "test_id", trace, self.oracle)
            metrics, records = evaluate_generated(paired, Scores([0, 0, 0, 9]), self.oracle, 3)
            self.assertEqual(metrics["direct"]["success_rate"], 0)
            self.assertEqual(metrics["prefix_visit"]["success_rate"], 1)
            self.assertEqual(metrics["restart_visit"]["success_rate"], 1)
            self.assertEqual(metrics["prefix_visit_heuristic"]["mean_policy_decisions"], 0)
            self.assertEqual(records[0]["methods"]["prefix_visit"]["kept_prefix"], "right")
            out = root / "out"
            report = save_result(out, "test_id", trace, dataset, "policy-hash", "executor-hash",
                                 3, metrics, records)
            self.assertEqual(report["n"], 1)
            self.assertEqual(len((out / "traces.jsonl").read_text().splitlines()), 1)
            with self.assertRaises(FileExistsError):
                save_result(out, "test_id", trace, dataset, "policy-hash", "executor-hash",
                            3, metrics, records)
            source["modes"]["direct"]["candidate"] = "right right"
            write_jsonl(trace, [source])
            with self.assertRaises(ValueError):
                load_generated_candidates(dataset, "test_id", trace, self.oracle)


if __name__ == "__main__":
    unittest.main()
