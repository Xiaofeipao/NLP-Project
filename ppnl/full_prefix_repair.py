"""Keep an executor-validated path prefix and repair from its failure position.

This module uses the original full fine-tuning branch's saved generation traces
and existing one-action policy. It neither trains a new model nor searches a
graph at inference time.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

from .core import ACTIONS, Grid
from .data import read_jsonl, write_json, write_jsonl
from .evaluation import plan_metrics
from .policy import neighbors
from .prefix_state_repair import repair_generated as repair_from_prefix


def _key(grid):
    return grid.layout_id, grid.start, grid.goal


def load_generated_candidates(dataset_root, split, trace_path, oracle):
    """Match original full-model direct outputs to the frozen environment set."""
    if split not in ("dev", "test_id", "test_ood"):
        raise ValueError("Expected dev, test_id, or test_ood")
    rows = read_jsonl(Path(dataset_root) / f"{split}.jsonl")
    if split == "dev":
        rows = [row for row in rows if row["task"] == "verification"]
    expected = {}
    for row in rows:
        grid = Grid.from_dict(row["grid"])
        expected.setdefault(_key(grid), grid)
    if not expected:
        raise ValueError("No environments in split")
    saved = {}
    for row in read_jsonl(trace_path):
        grid = Grid.from_dict(row["grid"])
        key = _key(grid)
        if key in saved:
            raise ValueError("Duplicate environment in generated traces")
        saved[key] = row
    if set(saved) != set(expected):
        raise ValueError("Generated trace environments differ from frozen split")
    result = []
    for key, grid in expected.items():
        row = saved[key]
        direct = row.get("direct")
        if not isinstance(direct, dict) or not isinstance(direct.get("plan"), str):
            raise ValueError("Expected original full-model direct plan")
        candidate = direct["plan"]
        if direct.get("verdict") != oracle.check(grid, candidate).label():
            raise ValueError("Recorded direct verdict disagrees with official executor")
        result.append((grid, candidate, row.get("source_id")))
    return result


def repair_generated(grid, candidate, scorer, oracle, max_actions=36, check=lambda: None):
    """Adapt the shared prefix/visit rollout to the historical full-model trace."""
    if not isinstance(candidate, str):
        raise TypeError("Generated candidate must be text")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    result = repair_from_prefix(grid, candidate, scorer, oracle, "prefix_visit",
                                max_actions, check)

    def endpoint(plan):
        position = grid.start
        for action in plan.split():
            dr, dc = ACTIONS[action]
            position = position[0] + dr, position[1] + dc
        return list(position)

    initially_valid = result["feedback"] == "valid"
    stopped = result["stop_reason"]
    if initially_valid and stopped == "goal_reached_in_prefix":
        stopped = "already_valid_trimmed_to_goal"
    steps = [{"position": step["position"],
              "legal_actions": list(neighbors(grid, tuple(step["position"]))),
              "least_visited_choices": step["allowed_actions"],
              "action": step["action"], "next_position": step["next_position"]}
             for step in result["policy_trace"]]
    record = {"plan": result["plan"],
              "kept_prefix": candidate if stopped == "already_valid" else result["kept_prefix"],
              "failure_position": None if initially_valid else endpoint(result["observed_prefix"]),
              "initial_feedback": result["feedback"],
              "final_feedback": (result["feedback"] if stopped == "already_valid" else
                                 oracle.check(grid, result["plan"]).label()),
              "policy_calls": result["policy_decisions"], "steps": steps, "stop_reason": stopped}
    if stopped == "prefix_exceeds_budget":
        record["budget_position"] = endpoint(result["kept_prefix"])
    return record


def evaluate_generated(paired, scorer, oracle, max_actions=36, check=lambda: None):
    if not paired:
        raise ValueError("No generated candidates")
    grids, baseline, repaired, records = [], [], [], []
    for grid, candidate, source_id in paired:
        check()
        result = repair_generated(grid, candidate, scorer, oracle, max_actions, check)
        grids.append(grid)
        baseline.append(candidate)
        repaired.append(result["plan"])
        records.append({"source_id": source_id, "grid": asdict(grid),
                        "candidate": candidate, **result})
    metrics = {"direct": plan_metrics(grids, baseline),
               "prefix_state_repair": plan_metrics(grids, repaired)}
    metrics["direct"]["over_repair_budget_count"] = sum(
        len(candidate.split()) > max_actions for candidate in baseline)
    metrics["prefix_state_repair"]["prefix_over_budget_count"] = sum(
        row["stop_reason"] == "prefix_exceeds_budget" for row in records)
    metrics["prefix_state_repair"]["mean_policy_calls"] = sum(
        row["policy_calls"] for row in records) / len(records)
    metrics["prefix_state_repair"]["mean_kept_actions"] = sum(
        len(row["kept_prefix"].split()) for row in records) / len(records)
    return metrics, records


def save_result(directory, dataset_root, split, trace_path, policy_sha256,
                oracle_sha256, max_actions, metrics, records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(trace_path)
    dataset = Path(dataset_root) / f"{split}.jsonl"
    report = {"split": split, "n": len(records), "max_total_actions": max_actions,
              "budget_applies_to": "prefix_state_repair",
              "source_trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
              "policy_weights_sha256": policy_sha256, "official_executor_sha256": oracle_sha256,
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "repair_core_sha256": hashlib.sha256(
                  Path(__file__).with_name("prefix_state_repair.py").read_bytes()).hexdigest(),
              "metrics": metrics}
    write_json(directory / "results.json", report)
    write_jsonl(directory / "traces.jsonl", records)
    return report
