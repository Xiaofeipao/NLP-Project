"""Keep an executor-validated path prefix and repair from its failure position.

This module uses the original full fine-tuning branch's saved generation traces
and existing one-action policy. It neither trains a new model nor searches a
graph at inference time.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

from .core import ACTIONS, Grid
from .data import read_jsonl, write_json, write_jsonl
from .eval_harness import repair_prefix
from .evaluation import plan_metrics
from .policy import neighbors


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
    """Keep a physically valid prefix within budget, then repair one step at a time."""
    if not isinstance(candidate, str):
        raise TypeError("Generated candidate must be text")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    original = oracle.check(grid, candidate)
    if original.valid and len(candidate.split()) <= max_actions:
        return {"plan": candidate, "kept_prefix": candidate, "failure_position": None,
                "initial_feedback": original.label(), "final_feedback": original.label(),
                "policy_calls": 0, "steps": [], "stop_reason": "already_valid"}
    prefix, remaining = repair_prefix(grid, candidate)
    failure_position = (None if original.valid else
                        list(remaining.start if remaining is not None else grid.goal))
    plan = list(prefix[:max_actions])
    visits = Counter({grid.start: 1})
    cursor = grid.start
    for action in plan:
        cursor = neighbors(grid, cursor)[action]
        visits[cursor] += 1
    if len(prefix) > max_actions:
        final_plan = " ".join(plan)
        final = oracle.check(grid, final_plan)
        return {"plan": final_plan, "kept_prefix": final_plan,
                "failure_position": failure_position,
                "budget_position": list(cursor), "initial_feedback": original.label(),
                "final_feedback": final.label(), "policy_calls": 0, "steps": [],
                "stop_reason": "prefix_exceeds_budget"}
    if original.valid:
        final_plan = " ".join(plan)
        final = oracle.check(grid, final_plan)
        if not final.valid:
            raise AssertionError("First goal prefix of a valid candidate must remain valid")
        return {"plan": final_plan, "kept_prefix": final_plan, "failure_position": None,
                "initial_feedback": original.label(), "final_feedback": final.label(),
                "policy_calls": 0, "steps": [], "stop_reason": "already_valid_trimmed_to_goal"}
    position = remaining.start if remaining is not None else grid.goal
    steps = []
    if cursor != position:
        raise AssertionError("Executor prefix and failure position disagree")
    stop_reason = "goal_reached_in_prefix" if position == grid.goal else "action_budget"
    while position != grid.goal and len(plan) < max_actions:
        check()
        legal = neighbors(grid, position)
        if not legal:
            stop_reason = "no_legal_action"
            break
        if scorer is None:
            raise ValueError("A learned policy scorer is required for an unfinished path")
        scores = scorer.scores([(grid, position)])[0]
        if len(scores) != len(ACTIONS) or any(not math.isfinite(value) for value in scores):
            raise ValueError("Policy must return four finite action scores")
        least = min(visits[next_position] for next_position in legal.values())
        choices = [action for action, next_position in legal.items() if visits[next_position] == least]
        action = max(choices, key=lambda name: scores[list(ACTIONS).index(name)])
        next_position = legal[action]
        steps.append({"position": list(position), "legal_actions": list(legal),
                      "least_visited_choices": choices, "action": action,
                      "next_position": list(next_position)})
        plan.append(action)
        position = next_position
        visits[position] += 1
        if position == grid.goal:
            stop_reason = "goal_reached"
    final_plan = " ".join(plan)
    final = oracle.check(grid, final_plan)
    return {"plan": final_plan, "kept_prefix": " ".join(prefix),
            "failure_position": failure_position,
            "initial_feedback": original.label(), "final_feedback": final.label(),
            "policy_calls": len(steps), "steps": steps, "stop_reason": stop_reason}


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
              "metrics": metrics}
    write_json(directory / "results.json", report)
    write_jsonl(directory / "traces.jsonl", records)
    return report
