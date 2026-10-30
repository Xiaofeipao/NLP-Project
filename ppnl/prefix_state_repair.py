"""Repair generated plans from their executor-confirmed valid prefix.

The executor supplies the failure and current position. A learned one-action policy
chooses each subsequent move; no shortest-path oracle or graph search is used at
inference time. The restart control uses the same policy and action budget.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

from .core import ACTIONS, Grid, execute
from .data import read_jsonl, write_json, write_jsonl
from .eval_harness import repair_prefix
from .evaluation import plan_metrics
from .policy import neighbors

METHODS = ("prefix_raw", "prefix_legal", "prefix_visit",
           "prefix_visit_heuristic", "restart_visit")
SAVED_BASELINES = ("direct", "executor_one", "executor_three")


class GoalDirectionScores:
    """Symbolic one-step control with the same visit rule and action budget."""

    def scores(self, observations):
        result = []
        for grid, position in observations:
            values = []
            for dr, dc in ACTIONS.values():
                nxt = position[0] + dr, position[1] + dc
                values.append(-abs(nxt[0] - grid.goal[0]) - abs(nxt[1] - grid.goal[1]))
            result.append(values)
        return result


def grid_key(grid):
    return grid.layout_id, grid.start, grid.goal


def load_generated_candidates(dataset_root, split, trace_path, oracle):
    """Pair frozen generation environments with actual saved GVR outputs."""
    if split not in ("dev", "test_id", "test_ood"):
        raise ValueError("Expected dev, test_id, or test_ood")
    rows = read_jsonl(Path(dataset_root) / f"{split}.jsonl")
    if split == "dev":
        rows = [row for row in rows if row["task"] == "generation"]
    expected = {}
    for row in rows:
        grid = Grid.from_dict(row["grid"])
        expected.setdefault(grid_key(grid), grid)
    if not expected:
        raise ValueError("No generation environments in split")
    saved = {}
    for row in read_jsonl(trace_path):
        grid = Grid.from_dict(row["grid"])
        key = grid_key(grid)
        if key in saved:
            raise ValueError("Duplicate generated environment in saved traces")
        saved[key] = row
    if not set(expected) <= set(saved):
        raise ValueError("Saved traces omit generation environments")
    result = []
    for key, grid in expected.items():
        row = saved[key]
        direct = row["modes"]["direct"]
        candidate = row["initial_candidate"]
        if candidate != direct["candidate"]:
            raise ValueError("Direct and initial candidates disagree")
        if not isinstance(candidate, str):
            raise ValueError("Generated candidate must be text")
        if direct["truth"] != oracle.check(grid, candidate).label():
            raise ValueError("Saved direct verdict disagrees with executor")
        for name in SAVED_BASELINES[1:]:
            recorded = row["modes"][name]
            if recorded["truth"] != oracle.check(grid, recorded["candidate"]).label():
                raise ValueError(f"Saved {name} verdict disagrees with executor")
        result.append((grid, row))
    return result


def _rollout(grid, prefix, scorer, mode, max_steps, check):
    position = grid.start
    visits = Counter({position: 1})
    for action in prefix:
        position = neighbors(grid, position)[action]
        visits[position] += 1
    plan = list(prefix)
    trace = []
    if position == grid.goal:
        return " ".join(plan), trace, "goal_reached"
    for _ in range(max(0, max_steps - len(plan))):
        check()
        scores = scorer.scores([(grid, position)])[0]
        if len(scores) != len(ACTIONS) or any(not math.isfinite(score) for score in scores):
            raise ValueError("Policy must return four finite action scores")
        legal = neighbors(grid, position)
        choices = list(ACTIONS) if mode == "prefix_raw" else list(legal)
        if mode in ("prefix_visit", "prefix_visit_heuristic", "restart_visit") and choices:
            least = min(visits[legal[action]] for action in choices)
            choices = [action for action in choices if visits[legal[action]] == least]
        if not choices:
            return " ".join(plan), trace, "no_legal_action"
        action = max(choices, key=lambda item: scores[list(ACTIONS).index(item)])
        target = legal.get(action)
        trace.append({"position": list(position), "action": action,
                      "next_position": list(target) if target else None,
                      "allowed_actions": choices})
        plan.append(action)
        if target is None:
            return " ".join(plan), trace, "illegal_proposal"
        position = target
        visits[position] += 1
        if position == grid.goal:
            return " ".join(plan), trace, "goal_reached"
    return " ".join(plan), trace, "action_budget"


def repair_generated(grid, candidate, scorer, oracle, mode="prefix_visit",
                     max_steps=36, check=lambda: None):
    """Return a complete plan and audit trace under a total-action budget."""
    if mode not in METHODS or isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("Invalid repair method or action budget")
    source = oracle.check(grid, candidate)
    if source.valid:
        return {"plan": candidate, "kept_prefix": candidate, "feedback": source.label(),
                "policy_trace": [], "policy_decisions": 0, "stop_reason": "already_valid"}
    prefix, remaining = repair_prefix(grid, candidate)
    if len(prefix) > max_steps:
        raise ValueError("Executor-valid prefix exceeds total action budget")
    if remaining is None:
        plan, trace, stopped = " ".join(prefix), [], "goal_reached_in_prefix"
    else:
        kept = [] if mode == "restart_visit" else prefix
        chosen_scorer = GoalDirectionScores() if mode == "prefix_visit_heuristic" else scorer
        plan, trace, stopped = _rollout(grid, kept, chosen_scorer, mode, max_steps, check)
    return {"plan": plan, "kept_prefix": " ".join(prefix), "feedback": source.label(),
            "policy_trace": trace,
            "policy_decisions": 0 if mode == "prefix_visit_heuristic" else len(trace),
            "stop_reason": stopped}


def evaluate_generated(paired, scorer, oracle, max_steps=36, check=lambda: None,
                       methods=METHODS):
    """Compare paired repairs of exactly the same model-generated paths."""
    if not paired:
        raise ValueError("No generated environments to evaluate")
    if not methods or any(name not in METHODS for name in methods) or len(set(methods)) != len(methods):
        raise ValueError("Choose distinct supported repair methods")
    if scorer is None and any(name != "prefix_visit_heuristic" for name in methods):
        raise ValueError("Neural repair methods require a policy scorer")
    grids = [grid for grid, _ in paired]
    saved = {name: [row["modes"][name]["candidate"] for _, row in paired]
             for name in SAVED_BASELINES}
    predicted = {name: [] for name in methods}
    records = []
    for grid, row in paired:
        check()
        initial = row["initial_candidate"]
        record = {"grid": asdict(grid), "source_id": row.get("source_id"),
                  "initial_candidate": initial, "initial_feedback": oracle.check(grid, initial).label(),
                  "methods": {}}
        for name in methods:
            result = repair_generated(grid, initial, scorer, oracle, name, max_steps, check)
            result["final_feedback"] = oracle.check(grid, result["plan"]).label()
            record["methods"][name] = result
            predicted[name].append(result["plan"])
        records.append(record)
    metrics = {name: plan_metrics(grids, plans) for name, plans in {**saved, **predicted}.items()}
    for name in SAVED_BASELINES:
        metrics[name]["mean_model_calls"] = sum(
            row["modes"][name]["model_calls"] for _, row in paired) / len(paired)
    for name in methods:
        metrics[name]["mean_policy_decisions"] = sum(
            record["methods"][name]["policy_decisions"] for record in records) / len(records)
        metrics[name]["mean_step_decisions"] = sum(
            len(record["methods"][name]["policy_trace"]) for record in records) / len(records)
    return metrics, records


def save_result(directory, split, trace_path, dataset_root, policy_weights_sha256,
                executor_sha256, max_steps, metrics, records, policy_config_sha256=None):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Choose a fresh evaluation directory")
    directory.mkdir(parents=True)
    source = Path(trace_path)
    dataset = Path(dataset_root) / f"{split}.jsonl"
    report = {"split": split, "n": len(records), "max_total_actions": max_steps,
              "policy_weights_sha256": policy_weights_sha256,
              "policy_config_sha256": policy_config_sha256,
              "executor_sha256": executor_sha256,
              "source_trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "policy_code_sha256": hashlib.sha256(
                  Path(__file__).with_name("policy.py").read_bytes()).hexdigest(),
              "modeling_code_sha256": hashlib.sha256(
                  Path(__file__).with_name("modeling.py").read_bytes()).hexdigest(),
              "definitions": {"prefix": "keep actions before the first failure; discard failing action and suffix",
                              "restart_visit": "same policy and total-action limit, starting at original start",
                              "policy": "one learned action per step; no BFS labels or graph search at inference",
                              "prefix_visit_heuristic": "goal-distance one-step control; no learned scores or search"},
              "metrics": metrics}
    write_json(directory / "results.json", report)
    write_jsonl(directory / "traces.jsonl", records)
    return report
