"""Recompute GVR metrics from frozen rows and saved predictions, without a GPU."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path

from .core import Grid, execute, parse_verdict, shortest_path
from .data import read_jsonl, write_json
from .evaluation import plan_metrics, verification_metrics
from .gvr_evaluation import evaluation_subsets, repeated_action_runs


def _unique_by_id(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} in saved traces")
    return result


def _grid_key(row):
    grid = Grid.from_dict(row["grid"])
    return grid.layout_id, grid.start, grid.goal


def recompute_metrics(dataset_root, split, evaluation_dir, output, oracle=None):
    if split not in ("dev", "test_id", "test_ood"):
        raise ValueError("Unsupported GVR split")
    dataset = Path(dataset_root) / f"{split}.jsonl"
    directory = Path(evaluation_dir)
    candidate_file = directory / "candidate_traces.jsonl"
    pipeline_file = directory / "pipeline_traces.jsonl"
    rows = read_jsonl(dataset)
    verify_rows, repair_rows, generation_rows = evaluation_subsets(rows)
    candidates = _unique_by_id(read_jsonl(candidate_file), "id")
    expected_candidate_ids = {row["id"] for row in rows if "candidate" in row}
    if set(candidates) != expected_candidate_ids:
        raise ValueError("Saved candidate IDs differ from frozen dataset")
    for row in rows:
        if "candidate" in row:
            trace = candidates[row["id"]]
            if trace["candidate"] != row["candidate"] or trace["gold_feedback"] != row["label"]:
                raise ValueError(f"Candidate or gold feedback changed: {row['id']}")
    verify_predictions = [candidates[row["id"]]["model_feedback"] for row in verify_rows]
    repair_predictions = [candidates[row["id"]]["repair"] for row in repair_rows]
    if any(not isinstance(value, str) for value in verify_predictions + repair_predictions):
        raise ValueError("Missing saved Verify or Repair prediction")
    repair_grids = [Grid.from_dict(row["grid"]) for row in repair_rows]
    if oracle:
        for grid, plan in zip(repair_grids, repair_predictions):
            oracle.check(grid, plan)
    verification = verification_metrics(verify_rows, verify_predictions)
    labels = Counter("malformed" if parsed is None else "valid" if parsed[0] else "invalid"
                     for parsed in map(parse_verdict, verify_predictions))
    verification["predicted_label_counts"] = dict(labels)
    verification["all_valid_warning"] = labels["valid"] == len(verify_predictions)
    repair = plan_metrics(repair_grids, repair_predictions)
    repair["repair_success_rate"] = repair["success_rate"]
    verify_by_reason, repair_by_reason = defaultdict(list), defaultdict(list)
    for row, prediction in zip(verify_rows, verify_predictions):
        verify_by_reason[row["reason"] or "valid"].append((row, prediction))
    for row, prediction in zip(repair_rows, repair_predictions):
        repair_by_reason[row["reason"] or "valid"].append((row, prediction))
    by_error_type = {}
    for reason in sorted(verify_by_reason.keys() | repair_by_reason.keys()):
        verification_group = verify_by_reason[reason]
        repair_group = repair_by_reason[reason]
        by_error_type[reason] = {
            "verification": verification_metrics([row for row, _ in verification_group],
                                                  [prediction for _, prediction in verification_group]),
            "repair": plan_metrics([Grid.from_dict(row["grid"]) for row, _ in repair_group],
                                   [prediction for _, prediction in repair_group])}

    pipeline_rows = read_jsonl(pipeline_file)
    # Source IDs can repeat across different start/goal environments; grid keys cannot.
    pipelines = {_grid_key(row): row for row in pipeline_rows}
    if len(pipelines) != len(pipeline_rows):
        raise ValueError("Duplicate environment in saved pipeline traces")
    if not set(pipelines) <= {_grid_key(row) for row in rows}:
        raise ValueError("Saved pipeline contains an environment outside the frozen split")
    selected_keys = {_grid_key(row) for row in generation_rows}
    if not selected_keys <= set(pipelines):
        raise ValueError("Saved pipeline traces omit generation environments")
    # The original evaluator deduplicated environments before generation.
    selected = [pipelines[key] for key in dict.fromkeys(_grid_key(row) for row in generation_rows)]
    grids = [Grid.from_dict(row["grid"]) for row in selected]
    for row, grid in zip(selected, grids):
        if row["initial_candidate"] != row["modes"]["direct"]["candidate"]:
            raise ValueError("Saved direct candidate differs from paired initial candidate")
        best = shortest_path(grid)
        for mode in ("direct", "self", "executor_one", "executor_three"):
            record = row["modes"][mode]
            plan = record["candidate"]
            verdict = oracle.check(grid, plan) if oracle else execute(grid, plan)
            expected = {"truth": verdict.label(), "success": verdict.valid,
                        "feasible": verdict.feasible,
                        "optimal": verdict.valid and len(plan.split()) == len(best)}
            if any(record.get(field) != value for field, value in expected.items()):
                raise ValueError(f"Saved {mode} score disagrees with executor")
    metrics = {"verification": verification, "repair": repair, "by_error_type": by_error_type,
               "subset_counts": {"verification": len(verify_rows), "repair": len(repair_rows),
                                 "generation": len(selected)}}
    for mode in ("direct", "self", "executor_one", "executor_three"):
        plans = [row["modes"][mode]["candidate"] for row in selected]
        scores = plan_metrics(grids, plans)
        scores["stop_reason_counts"] = dict(Counter(row["modes"][mode]["stop_reason"] for row in selected))
        scores["repeated_action_run_ge_3"] = sum(repeated_action_runs(plan) >= 3 for plan in plans)
        scores["false_valid_accepts"] = sum(
            mode == "self" and row["modes"][mode]["stop_reason"] == "feedback_valid" and
            row["modes"][mode]["truth"] != "valid" for row in selected)
        for field in ("model_calls", "executor_calls", "input_tokens", "output_tokens", "active_seconds",
                      "executor_seconds"):
            values = [row["modes"][mode][field] for row in selected if field in row["modes"][mode]]
            if len(values) == len(selected):
                scores[f"mean_{field}"] = sum(values) / len(values)
        metrics[mode] = scores
    if oracle:
        metrics["official_executor_sha256"] = oracle.sha256
    metrics["source"] = {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                         "candidate_traces_sha256": hashlib.sha256(candidate_file.read_bytes()).hexdigest(),
                         "pipeline_traces_sha256": hashlib.sha256(pipeline_file.read_bytes()).hexdigest(),
                         "method": "CPU recomputation from saved predictions; no model calls"}
    write_json(output, metrics)
    return metrics
