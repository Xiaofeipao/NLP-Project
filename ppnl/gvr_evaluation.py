"""Independent candidate and end-to-end GVR evaluation."""
from __future__ import annotations

from collections import defaultdict
from collections import Counter
from pathlib import Path

from .core import Grid, parse_verdict
from .data import task_index, write_json, write_jsonl
from .evaluation import plan_metrics, verification_metrics
from .gvr_harness import create_gvr_session, run_gvr_session, score_gvr_session


def evaluation_subsets(rows):
    """Honor the fixed 40/100/60 dev quota; test rows serve all tasks."""
    tasks = {r.get("task") for r in rows}
    if tasks == {None}:
        if len(rows) == 200 and sum(r["valid"] for r in rows) != 100:
            raise ValueError("Candidate test split must contain 100 valid and 100 invalid rows")
        verify = list(rows)
        repair = [r for r in rows if not r["valid"]]
        generate = list(rows)
    elif tasks == {"generation", "verification", "repair"}:
        if len(rows) == 200 and Counter(r["task"] for r in rows) != {
                "generation": 40, "verification": 100, "repair": 60}:
            raise ValueError("Development split violates fixed G/V/R quota")
        verify = [r for r in rows if r["task"] == "verification"]
        repair = [r for r in rows if r["task"] == "repair"]
        generate = [r for r in rows if r["task"] == "generation"]
        if any(r["valid"] for r in repair):
            raise ValueError("Development repair subset contains valid candidates")
    else:
        raise ValueError("Expected a complete dev split or a candidate test split")
    if not verify or not repair or not generate:
        raise ValueError("Empty GVR evaluation subset")
    return verify, repair, generate


def repeated_action_runs(plan):
    longest = current = 0
    previous = None
    for token in plan.split():
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    return longest


def dev_model_diagnostics(model, dataset_root, check=lambda: None):
    """Check each loaded adapter on its own frozen development task."""
    tasks = {task: task_index(dataset_root, "dev", task)
             for task in ("generation", "verification", "repair")}
    outputs = {task: [model.predict(task, Grid.from_dict(row["grid"]), row.get("candidate", ""),
                                    row["label"] if task == "repair" else "", check) for row in rows]
               for task, rows in tasks.items()}
    verification = verification_metrics(tasks["verification"], outputs["verification"])
    counts = Counter("malformed" if parsed is None else "valid" if parsed[0] else "invalid"
                     for parsed in map(parse_verdict, outputs["verification"]))
    verification["predicted_label_counts"] = dict(counts)
    verification["all_valid_warning"] = counts["valid"] == len(tasks["verification"])
    result = {"verification": verification}
    for task in ("generation", "repair"):
        metrics = plan_metrics([Grid.from_dict(row["grid"]) for row in tasks[task]], outputs[task])
        metrics["repeated_action_run_ge_3"] = sum(repeated_action_runs(plan) >= 3 for plan in outputs[task])
        result[task] = metrics
    return result


def evaluate_gvr(model, rows, output, identity, oracle, check=lambda: None):
    if not rows:
        raise ValueError("No evaluation rows")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh evaluation output directory")
    out.mkdir(parents=True, exist_ok=True)
    verify_rows, repair_rows, generation_rows = evaluation_subsets(rows)
    verify_grids = [Grid.from_dict(r["grid"]) for r in verify_rows]
    repair_grids = [Grid.from_dict(r["grid"]) for r in repair_rows]
    verdicts = [model.predict("verification", g, r["candidate"], check=check)
                for g, r in zip(verify_grids, verify_rows)]
    repairs = [model.predict("repair", g, r["candidate"], r["label"], check)
               for g, r in zip(repair_grids, repair_rows)]
    for grid, plan in zip(repair_grids, repairs):
        oracle.check(grid, plan)
    metrics = {"verification": verification_metrics(verify_rows, verdicts),
               "repair": plan_metrics(repair_grids, repairs), "by_error_type": {}}
    metrics["repair"]["repair_success_rate"] = metrics["repair"]["success_rate"]
    parsed = [parse_verdict(value) for value in verdicts]
    label_counts = Counter("malformed" if value is None else "valid" if value[0] else "invalid"
                           for value in parsed)
    metrics["verification"]["predicted_label_counts"] = dict(label_counts)
    metrics["verification"]["all_valid_warning"] = label_counts["valid"] == len(verdicts)
    verify_by_reason, repair_by_reason = defaultdict(list), defaultdict(list)
    for row, prediction in zip(verify_rows, verdicts):
        verify_by_reason[row["reason"] or "valid"].append((row, prediction))
    for row, plan in zip(repair_rows, repairs):
        repair_by_reason[row["reason"] or "valid"].append((row, plan))
    for reason in sorted(verify_by_reason.keys() | repair_by_reason.keys()):
        verification = verify_by_reason[reason]
        repair = repair_by_reason[reason]
        metrics["by_error_type"][reason] = {
            "verification": verification_metrics([r for r, _ in verification], [p for _, p in verification]),
            "repair": plan_metrics([Grid.from_dict(r["grid"]) for r, _ in repair], [p for _, p in repair])}
    verification_by_id = dict(zip((r["id"] for r in verify_rows), verdicts))
    repair_by_id = dict(zip((r["id"] for r in repair_rows), repairs))
    candidate_traces = [{"id": r["id"], "task": r.get("task", "candidate"), "reason": r["reason"],
                         "candidate": r["candidate"], "gold_feedback": r["label"],
                         "model_feedback": verification_by_id.get(r["id"]),
                         "repair": repair_by_id.get(r["id"])}
                        for r in rows if "candidate" in r]
    write_jsonl(out / "candidate_traces.jsonl", candidate_traces)

    unique = {}
    for row in generation_rows:
        grid = Grid.from_dict(row["grid"])
        unique.setdefault((grid.layout_id, grid.start, grid.goal), (row, grid))
    traces = []
    for row, grid in unique.values():
        record = {"source_id": row["source_id"], "grid": row["grid"], "modes": {}}
        for mode in ("direct", "self", "executor_one", "executor_three"):
            trial = out / "sessions" / row["id"] / mode
            trial.parent.mkdir(parents=True, exist_ok=True)
            create_gvr_session(trial, grid, mode, identity)
            run_gvr_session(trial, model, identity, check)
            record["modes"][mode] = score_gvr_session(trial, oracle)
        # The final candidates can differ, but all four runs must start from
        # the same deterministic Generate output for a paired comparison.
        from .harness import read_session
        initials = []
        for mode in ("direct", "self", "executor_one", "executor_three"):
            _, state, _, _ = read_session(out / "sessions" / row["id"] / mode)
            initials.append(state["candidate_history"][0])
        if len(set(initials)) != 1:
            raise ValueError("Generate outputs differed across paired modes")
        record["initial_candidate"] = initials[0]
        traces.append(record)
    for mode in ("direct", "self", "executor_one", "executor_three"):
        plans = [r["modes"][mode]["candidate"] for r in traces]
        scores = plan_metrics([Grid.from_dict(r["grid"]) for r in traces], plans)
        scores["mean_model_calls"] = sum(r["modes"][mode]["model_calls"] for r in traces) / len(traces)
        scores["mean_executor_calls"] = sum(r["modes"][mode]["executor_calls"] for r in traces) / len(traces)
        scores["mean_executor_seconds"] = sum(r["modes"][mode]["executor_seconds"] for r in traces) / len(traces)
        scores["mean_input_tokens"] = sum(r["modes"][mode]["input_tokens"] for r in traces) / len(traces)
        scores["mean_output_tokens"] = sum(r["modes"][mode]["output_tokens"] for r in traces) / len(traces)
        scores["mean_active_seconds"] = sum(r["modes"][mode]["active_seconds"] for r in traces) / len(traces)
        scores["stop_reason_counts"] = dict(Counter(r["modes"][mode]["stop_reason"] for r in traces))
        scores["repeated_action_run_ge_3"] = sum(repeated_action_runs(p) >= 3 for p in plans)
        scores["false_valid_accepts"] = sum(r["modes"][mode]["false_valid_accept"] for r in traces)
        metrics[mode] = scores
    metrics["subset_counts"] = {"verification": len(verify_rows), "repair": len(repair_rows),
                                "generation": len(traces)}
    metrics["definitions"] = {"self_feedback": "model output only; truth used after stop for scoring",
                              "paired_id_ood": "generation grids are paired; candidate corruption differs",
                              "failure_denominator": "all final candidates including malformed output"}
    metrics["official_executor_sha256"] = oracle.sha256
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "pipeline_traces.jsonl", traces)
    cases = [r for r in traces if r["modes"]["direct"]["candidate"] != r["modes"]["executor_one"]["candidate"]]
    write_json(out / "case_studies.json", cases[:10])
    return metrics
