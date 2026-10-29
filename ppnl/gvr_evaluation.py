"""Independent candidate and end-to-end GVR evaluation."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .core import Grid
from .data import write_json, write_jsonl
from .evaluation import plan_metrics, verification_metrics
from .gvr_harness import create_gvr_session, run_gvr_session, score_gvr_session


def evaluate_gvr(model, rows, output, identity, oracle, check=lambda: None):
    if not rows:
        raise ValueError("No evaluation rows")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh evaluation output directory")
    out.mkdir(parents=True, exist_ok=True)
    candidates = [r for r in rows if "candidate" in r]
    grids = [Grid.from_dict(r["grid"]) for r in candidates]
    verdicts = [model.predict("verification", g, r["candidate"], check=check) for g, r in zip(grids, candidates)]
    invalid = [i for i, r in enumerate(candidates) if not r["valid"]]
    repairs = [model.predict("repair", grids[i], candidates[i]["candidate"], candidates[i]["label"], check)
               for i in invalid]
    for i, plan in zip(invalid, repairs):
        oracle.check(grids[i], plan)
    metrics = {"verification": verification_metrics(candidates, verdicts),
               "repair": plan_metrics([grids[i] for i in invalid], repairs), "by_error_type": {}}
    metrics["repair"]["repair_success_rate"] = metrics["repair"]["success_rate"]
    repair_by_index = dict(zip(invalid, repairs))
    groups = defaultdict(list)
    for i, row in enumerate(candidates):
        groups[row["reason"] or "valid"].append(i)
    for reason, ids in sorted(groups.items()):
        bad = [i for i in ids if i in repair_by_index]
        metrics["by_error_type"][reason] = {
            "verification": verification_metrics([candidates[i] for i in ids], [verdicts[i] for i in ids]),
            "repair": plan_metrics([grids[i] for i in bad], [repair_by_index[i] for i in bad])}
    candidate_traces = [{"id": r["id"], "reason": r["reason"], "candidate": r["candidate"],
                         "gold_feedback": r["label"], "model_feedback": verdicts[i],
                         "repair": repair_by_index.get(i)} for i, r in enumerate(candidates)]
    write_jsonl(out / "candidate_traces.jsonl", candidate_traces)

    unique = {}
    for row in rows:
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
        scores["mean_input_tokens"] = sum(r["modes"][mode]["input_tokens"] for r in traces) / len(traces)
        scores["mean_output_tokens"] = sum(r["modes"][mode]["output_tokens"] for r in traces) / len(traces)
        scores["mean_active_seconds"] = sum(r["modes"][mode]["active_seconds"] for r in traces) / len(traces)
        metrics[mode] = scores
    metrics["definitions"] = {"self_feedback": "model output only; truth used after stop for scoring",
                              "paired_id_ood": "generation grids are paired; candidate corruption differs",
                              "failure_denominator": "all final candidates including malformed output"}
    metrics["official_executor_sha256"] = oracle.sha256
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "pipeline_traces.jsonl", traces)
    cases = [r for r in traces if r["modes"]["direct"]["candidate"] != r["modes"]["executor_one"]["candidate"]]
    write_json(out / "case_studies.json", cases[:10])
    return metrics
