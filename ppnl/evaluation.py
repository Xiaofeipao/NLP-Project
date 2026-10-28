"""Candidate-level verification/repair and environment-level end-to-end evaluation."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .core import Grid, execute, parse_verdict, shortest_path
from .data import write_json, write_jsonl
from .prompts import prompt


def mean(values):
    return sum(values) / len(values) if values else None


def plan_metrics(grids, plans):
    successes, feasible, optimal, efficiency = [], [], [], []
    for grid, plan in zip(grids, plans):
        verdict = execute(grid, plan)
        best = shortest_path(grid)
        successes.append(verdict.valid)
        feasible.append(verdict.feasible)
        optimal.append(verdict.valid and len(plan.split()) == len(best))
        if verdict.valid:
            efficiency.append(len(best) / len(plan.split()))
    return {"n": len(plans), "success_rate": mean(successes), "feasibility": mean(feasible),
            "optimality": mean(optimal), "path_efficiency_on_success": mean(efficiency)}


def verification_metrics(rows, predictions):
    parsed = [parse_verdict(p) for p in predictions]
    truth = [r["valid"] for r in rows]
    f1 = []
    for cls in (False, True):
        tp = sum(t == cls and p is not None and p[0] == cls for t, p in zip(truth, parsed))
        fp = sum(t != cls and p is not None and p[0] == cls for t, p in zip(truth, parsed))
        fn = sum(t == cls and (p is None or p[0] != cls) for t, p in zip(truth, parsed))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    invalid = [(r, p) for r, p in zip(rows, parsed) if not r["valid"]]
    return {"n": len(rows), "macro_f1": mean(f1),
            "format_accuracy": mean([p is not None for p in parsed]),
            "first_error_localization_accuracy": mean([p is not None and p[0] is False and p[1] == r["step"]
                                                        for r, p in invalid]),
            "failure_type_accuracy": mean([p is not None and p[0] is False and p[2] == r["reason"]
                                            for r, p in invalid]),
            "joint_error_accuracy": mean([p == (False, r["step"], r["reason"]) for r, p in invalid])}


def diagnose(grid, plan):
    verdict = execute(grid, plan)
    return {"plan": plan, "verdict": verdict.label(), "trace": verdict.trace}


def evaluate(runner, rows, output, check=lambda: None, full=False, rounds=3, oracle=None):
    if not rows:
        raise ValueError("No evaluation examples")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a fresh evaluation output directory: {out}")
    grids = [Grid.from_dict(r["grid"]) for r in rows]
    predicted_verdicts = runner.predict([prompt("verification", g, r["candidate"])
                                        for g, r in zip(grids, rows)], check)
    invalid_indices = [i for i, r in enumerate(rows) if not r["valid"]]
    repaired = runner.predict([prompt("repair", grids[i], rows[i]["candidate"], rows[i]["label"])
                               for i in invalid_indices], check)
    repaired_by_index = dict(zip(invalid_indices, repaired))
    metrics = {"verification": verification_metrics(rows, predicted_verdicts),
               "repair": plan_metrics([grids[i] for i in invalid_indices], repaired)}
    metrics["repair"]["repair_success_rate"] = metrics["repair"]["success_rate"]
    by_reason = defaultdict(list)
    for i, row in enumerate(rows):
        by_reason[row["reason"] or "valid"].append(i)
    metrics["by_failure_type"] = {}
    for reason, indices in sorted(by_reason.items()):
        bad = [i for i in indices if i in repaired_by_index]
        metrics["by_failure_type"][reason] = {
            "verification": verification_metrics([rows[i] for i in indices], [predicted_verdicts[i] for i in indices]),
            "repair": plan_metrics([grids[i] for i in bad], [repaired_by_index[i] for i in bad])}
    records = []
    for i, row in enumerate(rows):
        record = {"id": row["id"], "layout_id": row["layout_id"], "grid": row["grid"],
                  "candidate": diagnose(grids[i], row["candidate"]), "gold_verdict": row["label"],
                  "predicted_verdict": predicted_verdicts[i], "edits": row.get("edits", [])}
        if i in repaired_by_index:
            record["repair"] = diagnose(grids[i], repaired_by_index[i])
        records.append(record)

    unique = {}
    for row, grid in zip(rows, grids):
        unique.setdefault((grid.layout_id, grid.start, grid.goal), (row, grid))
    envs = list(unique.values())
    env_grids = [g for _, g in envs]
    generated = runner.predict([prompt("generation", g) for g in env_grids], check)
    metrics["direct_generation"] = plan_metrics(env_grids, generated)
    pipeline_records = [{"source_id": r["source_id"], "grid": r["grid"], "direct": diagnose(g, p)}
                        for (r, g), p in zip(envs, generated)]
    if full:
        if rounds < 2:
            raise ValueError("Improvement requires at least two repair rounds")
        feedbacks = runner.predict([prompt("verification", g, p) for g, p in zip(env_grids, generated)], check)
        self_plans = list(generated)
        self_indices = [i for i, feedback in enumerate(feedbacks) if parse_verdict(feedback) != (True, None, None)]
        outputs = runner.predict([prompt("repair", env_grids[i], generated[i], feedbacks[i]) for i in self_indices], check)
        for i, plan in zip(self_indices, outputs):
            self_plans[i] = plan
        metrics["self_verification_repair"] = plan_metrics(env_grids, self_plans)
        executor_plans = list(generated)
        for i in range(len(envs)):
            pipeline_records[i].update(self_feedback=feedbacks[i],
                                       self_repair=diagnose(env_grids[i], self_plans[i]), iterations=[])
        for iteration in range(rounds):
            invalid = [i for i, (g, p) in enumerate(zip(env_grids, executor_plans)) if not execute(g, p).valid]
            outputs = runner.predict([prompt("repair", env_grids[i], executor_plans[i],
                                             execute(env_grids[i], executor_plans[i]).label()) for i in invalid], check)
            for i, plan in zip(invalid, outputs):
                previous = executor_plans[i]
                executor_plans[i] = plan
                pipeline_records[i]["iterations"].append({"round": iteration + 1,
                    "feedback": execute(env_grids[i], previous).label(), "result": diagnose(env_grids[i], plan)})
            if iteration == 0:
                first_executor_plans = list(executor_plans)
                metrics["executor_feedback_repair"] = plan_metrics(env_grids, executor_plans)
                # Focused ablation removes repeat repairs, same checkpoint and same first pass.
                metrics["ablation_one_round"] = dict(metrics["executor_feedback_repair"])
        metrics["improved_iterative_repair"] = plan_metrics(env_grids, executor_plans)
        grouped = defaultdict(list)
        for i, (g, p) in enumerate(zip(env_grids, generated)):
            grouped[execute(g, p).reason or "valid"].append(i)
        metrics["pipeline_by_initial_failure"] = {
            reason: {name: plan_metrics([env_grids[i] for i in indices], [plans[i] for i in indices])
                     for name, plans in (("direct", generated), ("self", self_plans),
                                         ("executor_one_round", first_executor_plans), ("iterative", executor_plans))}
            for reason, indices in grouped.items()}
        # Also evaluate the improvement on supplied single/double-error candidates.
        candidate_plans = [r["candidate"] for r in rows]
        for iteration in range(rounds):
            bad = [i for i, (g, p) in enumerate(zip(grids, candidate_plans)) if not execute(g, p).valid]
            outputs = runner.predict([prompt("repair", grids[i], candidate_plans[i], execute(grids[i], candidate_plans[i]).label())
                                      for i in bad], check)
            for i, plan in zip(bad, outputs):
                candidate_plans[i] = plan
                records[i].setdefault("iterative_repair", []).append({"round": iteration + 1,
                                                                    "result": diagnose(grids[i], plan)})
        metrics["improved_candidate_repair"] = plan_metrics([grids[i] for i in invalid_indices],
                                                            [candidate_plans[i] for i in invalid_indices])
        metrics["candidate_ablation_one_round"] = dict(metrics["repair"])
        for reason, indices in by_reason.items():
            bad = [i for i in indices if not rows[i]["valid"]]
            metrics["by_failure_type"][reason]["improved_candidate_repair"] = plan_metrics(
                [grids[i] for i in bad], [candidate_plans[i] for i in bad])
    # Actual generated candidates are independent of corruption: shared paired grids
    # intentionally have identical generation metrics across ID and OOD sets.
    metrics["definitions"] = {"feasibility": "all tokens legal, no obstacle or boundary violation; goal not required",
                              "optimality": "successful and action count equals BFS shortest length, divided by all plans",
                              "first_error_localization": "correct invalid prediction and step / all invalid candidates",
                              "macro_f1": "fixed valid/invalid classes; malformed predictions count as false negatives",
                              "e2e": "success_rate over unique start/goal environments for each pipeline",
                              "paired_sets": "ID/OOD differ in supplied candidates; generation uses the same held-out grids"}
    if oracle is None:
        raise ValueError("Official oracle required for reported model evaluation")

    def check_plans(value, grid):
        if isinstance(value, dict):
            if "plan" in value:
                oracle.check(grid, value["plan"])
            for child in value.values():
                check_plans(child, grid)
        elif isinstance(value, list):
            for child in value:
                check_plans(child, grid)

    for record in records + pipeline_records:
        check_plans(record, Grid.from_dict(record["grid"]))
    metrics["official_executor_sha256"] = oracle.sha256
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "candidate_traces.jsonl", records)
    write_jsonl(out / "pipeline_traces.jsonl", pipeline_records)
    # Prioritize real failures for later case studies, with no fabricated outcomes.
    cases = [r for r in records if r["predicted_verdict"].strip() != r["gold_verdict"] or
             ("repair" in r and r["repair"]["verdict"] != "valid")]
    cases.extend(r for r in records if "repair" in r and r not in cases)
    write_json(out / "case_studies.json", cases[:max(2, min(10, len(cases)))])
    return metrics
