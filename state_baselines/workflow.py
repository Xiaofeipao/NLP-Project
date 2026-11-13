"""Shared one-attempt repair driven by the supplied first-error diagnosis."""

from collections import Counter

from ppnl.core import execute, parse_verdict
from ppnl.evaluation import plan_metrics, verification_metrics
from ppnl.local_finish import _bridge
from ppnl.prompts import prompt


def repair(grid, candidate, feedback):
    verdict = parse_verdict(feedback)
    if verdict is None:
        return candidate, "malformed_feedback"
    valid, step, reason = verdict
    if valid:
        return candidate, "accepted"
    actions = candidate.split()
    if (reason == "goal_not_reached" and step != len(actions) + 1
            or reason != "goal_not_reached" and step > len(actions)):
        return candidate, "bad_step"
    prefix = actions[:step - 1]
    trace = execute(grid, " ".join(prefix))
    if not trace.feasible:
        return candidate, "illegal_reported_prefix"
    positions = [grid.start] + [item["to"] for item in trace.trace if item["reason"] is None]
    for keep in range(len(prefix), max(-1, len(prefix) - 3), -1):
        suffix = _bridge(grid, positions[keep], grid.goal, 3)
        if suffix is not None:
            return " ".join([*prefix[:keep], *suffix]), "repaired"
    return candidate, "unresolved"


def verify(runner, grids, plans, check):
    return runner.predict([prompt("verification", grid, plan)
                           for grid, plan in zip(grids, plans, strict=True)], check)


def verifier_report(rows, predictions):
    parsed = [parse_verdict(value) for value in predictions]
    groups = sorted({row["reason"] or "valid" for row in rows})
    return {
        **verification_metrics(rows, predictions),
        "predicted_labels": dict(Counter("malformed" if p is None else p[2] or "valid" for p in parsed)),
        "by_error": {
            reason: verification_metrics(
                [rows[i] for i in indices], [predictions[i] for i in indices])
            for reason in groups
            if (indices := [i for i, row in enumerate(rows) if (row["reason"] or "valid") == reason])
        },
    }


def compare(runner, grids, plans, oracle, check):
    predictions = verify(runner, grids, plans, check)
    records = []
    for grid, plan, prediction in zip(grids, plans, predictions, strict=True):
        truth = oracle.check(grid, plan)
        record = {"direct": plan, "gold": truth.label(), "reason": truth.reason}
        for method, feedback in (("self", prediction), ("executor", truth.label())):
            revised, status = repair(grid, plan, feedback)
            record[method] = {"feedback": feedback, "plan": revised, "status": status,
                              "final_verdict": oracle.check(grid, revised).label()}
        records.append(record)
    invalid = [i for i, row in enumerate(records) if row["reason"] is not None]
    metrics = {"direct": plan_metrics(grids, plans)}
    for method in ("self", "executor"):
        revised = [row[method]["plan"] for row in records]
        metrics[method] = {
            **plan_metrics(grids, revised),
            "invalid_candidate_repair": plan_metrics([grids[i] for i in invalid], [revised[i] for i in invalid]),
            "outcomes": dict(Counter(row[method]["status"] for row in records)),
        }
    metrics["verification"] = verifier_report(
        [{"valid": (parsed := parse_verdict(row["gold"]))[0], "step": parsed[1], "reason": parsed[2]}
         for row in records], predictions)
    return metrics, records
