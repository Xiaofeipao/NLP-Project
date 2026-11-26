"""Recompute the uniform checkpoint's error analysis from saved traces."""

import argparse
import json
from ast import literal_eval
from collections import Counter
from pathlib import Path

from ppnl.core import Grid, execute, parse_verdict
from .protocol import ERROR_SETS, MOVES, ROOT, digest, gold_rows, point, put, read, rows_at

PUBLISHED = ROOT / "artifacts/part1_baseline"
TRAINING = ROOT / "data/coordinate_gcv_review/train.jsonl"


def expected_verdict(grid, cell):
    if cell == "no_coordinate":
        return "invalid_action_token"
    destination = literal_eval(cell)
    if not grid.inside(destination):
        return "out_of_bounds"
    if destination in grid.obstacles:
        return "obstacle_collision"
    return "valid" if destination == grid.goal else "goal_not_reached"


def first_error_case(gold, raw, score, grid):
    predicted = parse_verdict(score["prediction"])
    trace = raw["verification_trace"]
    if predicted is None or not trace:
        return dict(id=gold["id"], gold=gold["label"], prediction=score["prediction"], timing="model_error")
    valid, step, _ = predicted
    timing = "false_valid" if valid else "wrong_type"
    if not valid and step != gold["step"]:
        timing = "early" if step < gold["step"] else "late"
    index = min(gold["step"], step or gold["step"], len(trace)) - 1
    event = trace[index]
    reference = execute(grid, gold["candidate"]).trace[index]
    true_cell = "no_coordinate" if reference["reason"] == "invalid_action_token" else point(reference["to"])
    expected_v = expected_verdict(grid, event["cell"])
    component = ("C" if event["cell"] != true_cell else "") + ("V" if event["verdict"] != expected_v else "")
    return dict(
        id=gold["id"],
        gold=gold["label"],
        prediction=score["prediction"],
        timing=timing,
        decisive_step=index + 1,
        action=event["action"],
        true_cell=true_cell,
        predicted_cell=event["cell"],
        expected_V_on_predicted_cell=expected_v,
        predicted_V=event["verdict"],
        error_component=component or "other",
    )


def split_diagnostics(work, split, role):
    folder = work / "scores/flan_t5_base" / split
    predictions = rows_at(folder / "candidate_predictions.jsonl")
    scores = rows_at(folder / "candidate_scores.jsonl")
    receipt = read(folder / "done.json")
    assert all(digest(folder / name) == checksum for name, checksum in receipt["files"].items())
    cases, returned_prefixes = [], []
    missing_cell, lexical, stops, scoring = Counter(), Counter(), Counter(), Counter()
    transitions = Counter()
    for gold, raw, score in zip(gold_rows(split), predictions, scores, strict=True):
        assert gold["id"] == raw["id"] == score["id"]
        trace = raw["verification_trace"]
        missing_cell.update(event["verdict"] for event in trace if event["cell"] == "no_coordinate")
        grid = Grid.from_dict(gold["grid"])
        if gold["reason"] == "invalid_action_token":
            action = gold["candidate"].split()[gold["step"] - 1]
            kind = "upper" if action in {word.upper() for word in MOVES} else "other"
            lexical[kind] += 1
            lexical[kind + "_exact"] += score["joint"]
        if not score["joint"]:
            case = first_error_case(gold, raw, score, grid)
            cases.append(case)
            if "true_cell" in case:
                transitions[expected_verdict(grid, case["true_cell"]), case["predicted_V"]] += 1
        elif not score["repair"]["success"]:
            status = raw["repair_status"]
            verdict = score["repair"]["actual_verdict"]
            scoring[parse_verdict(verdict)[2]] += 1
            stops[status] += 1
            if status == "returned_prefix":
                path = raw["repaired_path"]
                replay = execute(grid, path)
                tail = trace[len(path.split()) - 1]
                actual_cell = replay.trace[len(path.split()) - 1]["to"]
                remaining = sum(abs(a - b) for a, b in zip(actual_cell, grid.goal, strict=True))
                returned_prefixes.append(
                    dict(
                        id=gold["id"],
                        path=path,
                        cell=tail["cell"],
                        goal=point(grid.goal),
                        offline_cell=point(actual_cell),
                        goal_distance_manhattan=remaining,
                        predicted_V=tail["verdict"],
                        offline_verdict=replay.label(),
                        feasible=replay.feasible,
                        new_move_attempts=len(raw["repair_trace"]),
                    )
                )
    return dict(
        role=role,
        n=len(scores),
        correct_first_errors=sum(score["joint"] for score in scores),
        first_error_mismatches=Counter(c["timing"] for c in cases),
        component=Counter(c.get("error_component", "other") for c in cases),
        incorrect_transitions=[dict(gold=a, predicted=b, count=n) for (a, b), n in transitions.items()],
        lexical=lexical,
        no_coordinate_verdicts=missing_cell,
        repair_failures_after_correct_verification=stops,
        scored_path_errors=scoring,
        returned_prefix_failures=returned_prefixes,
        cases=cases,
        predictions_sha256=receipt["files"]["candidate_predictions.jsonl"],
        scores_sha256=receipt["files"]["candidate_scores.jsonl"],
    )


def diagnose(work, uniform_dev):
    generation = [row for row in rows_at(uniform_dev) if row["task"] == "G"]
    training = rows_at(TRAINING)
    directions = {}
    for move in MOVES:
        cases = [r for r in generation if r["gold"] == [move]]
        directions[move] = dict(
            n=len(cases),
            correct=sum(r["raw"] == move for r in cases),
            train=sum(r["task"] == "G" and r["answers"] == [move] for r in training),
        )
    wrong = Counter(r["raw"] for r in generation if r["raw"] not in r["gold"])
    upper = {move.upper() for move in MOVES}
    lexical = (
        r["text"].splitlines()[1].removeprefix("Action: ")
        for r in training
        if r["task"] == "C" and r["answers"] == ["no_coordinate"]
    )
    protocol = read(work / "data/protocol.json")
    report = dict(
        G=dict(
            n=len(generation),
            wrong=sum(wrong.values()),
            wrong_predictions=wrong,
            unique_directions=directions,
            training_uppercase=Counter(word for word in lexical if word in upper),
        ),
        splits={s: split_diagnostics(work, s, protocol["held_out"][s]["role"]) for s in ERROR_SETS},
    )
    put(work / "first_group_errors.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=PUBLISHED)
    parser.add_argument(
        "--uniform-dev", type=Path, default=PUBLISHED / "original_uniform/epoch3/dev_predictions.jsonl"
    )
    args = parser.parse_args()
    result = diagnose(args.work, args.uniform_dev)
    print(json.dumps({split: row["first_error_mismatches"] for split, row in result["splits"].items()}))
