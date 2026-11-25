"""Blind verification and revising. Only the three model replies advance a path."""
from collections import defaultdict
import re

from texts import ACTIONS, CLASSES, point

CELL = re.compile(r"\((?:0|-?[1-9][0-9]*),(?:0|-?[1-9][0-9]*)\)")
LEGAL = {"valid", "goal_not_reached"}
REPAIR_LIMIT = 36


def episode(case, templates, limit=REPAIR_LIMIT):
    grid = case["grid"]
    current, goal = point(grid["start"]), point(grid["goal"])
    walls = "[" + ", ".join(point(w) for w in sorted(grid["obstacles"])) + "]"
    actions = case["candidate"].split()
    prefix, verify_trace, repair_trace = [], [], []
    rejected = defaultdict(set)

    def judge(before, action):
        cell = yield "C", templates["C"].format(current=before, action=action)
        event = dict(before=before, action=action, cell=cell, verdict=None)
        if cell != "no_coordinate" and not CELL.fullmatch(cell):
            event["failure"] = "malformed_C"
            return event
        verdict = yield "V", templates["V"].format(current=cell, goal=goal, walls=walls)
        event["verdict"] = verdict
        if verdict not in CLASSES or (cell == "no_coordinate" and verdict in LEGAL):
            event["failure"] = "incompatible_V"
        return event

    def finish(verification, status):
        return dict(id=case["id"], verification=verification, repair_status=status,
                    repaired_path=" ".join(prefix), verification_trace=verify_trace, repair_trace=repair_trace)

    if not actions:
        raise ValueError("The registered error-path sets contain only nonempty paths")
    arrived = False
    for k, action in enumerate(actions, 1):
        event = yield from judge(current, action)
        verify_trace.append(event)
        if "failure" in event:
            return finish(dict(label="model_error", step=None, reason=None), event["failure"])
        verdict = event["verdict"]
        if verdict not in LEGAL:
            verification = dict(label="invalid", step=k, reason=verdict)
            if action in ACTIONS:
                rejected[current].add(action)
            break
        prefix.append(action)
        current, arrived = event["cell"], verdict == "valid"
    else:
        if arrived:
            return finish(dict(label="valid", step=None, reason=None), "unchanged")
        verification = dict(label="invalid", step=len(actions) + 1, reason="goal_not_reached")

    if arrived:
        return finish(verification, "returned_prefix")
    for _ in range(limit):
        ranked = yield "G", templates["G"].format(current=current, goal=goal, walls=walls)
        action = next((a for a in ranked if a not in rejected[current]), None)
        if action is None:
            return finish(verification, "directions_exhausted")
        event = yield from judge(current, action)
        repair_trace.append(event)
        if "failure" in event:
            return finish(verification, event["failure"])
        if event["verdict"] not in LEGAL:
            rejected[current].add(action)
            continue
        prefix.append(action)
        current = event["cell"]
        if event["verdict"] == "valid":
            return finish(verification, "returned_plan")
    return finish(verification, "attempt_limit")


def run_paths(cases, templates, predict):
    """Batch pending questions by task and reuse the model's own deterministic replies."""
    engines = [episode(case, templates) for case in cases]
    waiting = {i: next(engine) for i, engine in enumerate(engines)}
    cache, results = {}, {}
    while waiting:
        unique = dict.fromkeys(waiting.values())
        for task in ("G", "C", "V"):
            missing = [key for key in unique if key[0] == task and key not in cache]
            if missing:
                answers = predict(task, [text for _, text in missing])
                cache.update(zip(missing, answers, strict=True))
        for i, key in list(waiting.items()):
            try:
                waiting[i] = engines[i].send(cache[key])
            except StopIteration as done:
                results[i] = done.value
                del waiting[i]
    return [results[i] for i in range(len(cases))]
