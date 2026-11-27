"""Prompt-only G/C/V inference. The optional feedback callback is exclusive to the oracle arm."""

from collections import Counter, defaultdict
import re

from .protocol import LABELS, MOVES, point

CELL = re.compile(r"\((?:0|-?[1-9][0-9]*),(?:0|-?[1-9][0-9]*)\)")
LEGAL = {"goal_not_reached", "valid"}
DELIVERED = {"returned_plan", "returned_prefix", "unchanged"}
LIMIT = 36


def fields(grid):
    return dict(
        current=point(grid["start"]),
        goal=point(grid["goal"]),
        walls="[" + ", ".join(map(point, sorted(grid["obstacles"]))) + "]",
    )


def direct(case, prompts, limit=LIMIT):
    state, moves, trace = fields(case["grid"]), [], []
    status = "attempt_limit"
    for _ in range(limit):
        ranked = yield "G", prompts["G"].format(**state)
        action = ranked[0]
        cell = yield "C", prompts["C"].format(current=state["current"], action=action)
        moves.append(action)
        trace.append(dict(before=state["current"], action=action, cell=cell))
        if not CELL.fullmatch(cell):
            status = "malformed_C"
            break
        state["current"] = cell
        if cell == state["goal"]:
            status = "predicted_goal"
            break
    return dict(id=case["id"], plan=" ".join(moves), status=status, trace=trace)


def episode(case, prompts, limit=LIMIT, feedback=None, *, soft_revisit=False):
    state = fields(case["grid"])
    actions = case["candidate"].split()
    prefix, verification_trace, repair_trace = [], [], []
    rejected = defaultdict(set)

    def judge(action):
        current = state["current"]
        if feedback is not None:
            return feedback(case["grid"], prefix, current, action)
        cell = yield "C", prompts["C"].format(current=current, action=action)
        event = dict(before=current, action=action, cell=cell, verdict=None)
        if cell != "no_coordinate" and not CELL.fullmatch(cell):
            event["failure"] = "malformed_C"
            return event
        verdict = yield "V", prompts["V"].format(**dict(state, current=cell))
        event["verdict"] = verdict
        if verdict not in LABELS or (cell == "no_coordinate" and verdict in LEGAL):
            event["failure"] = "incompatible_V"
        return event

    def finish(verification, status):
        return dict(
            id=case["id"],
            verification=verification,
            repair_status=status,
            repaired_path=" ".join(prefix),
            verification_trace=verification_trace,
            repair_trace=repair_trace,
        )

    visits = Counter([state["current"]])
    arrived = False
    for step, action in enumerate(actions, 1):
        event = yield from judge(action)
        verification_trace.append(event)
        if "failure" in event:
            return finish(dict(label="model_error", step=None, reason=None), event["failure"])
        verdict = event["verdict"]
        if verdict not in LEGAL:
            verification = dict(label="invalid", step=step, reason=verdict)
            if action in MOVES:
                rejected[state["current"]].add(action)
            break
        if soft_revisit:
            visits[event["cell"]] += 1
        prefix.append(action)
        state["current"], arrived = event["cell"], verdict == "valid"
    else:
        # Source grids require distinct Start and Goal, so an empty path ends short at step 1.
        if arrived:
            return finish(dict(label="valid", step=None, reason=None), "unchanged")
        verification = dict(label="invalid", step=len(actions) + 1, reason="goal_not_reached")

    if arrived:
        return finish(verification, "returned_prefix")
    for _ in range(limit):
        ranked = yield "G", prompts["G"].format(**state)
        candidates = [a for a in ranked if a not in rejected[state["current"]]]
        action = next(iter(candidates), None)
        if soft_revisit and len(candidates) > 1:
            costs = []
            for rank, candidate in enumerate(candidates):
                destination = yield "C", prompts["C"].format(current=state["current"], action=candidate)
                costs.append(rank + 2 * visits[destination])
            action = candidates[min(range(len(costs)), key=costs.__getitem__)]
        if action is None:
            return finish(verification, "directions_exhausted")
        event = yield from judge(action)
        repair_trace.append(event)
        if "failure" in event:
            return finish(verification, event["failure"])
        if event["verdict"] not in LEGAL:
            rejected[state["current"]].add(action)
            continue
        if soft_revisit:
            visits[event["cell"]] += 1
        prefix.append(action)
        state["current"] = event["cell"]
        if event["verdict"] == "valid":
            return finish(verification, "returned_plan")
    return finish(verification, "attempt_limit")


def run(engines, predict, cache=None):
    """Batch requests in the original model/task order. Oracle-only episodes may finish at entry."""
    cache = {} if cache is None else cache
    waiting, results = {}, [None] * len(engines)

    def advance(i, value):
        try:
            waiting[i] = engines[i].send(value)
        except StopIteration as done:
            waiting.pop(i, None)
            results[i] = done.value

    for i in range(len(engines)):
        advance(i, None)
    while waiting:
        unique = dict.fromkeys(waiting.values())
        for task in ("G", "C", "V"):
            missing = [k for k in unique if k[0] == task and k not in cache]
            if missing:
                answers = predict(task, [text for _, text in missing])
                cache.update(zip(missing, answers, strict=True))
        for i, key in list(waiting.items()):
            advance(i, cache[key])
    return results


def line(verification):
    if verification["label"] == "invalid":
        return f"invalid | step={verification['step']} | reason={verification['reason']}"
    return verification["label"]
