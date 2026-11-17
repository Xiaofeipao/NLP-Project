"""R-only alternatives: probability traces, loop erasure and bounded branching."""

from collections import Counter
import math

from ppnl.core import ACTIONS, execute, parse_verdict
from ppnl.policy import NAMES, neighbors

VARIANTS = [
    {"name": "raw_probability", "mode": "raw"},
    {"name": "visit_probability", "mode": "visit"},
    {"name": "raw_erase", "mode": "raw", "erase": True},
    {"name": "visit_erase", "mode": "visit", "erase": True},
    {"name": "top3_budget36", "mode": "beam", "score_cap": 36},
    {"name": "top3_budget108", "mode": "beam", "score_cap": 108},
    {"name": "soft_cycle_penalty", "mode": "soft"},
    {"name": "hard_cycle_ban", "mode": "ban"},
    {"name": "top3_soft108", "mode": "beam_soft", "score_cap": 108},
    {"name": "top3_soft108_erase", "mode": "beam_soft", "score_cap": 108, "erase": True},
]


def probabilities(scores):
    if len(scores) != 4 or not all(math.isfinite(x) for x in scores):
        raise ValueError("Expected four finite action logits")
    largest = max(scores)
    weights = [math.exp(x - largest) for x in scores]
    return {a: v / sum(weights) for a, v in zip(NAMES, weights)}


def erase_loops(grid, plan):
    """Static-grid closed walks can be removed without changing the endpoint.

    Illegal paths remain byte-for-byte unchanged; never erase a collision away.
    Consecutive same-direction actions are NOT loops.
    """
    if not execute(grid, plan).feasible:
        return plan, 0
    positions = [grid.start]
    tokens = []
    index = {grid.start: 0}
    for action in plan.split():
        dr, dc = ACTIONS[action]
        nxt = positions[-1][0] + dr, positions[-1][1] + dc
        if nxt in index:
            keep = index[nxt]
            for p in positions[keep + 1 :]:
                index.pop(p)
            positions = positions[: keep + 1]
            tokens = tokens[:keep]
        else:
            tokens.append(action)
            positions.append(nxt)
            index[nxt] = len(tokens)
    result = " ".join(tokens)
    assert execute(grid, result).valid == execute(grid, plan).valid
    return result, len(plan.split()) - len(tokens)


def repaired_prefix(grid, candidate, feedback):
    q = parse_verdict(feedback)
    if q is None:
        return None, "malformed_feedback"
    valid, step, reason = q
    if valid:
        return None, "accepted"
    tokens = candidate.split()
    if (
        reason == "goal_not_reached"
        and step != len(tokens) + 1
        or reason != "goal_not_reached"
        and step > len(tokens)
    ):
        return None, "bad_step"
    prefix = tokens[: step - 1]
    verdict = execute(grid, " ".join(prefix))
    if not verdict.feasible:
        return None, "illegal_prefix"
    positions = [grid.start]
    for a in prefix:
        dr, dc = ACTIONS[a]
        positions.append((positions[-1][0] + dr, positions[-1][1] + dc))
    return (prefix, positions), "ready"


def action_choices(grid, positions, probs, mode):
    legal = neighbors(grid, positions[-1])
    visits = Counter(positions)
    choices = list(NAMES) if mode == "raw" else list(legal)
    penalty = {a: 0.0 for a in choices}
    if mode == "visit" and choices:
        least = min(visits[legal[a]] for a in choices)
        choices = [a for a in choices if visits[legal[a]] == least]
    if mode in ("soft", "beam_soft", "ban"):
        recent = set(positions[-4:])
        if mode == "ban":
            choices = [a for a in choices if legal[a] not in recent]
        else:
            for a in choices:
                penalty[a] = 1.5 * visits[legal[a]] + (
                    2.0 if len(positions) > 1 and legal[a] == positions[-2] else 0
                )
    adjusted = {a: math.log(max(probs[a], 1e-300)) - penalty[a] for a in choices}
    return sorted(choices, key=lambda a: adjusted[a], reverse=True), adjusted, penalty


def repair(grid, candidate, feedback, scorer, spec, max_new=36):
    prepared, status = repaired_prefix(grid, candidate, feedback)
    telemetry = {"score_rows": 0, "probability_trace": [], "removed_actions": 0}
    if prepared is None:
        return candidate, status, telemetry
    prefix, positions = prepared
    # Beam ranks by cumulative model log-probability, with optional declared
    # cycle penalty. No Manhattan/BFS/reference path is used at inference.
    frontier = [(prefix, positions, 0.0)]
    mode = spec["mode"]
    beam = mode in ("beam", "beam_soft")
    cap = spec.get("score_cap", max_new)
    best = frontier[0]
    status = "action_budget"
    for depth in range(max_new + 1):
        finished = [b for b in frontier if b[1][-1] == grid.goal]
        if finished:
            best = max(finished, key=lambda b: b[2])
            status = "success"
            break
        if depth == max_new or telemetry["score_rows"] >= cap:
            best = max(frontier, key=lambda b: b[2])
            status = "score_or_action_budget"
            break
        active = frontier[: cap - telemetry["score_rows"]]
        logits = scorer.scores([(grid, b[1][-1]) for b in active])
        telemetry["score_rows"] += len(active)
        expanded = []
        for state, scores in zip(active, logits):
            path, visited, total = state
            probs = probabilities(scores)
            choices, adjusted, penalty = action_choices(grid, visited, probs, mode)
            telemetry["probability_trace"].append(
                {
                    "depth": depth,
                    "position": visited[-1],
                    "probabilities": probs,
                    "ranked_allowed": choices,
                    "cycle_penalties": penalty,
                }
            )
            for a in choices[: 3 if beam else 1]:
                dr, dc = ACTIONS[a]
                nxt = visited[-1][0] + dr, visited[-1][1] + dc
                node = (path + [a], visited + [nxt], total + adjusted[a])
                if not grid.inside(nxt) or nxt in grid.obstacles:
                    best = node
                    status = "illegal_action"
                    continue
                expanded.append(node)
        if not expanded:
            if status != "illegal_action":
                best = frontier[0]
                status = "all_actions_blocked"
            break
        frontier = sorted(expanded, key=lambda b: b[2], reverse=True)[: 3 if beam else 1]
        best = frontier[0]
    output = " ".join(best[0])
    if spec.get("erase"):
        output, telemetry["removed_actions"] = erase_loops(grid, output)
    telemetry["new_actions_before_erasure"] = max(0, len(best[0]) - len(prefix))
    return output, status, telemetry
