"""Adaptive prefix rollback; shortest-route labels are offline analysis only."""

from collections import deque

from ppnl.core import ACTIONS, execute, parse_verdict


def choose_prefix(grid, candidate, feedback, backoff=0, rewind_cycle=False):
    """Inference uses supplied feedback, prefix execution and observed visits.

    backoff is fixed beforehand (or chosen on dev), never read from test gold.
    Prefix correction is explicit symbolic assistance, not learned diagnosis.
    """
    if not isinstance(backoff, int) or backoff < 0:
        raise ValueError("backoff must be a nonnegative integer")
    parsed = parse_verdict(feedback)
    if parsed is None or parsed[0]:
        return None, {"status": "accepted" if parsed else "malformed_feedback", "executor_calls": 0}
    tokens = candidate.split()
    anchor = min(parsed[1] - 1, len(tokens))
    keep = max(0, anchor - backoff)
    verdict = execute(grid, " ".join(tokens[:keep]))
    corrected = not verdict.feasible
    if corrected:
        keep = verdict.step - 1
    cycle_keep = None
    if rewind_cycle:
        seen = {grid.start: 0}
        position = grid.start
        for step, action in enumerate(tokens[:keep], 1):
            dr, dc = ACTIONS[action]
            position = position[0] + dr, position[1] + dc
            if position in seen:
                cycle_keep = seen[position]
                keep = cycle_keep
                break
            seen[position] = step
    return " ".join(tokens[:keep]), {
        "status": "ready",
        "anchor": anchor,
        "keep_actions": keep,
        "backoff_steps": anchor - keep,
        "cycle_keep": cycle_keep,
        "symbolic_prefix_correction": corrected,
        "executor_calls": 1,
    }


def offline_diagnosis(grid, candidate, feedback):
    """Gold/BFS instrumentation. MUST NOT be used to select test repair cuts."""
    distances = {grid.goal: 0}
    queue = deque([grid.goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTIONS.values():
            nxt = r + dr, c + dc
            if grid.inside(nxt) and nxt not in grid.obstacles and nxt not in distances:
                distances[nxt] = distances[(r, c)] + 1
                queue.append(nxt)
    if grid.start not in distances:
        raise ValueError("Offline optimality needs a reachable goal")
    truth = execute(grid, candidate)
    pred = parse_verdict(feedback)
    position = grid.start
    first_suboptimal = None
    legal = 0
    for step, a in enumerate(candidate.split(), 1):
        if a not in ACTIONS:
            break
        dr, dc = ACTIONS[a]
        nxt = position[0] + dr, position[1] + dc
        if not grid.inside(nxt) or nxt in grid.obstacles:
            break
        legal = step
        position = nxt
        if (
            first_suboptimal is None
            and step + distances.get(nxt, float("inf")) > distances[grid.start]
        ):
            first_suboptimal = step
    optimum_limit = first_suboptimal - 1 if first_suboptimal else legal
    # Latest prefix which is both executable and compatible with ANY shortest route.
    limit = min(legal, optimum_limit)
    anchor = min(pred[1] - 1, len(candidate.split())) if pred is not None and not pred[0] else None
    return {
        "true_step": truth.step,
        "predicted_step": pred[1] if pred else None,
        "signed_step_offset": (
            pred[1] - truth.step if pred is not None and not pred[0] and truth.step else None
        ),
        "first_suboptimal_action": first_suboptimal,
        "latest_optimal_legal_prefix": limit,
        "oracle_backoff": max(0, anchor - limit) if anchor is not None else None,
        "warning": "oracle_backoff is an offline upper-bound label, not inference input",
    }
