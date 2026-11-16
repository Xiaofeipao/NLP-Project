"""Explicit symbolic assistance for learned verification, never a neural gain."""

from .core import ACTIONS, execute

MODES = ("model", "vocabulary", "terminal", "trace")


def guard_verdict(grid, candidate, prediction, mode="trace"):
    """Partial checks only override when their prefix is safe to diagnose.

    Full trace replaces the diagnosis, including false rejection of valid plans.
    It is intentionally equivalent to the executor, not an improved learned V.
    """
    if mode not in MODES:
        raise ValueError("Unknown guard mode")
    if mode == "model":
        return prediction
    pos = grid.start
    for step, action in enumerate(candidate.split(), 1):
        if action not in ACTIONS:
            return f"invalid | step={step} | reason=invalid_action_token"
        dr, dc = ACTIONS[action]
        nxt = pos[0] + dr, pos[1] + dc
        reason = (
            "out_of_bounds"
            if not grid.inside(nxt)
            else "obstacle_collision" if nxt in grid.obstacles else None
        )
        if reason:
            return execute(grid, candidate).label() if mode == "trace" else prediction
        pos = nxt
    if mode in ("terminal", "trace") and pos != grid.goal:
        return f"invalid | step={len(candidate.split()) + 1} | reason=goal_not_reached"
    return "valid" if mode == "trace" else prediction


def traced_verification_prompt(grid, candidate):
    """Tool-assisted model V: deterministic trace explicitly contains diagnosis.

    This measures reading/formatting executor evidence, not unaided reasoning.
    Gold row labels, reference routes and BFS are never read here.
    """
    verdict = execute(grid, candidate)
    lines = [
        f"step={r['step']} from={r['from']} action={r['action']} to={r['to']} outcome={r['reason'] or 'ok'}"
        for r in verdict.trace
    ]
    return (
        grid.description()
        + "\nCandidate: "
        + candidate
        + "\nExecutor observations:\n"
        + "\n".join(lines)
        + "\nFinal goal reached: "
        + str(verdict.valid).lower()
        + "\nReturn valid, or invalid | step=<first failure index> | reason=<error type>. "
        "Use obstacle_collision, out_of_bounds, invalid_action_token, or goal_not_reached."
    )
