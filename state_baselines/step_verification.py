"""Scan a candidate one action at a time with a separate verifier."""

import json

from ppnl.core import parse_verdict
from ppnl.policy import neighbors


def step_prompt(grid, position, action):
    return (
        f"Verify one step in a {grid.rows} by {grid.cols} grid. "
        "Coordinates are zero-based (row, column). Up decreases row; right increases column.\n"
        f"Current position: {tuple(position)}. Goal: {grid.goal}. Obstacles: {list(grid.obstacles)}.\n"
        f"Candidate action: {json.dumps(action)}. A null action means stop without moving.\n"
        "Return valid if the action is legal and ends at the goal (or stop at the goal). "
        "For an illegal action, return invalid | step=1 | reason=<error_type>, where the type is "
        "invalid_action_token, out_of_bounds, or obstacle_collision. "
        "If the action is legal but the goal has not been reached, "
        "return invalid | step=2 | reason=goal_not_reached. Use step=1 if stopping without moving."
    )


def verify_steps(runner, grids, plans, check=lambda: None):
    """Use runner.predict for every verdict; the simulator only advances positions."""
    candidates = [(grid, plan.split() or [None])
                  for grid, plan in zip(grids, plans, strict=True)]
    positions = [grid.start for grid, _ in candidates]
    verdicts = [None] * len(candidates)
    active = list(range(len(candidates)))
    for offset in range(max((len(actions) for _, actions in candidates), default=0)):
        if not active:
            break
        prompts = [step_prompt(candidates[i][0], positions[i], candidates[i][1][offset])
                   for i in active]
        continuing = []
        for i, prediction in zip(active, runner.predict(prompts, check), strict=True):
            parsed = parse_verdict(prediction)
            if parsed is None:
                verdicts[i] = prediction
                continue
            valid, _, reason = parsed
            grid, actions = candidates[i]
            action = actions[offset]
            step = offset + 1
            if not valid and reason != "goal_not_reached":
                verdicts[i] = f"invalid | step={step} | reason={reason}"
            elif step == len(actions):
                eos = step if action is None else step + 1
                verdicts[i] = "valid" if valid else f"invalid | step={eos} | reason={reason}"
            else:
                # A false acceptance cannot make an illegal move change the state.
                # Retain the model's decision for evaluation instead of using a rule verdict.
                positions[i] = neighbors(grid, positions[i]).get(action, positions[i])
                continuing.append(i)
        active = continuing
    return verdicts
