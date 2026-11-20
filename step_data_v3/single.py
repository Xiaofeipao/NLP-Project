"""One learned V decision per move or final STOP, composed into a whole-candidate verdict."""
import hashlib
import json

from ppnl.core import ACTIONS
from ppnl.policy import neighbors
from . import model_io
from .schema import State, render, state_lines

VERSION = "ppnl-single-step-v1"
LABELS = ("valid", "obstacle_collision", "out_of_bounds", "invalid_action_token", "goal_not_reached")


def step_input(state, action):
    if action is not None and (not isinstance(action, str) or action.split() != [action]):
        raise ValueError("Supply one action token or None for STOP")
    return "\n".join(["Task: v_step", *state_lines(state), "Action: " + json.dumps(action),
                      "For a move (up, down, left, right), check legality. For null (STOP), check arrival.",
                      "Output one: " + ", ".join(LABELS) + "."])


def template_hash():
    state = State(6, 6, ((2, 2),), (0, 0), (0, 2))
    return hashlib.sha256("\n\n".join([render("g_next", state), step_input(state, "right"),
                                      step_input(state, None)]).encode()).hexdigest()


def reference_label(state, action):
    """Construction and audits only. The learned scanner receives no reference verdict."""
    if action is None:
        return "valid" if state.current == state.goal else "goal_not_reached"
    if action not in ACTIONS:
        return "invalid_action_token"
    dr, dc = ACTIONS[action]
    r, c = state.current[0] + dr, state.current[1] + dc
    if not (0 <= r < state.rows and 0 <= c < state.cols):
        return "out_of_bounds"
    return "obstacle_collision" if (r, c) in state.obstacles else "valid"


def prepare_training(runner, dataset):
    return model_io.prepare_training(runner, dataset, version=VERSION, fingerprint=template_hash())


def batch_loss(runner, batch):
    # V has one text label. G still uses the full optimal-action probability mass.
    return runner.loss(batch.to(runner.device))


def generation_policy(runner, check=lambda: None):
    return model_io.StatePolicy(runner, check, fingerprint=template_hash())


def scan(grids, plans, decide):
    """decide accepts (State, action-or-None) pairs. Only its predictions produce verdicts."""
    cases = [(g, plan.split()) for g, plan in zip(grids, plans, strict=True)]
    positions = [g.start for g, _ in cases]
    results = [None] * len(cases)
    active = list(range(len(cases)))
    offset = 0
    while active:
        observations = []
        for i in active:
            grid, tokens = cases[i]
            action = tokens[offset] if offset < len(tokens) else None
            observations.append((State.from_grid(grid, positions[i]), action))
        continuing = []
        for i, (_, action), output in zip(active, observations, decide(observations), strict=True):
            label = output.strip()
            if label not in LABELS:
                results[i] = f"malformed local verdict: {output}"
            elif label != "valid":
                # Preserve even a wrong predicted reason so evaluation sees the model's mistake.
                results[i] = f"invalid | step={offset+1} | reason={label}"
            elif action is None:
                results[i] = "valid"
            else:
                # A rejected physical transition leaves the position unchanged. It never
                # corrects a false acceptance or injects an executor verdict.
                positions[i] = neighbors(cases[i][0], positions[i]).get(action, positions[i])
                continuing.append(i)
        active = continuing
        offset += 1
    return results


def verify_steps(runner, observations, check=lambda: None):
    model_io.check_checkpoint(runner, template_hash())
    return runner.predict([step_input(s, a) for s, a in observations], check)


def verify_plans(runner, grids, plans, check=lambda: None):
    return scan(grids, plans, lambda items: verify_steps(runner, items, check))
