"""Verify the final actual move from the state immediately before it."""
import hashlib
import json

from ppnl.core import ACTIONS
from ppnl.policy import neighbors
from . import model_io
from .schema import State, render, state_lines
from .single import LABELS, batch_loss

VERSION = "ppnl-last-move-v1"


def check_token(action):
    if not isinstance(action, str) or action.split() != [action]:
        raise ValueError("Expected exactly one actual action token")


def step_input(state, action, *, explicit_vocabulary=False):
    check_token(action)
    vocabulary = ["Allowed moves: " + " ".join(ACTIONS)] if explicit_vocabulary else []
    return "\n".join(["Task: v_step", *state_lines(state), *vocabulary, "Last move: " + json.dumps(action),
                      "Output one label: valid if the move reaches the goal, goal_not_reached if legal elsewhere,"
                      " otherwise obstacle_collision, out_of_bounds or invalid_action_token."])


def template_hash(renderer=step_input):
    sample = State(6, 6, ((2, 2),), (0, 0), (0, 2))
    return hashlib.sha256((render("g_next", sample)+"\n\n"+renderer(sample,"right")).encode()).hexdigest()


def transition(state, action):
    """Data reference T(s,a). Return the destination for a legal move, otherwise None."""
    check_token(action)
    if action not in ACTIONS:
        return None, "invalid_action_token"
    dr, dc = ACTIONS[action]
    dest = state.current[0] + dr, state.current[1] + dc
    if not (0 <= dest[0] < state.rows and 0 <= dest[1] < state.cols):
        return None, "out_of_bounds"
    if dest in state.obstacles:
        return None, "obstacle_collision"
    return dest, "valid" if dest == state.goal else "goal_not_reached"


def prepare_training(runner, dataset):
    return model_io.prepare_training(runner,dataset,version=VERSION,fingerprint=template_hash())


def generation_policy(runner, check=lambda: None):
    return model_io.StatePolicy(runner,check,fingerprint=template_hash())


def verify_moves(runner, observations, check=lambda: None, *, renderer=step_input):
    """One (state_before_last_move, last_move) per independent candidate."""
    model_io.check_checkpoint(runner,template_hash(renderer))
    return runner.predict([renderer(state,a) for state,a in observations],check)


def verify_last(runner, state_before_last, last_action, check=lambda: None):
    """Online decision for an already checked prefix. A single valid label permits emission."""
    label, = verify_moves(runner,[(state_before_last,last_action)],check)
    return label


def path_steps(grid, path):
    """Hard-coded position tracking. Yield the state before each real move, without a verdict."""
    position = grid.start
    for action in path.split():
        yield State.from_grid(grid,position),action
        # A physically impossible move leaves the position unchanged. The model
        # still supplies every learned verdict, including any false acceptance.
        position = neighbors(grid,position).get(action,position)


def scan(grids, plans, decide):
    """Benchmark complete fixed candidates, querying real moves only. No decision at EOS."""
    cases = [(grid,list(path_steps(grid,path))) for grid,path in zip(grids,plans,strict=True)]
    results = [None if steps else ("valid" if g.start==g.goal else "invalid | step=1 | reason=goal_not_reached")
               for g,steps in cases]
    active = [i for i,(_,steps) in enumerate(cases) if steps]
    offset = 0
    while active:
        observations = [cases[i][1][offset] for i in active]
        following = []
        for i,text in zip(active,decide(observations),strict=True):
            label, step = text.strip(), offset+1
            if label not in LABELS:
                results[i] = f"malformed local verdict: {text}"
            elif label not in ("valid","goal_not_reached"):
                results[i] = f"invalid | step={step} | reason={label}"
            elif step == len(cases[i][1]):
                results[i] = "valid" if label=="valid" else f"invalid | step={step+1} | reason=goal_not_reached"
            else:
                following.append(i)
        active, offset = following, offset+1
    return results


def verify_plans(runner, grids, plans, check=lambda: None):
    return scan(grids,plans,lambda observations:verify_moves(runner,observations,check))
