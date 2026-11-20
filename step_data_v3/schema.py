"""The same renderer is used for building rows and querying the model."""
from dataclasses import asdict, dataclass
import hashlib
import json

from ppnl.core import ACTIONS, Grid, execute
from ppnl.policy import neighbors


VERSION = "ppnl-step-semantic-v3.1"
TASK = {"g_next": "generation", "v_step": "verification", "v_plan": "verification"}
REASON_SPACE = "obstacle_collision, out_of_bounds, invalid_action_token, goal_not_reached"
VERDICT = "valid or invalid | step=N | reason=R. R: " + REASON_SPACE + "."
INSTRUCTIONS = {
    "g_next": "One shortest next move: up, down, left or right.",
    "v_step": (VERDICT + ' Empty attempt means stop. Illegal move: N=1. Legal move ending'
               ' away: N=2. Stop away: N=1. End at goal: valid.'),
    "v_plan": (VERDICT + " Check the whole path. First error is 1-based. Unfinished fails at length+1."),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class State:
    rows: int
    cols: int
    obstacles: tuple
    current: tuple
    goal: tuple

    @classmethod
    def from_grid(cls, grid, current=None, goal=None):
        return cls(grid.rows, grid.cols, tuple(sorted(grid.obstacles)),
                   grid.start if current is None else tuple(current),
                   grid.goal if goal is None else tuple(goal))

    @classmethod
    def from_record(cls, row):
        return cls(row["rows"], row["cols"], tuple(map(tuple, row["obstacles"])),
                   tuple(row["current"]), tuple(row["goal"]))

    @property
    def id(self):
        return digest([self.rows, self.cols, self.obstacles, self.current, self.goal])

    def grid(self):
        return Grid(self.rows, self.cols, self.obstacles, self.current, self.goal)

    def witness(self):
        """A real grid and an executable prefix witnessing the current position."""
        if self.current != self.goal:
            return self.grid(), []
        for action, (dr, dc) in ACTIONS.items():
            start = self.goal[0] - dr, self.goal[1] - dc
            if (0 <= start[0] < self.rows and 0 <= start[1] < self.cols
                    and start not in self.obstacles):
                return Grid(self.rows, self.cols, self.obstacles, start, self.goal), [action]
        raise ValueError("A terminal state needs a reachable arrival witness")

    def record(self):
        grid, _ = self.witness()
        return dict(id=self.id, layout_id=grid.layout_id, **asdict(self))


def point(pair):
    return f"{pair[0]},{pair[1]}"


def state_lines(state):
    return [f"Grid: {state.rows}x{state.cols} | zero-based row,col | up=-row, right=+col",
            f"Current: {point(state.current)}", f"Goal: {point(state.goal)}",
            "Walls: " + (" ".join(map(point, state.obstacles)) or "none")]


def render(mode, state, candidate=None):
    """No labels, reference plan, optimal sets, IDs, or simulator features as inputs."""
    if (mode == "g_next") != (candidate is None):
        raise ValueError("Supply a candidate exactly for verification")
    lines = [f"Task: {mode}", *state_lines(state)]
    if candidate is not None:
        tokens = candidate.split()
        if mode == "v_step" and len(tokens) > 1:
            raise ValueError("Local verification accepts exactly one move or the empty attempt")
        lines.append("Attempt: " + json.dumps(" ".join(tokens), ensure_ascii=False))
    lines.append("Output: " + INSTRUCTIONS[mode])
    return "\n".join(lines)


def template_hash():
    state = State(6, 6, ((2, 2),), (0, 0), (0, 2))
    examples = [render("g_next", state), render("v_step", state, ""),
                render("v_step", state, "right"), render("v_plan", state, "right right")]
    return hashlib.sha256("\n\n".join(examples).encode("utf-8")).hexdigest()


def step_labels(state, candidate):
    tokens = candidate.split()
    if len(tokens) > 1:
        raise ValueError("Expected a single action or stop")
    grid, prefix = state.witness()
    verdict = execute(grid, " ".join(prefix + tokens))
    local_step = None if verdict.valid else verdict.step - len(prefix)
    action_legal = verdict.feasible if tokens else None
    position = state.current
    if tokens and action_legal:
        position = neighbors(grid, position)[tokens[0]]
    labels = {"action_legal": action_legal, "position_after": list(position) if verdict.feasible else None,
              "at_goal_after": position == state.goal if verdict.feasible else None,
              "valid": verdict.valid, "step": local_step, "reason": verdict.reason}
    text = "valid" if verdict.valid else f"invalid | step={local_step} | reason={verdict.reason}"
    return text, labels
