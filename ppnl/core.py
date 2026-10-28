"""Deterministic grid semantics and a safe adapter to the upstream executor."""
from __future__ import annotations

import ast
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
REASONS = ("obstacle_collision", "out_of_bounds", "invalid_action_token", "goal_not_reached")
UPSTREAM_REVISION = "5fd0376fc50ae9860bc986085165ebac5675fd44"


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    obstacles: tuple[tuple[int, int], ...]
    start: tuple[int, int]
    goal: tuple[int, int]

    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Grid dimensions must be positive")
        points = (*self.obstacles, self.start, self.goal)
        if any(not self.inside(p) for p in points):
            raise ValueError("Coordinates outside grid")
        if self.start == self.goal or self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Start and goal must be distinct free cells")

    def inside(self, p):
        return 0 <= p[0] < self.rows and 0 <= p[1] < self.cols

    @property
    def layout_id(self):
        # Deliberately excludes start/goal: all paths on one obstacle layout stay together.
        value = [self.rows, self.cols, sorted(set(self.obstacles))]
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:20]

    def world(self):
        result = [[0] * self.cols for _ in range(self.rows)]
        for r, c in self.obstacles:
            result[r][c] = 1
        result[self.start[0]][self.start[1]] = 2
        result[self.goal[0]][self.goal[1]] = 3
        return result

    @classmethod
    def from_world(cls, world):
        if not world or not world[0] or any(len(row) != len(world[0]) for row in world):
            raise ValueError("Expected a rectangular world")
        if any(v not in (0, 1, 2, 3) for row in world for v in row):
            raise ValueError("Unknown world cell")
        find = lambda value: [(r, c) for r, row in enumerate(world) for c, v in enumerate(row) if v == value]
        starts, goals = find(2), find(3)
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("Only one start and one goal are supported")
        return cls(len(world), len(world[0]), tuple(find(1)), starts[0], goals[0])

    @classmethod
    def from_dict(cls, value):
        return cls(value["rows"], value["cols"], tuple(map(tuple, value["obstacles"])),
                   tuple(value["start"]), tuple(value["goal"]))

    def description(self):
        return (f"You are in a {self.rows} by {self.cols} world. Coordinates are zero-based "
                f"(row, column); up decreases row. Avoid obstacles at {list(self.obstacles)}. "
                f"Go from {self.start} to {self.goal}. Execute the entire plan.")


@dataclass
class Verdict:
    valid: bool
    feasible: bool
    step: int | None
    reason: str | None
    trace: list

    def label(self):
        return "valid" if self.valid else f"invalid | step={self.step} | reason={self.reason}"


def execute(grid: Grid, plan: str) -> Verdict:
    """One-based failure indices; unfinished plan fails at len(actions)+1 (EOS)."""
    pos, trace = grid.start, []
    tokens = plan.split()
    for step, token in enumerate(tokens, 1):
        record = {"step": step, "from": pos, "action": token}
        reason = None
        if token not in ACTIONS:
            target, reason = pos, "invalid_action_token"
        else:
            dr, dc = ACTIONS[token]
            target = pos[0] + dr, pos[1] + dc
            if not grid.inside(target):
                reason = "out_of_bounds"
            elif target in grid.obstacles:
                reason = "obstacle_collision"
        record.update(to=target, reason=reason)
        trace.append(record)
        if reason:
            return Verdict(False, False, step, reason, trace)
        pos = target
    if pos != grid.goal:
        trace.append({"step": len(tokens) + 1, "from": pos, "action": "<eos>",
                      "to": pos, "reason": "goal_not_reached"})
        return Verdict(False, True, len(tokens) + 1, "goal_not_reached", trace)
    return Verdict(True, True, None, None, trace)


def shortest_path(grid: Grid) -> list[str] | None:
    queue, seen = deque([(grid.start, [])]), {grid.start}
    while queue:
        pos, path = queue.popleft()
        if pos == grid.goal:
            return path
        for token, (dr, dc) in ACTIONS.items():
            nxt = pos[0] + dr, pos[1] + dc
            if grid.inside(nxt) and nxt not in grid.obstacles and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [token]))
    return None


def parse_verdict(text):
    if text.strip() == "valid":
        return True, None, None
    match = re.fullmatch(r"invalid\s*\|\s*step=([1-9]\d*)\s*\|\s*reason=(\w+)", text.strip())
    if match and match[2] in REASONS:
        return False, int(match[1]), match[2]
    return None  # malformed predictions remain incorrect, never silently become invalid.


class OfficialOracle:
    """Load only is_goal's AST: upstream script otherwise executes a CLI on import.

    Upstream silently ignores unknown tokens. Strict action vocabulary validation is
    an explicit project extension, applied before calling upstream on legal tokens.
    """
    def __init__(self, path):
        source = Path(path).read_text(encoding="utf-8")
        self.sha256 = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_goal"]
        if len(functions) != 1:
            raise ValueError("Official source must contain exactly one is_goal function")
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        self.is_goal = namespace["is_goal"]

    def check(self, grid, plan):
        verdict = execute(grid, plan)
        if all(t in ACTIONS for t in plan.split()):
            result = self.is_goal(grid.world(), grid.start, plan, grid.rows, grid.cols)
            if (result == 1) != verdict.valid:
                raise AssertionError(f"Upstream/local disagreement: {plan!r}")
            if (result == 2) != (verdict.reason == "obstacle_collision"):
                raise AssertionError("Upstream collision disagreement")
        else:
            # Official no-op token behavior cannot supply the required token label.
            assert verdict.reason in REASONS
        return verdict


def jsonable(value):
    return asdict(value)
