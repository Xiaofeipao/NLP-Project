"""Bounded completion of an already emitted action sequence."""

from collections import deque
from dataclasses import dataclass

from .core import ACTIONS, Grid, execute


@dataclass(frozen=True)
class LocalFinish:
    plan: str
    kept_actions: int
    added_actions: int


def _bridge(grid, start, goal, max_steps):
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        position, path = queue.popleft()
        if position == goal:
            return path
        remaining = max_steps - len(path)
        distance = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
        if not remaining or distance > remaining:
            continue
        for action, (dr, dc) in ACTIONS.items():
            target = position[0] + dr, position[1] + dc
            if target not in seen and grid.inside(target) and target not in grid.obstacles:
                seen.add(target)
                queue.append((target, (*path, action)))
    return None


def finish_locally(grid: Grid, candidate: str, *, max_bridge=3, max_rollback=2):
    """Complete at the first reachable anchor, starting with the longest legal prefix.

    Return None when the goal is beyond the local budget. Callers can keep the
    original candidate and its feedback in that case.
    """
    if any(type(limit) is not int or limit < 0 for limit in (max_bridge, max_rollback)):
        raise ValueError("Local finish budgets must be nonnegative integers")
    tokens = candidate.split()
    verdict = execute(grid, candidate)
    if verdict.valid:
        return LocalFinish(candidate, len(tokens), 0)
    positions = [grid.start]
    for step in verdict.trace:
        if step["reason"]:
            break
        positions.append(step["to"])
        if positions[-1] == grid.goal:
            count = len(positions) - 1
            return LocalFinish(" ".join(tokens[:count]), count, 0)
    longest = len(positions) - 1
    for count in range(longest, max(-1, longest - max_rollback - 1), -1):
        bridge = _bridge(grid, positions[count], grid.goal, max_bridge)
        if bridge is not None:
            return LocalFinish(" ".join((*tokens[:count], *bridge)), count, len(bridge))
    return None


def shorten_valid_path(grid: Grid, candidate: str):
    """Replace detours between visited cells with legal bridges of up to three steps."""
    verdict = execute(grid, candidate)
    if not verdict.valid:
        raise ValueError("Shortcut selection requires a valid plan")
    actions = candidate.split()
    positions = [grid.start, *(step["to"] for step in verdict.trace)]
    while True:
        best = None
        saved = 0
        for start, position in enumerate(positions):
            for end in range(start + 1, len(positions)):
                goal = positions[end]
                distance = sum(abs(a - b) for a, b in zip(position, goal))
                gap = end - start
                if distance > 3 or gap - distance <= saved:
                    continue
                bridge = _bridge(grid, position, goal, min(3, gap - saved - 1))
                if bridge is not None:
                    saved = gap - len(bridge)
                    best = start, end, bridge
        if best is None:
            return " ".join(actions)
        start, end, bridge = best
        actions[start:end] = bridge
        positions = [grid.start, *(step["to"] for step in execute(grid, " ".join(actions)).trace)]
