"""CPU-only score controls for the same bounded action decoder."""
from __future__ import annotations

from .constrained_decoding import DecodeSpec, decode_actions
from .core import ACTIONS


def rule_repair(grid, kind, spec=DecodeSpec(), check=lambda: None):
    if not spec.state_constraints or kind not in ("uniform", "manhattan"):
        raise ValueError("Rule controls require executor-constrained decoding")
    # One extra action must cost more than the largest possible sum of
    # Manhattan-distance differences over the entire search horizon.
    length_cost = spec.max_actions * (grid.rows + grid.cols) + 1

    def score(prefixes):
        rows = []
        for prefix in prefixes:
            position = grid.start
            for action in prefix:
                dr, dc = ACTIONS[action]
                position = (position[0] + dr, position[1] + dc)
            values = {}
            for action, (dr, dc) in ACTIONS.items():
                next_position = (position[0] + dr, position[1] + dc)
                distance = abs(next_position[0] - grid.goal[0]) + abs(next_position[1] - grid.goal[1])
                values[action] = -1 if kind == "uniform" else -(length_cost + distance)
            values["<eos>"] = 0
            rows.append(values)
        return rows

    return decode_actions(grid, score, spec, check)
