"""Pre-move coordinates from the candidate actions. This module needs only the grid."""

DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def destination(rows, cols, walls, current, action):
    delta = DELTAS.get(action)
    if delta is None:
        return None
    row = current[0] + delta[0]
    col = current[1] + delta[1]
    if 0 <= row < rows and 0 <= col < cols and (row, col) not in walls:
        return row, col
    return None


def pre_move_positions(rows, cols, walls, start, actions):
    """Yield before moving. An impossible move leaves the next position at the same cell."""
    blocked = frozenset(map(tuple, walls))
    current = tuple(start)
    for action in actions:
        yield current, action
        next_cell = destination(rows, cols, blocked, current, action)
        if next_cell is not None:
            current = next_cell


def blind_case(case_id, grid, candidate):
    fields = {key: grid[key] for key in ("rows", "cols", "obstacles", "start", "goal")}
    actions = candidate.split()
    sequence = pre_move_positions(fields["rows"], fields["cols"], fields["obstacles"], fields["start"], actions)
    return {"id": case_id, "grid": fields, "candidate": candidate,
            "current_before": [list(current) for current, _ in sequence]}
