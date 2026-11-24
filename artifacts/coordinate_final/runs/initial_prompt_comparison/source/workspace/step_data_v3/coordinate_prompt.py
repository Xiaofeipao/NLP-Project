"""Render the shared-state five-label verifier input from the supplied pre-move position."""

TEMPLATE = """Current: {current}
Goal: {goal}
Walls: {walls}
Action: {action}"""


def pair(coords):
    row, col = coords
    return f"({row},{col})"


def render(rows, cols, current, goal, walls, action):
    if (rows, cols) != (6, 6):
        raise ValueError("The coordinate-only task uses a fixed 6x6 grid")
    if not isinstance(action, str) or action.split() != [action]:
        raise ValueError("Expected one candidate action token")
    return TEMPLATE.format(
        current=pair(current), goal=pair(goal), walls="[" + ", ".join(map(pair, walls)) + "]",
        action=action,
    )
