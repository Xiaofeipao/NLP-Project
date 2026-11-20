"""Sample from the supplied split only. Geometry drives paired observations."""
from collections import defaultdict, deque
from functools import lru_cache

from ppnl.core import ACTIONS, Grid, execute
from ppnl.policy import neighbors
from ppnl.policy_v3_data import distances_to_goal, optimal_moves
from .schema import State


DIST = lru_cache(maxsize=12000)(distances_to_goal)
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
BAD_WORDS = ("jump", "wait", "stay", "spin", "walk", "north", "east", "turn")


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def forced_detour(grid, position, moves):
    options = neighbors(grid, position)
    return all(manhattan(options[a], grid.goal) > manhattan(position, grid.goal) for a in moves)


class Donors:
    def __init__(self, rows):
        self.by_layout = defaultdict(list)
        self.traces = defaultdict(lambda: defaultdict(list))
        by_walls = {}
        for row in rows:
            grid = Grid.from_dict(row["grid"])
            self.by_layout[grid.layout_id].append((grid, row["id"]))
            by_walls[frozenset(grid.obstacles)] = grid.layout_id
            for tick in row.get("trace", []):
                action = tick["action"]
                if action not in ACTIONS or tuple(tick["from"]) == grid.goal:
                    continue
                reason = tick["reason"]
                if reason == "out_of_bounds":
                    family = "bounds"
                elif reason == "obstacle_collision":
                    family = "collision"
                elif reason is None:
                    family = "arrive" if tuple(tick["to"]) == grid.goal else "continue"
                else:
                    continue
                self.traces[family][action].append((grid, row["id"], tick))
        self.layouts = sorted(self.by_layout)
        self.wall_pairs = [(by_walls[walls - {point}], larger, point)
                           for walls, larger in sorted(by_walls.items(), key=lambda item: item[1])
                           for point in sorted(walls) if walls - {point} in by_walls]

    def sample(self, rng, layout=None):
        return rng.choice(self.by_layout[layout or rng.choice(self.layouts)])

    def observed(self, family, rng):
        bucket = self.traces["continue" if family == "move_goal" else family]
        action = rng.choice(sorted(bucket))
        grid, donor, tick = rng.choice(bucket[action])
        state = State.from_grid(grid, tick["from"])
        result = [(state, action, donor, {"step": tick["step"], "goal_changed": False})]
        if family == "move_goal":
            alternative = State.from_grid(grid, tick["from"], tick["to"])
            result.append((alternative, action, donor, {"step": tick["step"], "goal_changed": True}))
        return result


def random_path(grid, rng):
    distances = DIST(grid)
    moves, position = [], grid.start
    while position != grid.goal:
        action = rng.choice(optimal_moves(grid, position, distances))
        moves.append(action)
        position = neighbors(grid, position)[action]
    return moves


def positions(grid, moves):
    result = [grid.start]
    for action in moves:
        result.append(neighbors(grid, result[-1])[action])
    return result


def longer_path(grid, base, kind, rng):
    points = positions(grid, base)
    if kind == "backtrack":
        index = rng.randrange(len(base))
        action = rng.choice(list(neighbors(grid, points[index])))
        return base[:index] + [action, OPPOSITE[action]] + base[index:]
    if kind == "square":
        choices = []
        for i, start in enumerate(points[:-1]):
            for first, p in neighbors(grid, start).items():
                for second, q in neighbors(grid, p).items():
                    if second in (first, OPPOSITE[first]):
                        continue
                    r = neighbors(grid, q).get(OPPOSITE[first])
                    if r is not None and neighbors(grid, r).get(OPPOSITE[second]) == start:
                        choices.append((i, [first, second, OPPOSITE[first], OPPOSITE[second]]))
        if choices:
            index, cycle = rng.choice(choices)
            return base[:index] + cycle + base[index:]
        return None
    if kind != "simple_detour":
        raise ValueError(kind)
    # Find a longer path with no repeated cell by avoiding one edge of the shortest path.
    edges = list(zip(points, points[1:]))
    rng.shuffle(edges)
    for p, q in edges:
        queue, previous = deque([grid.start]), {grid.start: None}
        while queue and grid.goal not in previous:
            current = queue.popleft()
            options = list(neighbors(grid, current).items())
            rng.shuffle(options)
            for action, nxt in options:
                if (current, nxt) in ((p, q), (q, p)) or nxt in previous:
                    continue
                previous[nxt] = (current, action)
                queue.append(nxt)
        if grid.goal in previous:
            path, at = [], grid.goal
            while at != grid.start:
                at, action = previous[at]
                path.append(action)
            if len(path) > len(base):
                return path[::-1]
    return None


def move_pair(family, donors, rng, words):
    """Each result has state, candidate action (empty for stopping), and source row ID."""
    if family == "wall":
        small, large, dest = rng.choice(donors.wall_pairs)
        open_grid, a = donors.sample(rng, small)
        blocked, b = donors.sample(rng, large)
        reachable = list(DIST(blocked))
        choices = []
        for action, (dr, dc) in ACTIONS.items():
            origin = (dest[0] - dr, dest[1] - dc)
            if origin in reachable:
                choices.append((origin, action))
        if not choices or len(reachable) < 2:
            return []
        origin, action = rng.choice(choices)
        goal = rng.choice([p for p in reachable if p != origin])
        return [(State.from_grid(g, origin, goal), action, donor)
                for g, donor in ((open_grid, a), (blocked, b))]

    grid, donor = donors.sample(rng)
    cells = list(DIST(grid))
    origin = rng.choice(cells)
    legal = neighbors(grid, origin)
    if not legal:
        return []
    action, dest = rng.choice(list(legal.items()))
    other = [p for p in cells if p != origin and p != dest]
    if not other:
        return []
    goal = rng.choice(other)
    at = lambda now, end, token: (State.from_grid(grid, now, end), token, donor)
    if family == "stop_goal":
        return [at(origin, origin, ""), at(origin, goal, "")]
    if family == "token":
        return [at(origin, dest, action), at(origin, dest, rng.choice(words))]
    if family == "edge":
        options = []
        for token, (dr, dc) in ACTIONS.items():
            toward = (origin[0] + dr, origin[1] + dc)
            if not grid.inside(toward):
                inner = legal.get(OPPOSITE[token])
                if inner is not None:
                    options.append((token, inner))
        if not options:
            return []
        token, inner = rng.choice(options)
        goals = [p for p in cells if p not in (origin, inner)]
        if not goals:
            return []
        goal = rng.choice(goals)
        return [at(origin, goal, token), at(inner, goal, token)]
    if family == "unknown":
        return [at(origin, goal, rng.choice(words))]
    raise ValueError(family)


def apply_edit(base, edit):
    i, operation = edit["offset"], edit["operation"]
    if operation == "truncate":
        return base[:i]
    if operation == "delete":
        return base[:i] + base[i+1:]
    if operation == "insert":
        return base[:i] + [edit["token"]] + base[i:]
    if operation == "replace":
        return base[:i] + [edit["token"]] + base[i+1:]
    raise ValueError(operation)


def single_error(grid, base, reason, operation, rng, words):
    if operation == "truncate":
        offset = rng.randrange(len(base))
        edit = {"operation": operation, "offset": offset}
        return apply_edit(base, edit), edit
    edits = []
    for i in range(len(base) + (operation == "insert")):
        if operation == "delete":
            edits.append({"operation": operation, "offset": i})
        else:
            tokens = [rng.choice(words)] if reason == "invalid_action_token" else list(ACTIONS)
            edits.extend({"operation": operation, "offset": i, "token": token} for token in tokens
                         if operation != "replace" or token != base[i])
    rng.shuffle(edits)
    for edit in edits:
        changed = apply_edit(base, edit)
        if execute(grid, " ".join(changed)).reason == reason:
            return changed, edit
    return None
