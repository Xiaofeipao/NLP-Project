"""State-based neural planning. Inference never receives a reference path."""
from __future__ import annotations

import heapq
import itertools
import math

from .core import ACTIONS

NAMES = tuple(ACTIONS)


def neighbors(grid, position):
    result = {}
    for action, (dr, dc) in ACTIONS.items():
        nxt = position[0] + dr, position[1] + dc
        if grid.inside(nxt) and nxt not in grid.obstacles:
            result[action] = nxt
    return result


def state_prompt(grid, position, local_status=False):
    dr, dc = grid.goal[0] - position[0], grid.goal[1] - position[1]
    vertical = "below" if dr > 0 else "above" if dr < 0 else "same row"
    horizontal = "right" if dc > 0 else "left" if dc < 0 else "same column"
    legal = " ".join(neighbors(grid, position))
    rows = []
    for r in range(grid.rows):
        rows.append(" ".join("G" if (r, c) == grid.goal else "P" if (r, c) == position
                             else "#" if (r, c) in grid.obstacles else "." for c in range(grid.cols)))
    status = ""
    if local_status:
        items = []
        for action, (r, c) in ACTIONS.items():
            target = position[0] + r, position[1] + c
            outcome = "outside wall" if not grid.inside(target) else "hits obstacle" if target in grid.obstacles else "free"
            items.append(f"{action}: {outcome}")
        status = "Immediate move status: " + "; ".join(items) + ".\n"
    return (f"Choose the next move toward the goal. Return one action: up down left right.\n"
            f"Current row {position[0]} column {position[1]}. Goal row {grid.goal[0]} column {grid.goal[1]}.\n"
            f"Goal direction: {vertical}, {horizontal}. Row difference {dr}. Column difference {dc}.\n"
            f"Legal moves: {legal}. Up decreases row; right increases column.\n"
            + status + "Map P=current G=goal #=obstacle .=free:\n" + "\n".join(rows))


class NeuralPolicy:
    """Score four single-token actions in one decoder step, batched across grids."""
    def __init__(self, runner, check=lambda: None):
        self.runner, self.check = runner, check
        ids = [runner.tokenizer.encode(a, add_special_tokens=False) for a in NAMES]
        if any(len(x) != 1 for x in ids):
            raise ValueError("This policy head requires one-token action words; validated for Flan-T5")
        self.ids = [x[0] for x in ids]
        self.cache = {}

    def scores(self, observations):
        import torch
        keys = [(g.layout_id, g.goal, tuple(p)) for g, p in observations]
        unseen = {}
        for key, observation in zip(keys, observations):
            if key not in self.cache:
                unseen.setdefault(key, observation)
        items = list(unseen.items())
        runner = self.runner
        runner.model.eval()
        batch_size = runner.config["training"]["eval_batch_size"]
        for offset in range(0, len(items), batch_size):
            self.check()
            batch = items[offset:offset + batch_size]
            inputs = runner.tokenizer([state_prompt(g, p, runner.config.get("policy_local_status", False))
                                       for _, (g, p) in batch], padding=True,
                                      truncation=False, pad_to_multiple_of=8, return_tensors="pt")
            if inputs.input_ids.shape[1] > runner.input_limit:
                raise ValueError("State prompt exceeds input limit; refusing truncation")
            decoder = torch.full((len(batch), 1), runner.model.config.decoder_start_token_id,
                                 dtype=torch.long, device=runner.device)
            with torch.inference_mode(), runner.autocast():
                logits = runner.model(**inputs.to(runner.device), decoder_input_ids=decoder,
                                      use_cache=False).logits[:, 0, self.ids]
            for (key, _), values in zip(batch, logits.float().cpu().tolist()):
                self.cache[key] = values
        return [self.cache[k] for k in keys]


def greedy_plans(grids, scorer, masked, max_steps=36):
    positions = [g.start for g in grids]
    plans, visited, done = [[] for _ in grids], [{g.start} for g in grids], set()
    for _ in range(max_steps):
        active = []
        choices = {}
        for i, g in enumerate(grids):
            if i in done:
                continue
            if positions[i] == g.goal:
                done.add(i)
                continue
            options = (list(a for a, p in neighbors(g, positions[i]).items() if p not in visited[i])
                       if masked else list(NAMES))
            if not options:
                done.add(i)
                continue
            choices[i] = options
            active.append(i)
        if not active:
            break
        values = scorer.scores([(grids[i], positions[i]) for i in active])
        for i, logits in zip(active, values):
            action = max(choices[i], key=lambda a: logits[NAMES.index(a)])
            plans[i].append(action)
            legal = neighbors(grids[i], positions[i])
            if action not in legal:
                done.add(i)
                continue
            positions[i] = legal[action]
            visited[i].add(positions[i])
    return [" ".join(p) for p in plans]


def search_plans(grids, scorer=None, weight=0.25, max_expansions=36):
    """Best-first graph search, optionally ranked by neural action scores.

    Symbolic legality and Manhattan distance are explicit. At most one expansion
    per cell. This is a hybrid planner, not evidence of standalone LM reasoning.
    With enough expansions, the uniform-score ablation can solve these grids too.
    """
    counter = itertools.count()
    frontiers = [[(0.0, next(counter), 0.0, g.start, [])] for g in grids]
    closed = [set() for _ in grids]
    plans, expansions = [None] * len(grids), [0] * len(grids)
    for _ in range(max_expansions + 1):
        active = []
        for i, grid in enumerate(grids):
            if plans[i] is not None:
                continue
            while frontiers[i] and frontiers[i][0][3] in closed[i]:
                heapq.heappop(frontiers[i])
            if not frontiers[i]:
                continue
            _, _, cost, position, path = heapq.heappop(frontiers[i])
            if position == grid.goal:
                plans[i] = " ".join(path)
                continue
            if expansions[i] >= max_expansions:
                continue
            closed[i].add(position)
            expansions[i] += 1
            active.append((i, position, path, cost))
        if not active:
            break
        scores = scorer.scores([(grids[i], p) for i, p, _, _ in active]) if scorer else [[0.0] * 4 for _ in active]
        for (i, position, path, cost), logits in zip(active, scores):
            maximum = max(logits)
            log_z = maximum + math.log(sum(math.exp(v - maximum) for v in logits))
            for action, nxt in neighbors(grids[i], position).items():
                if nxt in closed[i]:
                    continue
                new_cost = cost + 1 + weight * (log_z - logits[NAMES.index(action)])
                h = abs(nxt[0] - grids[i].goal[0]) + abs(nxt[1] - grids[i].goal[1])
                heapq.heappush(frontiers[i], (new_cost + h, next(counter), new_cost, nxt, path + [action]))
    # A bounded failed search returns an empty candidate, never an oracle fallback.
    return [p if p is not None else "" for p in plans], expansions


def optimal_search_plans(grids, scorer=None, goal_tiebreak=False, max_expansions=36):
    """Admissible A*: neural scores ONLY break ties between equal g+Manhattan.

    Unlike additive neural edge costs, this preserves unit-action shortest-path
    optimality. A strong non-neural ablation breaks ties by distance to goal.
    """
    counter = itertools.count()
    frontiers = [[(0, 0.0, next(counter), 0, 0.0, g.start, [])] for g in grids]
    closed = [set() for _ in grids]
    plans, expansions = [None] * len(grids), [0] * len(grids)
    for _ in range(max_expansions + 1):
        active = []
        for i, grid in enumerate(grids):
            if plans[i] is not None:
                continue
            while frontiers[i] and frontiers[i][0][5] in closed[i]:
                heapq.heappop(frontiers[i])
            if not frontiers[i]:
                continue
            _, _, _, length, nll, position, path = heapq.heappop(frontiers[i])
            if position == grid.goal:
                plans[i] = " ".join(path)
                continue
            if expansions[i] >= max_expansions:
                continue
            closed[i].add(position)
            expansions[i] += 1
            active.append((i, length, nll, position, path))
        if not active:
            break
        scores = scorer.scores([(grids[i], pos) for i, _, _, pos, _ in active]) if scorer else [[0.0] * 4 for _ in active]
        for (i, length, nll, position, path), logits in zip(active, scores):
            maximum = max(logits)
            log_z = maximum + math.log(sum(math.exp(v - maximum) for v in logits))
            for action, nxt in neighbors(grids[i], position).items():
                if nxt in closed[i]:
                    continue
                new_nll = nll + log_z - logits[NAMES.index(action)]
                h = abs(nxt[0] - grids[i].goal[0]) + abs(nxt[1] - grids[i].goal[1])
                tie = new_nll if scorer else h if goal_tiebreak else 0.0
                heapq.heappush(frontiers[i], (length + 1 + h, tie, next(counter), length + 1,
                                              new_nll, nxt, path + [action]))
    return [p if p is not None else "" for p in plans], expansions
