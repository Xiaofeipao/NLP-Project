"""Train-only critical-state sampling and labels allowing all shortest next moves."""
from collections import deque
import json
from pathlib import Path
import random

from .core import ACTIONS, Grid
from .data import read_jsonl, write_json, write_jsonl
from .policy import neighbors, state_prompt


def distances_to_goal(grid):
    distances = {grid.goal: 0}
    queue = deque([grid.goal])
    while queue:
        current = queue.popleft()
        for nxt in neighbors(grid, current).values():
            if nxt not in distances:
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
    return distances


def optimal_moves(grid, position, distances):
    return [a for a, p in neighbors(grid, position).items()
            if distances.get(p) == distances[position] - 1]


def critical(grid, position):
    dr, dc = grid.goal[0] - position[0], grid.goal[1] - position[1]
    toward = []
    if dr:
        toward.append("down" if dr > 0 else "up")
    if dc:
        toward.append("right" if dc > 0 else "left")
    return any((position[0] + ACTIONS[a][0], position[1] + ACTIONS[a][1]) in grid.obstacles for a in toward)


def prepare(source, output, seed=42):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    rng = random.Random(seed)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["parent_dataset"] = str(source.resolve())
    manifest["representation"] = "one reachable state per source planning/repair example; 50% critical-state sampling on train only"
    manifest["source_counts"] = json.loads(json.dumps(manifest["counts"]))
    manifest["source_reason_counts"] = manifest.pop("reason_counts", {})
    manifest["critical_states_sampled"] = {}
    for split in ("train", "dev"):
        rows, result, hard_count = read_jsonl(source / f"{split}.jsonl"), [], 0
        for row in rows:
            if row["task"] == "verification":
                result.append(row)
                continue
            grid = Grid.from_dict(row["grid"])
            distances = distances_to_goal(grid)
            states = sorted(p for p in distances if p != grid.goal)
            hard = [p for p in states if critical(grid, p)]
            if split == "train" and hard and rng.random() < 0.5:
                position = rng.choice(hard)
                hard_count += 1
            else:
                position = rng.choice(states)
            actions = optimal_moves(grid, position, distances)
            if not actions:
                raise AssertionError("Reachable non-goal state must have a shortest next move")
            result.append({"id": row["id"], "source_id": row["source_id"], "layout_id": row["layout_id"],
                           "grid": row["grid"], "task": "generation", "parent_task": row["task"],
                           "policy_position": position, "optimal_actions": actions,
                           "input": state_prompt(grid, position, local_status=True), "target": actions[0]})
        write_jsonl(output / f"{split}.jsonl", result)
        manifest["counts"][split] = {"generation": sum(r["task"] == "generation" for r in result),
                                     "verification": sum(r["task"] == "verification" for r in result), "repair": 0}
        manifest["critical_states_sampled"][split] = hard_count
    for split in ("test_id", "test_ood"):
        write_jsonl(output / f"{split}.jsonl", read_jsonl(source / f"{split}.jsonl"))
    manifest["label_rules"] = "BFS is used only to construct training/development labels; inference only sees local status and known grid coordinates"
    write_json(output / "manifest.json", manifest)
