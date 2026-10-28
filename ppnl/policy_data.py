"""Re-express existing generation/repair rows as state-action supervision."""
from __future__ import annotations

import json
from pathlib import Path
import random

from .core import ACTIONS, Grid, execute
from .data import read_jsonl, write_json, write_jsonl
from .policy import state_prompt


def prepare_policy(source, output, seed=42):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    rng = random.Random(seed)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["parent_dataset"] = str(source.resolve())
    manifest["representation"] = "one sampled state-action observation per existing generation/repair example"
    manifest["no_new_environments_or_extra_rows"] = True
    for split in ("train", "dev"):
        rows = read_jsonl(source / f"{split}.jsonl")
        result = []
        for row in rows:
            if row["task"] == "verification":
                result.append(row)
                continue
            grid = Grid.from_dict(row["grid"])
            plan = row["target"] if row["task"] == "generation" else row["target_plan"]
            if not execute(grid, plan).valid:
                raise ValueError("Source training target must be feasible")
            tokens = plan.split()
            index = rng.randrange(len(tokens))
            position = grid.start
            for action in tokens[:index]:
                dr, dc = ACTIONS[action]
                position = position[0] + dr, position[1] + dc
            # No remaining path, BFS distance, or reference action is included in input.
            result.append({"id": row["id"], "source_id": row["source_id"], "layout_id": row["layout_id"],
                           "grid": row["grid"], "task": "generation", "parent_task": row["task"],
                           "policy_position": position, "policy_prefix_length": index,
                           "input": state_prompt(grid, position), "target": tokens[index]})
        write_jsonl(output / f"{split}.jsonl", result)
        manifest["counts"][split] = {"generation": sum(r["task"] == "generation" for r in result),
                                     "verification": sum(r["task"] == "verification" for r in result), "repair": 0}
    # Immutable test candidates are copied only for split auditing, never expanded into training.
    for split in ("test_id", "test_ood"):
        write_jsonl(output / f"{split}.jsonl", read_jsonl(source / f"{split}.jsonl"))
    write_json(output / "manifest.json", manifest)
