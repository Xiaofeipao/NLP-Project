"""Build one joint training row per source row, with state feedback on G rows."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ppnl.core import Grid, OfficialOracle
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.experiment_freeze import audit_splits, file_hash
from ppnl.policy_v3_data import distances_to_goal
from state_baselines.prepare_step_verification import label_counts, observations


def generation_row(row, grid, oracle):
    position = tuple(row["policy_position"])
    distances = distances_to_goal(grid)
    if grid.start not in distances or position not in distances or position == grid.goal:
        raise ValueError(f"Expected a reachable unfinished state for {row['id']}")
    # Ask what happens if we stop here, before choosing the next G action.
    stop = oracle.check(replace(grid, start=position), "")
    if not stop.feasible or stop.reason != "goal_not_reached" or stop.step != 1:
        raise ValueError(f"Unexpected state verdict for {row['id']}")
    return dict(row, state_verification={"valid": stop.valid, "reason": stop.reason,
                                         "local_stop_target": stop.label()})


def verification_row(row, grid, oracle):
    plan_field = "candidate"
    verdict = oracle.check(grid, row[plan_field])
    if verdict.label() != row["label"]:
        raise ValueError(f"Stale candidate label for {row['id']}")
    if verdict.reason == "goal_not_reached":
        plan_field = "target_plan"
        verdict = oracle.check(grid, row[plan_field])
        if not verdict.valid:
            raise ValueError(f"Reference path does not reach the goal for {row['id']}")
    sample = list(observations(grid, verdict))[-1]
    if sample["action"] is None or (sample["valid"], sample["reason"]) != (verdict.valid, verdict.reason):
        raise ValueError(f"Unexpected final observation for {row['id']}")
    return dict(sample, id=f"{row['id']}-step", task="verification", parent_id=row["id"],
                grid=row["grid"], source_id=row["source_id"], layout_id=row["layout_id"],
                source_plan_field=plan_field, path_length=len(row[plan_field].split()))


def prepare(data, executor, out):
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    source = audit_splits(data)
    manifest = json.loads((data / "manifest.json").read_bytes())
    oracle = OfficialOracle(executor)
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Executor does not match the source manifest")

    original = read_jsonl(data / "train.jsonl")
    rows = []
    for row in original:
        grid = Grid.from_dict(row["grid"])
        if row["task"] == "generation":
            rows.append(generation_row(row, grid, oracle))
        elif row["task"] == "verification":
            rows.append(verification_row(row, grid, oracle))
        else:
            raise ValueError(f"Unexpected task in {row['id']}: {row['task']}")

    state_labels = [row["state_verification"] for row in rows if row["task"] == "generation"]
    direct = [row for row in rows if row["task"] == "verification"]
    before = label_counts([row for row in original if row["task"] == "verification"])
    expected = Counter(before)
    expected["valid"] += expected.pop("goal_not_reached", 0)
    direct_counts = label_counts(direct)
    if direct_counts != dict(expected) or len({row["id"] for row in rows}) != len(original):
        raise ValueError("Unexpected direct verification mix or duplicated source rows")

    state_counts = label_counts(state_labels)
    sources = {f"{split}.jsonl": info["sha256"] for split, info in source["splits"].items()}
    sources["manifest.json"] = source["manifest_sha256"]
    if any(file_hash(data / name) != digest for name, digest in sources.items()):
        raise ValueError("Source data changed during preparation")
    report = {
        "schema": "dual-label-step-verification-v2", "training_rows": len(rows),
        "generation_rows": len(state_labels), "direct_verification_rows": len(direct),
        "source_direct_labels": before, "direct_verification_labels": direct_counts,
        "auxiliary_state_labels": state_counts,
        "all_verification_labels": dict(Counter(direct_counts) + Counter(state_counts)),
        "state_scope": "Stop at policy_position before taking any action. The local empty suffix has EOS index 1; the original prefix length is unknown.",
        "direct_plan_fields": dict(Counter(row["source_plan_field"] for row in direct)),
        "unique_direct_inputs": len({row["input"] for row in direct}),
        "source_sha256": sources, "executor_sha256": oracle.sha256,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temporary:
        stage = Path(temporary) / "dataset"
        write_jsonl(stage / "train.jsonl", rows)
        report["training_sha256"] = file_hash(stage / "train.jsonl")
        write_json(stage / "audit.json", report)
        stage.rename(out)
    print(f"{len(rows)} rows, {len(state_labels)} G state labels, direct V labels {direct_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/policy_v3"))
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.executor, args.out)
