"""Build a verifier-only training file and replay frozen candidates on CPU."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ppnl.core import Grid, OfficialOracle, parse_verdict
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.experiment_freeze import SPLITS, audit_splits, file_hash
from state_baselines.step_verification import step_prompt, verify_steps


def observations(grid, verdict):
    for item in verdict.trace:
        position = item["from"]
        action, reason = item["action"], item["reason"]
        if reason == "goal_not_reached":
            if item["step"] != 1:
                break  # The preceding action already represents the unfinished state.
            action = None
        if reason:
            target = f"invalid | step=1 | reason={reason}"
        elif tuple(item["to"]) == grid.goal:
            target = "valid"
        else:
            target = "invalid | step=2 | reason=goal_not_reached"
        valid, step, reason = parse_verdict(target)
        yield {
            "path_step": item["step"], "verification_position": position, "action": action,
            "input": step_prompt(grid, position, action), "target": target,
            "label": target, "valid": valid, "step": step, "reason": reason,
        }


class ReferenceReplies:
    """Replay executor-derived responses through the model-facing scan for audit only."""

    def __init__(self):
        self.labels = {}
        self.calls = 0

    def add(self, sample):
        prompt, target = sample["input"], sample["target"]
        if self.labels.setdefault(prompt, target) != target:
            raise ValueError("Conflicting labels for the same verification input")

    def predict(self, prompts, check):
        check()
        self.calls += len(prompts)
        return [self.labels[prompt] for prompt in prompts]


def label_counts(rows):
    return dict(sorted(Counter(row["reason"] or "valid" for row in rows).items()))


def prepare(data, executor, out):
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    source_files = [data / f"{split}.jsonl" for split in SPLITS] + [data / "manifest.json"]
    hashes = {path.name: file_hash(path) for path in source_files}
    split_audit = audit_splits(data)
    oracle = OfficialOracle(executor)
    manifest = json.loads(source_files[-1].read_bytes())
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Executor does not match the original dataset manifest")
    train, reports = [], {}
    for split in SPLITS:
        path = data / f"{split}.jsonl"
        original = read_jsonl(path)
        rows = [row for row in original if "candidate" in row]
        grids = [Grid.from_dict(row["grid"]) for row in rows]
        plans = [row["candidate"] for row in rows]
        reference = ReferenceReplies()
        samples, examples = [], {}
        for row, grid, plan in zip(rows, grids, plans, strict=True):
            verdict = oracle.check(grid, plan)
            if verdict.label() != row["label"]:
                raise ValueError(f"{split}: stale label for {row['id']}")
            steps = list(observations(grid, verdict))
            for step in steps:
                reference.add(step)
            terminal = dict(steps[-1], id=f"{row['id']}-step", task="verification",
                            parent_id=row["id"], source_id=row["source_id"], layout_id=row["layout_id"],
                            grid=row["grid"], path_length=len(plan.split()))
            if (terminal["valid"], terminal["reason"]) != (verdict.valid, verdict.reason):
                raise ValueError(f"{split}: terminal label changed for {row['id']}")
            samples.append(terminal)
            case_types = [verdict.reason or "valid"]
            if not plan.split():
                case_types.append("empty_candidate")
            if not verdict.valid and any(step["valid"] for step in steps[:-1]):
                case_types.append("goal_before_final_invalid")
            if row["valid"] and row.get("valid_kind") == "nonoptimal":
                case_types.append("valid_nonoptimal")
            for case in case_types:
                if case not in examples:
                    examples[case] = {
                        "id": row["id"], "grid": row["grid"], "candidate": plan,
                        "step_replies": [{k: v for k, v in step.items()
                                          if k in ("path_step", "verification_position", "action", "target")}
                                         for step in steps],
                        "final_verdict": verdict.label(),
                    }

        predictions = verify_steps(reference, grids, plans)
        if predictions != [row["label"] for row in rows]:
            raise ValueError(f"{split}: step scan changed an original verdict")
        if split == "train":
            train = samples
        reports[split] = {
            "original_rows": len(original), "verification_candidates": len(rows),
            "executor_replay_exact": len(predictions), "step_inputs_replayed": reference.calls,
            "terminal_labels": label_counts(samples),
            "terminal_actions": dict(Counter(row["action"] or "<stop>" for row in samples)),
            "terminal_positions_in_path": dict(sorted(Counter(row["path_step"] for row in samples).items())),
            "source_path_lengths": dict(sorted(Counter(row["path_length"] for row in samples).items())),
            "unique_terminal_inputs": len({row["input"] for row in samples}),
            "layouts": len({row["layout_id"] for row in samples}), "examples": examples,
        }

    if any(file_hash(path) != hashes[path.name] for path in source_files):
        raise ValueError("Original data changed during preparation")
    report = {
        "training_rows": len(train),
        "sampling": "One terminal observation per original verification training candidate",
        "feedback_source": "Executor-derived replay on CPU; no model predictions",
        "source_sha256": hashes, "executor_sha256": oracle.sha256,
        "source_layout_counts": {name: len(item["layout_ids"]) for name, item in split_audit["splits"].items()},
        "splits": reports,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temporary:
        stage = Path(temporary) / "dataset"
        write_jsonl(stage / "train.jsonl", train)
        report["training_sha256"] = file_hash(stage / "train.jsonl")
        write_json(stage / "audit.json", report)
        stage.rename(out)
    print({split: {"matched": r["executor_replay_exact"], "labels": r["terminal_labels"]}
           for split, r in reports.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/policy_v3"))
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.executor, args.out)


if __name__ == "__main__":
    main()
