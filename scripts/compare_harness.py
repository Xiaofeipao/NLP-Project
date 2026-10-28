"""Paired agent trials with identical budgets and fixed input order; no training.

Uses only grid fields from the dataset. Reference paths are used only by the
post-run metric function. Whole-model and policy contributions stay separate.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid
from ppnl.data import read_jsonl, write_json
from ppnl.evaluation import plan_metrics
from ppnl.harness import Budgets, create_session, fingerprint, run_session
from ppnl.harness_backends import load_backend
from ppnl.modeling import gpu_budget
from run_harness import prepare


def clear_cache(backend):
    policy = getattr(backend, "policy", backend)
    scorer = getattr(policy, "scorer", None)
    if scorer is not None:
        scorer.cache.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/harness/repair_policy.json")
    parser.add_argument("--split", choices=("dev", "test_id", "test_ood"), default="dev")
    parser.add_argument("--limit", type=int, default=20, help="first N unique grids; 0 means all")
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--methods", nargs="+", choices=("repair", "repair_policy", "policy"),
                        default=["repair", "repair_policy", "policy"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.limit < 0 or len(set(args.methods)) != len(args.methods):
        parser.error("limit must be nonnegative and methods must be distinct")
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if spec["backend"] != "repair_policy":
        parser.error("comparison spec must define repair_policy and both checkpoints")
    spec["budgets"]["model_calls"] = spec["budgets"]["tool_calls"] = args.calls
    Budgets(**spec["budgets"])
    output = Path(args.out).resolve()
    if Path("outputs/agent_harness").resolve() not in output.parents:
        parser.error("output must be a fresh directory under outputs/agent_harness")
    config, _, _, _, _ = prepare(spec)
    dataset = Path(config["dataset_root"]) / f"{args.split}.jsonl"
    unique = {}
    for row in read_jsonl(dataset):
        if args.split == "dev" and row["task"] != "verification":
            continue
        grid = Grid.from_dict(row["grid"])
        unique.setdefault((grid.layout_id, grid.start, grid.goal), (row["id"], grid))
    samples = list(unique.values())
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("No evaluation grids")
    output.mkdir(parents=True, exist_ok=False)
    results = {"split": args.split, "dataset_sha256": fingerprint(read_jsonl(dataset)),
               "selection": "first unique grids in source order; no success filtering",
               "sample_ids": [identifier for identifier, _ in samples],
               "budgets": spec["budgets"], "methods": {},
               "note": "repair_policy uses a second trained model, no graph search; policy is its ablation"}
    write_json(output / "inputs.json", {"spec": spec, "evaluation": results})
    for method in args.methods:
        selected = deepcopy(spec)
        selected["backend"] = method
        if method == "policy":
            selected.update(model_config=spec["recovery_config"], checkpoint=spec["recovery_checkpoint"], legal_mask=True)
        settings, _, budgets, oracle, identity = prepare(selected)
        records = []
        with gpu_budget(settings["gpu_ledger"], settings["gpu_hours_cap"], f"compare_harness:{method}:{args.split}") as check:
            backend = load_backend(selected, settings, check)
            for index, (identifier, grid) in enumerate(samples):
                check()
                clear_cache(backend)
                directory = output / method / f"{index:04d}"
                trial_spec = deepcopy(selected)
                trial_spec["grid"] = str(directory / "grid.json")
                trial_identity = deepcopy(identity)
                trial_identity["spec_sha256"] = fingerprint(trial_spec)
                create_session(directory, grid, budgets, trial_identity,
                               {"kind": "official", "sha256": oracle.sha256})
                write_json(directory / "grid.json", asdict(grid))
                write_json(directory / "spec.json", trial_spec)
                state = run_session(directory, backend, trial_identity, oracle, check)
                verdict = oracle.check(grid, state["plan"])
                if verdict.valid != (state["status"] == "success"):
                    raise AssertionError("Environment and runtime success disagree")
                records.append({"id": identifier, "grid": asdict(grid), "state": state})
                write_json(output / f"{method}_trials.json", records)
                if (index + 1) % 5 == 0 or index + 1 == len(samples):
                    successes = sum(r["state"]["status"] == "success" for r in records)
                    print(f"{method}: {index + 1}/{len(samples)}, successes={successes}", flush=True)
            del backend
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        metrics = plan_metrics([g for _, g in samples], [r["state"]["plan"] for r in records])
        metrics.update(mean_model_calls=sum(r["state"]["model_calls"] for r in records) / len(records),
                       mean_invalid_calls=sum(r["state"]["invalid_calls"] for r in records) / len(records),
                       active_seconds=sum(r["state"]["active_seconds"] for r in records),
                       weights=identity)
        results["methods"][method] = metrics
        write_json(output / "results.json", results)
        print(json.dumps({method: {k: v for k, v in metrics.items() if k != "weights"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
