"""Evaluate the state policy and explicit symbolic-search ablations."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, OfficialOracle, execute
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.policy import NeuralPolicy, greedy_plans, search_plans, optimal_search_plans


def key(grid):
    return grid.layout_id, grid.start, grid.goal


def repair_prefix(grid, candidate):
    """Keep only physically executed actions; discard the failing action and suffix."""
    verdict = execute(grid, candidate)
    prefix, position = [], grid.start
    for event in verdict.trace:
        if event["reason"]:
            break
        prefix.append(event["action"])
        position = tuple(event["to"])
        if position == grid.goal:
            return prefix, None
    return prefix, replace(grid, start=position)


def evaluate(config, split):
    root = Path(config["output_root"])
    destination = root / f"eval_{split}"
    if destination.exists():
        raise FileExistsError(destination)
    rows = read_jsonl(Path(config["policy_source_dataset"]) / f"{split}.jsonl")
    if split == "dev":
        rows = [r for r in rows if r["task"] == "verification"]
    unique = {}
    for row in rows:
        grid = Grid.from_dict(row["grid"])
        unique.setdefault(key(grid), grid)
    grids = list(unique.values())
    baseline_dir = Path(config["baseline_root"]) / ("eval_dev" if split == "dev" else f"eval_{split}_full")
    original = {key(Grid.from_dict(r["grid"])): r["direct"]["plan"]
                for r in read_jsonl(baseline_dir / "pipeline_traces.jsonl")}
    baseline_plans = [original[key(g)] for g in grids]
    weights = Path(config.get("policy_checkpoint", str(root / "flan_t5_small" / "train" / "best")))
    with (weights / "model.safetensors").open("rb") as handle:
        weight_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if split != "dev":
        selection = json.loads((root / "policy_selection.json").read_text())
        if selection["weights_sha256"] != weight_hash:
            raise ValueError("Policy changed after development selection")
    require_cuda()
    manifest = json.loads((Path(config["dataset_root"]) / "manifest.json").read_text())
    oracle = OfficialOracle(manifest["executor_path"])
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Official executor changed")
    methods, times, expansion_counts = {"original_whole_plan": baseline_plans}, {}, {}
    traces = []
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], f"policy_evaluation:{split}") as check:
        runner = Seq2Seq(str(weights), config)
        scorer = NeuralPolicy(runner, check)
        methods_to_run = ["state_policy", "legal_mask_no_search", "neural_best_first", "symbolic_astar_no_model"]
        if config.get("policy_optimal_search"):
            methods_to_run += ["neural_optimal_astar", "symbolic_astar_goal_tiebreak"]
        for method in methods_to_run:
            check()
            started = time.perf_counter()
            if method == "state_policy":
                plans = greedy_plans(grids, scorer, masked=False)
            elif method == "legal_mask_no_search":
                plans = greedy_plans(grids, scorer, masked=True)
            elif method in ("neural_optimal_astar", "symbolic_astar_goal_tiebreak"):
                plans, expansions = optimal_search_plans(grids, scorer if method == "neural_optimal_astar" else None,
                                                          goal_tiebreak=method == "symbolic_astar_goal_tiebreak")
                expansion_counts[method] = sum(expansions) / len(expansions)
            else:
                plans, expansions = search_plans(grids, scorer if method == "neural_best_first" else None,
                                                 weight=0.25 if method == "neural_best_first" else 0.0)
                expansion_counts[method] = sum(expansions) / len(expansions)
            methods[method] = plans
            times[method] = time.perf_counter() - started
        metrics = {method: plan_metrics(grids, plans) for method, plans in methods.items()}
        invalid = [row for row in rows if not row["valid"]]
        prefixes, repair_grids, indices = {}, [], []
        for i, row in enumerate(invalid):
            grid = Grid.from_dict(row["grid"])
            prefix, remaining_grid = repair_prefix(grid, row["candidate"])
            prefixes[i] = prefix
            if remaining_grid is not None:
                repair_grids.append(remaining_grid)
                indices.append(i)
        repairs = {}
        for method in methods_to_run:
            check()
            if method in ("state_policy", "legal_mask_no_search"):
                suffixes = greedy_plans(repair_grids, scorer, masked=method == "legal_mask_no_search")
            elif method in ("neural_optimal_astar", "symbolic_astar_goal_tiebreak"):
                suffixes, _ = optimal_search_plans(repair_grids, scorer if method == "neural_optimal_astar" else None,
                                                  goal_tiebreak=method == "symbolic_astar_goal_tiebreak")
            else:
                suffixes, _ = search_plans(repair_grids, scorer if method == "neural_best_first" else None,
                                           weight=0.25 if method == "neural_best_first" else 0.0)
            by_index = dict(zip(indices, suffixes))
            repairs[method] = [" ".join(prefixes[i] + by_index.get(i, "").split()) for i in range(len(invalid))]
        repair_metrics = {method: plan_metrics([Grid.from_dict(r["grid"]) for r in invalid], plans)
                          for method, plans in repairs.items()}
        for i, grid in enumerate(grids):
            result = {"grid": asdict(grid), "plans": {}}
            for method, plans in methods.items():
                verdict = oracle.check(grid, plans[i])
                result["plans"][method] = {"plan": plans[i], "verdict": verdict.label(), "trace": verdict.trace}
            traces.append(result)
        repair_traces = []
        for i, row in enumerate(invalid):
            grid = Grid.from_dict(row["grid"])
            repair_traces.append({"id": row["id"], "grid": row["grid"], "candidate": row["candidate"],
                                  "plans": {m: {"plan": p[i], "verdict": oracle.check(grid, p[i]).label()}
                                            for m, p in repairs.items()}})
    output = {"split": split, "weights_sha256": weight_hash, "planning": metrics,
              "prefix_repair": repair_metrics, "seconds_by_planner": times,
              "mean_expansions": expansion_counts,
              "inference": {"max_steps": 36, "max_expansions": 36, "neural_weight": 0.25,
                            "reference_path_used": False, "symbolic_legality": True,
                            "search_uses_manhattan_heuristic": True,
                            "cached_policy_states": len(scorer.cache)},
              "definitions": {"state_policy": "learned one-action policy with explicit state observations; no movement mask or search",
                              "legal_mask_no_search": "same policy plus legal unvisited-action masking",
                              "neural_best_first": "hybrid best-first graph search ranked by learned action likelihood and Manhattan distance",
                              "symbolic_astar_no_model": "A* without any neural scores; necessary search ablation",
                              "neural_optimal_astar": "unit-cost A*, neural likelihood only breaks equal g+Manhattan ties",
                              "symbolic_astar_goal_tiebreak": "strong pure A* ablation: lower Manhattan breaks equal f ties",
                              "repair": "executor-valid prefix plus replanned suffix, invalid candidates only",
                              "timings": "shared state-score cache; times are incremental, not independent cold-start comparisons",
                              "paired_test": "ID/OOD use same grids; planning scores should match, prefix repair candidates differ"}}
    write_json(destination / "metrics.json", output)
    write_jsonl(destination / "planning_traces.jsonl", traces)
    write_jsonl(destination / "repair_traces.jsonl", repair_traces)
    if split == "dev":
        eligible = ("state_policy", "legal_mask_no_search", "neural_optimal_astar" if config.get("policy_optimal_search") else "neural_best_first")
        selected = max(eligible, key=lambda name: (metrics[name]["success_rate"], metrics[name]["optimality"]))
        write_json(root / "policy_selection.json", {"method": selected, "weights_sha256": weight_hash,
                                                    "rule": "development success then optimality; ties retain simpler earlier method",
                                                    "dev_metrics_sha256": hashlib.sha256((destination / "metrics.json").read_bytes()).hexdigest()})
    print(json.dumps({"split": split, "planning": metrics, "prefix_repair": repair_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.policy.json")
    parser.add_argument("--split", choices=["dev", "test_id", "test_ood"], required=True)
    args = parser.parse_args()
    evaluate(json.loads(Path(args.config).read_text(encoding="utf-8")), args.split)
