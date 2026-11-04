"""Validate and summarize the latest development-selected planning experiment."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    root = Path("outputs/planning_v3")
    status = read(root / "status.json")
    if "completed_unix" not in status or any(stage["returncode"] for stage in status["stages"]):
        raise ValueError("Study is not complete")
    selection = read(root / "selection.json")
    selected = selection["candidate"]
    candidates = {}
    for name in ("v2_reference", "single", "multi"):
        config = read(f"config.policy.v3.{name}.json")
        path = Path(config["output_root"])
        frozen = read(path / "policy_selection.json")
        weights = Path(config.get("policy_checkpoint", str(path / "flan_t5_small/train/best"))) / "model.safetensors"
        if digest(weights) != frozen["weights_sha256"] or digest(path / "eval_dev/metrics.json") != frozen["dev_metrics_sha256"]:
            raise ValueError(f"Candidate changed after development: {name}")
        metrics = read(path / "eval_dev/metrics.json")
        score = [metrics["planning"]["state_policy"]["success_rate"], metrics["planning"]["legal_mask_no_search"]["success_rate"],
                 metrics["planning"]["state_policy"]["optimality"]]
        if score != selection["development_scores"][name]:
            raise ValueError("Selection scores changed")
        candidates[name] = {"config": config, "dev": metrics, "weights_sha256": digest(weights)}
        if name != "v2_reference":
            candidates[name]["training"] = read(path / "flan_t5_small/train/training.json")
    chosen = Path(candidates[selected]["config"]["output_root"])
    prior = read("outputs/planning_v2/results.json")
    if digest(prior["weights"]) != prior["weights_sha256"]:
        raise ValueError("Previous policy weights were overwritten")
    for baseline in read("outputs/study_results.json")["baselines"].values():
        if digest(baseline["weights"]) != baseline["weights_sha256"]:
            raise ValueError("Original formal baseline weights changed")
    for split in ("test_id", "test_ood"):
        if digest(Path("data/policy_v3") / f"{split}.jsonl") != digest(Path("data/derived") / f"{split}.jsonl"):
            raise ValueError("Test candidates changed")
    result = {"selection": selection, "candidates": candidates,
              "selected_test": {s: read(chosen / f"eval_{s}/metrics.json") for s in ("test_id", "test_ood")},
              "previous_test": prior["evaluations"]["test_id"],
              "deployment_config": "config.policy.best.json", "study_wall_seconds": status["completed_unix"] - status["started_unix"],
              "integrity": {"v2_weights_unchanged": True, "test_files_unchanged": True,
                            "original_baselines_unchanged": True, "candidate_selection_development_only": True},
              "limitations": ["Same previously evaluated 200 grid environments; not a new blind test.",
                  "Local-status observations and critical-state sampling are shared changes; only single vs set-valued loss is a focused ablation.",
                  "Four-action losses use per-example normalization, so loss magnitudes are not directly comparable to the prior text-token loss.",
                  "Search remains an explicit symbolic component. Strong pure A* is reported and may expand fewer states than neural search.",
                  "ID/OOD grids are paired; generation scores should agree while prefix-repair candidates differ."]}
    result["total_gpu_reserved_wall_hours"] = sum(r["seconds"] for r in read("outputs/gpu_ledger.json")["runs"]) / 3600
    write_path = root / "results.json"
    write_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for method, metrics in result["selected_test"]["test_id"]["planning"].items():
        print(method, "success", metrics["success_rate"], "optimality", metrics["optimality"])
    print("Summary:", write_path.resolve())


if __name__ == "__main__":
    main()
