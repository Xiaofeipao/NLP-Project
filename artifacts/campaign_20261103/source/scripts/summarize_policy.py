"""Check policy-study integrity and collect all measured comparisons."""
import hashlib
import json
from pathlib import Path

from argparse import ArgumentParser


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.policy.json")
    args = parser.parse_args()
    config = read(args.config)
    root = Path(config["output_root"])
    status = read(root / "status.json")
    if "completed_unix" not in status or any(stage["returncode"] for stage in status["stages"]):
        raise ValueError("Policy study is incomplete")
    selection = read(root / "policy_selection.json")
    weights = root / "flan_t5_small/train/best/model.safetensors"
    if sha(weights) != selection["weights_sha256"]:
        raise ValueError("Policy checkpoint changed since development selection")
    if sha(root / "eval_dev/metrics.json") != selection["dev_metrics_sha256"]:
        raise ValueError("Development metrics changed since selection")
    baseline = read("outputs/study_results.json")
    for model in baseline["baselines"].values():
        if sha(model["weights"]) != model["weights_sha256"]:
            raise ValueError("Original baseline weights changed")
    for split in ("test_id", "test_ood"):
        if sha(Path(config["dataset_root"]) / f"{split}.jsonl") != sha(Path(config["policy_source_dataset"]) / f"{split}.jsonl"):
            raise ValueError("Test candidates changed")
    for split in ("train", "dev"):
        source = [json.loads(line) for line in (Path(config["policy_source_dataset"]) / f"{split}.jsonl").read_text().splitlines()]
        transformed = [json.loads(line) for line in (Path(config["dataset_root"]) / f"{split}.jsonl").read_text().splitlines()]
        original = {r["id"]: (r["source_id"], r["layout_id"]) for r in source}
        modified = {r["id"]: (r["source_id"], r["layout_id"]) for r in transformed}
        if len(source) != len(transformed) or original != modified:
            raise ValueError("Policy data did not preserve source example IDs and layouts")
    result = {"selection": selection, "training": read(root / "flan_t5_small/train/training.json"),
              "evaluations": {s: read(root / f"eval_{s}/metrics.json") for s in ("dev", "test_id", "test_ood")},
              "weights": str(weights.resolve()), "weights_sha256": sha(weights),
              "study_wall_seconds": status["completed_unix"] - status["started_unix"],
              "integrity": {"baseline_weights_unchanged": True, "test_candidates_unchanged": True,
                            "selection_frozen_before_testing": True, "source_example_ids_and_layouts_preserved": True},
              "limitations": ["State policy receives exact current state, local legal moves and coordinate-derived goal direction; it is not the old one-shot natural-language baseline.",
                  "Search success includes symbolic graph search; pure A* also solves these environments.",
                  "ID/OOD share grids. OOD concerns corrupted candidate repair, not new grid sizes.",
                  "Original test aggregate performance was already known before this improvement; no per-test tuning was performed in this run.",
                  "Prefix repair includes symbolic execution and truncation before learned replanning.",
                  "Single seed, only one fixed retraining configuration; state count savings do not imply wall-clock speedup."]}
    ledger = read(config["gpu_ledger"])
    result["total_gpu_reserved_wall_seconds_all_studies"] = sum(r["seconds"] for r in ledger["runs"])
    paths = [Path("ppnl/policy.py"), Path("ppnl/policy_data.py"), Path("scripts/evaluate_policy.py"),
             Path("scripts/run_policy_study.py"), Path("scripts/plan_grid.py"), Path("config.policy.json")]
    result["source_sha256"] = {str(p): sha(p) for p in paths}
    (root / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Verified all artifacts. Summary:", (root / "results.json").resolve())


if __name__ == "__main__":
    main()
