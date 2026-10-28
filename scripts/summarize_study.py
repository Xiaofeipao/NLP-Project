"""Validate completed formal-study artifacts and collect their measured results."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = read(args.config)
    root = Path(config["output_root"])
    status = read(root / "study_status.json")
    if "completed_unix" not in status or any(s["returncode"] for s in status["stages"]):
        raise RuntimeError("Formal study has not completed successfully")
    selection = read(root / "selection.json")
    primary = selection["primary"]
    result = {"primary": primary, "selection": selection, "baselines": {}, "primary_full": {}}
    configs, data = [], set()
    for key in config["models"]:
        path = root / key / "train" / "training.json"
        training = read(path)
        expected = selection["training_fingerprints"][key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Training metadata changed after selection: {key}")
        configs.append(training["training_config"])
        data.add(training["data_fingerprint"])
        weights = root / key / "train" / "best" / "model.safetensors"
        with weights.open("rb") as handle:
            weights_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        best = min(training["history"], key=lambda x: x["dev_token_mean_loss"])
        result["baselines"][key] = {"training": training, "best_epoch": best["epoch"],
            "weights": str(weights.resolve()), "weights_sha256": weights_hash,
            "dev": read(root / key / "eval_dev" / "metrics.json"),
            "test_id": read(root / key / "eval_test_id" / "metrics.json")}
    if any(c != configs[0] for c in configs) or len(data) != 1:
        raise ValueError("Baseline training budgets or data do not match")
    for split in ("test_id", "test_ood"):
        result["primary_full"][split] = read(root / primary / f"eval_{split}_full" / "metrics.json")
    ledger = read(config["gpu_ledger"])
    result["gpu_reserved_wall_seconds_including_smoke"] = sum(r["seconds"] for r in ledger["runs"])
    result["study_wall_seconds"] = status["completed_unix"] - status["started_unix"]
    result["decoding_protocol"] = read(root / "decoding_protocol.json")
    result["limitations"] = ["One seed and one fixed training configuration; no test-driven retuning.",
        "Pilot and previous technical smoke updates are separately logged; final checkpoints start from original pretrained weights.",
        "Training report dev_exact_match is diagnostic; formal selection uses eval_dev task metrics.",
        "Paired ID/OOD grids share direct-generation and end-to-end inputs; the corruption shift affects supplied candidates."]
    (root / "study_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key, values in result["baselines"].items():
        metrics = values["test_id"]
        print(key, json.dumps({"test_verification_macro_f1": metrics["verification"]["macro_f1"],
                              "test_generation_success": metrics["direct_generation"]["success_rate"],
                              "test_repair_success": metrics["repair"]["success_rate"]}))
    print("Primary:", primary)
    print("Result:", (root / "study_results.json").resolve())


if __name__ == "__main__":
    main()
