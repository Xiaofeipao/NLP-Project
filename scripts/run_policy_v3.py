"""Compare two fixed loss variants on dev, then test only the selected policy."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.data import write_json, audit_dataset
from ppnl.policy_v3_data import prepare


def main():
    root = Path("outputs/planning_v3")
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    if not Path("data/policy_v3/manifest.json").exists():
        prepare("data/derived", "data/policy_v3")
    print("Dataset:", audit_dataset("data/policy_v3"), flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    status_path = root / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"started_unix": time.time(), "stages": []}
    base = json.loads(Path("config.policy.json").read_text())
    configs = {}
    for name, loss in (("single", "single_action"), ("multi", "optimal_actions"), ("v2_reference", None)):
        config = json.loads(json.dumps(base))
        config["output_root"] = str(root / name)
        config["policy_optimal_search"] = True
        if loss:
            config["dataset_root"] = "data/policy_v3"
            config["policy_local_status"] = True
            config["training"]["policy_loss"] = loss
        else:
            config["policy_checkpoint"] = "outputs/planning_v2/flan_t5_small/train/best"
        path = Path(f"config.policy.v3.{name}.json")
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Do not change an in-progress experiment configuration")
        write_json(path, config)
        configs[name] = (path, config)

    def run(name, command, artifact):
        if artifact.exists():
            print("REUSE", name, flush=True)
            return
        print("START", name, flush=True)
        start = time.monotonic()
        with (logs / f"{name}.log").open("w", encoding="utf-8") as handle:
            result = subprocess.run([sys.executable, "-u", *command], stdout=handle, stderr=subprocess.STDOUT)
        previous = [s for s in status["stages"] if s["name"] == name]
        if previous:
            status.setdefault("previous_attempts", []).extend(previous)
            status["stages"] = [s for s in status["stages"] if s["name"] != name]
        status["stages"].append({"name": name, "seconds": time.monotonic() - start, "returncode": result.returncode})
        write_json(status_path, status)
        if result.returncode:
            print((logs / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-5000:])
            raise RuntimeError(name)
        print("DONE", name, flush=True)

    for name, (path, config) in configs.items():
        model_root = Path(config["output_root"])
        if name != "v2_reference":
            for operation, filename in (("pilot", "pilot.json"), ("train", "training.json")):
                run(f"{name}_{operation}", ["-m", "ppnl.cli", "--config", str(path), operation, "--model", "flan_t5_small"],
                    model_root / "flan_t5_small" / operation / filename)
        run(f"{name}_dev", ["scripts/evaluate_policy.py", "--config", str(path), "--split", "dev"],
            model_root / "eval_dev/metrics.json")
    scores = {}
    for name, (_, config) in configs.items():
        m = json.loads((Path(config["output_root"]) / "eval_dev/metrics.json").read_text())
        scores[name] = (m["planning"]["state_policy"]["success_rate"], m["planning"]["legal_mask_no_search"]["success_rate"],
                        m["planning"]["state_policy"]["optimality"])
    # Keep v2 on exact ties, avoiding extra complexity without development evidence.
    selected = max(("v2_reference", "single", "multi"), key=lambda name: scores[name])
    selection = {"candidate": selected, "development_scores": scores,
                 "rule": "dev unmasked policy success, then masked success, then raw optimality; exact ties keep v2 first",
                 "frozen_unix": time.time()}
    selection_path = root / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["candidate"] != selected:
            raise ValueError("Previously frozen development selection differs")
    else:
        write_json(selection_path, selection)
    print("SELECTED", selected, scores, flush=True)
    path, config = configs[selected]
    for split in ("test_id", "test_ood"):
        run(f"selected_{split}", ["scripts/evaluate_policy.py", "--config", str(path), "--split", split],
            Path(config["output_root"]) / f"eval_{split}/metrics.json")
    deployment = json.loads(json.dumps(config))
    deployment["deployment_method"] = "neural_optimal_astar"
    write_json("config.policy.best.json", deployment)
    status["completed_unix"] = time.time()
    status["selected"] = selected
    write_json(status_path, status)


if __name__ == "__main__":
    main()
