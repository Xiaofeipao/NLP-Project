"""Train the primary state policy and freeze development selection before testing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.data import audit_dataset, write_json
from ppnl.policy_data import prepare_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.policy.json")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = json.loads(path.read_text())
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not (Path(config["dataset_root"]) / "manifest.json").exists():
        prepare_policy(config["policy_source_dataset"], config["dataset_root"], config["seed"])
    print("Policy dataset:", audit_dataset(config["dataset_root"]), flush=True)
    snapshot = root / "config_snapshot.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != config:
        raise ValueError("Configuration changed; use a fresh output_root rather than reuse old stages")
    write_json(snapshot, config)
    status_path = root / "status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {"started_unix": time.time(), "stages": []}
    jobs = [("pilot", ["-m", "ppnl.cli", "--config", str(path), "pilot", "--model", "flan_t5_small"],
             root / "flan_t5_small/pilot/pilot.json"),
            ("train", ["-m", "ppnl.cli", "--config", str(path), "train", "--model", "flan_t5_small"],
             root / "flan_t5_small/train/training.json")]
    jobs.extend((split, ["scripts/evaluate_policy.py", "--config", str(path), "--split", split],
                 root / f"eval_{split}/metrics.json") for split in ("dev", "test_id", "test_ood"))
    for name, command, artifact in jobs:
        if artifact.exists():
            print("Reuse completed stage:", name, flush=True)
            continue
        previous = [stage for stage in state["stages"] if stage["name"] == name]
        if previous:
            state.setdefault("previous_attempts", []).extend(previous)
            state["stages"] = [stage for stage in state["stages"] if stage["name"] != name]
        print("START", name, flush=True)
        started = time.monotonic()
        with (logs / f"{name}.log").open("w", encoding="utf-8") as handle:
            result = subprocess.run([sys.executable, "-u", *command], stdout=handle, stderr=subprocess.STDOUT)
        state["stages"].append({"name": name, "seconds": time.monotonic() - started, "returncode": result.returncode})
        write_json(root / "status.json", state)
        if result.returncode:
            print((logs / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-5000:])
            raise RuntimeError(f"Failed: {name}")
        print("DONE", name, flush=True)
    state["completed_unix"] = time.time()
    write_json(root / "status.json", state)


if __name__ == "__main__":
    main()
