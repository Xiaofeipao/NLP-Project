"""Run both formal baselines and the selected model's complete study serially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    state = {"started_unix": time.time(), "config": str(config_path), "stages": []}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def run(name, command, artifact=None):
        if artifact and artifact.exists():
            print(f"Existing stage, reuse: {name}", flush=True)
            return
        print(f"START {name}", flush=True)
        start = time.monotonic()
        path = logs / f"{name}.log"
        with path.open("w", encoding="utf-8") as handle:
            result = subprocess.run([sys.executable, "-u", "-m", "ppnl.cli", "--config", str(config_path), *command],
                                    stdout=handle, stderr=subprocess.STDOUT)
        stage = {"name": name, "seconds": time.monotonic() - start, "returncode": result.returncode, "log": str(path)}
        state["stages"].append(stage)
        (root / "study_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        if result.returncode:
            print(path.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
            raise RuntimeError(f"Stage failed: {name}; see {path}")
        print(f"DONE {name}: {stage['seconds']:.1f}s", flush=True)

    for key in config["models"]:
        for command, file in (("pilot", "pilot.json"), ("train", "training.json")):
            run(f"{key}_{command}", [command, "--model", key], root / key / command / file)
        run(f"{key}_dev", ["evaluate", "--model", key, "--split", "dev"], root / key / "eval_dev" / "metrics.json")
    run("select", ["select"], root / "selection.json")
    for key in config["models"]:
        run(f"{key}_test_id", ["evaluate", "--model", key, "--split", "test_id"],
            root / key / "eval_test_id" / "metrics.json")
    primary = json.loads((root / "selection.json").read_text())["primary"]
    for split in ("test_id", "test_ood"):
        run(f"{primary}_{split}_full", ["evaluate", "--model", primary, "--split", split, "--full"],
            root / primary / f"eval_{split}_full" / "metrics.json")
    state["completed_unix"] = time.time()
    state["primary"] = primary
    (root / "study_status.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Study completed. Primary model: {primary}", flush=True)


if __name__ == "__main__":
    main()
