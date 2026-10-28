"""Agent harness: create/run, resume, or inspect an isolated grid-planning trial.

Run from the repository root. inspect and --dry-run never load a model.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, OfficialOracle
from ppnl.harness import Budgets, create_session, read_session, run_session
from ppnl.harness_backends import backend_identity, load_backend


def prepare(spec):
    if spec["backend"] not in ("policy", "repair", "repair_policy"):
        raise ValueError("backend must be policy, repair or repair_policy")
    if not isinstance(spec.get("legal_mask", False), bool):
        raise ValueError("legal_mask must be boolean")
    if not isinstance(spec.get("avoid_revisits", False), bool):
        raise ValueError("avoid_revisits must be boolean")
    config = json.loads(Path(spec["model_config"]).read_text(encoding="utf-8"))
    config["local_files_only"] = True
    value = json.loads(Path(spec["grid"]).read_text(encoding="utf-8"))
    grid = Grid.from_world(value) if isinstance(value, list) else (
        Grid.from_world(value["world"]) if "world" in value else Grid.from_dict(value))
    manifest = json.loads((Path(config["dataset_root"]) / "manifest.json").read_text(encoding="utf-8"))
    oracle = OfficialOracle(manifest["executor_path"])
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Official executor differs from the data manifest")
    return config, grid, Budgets(**spec.get("budgets", {})), oracle, backend_identity(spec, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("run")
    start.add_argument("--spec", required=True)
    start.add_argument("--out", required=True, help="new directory under outputs/agent_harness")
    start.add_argument("--dry-run", action="store_true", help="validate artifacts; write nothing and load no model")
    for command in ("resume", "inspect"):
        item = sub.add_parser(command)
        item.add_argument("--run", required=True, dest="directory")
    args = parser.parse_args()
    if args.command == "inspect":
        manifest, state, _, events = read_session(args.directory)
        print(json.dumps({"manifest": manifest, "state": state, "events": events}, ensure_ascii=False, indent=2))
        return
    if args.command == "run":
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        directory = Path(args.out).resolve()
        allowed = Path("outputs/agent_harness").resolve()
        if allowed not in directory.parents:
            raise ValueError("--out must be a new trial directory under outputs/agent_harness")
    else:
        directory = Path(args.directory)
        spec = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    config, grid, budgets, oracle, identity = prepare(spec)
    if args.command == "run":
        if args.dry_run:
            if directory.exists():
                raise ValueError("Trial directory already exists")
            print(json.dumps({"validated": True, "backend": identity, "gpu_loaded": False}, indent=2))
            return
        create_session(directory, grid, budgets, identity, {"kind": "official", "sha256": oracle.sha256})
        (directory / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    else:
        manifest, state, _, _ = read_session(directory)
        if manifest["backend"] != identity or manifest["grid"] != json.loads(json.dumps(grid.__dict__)):
            raise ValueError("Session inputs changed; create a new trial")
        if state["status"] in ("success", "stopped", "budget_exhausted"):
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return
    from ppnl.modeling import gpu_budget
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], "agent_harness") as check:
        backend = load_backend(spec, config, check)
        state = run_session(directory, backend, identity, oracle, check)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
