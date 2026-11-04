"""Run the G/V/R harness with the original full fine-tuned multi-task checkpoint."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, OfficialOracle
from ppnl.full_finetune_gvr import FullFinetuneGVRModel, checkpoint_identity
from ppnl.gvr_harness import MODES, create_gvr_session, run_gvr_session, score_gvr_session, validate_gvr_protocol
from ppnl.harness import read_session


def prepare(spec):
    config = json.loads(Path(spec["model_config"]).read_text(encoding="utf-8"))
    config["local_files_only"] = True
    raw = json.loads(Path(spec["grid"]).read_text(encoding="utf-8"))
    grid = Grid.from_world(raw) if isinstance(raw, list) else (
        Grid.from_world(raw["world"]) if "world" in raw else Grid.from_dict(raw))
    manifest = json.loads((Path(config["dataset_root"]) / "manifest.json").read_text(encoding="utf-8"))
    oracle = OfficialOracle(manifest["executor_path"])
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Official executor differs from the data manifest")
    identity = checkpoint_identity(spec["checkpoint"], config)
    return config, grid, oracle, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("run")
    start.add_argument("--checkpoint", required=True, help="local full fine-tuned model directory")
    start.add_argument("--grid", required=True, help="JSON grid or world")
    start.add_argument("--mode", choices=sorted(MODES), required=True)
    start.add_argument("--model-config", default="config.json")
    start.add_argument("--model-calls", type=int, default=8)
    start.add_argument("--active-seconds", type=float, default=300.0)
    start.add_argument("--out", required=True, help="new directory under outputs/full_finetune_gvr")
    start.add_argument("--dry-run", action="store_true")
    for name in ("resume", "inspect", "score"):
        sub.add_parser(name).add_argument("--run", required=True, dest="directory")
    args = parser.parse_args()

    if args.command == "inspect":
        manifest, state, _, events = read_session(args.directory)
        print(json.dumps({"manifest": manifest, "state": state, "events": events}, ensure_ascii=False, indent=2))
        return
    if args.command == "run":
        directory = Path(args.out).resolve()
        allowed = Path("outputs/full_finetune_gvr").resolve()
        if allowed not in directory.parents:
            raise ValueError("--out must be under outputs/full_finetune_gvr")
        spec = {"checkpoint": str(Path(args.checkpoint).resolve()),
                "grid": str(Path(args.grid).resolve()),
                "model_config": str(Path(args.model_config).resolve()),
                "mode": args.mode, "model_calls": args.model_calls,
                "active_seconds": args.active_seconds}
        if directory.exists():
            raise FileExistsError("Choose a new trial directory")
    else:
        directory = Path(args.directory)
        spec = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    config, grid, oracle, identity = prepare(spec)
    if args.command == "run":
        if args.dry_run:
            validate_gvr_protocol(spec["mode"], spec["model_calls"], spec["active_seconds"])
            print(json.dumps({"validated": True, "mode": spec["mode"],
                              "checkpoint_identity": identity, "gpu_loaded": False}, indent=2))
            return
        create_gvr_session(directory, grid, spec["mode"], identity,
                           spec["model_calls"], spec["active_seconds"])
        (directory / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    else:
        manifest, state, _, _ = read_session(directory)
        backend = manifest["backend"]
        if (backend["model"] != identity or backend["mode"] != spec["mode"] or
                backend["model_calls"] != spec["model_calls"] or
                backend["active_seconds"] != spec["active_seconds"] or
                manifest["grid"] != json.loads(json.dumps(grid.__dict__))):
            raise ValueError("Session inputs changed; create a new trial")
        if args.command == "score":
            if state["status"] not in ("stopped", "budget_exhausted"):
                raise ValueError("Score only a completed trial")
            print(json.dumps(score_gvr_session(directory, oracle), ensure_ascii=False, indent=2))
            return
        if state["status"] in ("stopped", "budget_exhausted"):
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return
    from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
    require_cuda()
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], "full_finetune_gvr") as check:
        model = FullFinetuneGVRModel(Seq2Seq(spec["checkpoint"], config))
        state = run_gvr_session(directory, model, identity, check)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
