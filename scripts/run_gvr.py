"""Frozen-task indexing, gated training, and GVR evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppnl.data import read_jsonl, task_index_manifest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("index", "pilot", "train", "verify", "evaluate"))
    parser.add_argument("--protocol", default="config.gvr.json")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--stage", choices=("generation", "verification", "repair"))
    parser.add_argument("--split", choices=("dev", "test_id", "test_ood"))
    parser.add_argument("--out")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    if args.command == "index":
        write_json(protocol["task_index"], task_index_manifest(protocol["dataset_root"]))
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.command in ("pilot", "train"):
        if not args.stage:
            parser.error("--stage is required")
        from ppnl.gvr_training import train_stage
        out = args.out or str(Path(protocol["output_root"]) / args.stage / args.command)
        print(json.dumps(train_stage(config, protocol, args.stage, out, args.command == "pilot"), indent=2))
        return
    if args.command == "evaluate" and not args.split:
        parser.error("--split is required")
    from ppnl.core import OfficialOracle
    from ppnl.gvr_evaluation import evaluate_gvr
    from ppnl.modeling import gpu_budget, require_cuda
    from ppnl.task_adapters import GVRModel
    require_cuda()
    operation = "gvr:verify" if args.command == "verify" else f"gvr:evaluate:{args.split}"
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], operation) as check:
        paths = {"verification": protocol["verification_adapter"], "repair": protocol["repair_adapter"]}
        model = GVRModel(protocol["generation_base"], paths, config)
        identity = {"base_sha256": model.base_hash, "adapters_sha256": model.adapter_hashes}
        if args.command == "verify":
            from ppnl.core import Grid
            from ppnl.modeling import Seq2Seq
            from ppnl.prompts import prompt
            sample = next(r for r in read_jsonl(Path(protocol["dataset_root"]) / "dev.jsonl") if r["task"] == "generation")
            grid = Grid.from_dict(sample["grid"])
            bare = Seq2Seq(protocol["generation_base"], config)
            baseline = bare.predict([prompt("generation", grid)], check)[0]
            disabled = model.predict("generation", grid, check=check)
            if baseline != disabled:
                raise RuntimeError("Disabling adapters changed the G output")
            print(json.dumps({"identity": identity, "disabled_adapter_matches_base": True,
                              "verification_output": model.predict("verification", grid, baseline, check=check),
                              "repair_output": model.predict("repair", grid, baseline,
                                                             "invalid | step=1 | reason=goal_not_reached", check)}, indent=2))
            return
        rows = read_jsonl(Path(protocol["dataset_root"]) / f"{args.split}.jsonl")
        oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
        out = args.out or str(Path(protocol["output_root"]) / "evaluation" / args.split)
        print(json.dumps(evaluate_gvr(model, rows, out, identity, oracle, check), indent=2))


if __name__ == "__main__":
    main()
