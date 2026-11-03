"""Evaluate executor-prefix repair on original main-branch generated paths."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import OfficialOracle
from ppnl.eval_harness import sha256_file, weights_dir
from ppnl.prefix_state_repair import evaluate_generated, load_generated_candidates, save_result


def prepare(config, split, trace_path):
    dataset = Path(config["policy_source_dataset"])
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    oracle = OfficialOracle(manifest["executor_path"])
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Official executor differs from frozen dataset manifest")
    paired = load_generated_candidates(dataset, split, trace_path, oracle)
    return dataset, oracle, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.policy.best.json")
    parser.add_argument("--split", choices=("dev", "test_id", "test_ood"), required=True)
    parser.add_argument("--trace", help="original full-model pipeline_traces.jsonl")
    parser.add_argument("--out", help="new directory under outputs/main_prefix_state_repair")
    parser.add_argument("--max-actions", type=int, default=36)
    parser.add_argument("--check-only", action="store_true", help="validate saved candidates without a model")
    args = parser.parse_args()
    if args.max_actions < 1:
        raise ValueError("--max-actions must be positive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config["local_files_only"] = True
    baseline = Path(config["baseline_root"])
    trace_path = Path(args.trace) if args.trace else (
        baseline / ("eval_dev" if args.split == "dev" else f"eval_{args.split}_full") / "pipeline_traces.jsonl")
    dataset, oracle, paired = prepare(config, args.split, trace_path)
    if args.check_only:
        print(json.dumps({"validated": True, "split": args.split, "n": len(paired),
                          "trace": str(trace_path), "official_executor_sha256": oracle.sha256}, indent=2))
        return
    if args.out is None:
        parser.error("--out is required unless --check-only is used")
    destination = Path(args.out).resolve()
    allowed = Path("outputs/main_prefix_state_repair").resolve()
    if allowed not in destination.parents or destination.exists():
        raise ValueError("--out must be a new directory under outputs/main_prefix_state_repair")
    weights = weights_dir(config)
    weight_hash = sha256_file(weights / "model.safetensors")
    selection = Path(config["output_root"]) / "policy_selection.json"
    if selection.exists():
        frozen = json.loads(selection.read_text(encoding="utf-8"))
        if frozen.get("weights_sha256") != weight_hash:
            raise ValueError("Policy weights differ from development selection")
    from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
    from ppnl.policy import NeuralPolicy
    require_cuda()
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"],
                    f"main_prefix_state_repair:{args.split}") as check:
        scorer = NeuralPolicy(Seq2Seq(str(weights), config), check)
        metrics, records = evaluate_generated(paired, scorer, oracle, args.max_actions, check)
    report = save_result(destination, dataset, args.split, trace_path, weight_hash,
                         oracle.sha256, args.max_actions, metrics, records)
    print(json.dumps({"output": str(destination), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
