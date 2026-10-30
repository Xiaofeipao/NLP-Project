"""Evaluate executor-prefix repair on saved, model-generated GVR plans."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppnl.core import OfficialOracle
from ppnl.prefix_state_repair import evaluate_generated, load_generated_candidates, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.policy.best.json",
                        help="development-selected state-policy configuration")
    parser.add_argument("--dataset-root", default="data/derived")
    parser.add_argument("--split", required=True, choices=("dev", "test_id", "test_ood"))
    parser.add_argument("--initial-traces", help="GVR pipeline_traces.jsonl; defaults to published artifacts")
    parser.add_argument("--out", help="fresh directory for results and per-environment traces")
    parser.add_argument("--max-actions", type=int, default=36,
                        help="maximum total actions in a repaired path")
    parser.add_argument("--check-only", action="store_true",
                        help="validate source data and report missing model prerequisites without CUDA")
    parser.add_argument("--symbolic-only", action="store_true",
                        help="run only the goal-direction and visit-rule control on CPU")
    parser.add_argument("--allow-unselected-weights", action="store_true",
                        help="explicitly permit an unselected policy checkpoint")
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if not args.check_only and not args.out:
        parser.error("--out is required for evaluation")
    if args.out and Path(args.out).exists():
        parser.error("--out must be a fresh directory")

    config_file = Path(args.config)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    manifest = json.loads((Path(args.dataset_root) / "manifest.json").read_text(encoding="utf-8"))
    oracle = OfficialOracle(manifest["executor_path"])
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Official executor does not match the dataset manifest")
    trace_path = Path(args.initial_traces or
                      f"artifacts/gvr/evaluation/{args.split}/pipeline_traces.jsonl")
    paired = load_generated_candidates(args.dataset_root, args.split, trace_path, oracle)

    root = Path(config["output_root"])
    weights = Path(config.get("policy_checkpoint", root / "flan_t5_small/train/best"))
    weight_file = weights / "model.safetensors"
    selection_path = root / "policy_selection.json"
    ledger = Path(config["gpu_ledger"])
    prerequisites = {"policy_checkpoint": weight_file.exists(),
                     "development_selection": selection_path.exists(),
                     "gpu_ledger": ledger.exists()}
    if args.check_only:
        print(json.dumps({"split": args.split, "n_generated_environments": len(paired),
                          "initial_traces": str(trace_path), "policy_weights": str(weights),
                          "prerequisites": prerequisites}, indent=2))
        return
    if args.symbolic_only:
        metrics, records = evaluate_generated(
            paired, None, oracle, args.max_actions, methods=("prefix_visit_heuristic",))
        report = save_result(args.out, args.split, trace_path, args.dataset_root, None,
                             oracle.sha256, args.max_actions, metrics, records,
                             hashlib.sha256(config_file.read_bytes()).hexdigest())
        print(json.dumps({"output_dir": args.out, "split": args.split, "n": report["n"],
                          "success_rate": {key: value["success_rate"] for key, value in metrics.items()}},
                         indent=2))
        return
    if not weight_file.exists():
        raise FileNotFoundError(f"Missing state-policy checkpoint: {weight_file}")
    if not selection_path.exists() and not args.allow_unselected_weights:
        raise FileNotFoundError(f"Missing development selection: {selection_path}")
    if not ledger.exists():
        raise FileNotFoundError(f"Missing shared GPU ledger: {ledger}; restore the historical ledger before evaluation")
    with weight_file.open("rb") as handle:
        weight_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["weights_sha256"] != weight_hash and not args.allow_unselected_weights:
            raise ValueError("Policy weights differ from the development-selected checkpoint")

    from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
    from ppnl.policy import NeuralPolicy

    require_cuda()
    config["local_files_only"] = True
    with gpu_budget(str(ledger), config["gpu_hours_cap"], f"prefix_state_repair:{args.split}") as check:
        scorer = NeuralPolicy(Seq2Seq(str(weights), config), check)
        metrics, records = evaluate_generated(paired, scorer, oracle, args.max_actions, check)
        report = save_result(args.out, args.split, trace_path, args.dataset_root, weight_hash,
                             oracle.sha256, args.max_actions, metrics, records,
                             hashlib.sha256(config_file.read_bytes()).hexdigest())
    print(json.dumps({"output_dir": args.out, "split": args.split, "n": report["n"],
                      "success_rate": {key: value["success_rate"] for key, value in metrics.items()}},
                     indent=2))


if __name__ == "__main__":
    main()
