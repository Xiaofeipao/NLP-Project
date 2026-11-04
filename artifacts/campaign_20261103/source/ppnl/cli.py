from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .data import audit_dataset, build_dataset, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description="PPNL single-goal project; model commands require CUDA")
    parser.add_argument("--config", default="config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("environment")
    p = sub.add_parser("prepare")
    p.add_argument("--official-data", required=True)
    p.add_argument("--executor", required=True)
    for name in ("pilot", "train", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--model", required=True)
    p.add_argument("--split", choices=["dev", "test_id", "test_ood"], required=True)
    p.add_argument("--full", action="store_true")
    sub.add_parser("select")
    sub.add_parser("audit")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root, dataset = Path(config["output_root"]), Path(config["dataset_root"])
    if args.command == "environment":
        import sys
        packages = {}
        for name in ("torch", "transformers", "sentencepiece", "numpy", "protobuf"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        print(json.dumps({"python": sys.version, "executable": sys.executable, "packages": packages}, indent=2))
    elif args.command == "prepare":
        build_dataset(config, args.official_data, args.executor, dataset)
        print(json.dumps(audit_dataset(dataset), indent=2))
    elif args.command == "audit":
        print(json.dumps(audit_dataset(dataset), indent=2))
    elif args.command in ("pilot", "train", "smoke"):
        from .modeling import train, smoke
        audit_dataset(dataset)
        if args.command == "smoke":
            report = smoke(config, args.model, dataset, root / args.model / "smoke")
        else:
            report = train(config, args.model, dataset, root / args.model / args.command,
                           pilot=args.command == "pilot")
        if args.command == "smoke":
            print(json.dumps({k: report[k] for k in ("purpose", "model", "before_loss", "after_loss",
                                                    "training_seconds", "peak_allocated_bytes", "precision_microbenchmark")}, indent=2))
            print(f"Report: {root / args.model / 'smoke' / 'smoke.json'}")
        else:
            print(json.dumps(report, indent=2))
    elif args.command == "evaluate":
        from .core import OfficialOracle
        from .evaluation import evaluate
        from .modeling import Seq2Seq, gpu_budget, require_cuda
        audit_dataset(dataset)
        key, split = args.model, args.split
        if split != "dev":
            selection = json.loads((root / "selection.json").read_text())
            if key != selection["primary"] and args.full:
                raise ValueError("Full study is limited to the development-selected primary model")
            fingerprint = hashlib.sha256((root / key / "train" / "training.json").read_bytes()).hexdigest()
            if selection["training_fingerprints"].get(key) != fingerprint:
                raise ValueError("Training changed after model selection")
        require_cuda()
        manifest = json.loads((dataset / "manifest.json").read_text())
        oracle = OfficialOracle(manifest["executor_path"])
        if oracle.sha256 != manifest["executor_sha256"]:
            raise ValueError("Official executor changed since data construction")
        rows = read_jsonl(dataset / f"{split}.jsonl")
        if split == "dev":
            if args.full:
                raise ValueError("Use baseline-only development evaluation for matched model selection")
            rows = [r for r in rows if r["task"] == "verification"]
        with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], f"evaluate:{key}:{split}") as check:
            runner = Seq2Seq(str(root / key / "train" / "best"), config)
            result = evaluate(runner, rows, root / key / f"eval_{split}{'_full' if args.full else ''}",
                              check, args.full, config["repair_rounds"], oracle)
            print(json.dumps(result, indent=2))
    elif args.command == "select":
        from .modeling import dataset_fingerprint
        scores, fingerprints, reports = {}, {}, {}
        for key in config["models"]:
            path = root / key / "train" / "training.json"
            training = json.loads(path.read_text())
            if training["data_fingerprint"] != dataset_fingerprint(dataset):
                raise ValueError("Data changed since training")
            fingerprints[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            reports[key] = training
            metrics = json.loads((root / key / "eval_dev" / "metrics.json").read_text())
            scores[key] = sum((metrics["direct_generation"]["success_rate"],
                               metrics["verification"]["macro_f1"], metrics["repair"]["success_rate"])) / 3
        if len(scores) < 2:
            raise ValueError("At least two baselines required")
        if len({json.dumps(r["training_config"], sort_keys=True) for r in reports.values()}) != 1:
            raise ValueError("Baseline training budgets do not match")
        primary = max(sorted(scores), key=scores.get)
        path = root / "selection.json"
        if path.exists():
            raise FileExistsError("Selection is frozen; use a new experiment directory to revise it")
        write_json(path, {"primary": primary, "dev_scores": scores, "training_fingerprints": fingerprints,
                          "rule": "mean of dev generation success, verification macro-F1, repair success; lexical tie break"})
        print(primary)


if __name__ == "__main__":
    main()
