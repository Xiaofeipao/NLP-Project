"""FirstPart-style stage dispatch for Part 2; inspect/audit never load CUDA."""

import argparse
from pathlib import Path
import subprocess
import sys

from part2.workflow import ROOT, prepare, verify_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=("prepare", "audit", "pilot", "train", "evaluate", "verify-results")
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--pretrained", type=Path)
    p.add_argument("--ledger", type=Path)
    p.add_argument("--executor", type=Path, default=ROOT / "vendor/ppnl/executor-point-sg.py")
    p.add_argument(
        "--model", default="flan_t5_base", choices=("flan_t5_base", "flan_t5_small", "bart_base")
    )
    args = p.parse_args()
    if args.command == "verify-results":
        print(
            verify_results(
                ROOT / "deliverables/part2_evidence_20261116.zip",
                2,
                args.executor if args.executor.exists() else None,
            )
        )
        return
    if args.out is None:
        p.error("--out is required")
    out = args.out.resolve()
    if args.command == "prepare":
        print(prepare(out))
        return
    if args.command == "evaluate":
        if args.checkpoint is None or args.ledger is None:
            p.error("evaluate requires --checkpoint and --ledger")
        command = [
            sys.executable,
            str(ROOT / "scripts/run_phase2_feedback.py"),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--out",
            str(out),
            "--ledger",
            str(args.ledger.resolve()),
            "--model-config",
            str(ROOT / "part2/protocol.json"),
        ]
        result = subprocess.run(command, cwd=ROOT)
    else:
        source = out / "source"
        if not source.is_dir():
            p.error("Run prepare first")
        command = [
            sys.executable,
            "-m",
            "state_baselines.run",
            args.command,
            "--out",
            str(out / "reproduction"),
            "--data",
            str(ROOT / "data/policy_v3"),
            "--executor",
            str(args.executor.resolve()),
        ]
        if args.command in ("pilot", "train"):
            if args.pretrained is None or args.ledger is None:
                p.error("Training requires --pretrained and --ledger")
            command += [
                "--model",
                args.model,
                "--pretrained",
                str(args.pretrained.resolve()),
                "--ledger",
                str(args.ledger.resolve()),
            ]
        result = subprocess.run(command, cwd=source)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
