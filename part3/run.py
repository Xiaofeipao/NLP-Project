"""FirstPart-style stage dispatch for Part 3."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from part2.workflow import ROOT
from part3.workflow import METHODS, verify_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("audit", "run", "plan", "verify-results"))
    p.add_argument("--out", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--ledger", type=Path)
    p.add_argument("--grid", type=Path, default=Path("examples/grid.json"))
    p.add_argument("--config", type=Path, default=Path("part3/best_overall.json"))
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--executor", type=Path, default=ROOT / "vendor/ppnl/executor-point-sg.py")
    args = p.parse_args()
    if args.command == "verify-results":
        print(
            verify_results(
                ROOT / "deliverables/part3_evidence_20261116.zip",
                3,
                args.executor if args.executor.exists() else None,
            )
        )
        return
    if args.command == "audit":
        protocol = json.loads((ROOT / "part3/protocol.json").read_text(encoding="utf-8"))
        if protocol["methods"] != METHODS:
            raise ValueError("Protocol and implementation differ")
        print({"methods": len(METHODS), "protocol_matches": True, "gpu_loaded": False})
        return
    if args.command == "run":
        if not (args.out and args.checkpoint and args.ledger):
            p.error("run requires --out, --checkpoint and --ledger")
        command = [
            sys.executable,
            "scripts/run_method_sweep.py",
            "--out",
            str(args.out),
            "--checkpoint",
            str(args.checkpoint),
            "--ledger",
            str(args.ledger),
            "--model-config",
            "part2/protocol.json",
        ]
    else:
        command = [
            sys.executable,
            "scripts/run_selected_method.py",
            "--config",
            str(args.config),
            "--grid",
            str(args.grid),
        ]
        if args.check_only:
            command += ["--check-only"]
        elif args.out:
            command += ["--out", str(args.out)]
        else:
            p.error("plan requires a new --out or --check-only")
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
