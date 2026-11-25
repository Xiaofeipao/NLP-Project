"""Run the selected Part 1 baseline on one single-goal grid."""

import argparse
import json
from pathlib import Path

import torch

from .model import Network
from .policy import DELIVERED, LIMIT, direct, episode, line, run
from .protocol import ROOT, digest, read, templates


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", choices=("generate", "verify", "repair", "pipeline"), default="pipeline")
    cli.add_argument("--grid", type=Path, required=True, help="JSON with rows, cols, start, goal, obstacles")
    cli.add_argument(
        "--candidate", help="Verify and revise this path. An empty string is a supported candidate."
    )
    cli.add_argument("--checkpoint", type=Path, help="Use a specific checkpoint directory")
    cli.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    cli.add_argument("--trace", action="store_true")
    args = cli.parse_args()
    if args.task in ("verify", "repair") and args.candidate is None:
        cli.error("--candidate is required for this task")
    if args.task == "generate" and args.candidate is not None:
        cli.error("Generation takes a grid only")
    checkpoint = args.checkpoint
    if checkpoint is None:
        work = ROOT / "outputs/part1_baseline"
        if not (work / "selection.json").exists():
            work = ROOT / "artifacts/part1_baseline"
        selection = read(work / "selection.json")
        family = selection["primary_model"]
        checkpoint = work / "models/expanded" / family
        if digest(checkpoint / "model.safetensors") != selection["scores"][family]["weights"]:
            raise ValueError("Local export differs from the dev-selected checkpoint")
    grid = read(args.grid)
    if [grid["rows"], grid["cols"]] != [6, 6] or grid["start"] == grid["goal"]:
        raise ValueError("Expected a 6x6 grid with distinct Start and Goal")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    net = Network(checkpoint, args.device)
    net.model.eval()
    ps = templates()
    case = dict(id="input", grid=grid)
    if args.candidate is None:
        g = run([direct(case, ps)], net.predict)[0]
        if args.task == "generate":
            print(json.dumps(g, indent=2) if args.trace else g["plan"])
            return
        case["candidate"] = g["plan"]
    else:
        case["candidate"] = args.candidate
    result = run([episode(case, ps, limit=0 if args.task == "verify" else LIMIT)], net.predict)[0]
    if args.task == "verify":
        verdict = line(result["verification"])
        print(
            json.dumps(dict(verdict=verdict, trace=result["verification_trace"]), indent=2)
            if args.trace
            else verdict
        )
    elif args.trace:
        print(json.dumps(result, indent=2))
    elif args.task == "repair":
        if result["repair_status"] not in DELIVERED:
            raise SystemExit("Repair ended without an accepted plan: " + result["repair_status"])
        print(result["repaired_path"])
    else:
        print(
            json.dumps(
                dict(
                    candidate=case["candidate"],
                    verification=line(result["verification"]),
                    repair_status=result["repair_status"],
                    repaired_path=result["repaired_path"],
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
