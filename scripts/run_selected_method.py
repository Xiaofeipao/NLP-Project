"""Run a retained dev-ranked method on one grid without retraining the sweep."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, OfficialOracle, execute
from ppnl.data import write_json
from ppnl.modeling import Seq2Seq, gpu_budget
from ppnl.policy import NeuralPolicy, greedy_plans
from ppnl.method_sweep import verification_input, extract_verdict, repair_once
from ppnl.verifier_guard import guard_verdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("part3/best_overall.json"))
    p.add_argument("--grid", type=Path, default=Path("examples/grid.json"))
    p.add_argument("--out", type=Path)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("generator_checkpoint", "verifier_checkpoint"):
        weights = Path(spec[key]) / "model.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != spec[key + "_sha256"]:
            raise ValueError("Checkpoint changed: " + key)
    if args.check_only:
        print({"validated": True, "method": spec["method"], "cuda_loaded": False})
        return
    if args.out is None or args.out.exists():
        p.error("Provide a new --out directory")
    raw = json.loads(args.grid.read_text(encoding="utf-8"))
    grid = (
        Grid.from_world(raw)
        if isinstance(raw, list)
        else Grid.from_world(raw["world"]) if "world" in raw else Grid.from_dict(raw)
    )
    config = json.loads(Path(spec["model_config"]).read_text())
    config["training"]["eval_batch_size"] = 4
    method = spec["method"]
    history = []
    oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
    with gpu_budget(spec["gpu_ledger"], 20, "selected_method:single_grid") as check:
        runner = Seq2Seq(spec["generator_checkpoint"], config)
        scorer = NeuralPolicy(runner, check)
        verifier = (
            runner
            if spec["verifier_checkpoint"] == spec["generator_checkpoint"]
            else Seq2Seq(spec["verifier_checkpoint"], config)
        )
        initial = greedy_plans([grid], scorer, masked=False)[0]
        value = initial
        remaining = 36
        for _ in range(max(1, method["rounds"])):
            if method["guard"] == "executor":
                feedback = execute(grid, value).label()
            else:
                pred = extract_verdict(
                    verifier.predict([verification_input(grid, value, method["style"])], check)[0],
                    method["style"],
                )
                feedback = guard_verdict(grid, value, pred, method["guard"])
            if not method["rounds"]:
                break
            revised, used, status = repair_once(
                grid,
                value,
                feedback,
                scorer,
                min(12, remaining) if method["rounds"] > 1 else remaining,
                method["decoder"],
            )
            history.append(
                {
                    "before": value,
                    "feedback": feedback,
                    "after": revised,
                    "status": status,
                    "new_actions": used,
                }
            )
            unchanged = value == revised
            value = revised
            remaining -= used
            if (
                status
                in ("accepted", "success", "malformed_feedback", "bad_step", "illegal_prefix")
                or unchanged
                or remaining <= 0
            ):
                break
        truth = oracle.check(grid, value)
    write_json(
        args.out / "result.json",
        {
            "method": method,
            "initial": initial,
            "plan": value,
            "verdict": truth.label(),
            "history": history,
        },
    )
    print(json.dumps({"method": method["name"], "plan": value, "verdict": truth.label()}))


if __name__ == "__main__":
    main()
