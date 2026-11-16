"""Replay frozen FirstPart model outputs with explicit verifier guard ablations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, OfficialOracle, parse_verdict
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import verification_metrics
from ppnl.verifier_guard import MODES, guard_verdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("outputs/phase2_verifier/source"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output directory")
    args.out.mkdir(parents=True)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    files = sorted(args.source.rglob("*.json")) + sorted(args.source.rglob("*.jsonl"))
    # Fixed matrix before replay; no tuning, selection or checkpoint changes.
    write_json(
        args.out / "protocol.json",
        {
            "model": "FirstPart flan_t5_base epoch 2",
            "modes": MODES,
            "source_sha256": {str(f.relative_to(args.source)): sha(f) for f in files},
            "code_sha256": {n: sha(Path(n)) for n in ["ppnl/verifier_guard.py", __file__]},
            "scope": "CPU replay of saved model predictions; reused exploratory sets; no training or repair claims",
        },
    )
    oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
    start = time.perf_counter()
    dev = {r["id"]: r for r in read_jsonl("data/policy_v3/dev.jsonl")}
    report = {}
    for split in ("dev", "test_id", "test_ood", "generated_id"):
        if split == "dev":
            saved = json.loads(
                next(args.source.rglob("flan_t5_base/dev_epoch_2.json")).read_text()
            )["verification"]
            records = [
                {
                    "id": r["id"],
                    "grid": dev[r["id"]]["grid"],
                    "direct": dev[r["id"]]["candidate"],
                    "gold": r["target"],
                    "self": {"feedback": r["prediction"]},
                }
                for r in saved
            ]
        else:
            name = "planning.jsonl" if split == "generated_id" else "candidates.jsonl"
            folder = "test_id" if split == "generated_id" else split
            records = read_jsonl(next(args.source.rglob(f"{folder}/flan_t5_base/{name}")))
        rows, predictions = [], {m: [] for m in MODES}
        detail = []
        for r in records:
            grid = Grid.from_dict(r["grid"])
            truth = oracle.check(grid, r["direct"])
            assert truth.label() == r["gold"], r["id"]
            rows.append({"valid": truth.valid, "step": truth.step, "reason": truth.reason})
            outputs = {m: guard_verdict(grid, r["direct"], r["self"]["feedback"], m) for m in MODES}
            for m in MODES:
                predictions[m].append(outputs[m])
            detail.append({**r, "guard_predictions": outputs})
        report[split] = {}
        for m, outputs in predictions.items():
            metrics = verification_metrics(rows, outputs)
            metrics["false_accepts"] = sum(
                not r["valid"] and parse_verdict(v) == (True, None, None)
                for r, v in zip(rows, outputs)
            )
            metrics["false_rejects"] = sum(
                r["valid"] and parse_verdict(v) != (True, None, None) for r, v in zip(rows, outputs)
            )
            metrics["joint_by_reason"] = {
                reason: {
                    "n": sum(r["reason"] == reason for r in rows),
                    "correct": sum(
                        r["reason"] == reason and parse_verdict(v) == (False, r["step"], reason)
                        for r, v in zip(rows, outputs)
                    ),
                }
                for reason in sorted({r["reason"] for r in rows if not r["valid"]})
            }
            # Generated pool may contain only valid candidates: absent-class F1 must be interpreted with counts.
            metrics["class_counts"] = dict(Counter(str(r["valid"]) for r in rows))
            report[split][m] = metrics
        write_jsonl(args.out / f"{split}_traces.jsonl", detail)
    write_json(
        args.out / "results.json",
        {
            "metrics": report,
            "cpu_seconds": time.perf_counter() - start,
            "gpu_calls": 0,
            "new_training_steps": 0,
            "warning": "trace mode is executor V, not improved neural reasoning",
        },
    )
    print(
        json.dumps(
            {
                s: {
                    m: {
                        k: v[k]
                        for k in (
                            "macro_f1",
                            "first_error_localization_accuracy",
                            "failure_type_accuracy",
                            "false_accepts",
                        )
                    }
                    for m, v in methods.items()
                }
                for s, methods in report.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
