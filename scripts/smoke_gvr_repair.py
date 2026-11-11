"""Fixed development generation slices for GVR repair sizing and diagnosis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppnl.constrained_decoding import DecodeSpec
from ppnl.core import Grid, OfficialOracle
from ppnl.data import read_jsonl, write_json
from ppnl.experiment_freeze import file_hash
from ppnl.full_finetune_gvr import FullFinetuneGVRModel
from ppnl.gvr_evaluation import evaluation_subsets
from ppnl.gvr_rule_controls import rule_repair
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/historical_models/flan_t5_small/train/best"
WEIGHT_SHA256 = "0b7be3b12238b363ba8c1a9304f687c6ae6f9edbecceff0076cebe7c260d62fa"


def preflight(checkpoint, limit, task):
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data = ROOT / "data/derived"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(data / "dev.jsonl")
    verify, repair, generate = evaluation_subsets(rows)
    if tuple(map(len, (verify, repair, generate))) != (100, 60, 40):
        raise ValueError("Development quotas changed")
    source = ROOT / "vendor/ppnl/executor-point-sg.py"
    if file_hash(source) != manifest["executor_sha256"]:
        raise ValueError("Official executor SHA differs from dataset")
    weight = checkpoint / "model.safetensors"
    if weight.stat().st_size < 1_000_000 or file_hash(weight) != WEIGHT_SHA256:
        raise ValueError("Selected full checkpoint is missing or differs from the LFS object")
    ledger = ROOT / config["gpu_ledger"]
    if not ledger.exists():
        raise FileNotFoundError("Reconcile the existing GPU ledger before model loading")
    used = sum(run["seconds"] for run in json.loads(ledger.read_text(encoding="utf-8"))["runs"])
    if used >= 20 * 3600:
        raise ValueError("Project GPU budget is exhausted")
    if (task == "generation" and limit not in (8, 40)) or (task == "repair" and limit != 60):
        raise ValueError("Use 8/40 generation environments or all 60 repair candidates")
    pool = generate if task == "generation" else repair
    selected = sorted(pool, key=lambda row: row["id"])[:limit]
    if len(selected) != limit:
        raise ValueError("Development generation count differs from requested slice")
    return config, manifest, selected, ledger, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/gvr_repair_study/smoke")
    parser.add_argument("--limit", type=int, choices=(8, 40, 60), default=8)
    parser.add_argument("--task", choices=("generation", "repair"), default="generation")
    args = parser.parse_args()
    config, manifest, rows, global_ledger, used = preflight(args.checkpoint, args.limit, args.task)
    if args.command == "preflight":
        print(json.dumps({"checkpoint_sha256": WEIGHT_SHA256, "dev_ids": [r["id"] for r in rows],
                          "prior_gpu_hours": used / 3600, "cuda_loaded": False}, indent=2))
        return
    out = args.out.resolve()
    allowed = (ROOT / "outputs/gvr_repair_study").resolve()
    if allowed not in out.parents:
        raise ValueError("Smoke output must be a new child of outputs/gvr_repair_study")
    if out.exists():
        raise FileExistsError("Smoke output already exists; do not overwrite a prior attempt")
    out.mkdir(parents=True)
    torch = require_cuda()
    oracle = OfficialOracle(ROOT / manifest["executor_path"])
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Official executor changed")
    phase = "smoke" if args.task == "generation" and args.limit == 8 else f"dev_{args.task}"
    cap = 0.25 if phase == "smoke" else 1.0
    phase_ledger = "smoke_gpu_ledger.json" if phase == "smoke" else "dev_gpu_ledger.json"
    with gpu_budget(global_ledger, 20, f"gvr_repair_{phase}") as global_check:
        with gpu_budget(out.parent / "study_gpu_ledger.json", 4, phase) as study_check:
            with gpu_budget(out.parent / phase_ledger, cap, phase) as smoke_check:
                def check():
                    global_check()
                    study_check()
                    smoke_check()

                torch.cuda.reset_peak_memory_stats()
                runner = Seq2Seq(str(args.checkpoint), config)
                greedy = FullFinetuneGVRModel(runner)
                bare = FullFinetuneGVRModel(runner, DecodeSpec(state_constraints=False))
                constrained = FullFinetuneGVRModel(runner, DecodeSpec())
                records = []
                for row in rows:
                    check()
                    grid = Grid.from_dict(row["grid"])
                    start = time.perf_counter()
                    initial = (greedy.predict("generation", grid, check=check)
                               if args.task == "generation" else row["candidate"])
                    verdict = oracle.check(grid, initial)
                    if args.task == "repair" and verdict.label() != row["label"]:
                        raise ValueError("Repair label differs from official executor")
                    result = {"id": row["id"], "initial": initial, "initial_valid": verdict.valid,
                              "generation_seconds": time.perf_counter() - start
                              if args.task == "generation" else 0.0,
                              "error_reason": row.get("reason"), "repairs": {}}
                    if not verdict.valid:
                        feedback = verdict.label()
                        for name, model in (("E1", greedy), ("B4", bare), ("C4", constrained)):
                            check()
                            start = time.perf_counter()
                            plan = model.predict("repair", grid, initial, feedback, check)
                            result["repairs"][name] = {
                                "plan": plan, "success": oracle.check(grid, plan).valid,
                                "seconds": time.perf_counter() - start,
                                "usage": dict(model.last_usage)}
                        for name, kind in (("U4", "uniform"), ("H4", "manhattan")):
                            start = time.perf_counter()
                            decoded = rule_repair(grid, kind, check=check)
                            result["repairs"][name] = {
                                "plan": decoded.plan, "success": oracle.check(grid, decoded.plan).valid,
                                "seconds": time.perf_counter() - start,
                                "usage": {"decode_status": decoded.stop_reason,
                                          "decoder_row_steps": decoded.decoder_row_steps}}
                    records.append(result)
                    write_json(out / "progress.json", {"rows": records,
                               "peak_memory_bytes": torch.cuda.max_memory_allocated()})
                write_json(out / "result.json", {"rows": records,
                           "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                           "purpose": "technical_sizing_only" if phase == "smoke" else
                                      f"{phase}_slice_not_full_matrix"})
    print(json.dumps({"completed": len(records), "output": str(out),
                      "peak_memory_bytes": torch.cuda.max_memory_allocated()}))


if __name__ == "__main__":
    main()
