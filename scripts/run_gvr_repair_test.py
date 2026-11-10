"""Frozen ID/OOD GVR comparison; OOD reuses the identical ID generation trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppnl.constrained_decoding import DecodeSpec
from ppnl.core import Grid, OfficialOracle
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics, verification_metrics
from ppnl.experiment_freeze import file_hash
from ppnl.full_finetune_gvr import FullFinetuneGVRModel
from ppnl.gvr_evaluation import evaluate_gvr, evaluation_subsets
from ppnl.gvr_rule_controls import rule_repair
from ppnl.gvr_study_lock import study_identity
from ppnl.harness import fingerprint
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.prefix_evaluation import validate_lock
from scripts.smoke_gvr_repair import CHECKPOINT, ROOT, preflight


class CachedGeneration:
    """One physical G inference per grid; sessions retain logical G calls."""
    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.last_usage = {}
        self.physical_calls = 0

    def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
        key = (grid.layout_id, grid.start, grid.goal)
        if task == "generation" and key in self.cache:
            self.last_usage = {"input_tokens": 0, "output_tokens": 0}
            return self.cache[key]
        value = self.model.predict(task, grid, candidate, feedback, check)
        self.last_usage = dict(self.model.last_usage)
        if task == "generation":
            self.cache[key] = value
            self.physical_calls += 1
        return value


def alternatives(grid, initial, feedback, bare, constrained, oracle, check):
    result = {}
    if oracle.check(grid, initial).valid:
        return {name: {"plan": initial, "success": True, "seconds": 0.0,
                       "usage": {"preserved_initial": True}}
                for name in ("B4", "C4", "U4", "H4")}
    for name, model in (("B4", bare), ("C4", constrained)):
        check()
        start = time.perf_counter()
        plan = model.predict("repair", grid, initial, feedback, check)
        result[name] = {"plan": plan, "success": oracle.check(grid, plan).valid,
                        "seconds": time.perf_counter() - start,
                        "usage": dict(model.last_usage)}
    for name, kind in (("U4", "uniform"), ("H4", "manhattan")):
        check()
        start = time.perf_counter()
        decoded = rule_repair(grid, kind, check=check)
        result[name] = {"plan": decoded.plan,
                        "success": oracle.check(grid, decoded.plan).valid,
                        "seconds": time.perf_counter() - start,
                        "usage": {"decode_status": decoded.stop_reason,
                                  "decoder_row_steps": decoded.decoder_row_steps}}
    return result


def compare_metrics(rows, records):
    grids = [Grid.from_dict(row["grid"]) for row in rows]
    return {name: plan_metrics(grids, [record[name]["plan"] for record in records])
            for name in ("B4", "C4", "U4", "H4")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test_id", "test_ood"), required=True)
    parser.add_argument("--freeze", type=Path, default=ROOT / "outputs/gvr_repair_study/freeze.json")
    parser.add_argument("--check-only", action="store_true", help="verify the lock and quotas without CUDA")
    args = parser.parse_args()
    identity = study_identity(ROOT)
    receipt = validate_lock(args.freeze, identity)  # Before any CUDA allocation.
    config, manifest, _, global_ledger, _ = preflight(CHECKPOINT, 40, "generation")
    rows = read_jsonl(ROOT / f"data/derived/{args.split}.jsonl")
    verify_rows, repair_rows, generation_rows = evaluation_subsets(rows)
    if tuple(map(len, (verify_rows, repair_rows, generation_rows))) != (200, 100, 200):
        raise ValueError("Test task quotas changed")
    oracle = OfficialOracle(ROOT / manifest["executor_path"])
    out = ROOT / "outputs/gvr_repair_study" / args.split
    if out.exists():
        raise FileExistsError("Test output already exists; do not overwrite previous evidence")
    if args.split == "test_ood":
        id_report = ROOT / "outputs/gvr_repair_study/test_id/results.json"
        if not id_report.exists():
            raise ValueError("Run paired ID generation before OOD candidate evaluation")
        id_data = json.loads(id_report.read_text(encoding="utf-8"))
        if id_data["freeze_receipt"] != receipt:
            raise ValueError("ID and OOD experiments use different locks")
    if args.check_only:
        print(json.dumps({"split": args.split, "quota": [len(verify_rows), len(repair_rows),
                                                 len(generation_rows)],
                          "freeze_receipt": receipt, "cuda_loaded": False}))
        return
    out.mkdir(parents=True)
    require_cuda()
    budget_root = ROOT / "outputs/gvr_repair_study"
    with gpu_budget(global_ledger, 20, f"gvr_repair:{args.split}") as global_check:
        with gpu_budget(budget_root / "study_gpu_ledger.json", 4, args.split) as study_check:
            with gpu_budget(budget_root / "test_gpu_ledger.json", 2, args.split) as test_check:
                def check():
                    global_check()
                    study_check()
                    test_check()

                runner = Seq2Seq(str(CHECKPOINT), config)
                base = FullFinetuneGVRModel(runner)
                bare = FullFinetuneGVRModel(runner, DecodeSpec(state_constraints=False))
                constrained = FullFinetuneGVRModel(runner, DecodeSpec())
                if args.split == "test_id":
                    cached = CachedGeneration(base)
                    legacy = evaluate_gvr(cached, rows, out / "legacy",
                                          {"study_sha256": fingerprint(identity)}, oracle, check)
                    legacy_candidates = {item["id"]: item["repair"] for item in
                                         read_jsonl(out / "legacy/candidate_traces.jsonl")}
                    traces = read_jsonl(out / "legacy/pipeline_traces.jsonl")
                    if len(traces) != 200 or cached.physical_calls != 200:
                        raise ValueError("Expected one physical generation per ID grid")
                    generated = []
                    for trace in traces:
                        check()
                        grid = Grid.from_dict(trace["grid"])
                        initial = trace["initial_candidate"]
                        feedback = oracle.check(grid, initial).label()
                        generated.append({"source_id": trace["source_id"], "grid": trace["grid"],
                                          "initial": initial, "feedback": feedback,
                                          "methods": alternatives(grid, initial, feedback, bare,
                                                                  constrained, oracle, check)})
                    write_jsonl(out / "generation_traces.jsonl", generated)
                    generation_records = [item["methods"] for item in generated]
                    generation_scores = compare_metrics(generation_rows, generation_records)
                    verification = legacy["verification"]
                    legacy_scores = {name: legacy[name] for name in
                                     ("direct", "self", "executor_one", "executor_three")}
                    physical_generations = cached.physical_calls
                else:
                    generated = read_jsonl(ROOT / "outputs/gvr_repair_study/test_id/generation_traces.jsonl")
                    if len(generated) != 200 or any(
                            item["grid"] != row["grid"] for item, row in zip(generated, generation_rows)):
                        raise ValueError("ID/OOD generation environments differ")
                    generation_scores = id_data["generation"]
                    legacy_scores = id_data["legacy_modes"]
                    physical_generations = 0
                    verdicts = [base.predict("verification", Grid.from_dict(row["grid"]),
                                             row["candidate"], check=check) for row in verify_rows]
                    verification = verification_metrics(verify_rows, verdicts)
                candidates = []
                legacy_repair = []
                for row in repair_rows:
                    check()
                    grid = Grid.from_dict(row["grid"])
                    if oracle.check(grid, row["candidate"]).label() != row["label"]:
                        raise ValueError("Candidate feedback differs from official executor")
                    old = (legacy_candidates[row["id"]] if args.split == "test_id" else
                           base.predict("repair", grid, row["candidate"], row["label"], check))
                    legacy_repair.append(old)
                    candidates.append({"id": row["id"], "reason": row["reason"],
                                       "E1": {"plan": old, "success": oracle.check(grid, old).valid},
                                       **alternatives(grid, row["candidate"], row["label"], bare,
                                                      constrained, oracle, check)})
                write_jsonl(out / "candidate_traces.jsonl", candidates)
                candidate_scores = compare_metrics(repair_rows, candidates)
                candidate_scores["E1"] = plan_metrics([Grid.from_dict(row["grid"]) for row in repair_rows],
                                                       legacy_repair)
                report = {"split": args.split, "freeze_receipt": receipt,
                          "test_status": "reused_exploratory", "generation": generation_scores,
                          "legacy_modes": legacy_scores, "verification": verification,
                          "candidate_repair": candidate_scores,
                          "logical_generation_cases": 200,
                          "physical_generation_calls": physical_generations,
                          "paired_generation_shared_with_id": args.split == "test_ood",
                          "id_generation_sha256": file_hash(
                              ROOT / "outputs/gvr_repair_study/test_id/generation_traces.jsonl")}
                write_json(out / "results.json", report)
    print(json.dumps({"split": args.split, "generation_success": {
        name: report["generation"][name]["success_rate"] for name in ("B4", "C4", "U4", "H4")},
        "candidate_success": {name: report["candidate_repair"][name]["success_rate"]
                              for name in ("E1", "B4", "C4", "U4", "H4")},
        "output": str(out)}))


if __name__ == "__main__":
    main()
