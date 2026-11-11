"""Run the frozen 40/100/60 development split through the four legacy GVR flows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppnl.core import OfficialOracle
from ppnl.data import read_jsonl
from ppnl.full_finetune_gvr import FullFinetuneGVRModel, checkpoint_identity
from ppnl.gvr_evaluation import evaluate_gvr
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from scripts.smoke_gvr_repair import CHECKPOINT, ROOT, preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/gvr_repair_study/dev_legacy")
    args = parser.parse_args()
    config, manifest, _, global_ledger, _ = preflight(args.checkpoint, 40, "generation")
    out = args.out.resolve()
    allowed = (ROOT / "outputs/gvr_repair_study").resolve()
    if allowed not in out.parents or out.exists():
        raise ValueError("Use a new development output under outputs/gvr_repair_study")
    prior_path = allowed / "dev_generation/result.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    initial_by_source = {row["id"]: row["initial"] for row in prior["rows"]}
    if len(initial_by_source) != 40:
        raise ValueError("Expected 40 earlier generated development candidates")
    oracle = OfficialOracle(ROOT / manifest["executor_path"])
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Official executor differs")
    identity = checkpoint_identity(args.checkpoint, config)
    rows = read_jsonl(ROOT / "data/derived/dev.jsonl")
    require_cuda()
    with gpu_budget(global_ledger, 20, "gvr_legacy_development") as global_check:
        with gpu_budget(allowed / "study_gpu_ledger.json", 4, "legacy_development") as study_check:
            with gpu_budget(allowed / "dev_gpu_ledger.json", 1, "legacy_development") as dev_check:
                def check():
                    global_check()
                    study_check()
                    dev_check()

                model = FullFinetuneGVRModel(Seq2Seq(str(args.checkpoint), config))
                metrics = evaluate_gvr(model, rows, out, identity, oracle, check)
    traces = read_jsonl(out / "pipeline_traces.jsonl")
    for row in traces:
        source = next(r for r in rows if r["source_id"] == row["source_id"] and
                      r["task"] == "generation")
        if row["initial_candidate"] != initial_by_source[source["id"]]:
            raise ValueError("Legacy and experimental initial G candidates differ")
    print(json.dumps({"modes": {mode: metrics[mode]["success_rate"] for mode in
                                 ("direct", "self", "executor_one", "executor_three")},
                      "verification_macro_f1": metrics["verification"]["macro_f1"],
                      "candidate_repair_success": metrics["repair"]["success_rate"],
                      "output": str(out)}))


if __name__ == "__main__":
    main()
