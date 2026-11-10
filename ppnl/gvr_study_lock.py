"""Development-only decision and immutable input identity for GVR repair tests."""
from __future__ import annotations

import json
from pathlib import Path

from .experiment_freeze import _tree_hashes, audit_splits, file_hash

METHODS = ("D0", "S1", "E1", "E3", "B4", "C4", "U4", "H4")


def study_identity(root):
    root = Path(root).resolve()
    checkpoint = root / "artifacts/historical_models/flan_t5_small/train/best"
    data = root / "data/derived"
    config = root / "config.json"
    source = {p.relative_to(root).as_posix(): file_hash(p)
              for folder in ("ppnl", "scripts")
              for p in sorted((root / folder).rglob("*.py"))}
    return {"schema_version": 1, "checkpoint_files": _tree_hashes(checkpoint),
            "config_sha256": file_hash(config), "split_audit": audit_splits(data),
            "executor_sha256": file_hash(root / "vendor/ppnl/executor-point-sg.py"),
            "source_sha256": source, "methods": list(METHODS), "seed": 42,
            "prompt_format": "vanilla", "beam_width": 4, "max_actions": 36,
            "test_status": "reused_exploratory"}


def development_report(root, identity):
    root = Path(root)
    out = root / "outputs/gvr_repair_study"
    files = {"generation": out / "dev_generation/result.json",
             "repair": out / "dev_repair/result.json",
             "legacy": out / "dev_legacy/metrics.json",
             "legacy_pipeline": out / "dev_legacy/pipeline_traces.jsonl",
             "legacy_candidates": out / "dev_legacy/candidate_traces.jsonl"}
    generation = json.loads(files["generation"].read_text(encoding="utf-8"))["rows"]
    repair = json.loads(files["repair"].read_text(encoding="utf-8"))["rows"]
    legacy = json.loads(files["legacy"].read_text(encoding="utf-8"))
    if len(generation) != 40 or len(repair) != 60 or legacy["subset_counts"] != {
            "verification": 100, "repair": 60, "generation": 40}:
        raise ValueError("Development task quotas differ")
    result = {task: {method: sum(row["initial_valid"] or row["repairs"][method]["success"]
                                   for row in rows)
                     for method in ("E1", "B4", "C4", "U4", "H4")}
              for task, rows in (("generation", generation), ("repair", repair))}
    for mode in ("direct", "self", "executor_one", "executor_three"):
        result[mode] = legacy[mode]["success_rate"]
    result["verification_macro_f1"] = legacy["verification"]["macro_f1"]
    result["legacy_candidate_success_rate"] = legacy["repair"]["success_rate"]
    gates = {"min_c4_generation_8": result["generation"]["C4"] >= 8,
             "net_vs_e1_4": result["generation"]["C4"] - result["generation"]["E1"] >= 4,
             "net_vs_b4_2": result["generation"]["C4"] - result["generation"]["B4"] >= 2,
             "repair_drop_at_most_3": result["repair"]["C4"] >= result["repair"]["E1"] - 3}
    if not all(gates.values()):
        raise ValueError(f"Development gates failed: {gates}")
    if abs(result["legacy_candidate_success_rate"] * 60 - result["repair"]["E1"]) > 1e-9:
        raise ValueError("Legacy candidate score differs from paired repair run")
    return {"split": "dev", "identity": identity, "metrics": result, "gates": gates,
            "source_files_sha256": {name: file_hash(path) for name, path in files.items()},
            "interpretation": "C4 improves GVR, but H4 is stronger or equal on both dev tasks; "
                              "model ranking advantage is unestablished",
            "test_status": "reused_exploratory"}
