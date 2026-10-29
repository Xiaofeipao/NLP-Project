"""Read-only handoff checks before a Project 6 GPU run."""
from __future__ import annotations

import json
from pathlib import Path

from .data import task_index_manifest


def preflight(config, protocol):
    checks = {}
    index_path = Path(protocol["task_index"])
    try:
        checks["task_index"] = (index_path.exists() and
                                json.loads(index_path.read_text(encoding="utf-8")) ==
                                task_index_manifest(protocol["dataset_root"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        checks["task_index"] = False
    try:
        from .core import OfficialOracle
        manifest = json.loads((Path(protocol["dataset_root"]) / "manifest.json").read_text(encoding="utf-8"))
        checks["official_executor"] = (OfficialOracle("vendor/ppnl/executor-point-sg.py").sha256 ==
                                       manifest["executor_sha256"])
    except (OSError, ValueError, KeyError):
        checks["official_executor"] = False
    ledger = Path(config["gpu_ledger"])
    try:
        runs = json.loads(ledger.read_text(encoding="utf-8"))["runs"]
        seconds = sum(run["seconds"] for run in runs)
        checks["shared_gpu_ledger"] = 0 <= seconds < config["gpu_hours_cap"] * 3600
    except (OSError, ValueError, KeyError, TypeError):
        seconds = None
        checks["shared_gpu_ledger"] = False
    item = config["models"]["flan_t5_small"]
    checks["manager_checkpoint_confirmation"] = bool(item.get("manager_confirmed") and item.get("confirmation_note"))
    checks["evidence_declared"] = all(protocol["evidence"].get(key) is True for key in
                                       ("gpu_ledger_reconciled", "checkpoint_manager_confirmed", "baseline_outputs_verified"))
    checks["base_checkpoint"] = Path(protocol["generation_base"]).joinpath("config.json").exists()
    checks["verification_adapter"] = Path(protocol["verification_adapter"]).joinpath("adapter_config.json").exists()
    checks["repair_adapter"] = Path(protocol["repair_adapter"]).joinpath("adapter_config.json").exists()
    return {"checks": checks, "gpu_seconds_recorded": seconds,
            "ready_for_training": all(checks[k] for k in ("task_index", "official_executor",
                                                         "shared_gpu_ledger", "manager_checkpoint_confirmation",
                                                         "evidence_declared")),
            "ready_for_evaluation": all(checks.values())}
