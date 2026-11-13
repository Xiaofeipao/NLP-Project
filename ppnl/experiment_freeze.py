"""CPU-only split auditing and reproducibility locks for GVR experiments.

These hashes detect accidental drift; they cannot prove that a person has never
looked at a test set, or defend against deliberate rewriting of the lock.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .core import Grid
from .data import audit_dataset, read_jsonl, task_index_manifest

SPLITS = ("train", "dev", "test_id", "test_ood")
ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_splits(directory):
    """Preserve historical sampling, export exact IDs, and check layout isolation."""
    root = Path(directory)
    audit_dataset(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = {split: read_jsonl(root / f"{split}.jsonl") for split in SPLITS}
    result = {}
    for split, examples in rows.items():
        group = "test" if split.startswith("test_") else split
        allowed = set(manifest["layout_splits"][group])
        if any(row["layout_id"] not in allowed for row in examples):
            raise ValueError(f"Rows outside declared layout split: {split}")
        expected = manifest["counts"][split]
        if isinstance(expected, dict):
            actual = Counter(row["task"] for row in examples)
            expected = Counter(expected)
        else:
            actual = len(examples)
        if actual != expected:
            raise ValueError(f"Count differs from manifest: {split}")
        result[split] = {
            "count": len(examples), "sha256": file_hash(root / f"{split}.jsonl"),
            "ids": [row["id"] for row in examples],
            "source_ids": sorted({row["source_id"] for row in examples}),
            "layout_ids": sorted({row["layout_id"] for row in examples}),
        }
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if set(result[left]["ids"]) & set(result[right]["ids"]):
                raise ValueError(f"Example ID leakage: {left}/{right}")
            if (left, right) != ("test_id", "test_ood"):
                if set(result[left]["source_ids"]) & set(result[right]["source_ids"]):
                    raise ValueError(f"Source ID leakage: {left}/{right}")
    def environments(examples):
        return [(g.layout_id, g.start, g.goal) for g in
                (Grid.from_dict(row["grid"]) for row in examples)]
    if environments(rows["test_id"]) != environments(rows["test_ood"]):
        raise ValueError("ID/OOD environments must remain paired in the same order")
    index = task_index_manifest(root)
    index_path = root / "gvr_task_index.json"
    if index_path.exists() and json.loads(index_path.read_text(encoding="utf-8")) != index:
        raise ValueError("Frozen task index differs from dataset")
    return {"schema_version": 1, "seed": manifest["seed"],
            "upstream_revision": manifest["upstream_revision"],
            "manifest_sha256": file_hash(root / "manifest.json"),
            "task_index": index, "splits": result,
            "sampling": "Historical layout-grouped sampling; unchanged task quotas; no HotpotQA labels",
            "test_relationship": "Paired ID/OOD environments; different candidate corruption",
            "test_status": "reused_exploratory"}


def _tree_hashes(directory):
    root = Path(directory)
    files = {p.relative_to(root).as_posix(): file_hash(p)
             for p in sorted(root.rglob("*")) if p.is_file()}
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError(f"No model weights: {root}")
    return files


def freeze_payload(config, protocol, selection_report, root=ROOT):
    """Bind all inference inputs, not just the selected weight files."""
    root = Path(root).resolve()
    report_path = Path(selection_report).resolve()
    # Output storage may live on another disk behind a workspace junction.
    # The report contents remain hash-bound; this is a local reproducibility lock.
    try:
        report_name = report_path.relative_to(root).as_posix()
    except ValueError:
        report_name = report_path.as_posix()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    from .prompts import protocol_model_config, check_checkpoint_prompt_format
    style = protocol_model_config(config, protocol)["prompt_format"]
    check_checkpoint_prompt_format(protocol["generation_base"], style)
    if report.get("prompt_format", "vanilla") != style:
        raise ValueError("Development evidence uses a different prompt format")
    if report.get("disabled_adapter_matches_base") is not True or not report.get("dev_diagnostics"):
        raise ValueError("Selection evidence must be a successful verify_dev report")
    from .task_adapters import checkpoint_hash
    identity = {"base_sha256": checkpoint_hash(protocol["generation_base"]),
                "adapters_sha256": {name: checkpoint_hash(protocol[f"{name}_adapter"])
                                    for name in ("verification", "repair")}}
    if report.get("identity") != identity:
        raise ValueError("Development verification does not match selected weights")
    trees = {name: _tree_hashes(protocol[key]) for name, key in (
        ("generation", "generation_base"), ("verification", "verification_adapter"),
        ("repair", "repair_adapter"))}
    code = {p.relative_to(root).as_posix(): file_hash(p)
            for folder in ("ppnl", "scripts") for p in sorted((root / folder).rglob("*.py"))}
    if not code:
        raise ValueError("No experiment source code found")
    canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {"split_audit": audit_splits(protocol["dataset_root"]),
            "config_sha256": canonical(config), "protocol_sha256": canonical(protocol),
            "models": trees, "identity": identity, "source_sha256": code,
            "executor_sha256": file_hash(root / "vendor/ppnl/executor-point-sg.py"),
            "selection_report": report_name, "selection_report_sha256": file_hash(report_path)}


def create_freeze(config, protocol, selection_report, note, root=ROOT):
    if not note.strip():
        raise ValueError("Describe the development-only checkpoint/setting selection")
    payload = freeze_payload(config, protocol, selection_report, root)
    manifest = json.loads((Path(protocol["dataset_root"]) / "manifest.json").read_text(encoding="utf-8"))
    if payload["executor_sha256"] != manifest["executor_sha256"]:
        raise ValueError("Executor differs from dataset manifest")
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_note": note, "payload": payload,
            "limitation": "Historical tests remain exploratory; freezing cannot restore blindness."}


def validate_freeze(path, config, protocol, root=ROOT):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or not record.get("selection_note", "").strip():
        raise ValueError("Invalid experiment freeze")
    frozen = record["payload"]
    actual = freeze_payload(config, protocol, Path(root) / frozen["selection_report"], root)
    if frozen != actual:
        changed = sorted(key for key in actual if frozen.get(key) != actual[key])
        raise ValueError(f"Experiment changed after freezing: {changed}")
    return {"freeze_sha256": file_hash(path), "test_status": actual["split_audit"]["test_status"]}


def write_new_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
