"""Shared workflow and CPU verification of public evidence, without model weights."""

import hashlib
import json
from pathlib import Path
import zipfile

from ppnl.core import Grid, execute, OfficialOracle
from ppnl.evaluation import plan_metrics, verification_metrics
from ppnl.core import parse_verdict
from scripts.run_phase2_feedback import continue_plan
from ppnl.verifier_guard import guard_verdict

ROOT = Path(__file__).resolve().parents[1]


def verify_results(archive, part, executor=None):
    """Check exported bytes and recompute all path/diagnostic summary metrics."""
    published = ROOT / "deliverables/part2_part3_manifest.json"
    if published.exists() and Path(archive).name in json.loads(published.read_text())["files"]:
        expected = json.loads(published.read_text())["files"][Path(archive).name]["sha256"]
        if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != expected:
            raise ValueError("Evidence archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("PUBLIC_EXPORT.json"))["files"]
        for name, record in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest() != record["export_sha256"]:
                raise ValueError("Evidence checksum mismatch: " + name)
        if any(name.endswith((".safetensors", ".bin", ".pt", ".pth")) for name in z.namelist()):
            raise ValueError("Evidence must not contain model weights")
        oracle = OfficialOracle(executor) if executor else None
        check = oracle.check if oracle else execute
        checks = cases = 0
        reports = (
            json.loads(z.read("results.json"))
            if part == 2
            else {
                s: json.loads(z.read(f"evaluation/{s}_results.json"))
                for s in ("dev", "test_id", "test_ood")
            }
        )
        for split, pools in reports.items():
            for pool, methods in pools.items():
                for method, metrics in methods.items():
                    if part == 2:
                        filename = f"{split}_{pool}_traces.jsonl"
                        rows = [
                            r
                            for line in z.read(filename).decode().splitlines()
                            if (r := json.loads(line))["method"] == method
                        ]
                    else:
                        filename = f"evaluation/{split}_{pool}_{method}_traces.jsonl"
                        rows = [json.loads(line) for line in z.read(filename).decode().splitlines()]
                    grids = [Grid.from_dict(r["grid"]) for r in rows]
                    candidates = [r["candidate"] for r in rows]
                    plans = [r["output" if part == 2 else "plan"] for r in rows]
                    truths = [check(g, c) for g, c in zip(grids, candidates)]
                    for r, t in zip(rows, truths):
                        if r["gold"] != t.label():
                            raise ValueError("Stored truth mismatch")
                    for g, plan in zip(grids, plans):
                        check(g, plan)
                    actual = plan_metrics(grids, plans)
                    if actual != metrics["planning"]:
                        raise ValueError("Path metric mismatch: " + filename)
                    bad = [i for i, t in enumerate(truths) if not t.valid]
                    repair = (
                        plan_metrics([grids[i] for i in bad], [plans[i] for i in bad])
                        if bad
                        else None
                    )
                    if repair != metrics["repair_invalid_only" if part == 2 else "repair"]:
                        raise ValueError("Repair metric mismatch")
                    gold = [{"valid": t.valid, "step": t.step, "reason": t.reason} for t in truths]
                    if (
                        verification_metrics(gold, [r["feedback"] for r in rows])
                        != metrics["verification"]
                    ):
                        raise ValueError("Verification metric mismatch")
                    checks += 3
                    cases += len(rows)
        return {
            "passed": True,
            "export_files": len(manifest),
            "metric_groups": checks,
            "method_cases": cases,
            "official_executor": oracle is not None,
            "weights_needed": False,
        }


def prepare(out):
    """Extract the exact FirstPart training code; reuse repository data separately."""
    if out.exists():
        raise FileExistsError("Use a new output directory")
    archive = ROOT / "deliverables/firstpart_training_source_f00fbdf.zip"
    expected = json.loads((ROOT / "deliverables/part2_part3_manifest.json").read_text())["files"][
        archive.name
    ]["sha256"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("FirstPart source archive changed")
    source = (out / "source").resolve()
    source.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target = (source / name).resolve()
            if source not in target.parents and target != source:
                raise ValueError("Unsafe archive member")
        z.extractall(source)
    return source
