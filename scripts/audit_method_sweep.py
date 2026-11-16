"""Recompute stored metrics with official execution and verify training/data locks."""

import json
import hashlib
import math
import argparse
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, OfficialOracle, shortest_path, parse_verdict
from ppnl.data import read_jsonl, write_json
from ppnl.method_sweep import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        help="Exact archived source for historical runs before CLI packaging",
    )
    args = parser.parse_args()
    root = args.root
    p = json.loads((root / "protocol.json").read_text())
    checks = cases = 0
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if args.source_snapshot:
        with zipfile.ZipFile(args.source_snapshot) as z:
            for n, h in p["code_sha256"].items():
                path = n.replace("\\", "/")
                marker = "scripts/" if "scripts/" in path else "ppnl/"
                relative = path[path.rfind(marker) :]
                assert hashlib.sha256(z.read(relative)).hexdigest() == h, n
    else:
        for n, h in p["code_sha256"].items():
            assert sha(n) == h, n
    for split, h in p["data_sha256"].items():
        assert sha(f"data/policy_v3/{split}.jsonl") == h
    original = read_jsonl("data/policy_v3/train.jsonl")
    v = [r["id"] for r in original if r["task"] == "verification"]
    for branch in ("continuation_control", "multitask", "augmentation"):
        path = root / "training" / branch
        budget = json.loads((path / "sample_budget.json").read_text())
        trained = json.loads((path / "training.json").read_text())
        assert set(budget["ids"]) == set(v[16:]) and len(budget["ids"]) == 1984
        assert set(budget["excluded_pilot_ids"]) == set(v[:16])
        assert trained["updates"] == 124 and trained["epochs"] == 1 and trained["examples"] == 1984
        assert sha(path / "best/model.safetensors") == trained["weights_sha256"]
        checks += 4
    oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
    reports = {}
    for split in ("dev", "test_id", "test_ood"):
        reports[split] = json.loads((root / "evaluation" / f"{split}_results.json").read_text())
        for pool, methods in reports[split].items():
            assert set(methods) == {m["name"] for m in METHODS}
            for name, m in methods.items():
                assert "not_run" not in m, (split, pool, name, m)
                rows = read_jsonl(root / "evaluation" / f"{split}_{pool}_{name}_traces.jsonl")
                n = len(rows)
                assert len({r["id"] for r in rows}) == n
                truths = []
                parsed = []
                good = feasible = optimal = repair = nbad = regress = 0
                for r in rows:
                    g = Grid.from_dict(r["grid"])
                    t = oracle.check(g, r["candidate"])
                    f = oracle.check(g, r["plan"])
                    assert t.label() == r["gold"]
                    truths.append(t)
                    parsed.append(parse_verdict(r["feedback"]))
                    good += f.valid
                    feasible += f.feasible
                    optimal += f.valid and len(r["plan"].split()) == len(shortest_path(g))
                    if not t.valid:
                        nbad += 1
                        repair += f.valid
                    regress += t.valid and not f.valid
                    assert sum(x["used"] for x in r["history"]) <= 36
                for key, value in [
                    ("success_rate", good / n),
                    ("feasibility", feasible / n),
                    ("optimality", optimal / n),
                ]:
                    assert math.isclose(m["planning"][key], value, abs_tol=1e-12)
                    checks += 1
                if nbad:
                    assert m["repair"]["success_rate"] == repair / nbad
                    checks += 1
                assert m["regressions"] == regress
                checks += 1
                f1 = []
                for cls in (False, True):
                    tp = sum(
                        t.valid == cls and q is not None and q[0] == cls
                        for t, q in zip(truths, parsed)
                    )
                    fp = sum(
                        t.valid != cls and q is not None and q[0] == cls
                        for t, q in zip(truths, parsed)
                    )
                    fn = sum(
                        t.valid == cls and (q is None or q[0] != cls)
                        for t, q in zip(truths, parsed)
                    )
                    f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
                assert math.isclose(m["verification"]["macro_f1"], sum(f1) / 2, abs_tol=1e-12)
                checks += 1
                assert (
                    m["verification"]["format_accuracy"] == sum(q is not None for q in parsed) / n
                )
                checks += 1
                invalid = [(t, q) for t, q in zip(truths, parsed) if not t.valid]
                for key, correct in [
                    (
                        "first_error_localization_accuracy",
                        lambda t, q: q is not None and not q[0] and q[1] == t.step,
                    ),
                    (
                        "failure_type_accuracy",
                        lambda t, q: q is not None and not q[0] and q[2] == t.reason,
                    ),
                    ("joint_error_accuracy", lambda t, q: q == (False, t.step, t.reason)),
                ]:
                    assert m["verification"][key] == sum(correct(t, q) for t, q in invalid) / len(
                        invalid
                    )
                    checks += 1
                cases += n
    dev = reports["dev"]
    selection = json.loads((root / "selection.json").read_text())
    assert sha(root / "evaluation/dev_results.json") == selection["dev_sha256"]

    def rank(m):
        c = dev["candidates"][m["name"]]
        g = dev["generated"][m["name"]]
        return (
            g["planning"]["success_rate"],
            c["repair"]["success_rate"],
            g["planning"]["optimality"],
            c["verification"]["joint_error_accuracy"],
        )

    best = max([m for m in METHODS if not m.get("control")], key=rank)
    assert best == selection["method"]
    write_json(
        root / "audit.json",
        {
            "passed": True,
            "metric_checks": checks,
            "method_cases": cases,
            "method_count": len(METHODS),
            "selected": best,
            "code_data_and_weight_hashes_match": True,
            "training_lineage_counts_checked": True,
            "all_methods_completed": True,
        },
    )
    print({"passed": True, "checks": checks, "method_cases": cases})


if __name__ == "__main__":
    main()
