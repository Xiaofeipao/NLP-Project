"""Replay all shortcut-study classifications, paired probes, and repair outputs."""

from collections import Counter
import hashlib
import json
from math import isclose
from pathlib import Path
import sys
import tarfile

from ppnl.core import ACTIONS, Grid, execute, parse_verdict
from ppnl.evaluation import verification_metrics, plan_metrics


def same(actual,expected):
    for key,value in actual.items():
        assert (isclose(value,expected[key],abs_tol=1e-12) if isinstance(value,float)
                else value==expected[key]),key


def main(root):
    results=json.loads((root/"results.json").read_text())
    counts=Counter()
    with tarfile.open(root/"evidence.tar.gz") as archive:
        def rows(name):
            return [json.loads(l) for l in archive.extractfile("./"+name)]
        pairs=rows("probes.jsonl")
        train=rows("original.jsonl")+rows("balanced.jsonl")
        for r in train+pairs:
            grid=Grid.from_dict(r["grid"])
            p=tuple(r["verification_position"])
            a=r["action"]
            if a is None:
                expected="valid" if p==grid.goal else "invalid | step=1 | reason=goal_not_reached"
            elif a not in ACTIONS:
                expected="invalid | step=1 | reason=invalid_action_token"
            else:
                dr,dc=ACTIONS[a]
                endpoint=(p[0]+dr,p[1]+dc)
                reason=("out_of_bounds" if not grid.inside(endpoint) else
                        "obstacle_collision" if endpoint in grid.obstacles else None)
                expected=(f"invalid | step=1 | reason={reason}" if reason else
                          "valid" if endpoint==grid.goal else "invalid | step=2 | reason=goal_not_reached")
            assert expected==r["target"],r["id"]
            counts["verified_training_and_probe_labels"]+=1
        groups={g:rows(f"candidates_{g}.jsonl") for g in ("dev","dev_generated","test_id","test_ood",
                                                      "g_peer_old","g_peer_new","g_old","g_new")}
        known={Grid.from_dict(r["grid"]).layout_id for r in train}
        heldout={Grid.from_dict(r["grid"]).layout_id for rs in (*groups.values(),pairs) for r in rs}
        assert known.isdisjoint(heldout)
        for arm,parts in results["arms"].items():
            for style,metrics in {**parts["metrics"],**parts["robustness"]}.items():
                for g,m in metrics.items():
                    candidates=groups[g]
                    records=rows(f"{arm}_{style}_{g}.jsonl")
                    grids=[Grid.from_dict(r["grid"]) for r in candidates]
                    truth=[execute(grid,r["candidate"]) for grid,r in zip(grids,candidates,strict=True)]
                    end=[execute(grid,r["plan"]) for grid,r in zip(grids,records,strict=True)]
                    for c,r,t,e in zip(candidates,records,truth,end,strict=True):
                        assert (c["id"],c["candidate"],t.label(),e.label())==(
                            r["id"],r["candidate"],r["gold"],r["final"])
                    pred=[r["prediction"] for r in records]
                    same(verification_metrics([vars(t) for t in truth],pred),m)
                    same(plan_metrics(grids,[r["plan"] for r in records]),m["planning"])
                    assert m["fixed"]==sum(not t.valid and e.valid for t,e in zip(truth,end))
                    assert m["broken"]==sum(t.valid and not e.valid for t,e in zip(truth,end))
                    counts["candidate_records"]+=len(records)
                probe=rows(f"{arm}_{style}_probes.jsonl")
                assert [(r["id"],r["target"]) for r in probe]==[(r["id"],r["target"]) for r in pairs]
                for k,p in parts["probes"][style].items():
                    indices=[i for i,r in enumerate(probe) if r["type"]==k]
                    assert p["exact"]==sum(probe[i]["target"]==probe[i]["prediction"].strip() for i in indices)
                    assert p["both_exact_pairs"]==sum(
                        all(probe[j]["target"]==probe[j]["prediction"].strip() for j in (i,i+1)) for i in indices[::2])
                    assert p["unchanged_pairs"]==sum(
                        probe[i]["prediction"].strip()==probe[i+1]["prediction"].strip() for i in indices[::2])
                counts["probe_records"]+=len(probe)
    with tarfile.open(root/"source.tar.gz") as archive:
        name="run_hierarchical_study.py" if "run_hierarchical_study.py" in archive.getnames() else "run_shortcut_study.py"
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==results["source"]
    (root/"audit.json").write_text(json.dumps(counts,indent=2),encoding="utf-8")
    print(json.dumps(counts,indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]))
