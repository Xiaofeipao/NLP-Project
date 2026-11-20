"""Recompute V and repair metrics from prompt-only study records."""

from collections import Counter
import hashlib
import json
from math import isclose
from pathlib import Path
import sys
import tarfile

from ppnl.core import Grid, execute
from ppnl.evaluation import plan_metrics, verification_metrics


def same(actual, expected):
    for key,value in actual.items():
        assert (isclose(value,expected[key],abs_tol=1e-12) if isinstance(value,float)
                else value==expected[key]), (key,value,expected[key])


def main(root):
    m=json.loads((root/"results.json").read_text())
    report=Counter()
    with tarfile.open(root/"evidence.tar.gz") as z:
        def rows(name):
            return [json.loads(s) for s in z.extractfile("./"+name)]
        train=rows("train_github.jsonl")
        for style in ("compact","explicit"):
            other=rows(f"train_{style}.jsonl")
            assert [{k:v for k,v in r.items() if k!="input"} for r in train]==[
                {k:v for k,v in r.items() if k!="input"} for r in other]
        for arm,v in m["arms"].items():
            for group,stored in v["metrics"].items():
                candidates=rows(f"candidates_{group}.jsonl")
                traces=rows(f"{arm}_{group}.jsonl")
                grids=[Grid.from_dict(r["grid"]) for r in candidates]
                truths=[execute(g,r["candidate"]) for g,r in zip(grids,candidates,strict=True)]
                outputs=[execute(g,r["plan"]) for g,r in zip(grids,traces,strict=True)]
                for a,b,t,y in zip(candidates,traces,truths,outputs,strict=True):
                    assert (a["id"],a["candidate"],t.label(),y.label())==(
                        b["id"],b["candidate"],b["gold"],b["final"])
                same(verification_metrics([vars(t) for t in truths],[r["prediction"] for r in traces]),stored)
                same(plan_metrics(grids,[r["plan"] for r in traces]),stored["planning"])
                assert stored["fixed"]==sum(not t.valid and y.valid for t,y in zip(truths,outputs))
                assert stored["broken"]==sum(t.valid and not y.valid for t,y in zip(truths,outputs))
                report["verified_candidate_records"]+=len(candidates)
                report["local_query_records"]+=len(rows(f"{arm}_{group}_local.jsonl"))
        report["training_rows_per_template"]=len(train)
    with tarfile.open(root/"source.tar.gz") as z:
        assert hashlib.sha256(z.extractfile("run_prompt_study.py").read()).hexdigest()==m["source"]
    (root/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]))
