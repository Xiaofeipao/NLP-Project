"""A lexical-only baseline that cannot read positions, obstacles, or goals."""

from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import sys
import tarfile
import zipfile

from ppnl.core import Grid,execute
from ppnl.evaluation import verification_metrics
from state_baselines.step_verification import verify_steps


class ActionOnly:
    def __init__(self,train):
        groups=defaultdict(Counter)
        total=Counter()
        for r in train:
            groups[r["action"]][r["target"]]+=1
            total[r["target"]]+=1
        self.rules={a:max(c,key=c.get) for a,c in groups.items()}
        self.default=max(total,key=total.get)

    def predict(self,prompts,check):
        return [self.rules.get(json.loads(re.search(r"Candidate action: (.*?)\. A null",p)[1]),self.default)
                for p in prompts]


def main(output):
    with zipfile.ZipFile("deliverables/step_verification_v1_v2_evidence.zip") as z:
        result={}
        for version in ("v1","v2"):
            raw=[json.loads(x) for x in z.read(f"{version}/train.jsonl").splitlines()]
            train=[r for r in raw if r["task"]=="verification"]
            obj=ActionOnly(train)
            result[version]={"label_counts":dict(Counter(r["reason"] or "valid" for r in train)),
                "action_rules":{str(k):v for k,v in obj.rules.items()},
                "null_auxiliary_negative":sum(r.get("state_verification",{}).get("valid") is False for r in raw),
                "literal_id_in_input":sum(r["id"] in r["input"] for r in train),
                "conflicting_input_targets":sum(len(ys)>1 for ys in (
                    {r["target"] for r in train if r["input"]==s} for s in {r["input"] for r in train})),
                "candidate_metrics":{}}
            root="results_full" if version=="v1" else "results"
            for group in ("dev","test_id","test_ood"):
                rs=[json.loads(x) for x in z.read(f"{version}/{root}/{group}/flan_t5_base/candidates.jsonl").splitlines()]
                grids=[Grid.from_dict(r["grid"]) for r in rs]
                pred=verify_steps(obj,grids,[r["direct"] for r in rs])
                truth=[execute(g,r["direct"]) for g,r in zip(grids,rs,strict=True)]
                result[version]["candidate_metrics"][group]=verification_metrics([vars(t) for t in truth],pred)
    output.mkdir(parents=True,exist_ok=True)
    (output/"lexical_baseline.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    for v,r in result.items():
        print(v,r["label_counts"],"ID/OOD F1",
              *(round(r["candidate_metrics"][g]["macro_f1"]*100,2) for g in ("test_id","test_ood")))


if __name__=="__main__":
    main(Path(sys.argv[1]))
