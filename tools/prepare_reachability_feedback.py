"""Create unlabelled human-review pairs on separate, layout-disjoint maps."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ppnl.core import Grid, shortest_path
from ppnl.data import read_jsonl, write_jsonl
from reachability_probe import load_cases


def main():
    out=ROOT/'data/reachability_feedback_pending'
    if out.exists(): raise FileExistsError(out)
    excluded={r['layout_id'] for split in ('train','dev','test_id','test_ood')
              for r in read_jsonl(ROOT/'data/derived'/f'{split}.jsonl')}
    excluded.update(g.layout_id for g in load_cases()[0].values())
    rng=random.Random(42)
    batches={}
    for split,count in (('train',16),('dev',8)):
        records=[]
        attempts=0
        while len(records)<count:
            attempts+=1
            if attempts>10000: raise RuntimeError('Could not fill quotas')
            obstacles=tuple((r,c) for r in range(6) for c in range(6)
                            if (r,c) not in ((0,0),(5,5)) and rng.random()<.35)
            grid=Grid(6,6,obstacles,(0,0),(5,5))
            path=shortest_path(grid)
            unreachable=len(records)%2==0
            if grid.layout_id in excluded or (path is None)!=unreachable: continue
            excluded.add(grid.layout_id)
            answers=(["NO_PATH", "right right right right right down down down down down"]
                     if unreachable else [" ".join(path),"NO_PATH"])
            rng.shuffle(answers)
            records.append({'id':f'{split}-{len(records):02d}','grid':asdict(grid),
                            'layout_id':grid.layout_id,'response_a':answers[0],'response_b':answers[1],
                            'candidate_source':'synthetic; BFS and fixed-plan controls',
                            'preference':None,'annotator':None,'human_rationale':None})
        batches[split]=records
    out.mkdir()
    for split,rows in batches.items(): write_jsonl(out/f'{split}.jsonl',rows)
    print('Created 16 train and 8 dev review pairs. All preferences remain null; no human training performed.')


if __name__=='__main__':main()
