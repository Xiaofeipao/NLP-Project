"""CPU-only paired double-error extension; never overwrite frozen single-error sets."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from ppnl.core import Grid,OfficialOracle,shortest_path,execute
from ppnl.data import read_jsonl,write_json,write_jsonl,make_candidate,edit_distance


def build(output,seed=20261112):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    started=time.perf_counter();rng=random.Random(seed)
    oracle=OfficialOracle('vendor/ppnl/executor-point-sg.py')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    sources=[Path('data/generalization_v1')/f'test_{name}.jsonl' for name in ['7x7','dense_6x6']]
    before={str(f):sha(f) for f in sources}
    datasets={}
    for source in sources:
        rows=read_jsonl(source);paired=[]
        for row in rows:
            g=Grid.from_dict(row['grid']);base=shortest_path(g)
            if row['valid']:
                new=dict(row) # same valid controls, including detour path
            else:
                candidate,verdict,edits=make_candidate(g,base,rng,oracle,False,2)
                assert len(edits)==2 and len({e['original_index'] for e in edits})==2
                assert edit_distance(base,candidate.split())==2
                assert execute(g,candidate).label()==verdict.label()
                new={**row,'candidate':candidate,'label':verdict.label(),'valid':False,
                     'reason':verdict.reason,'step':verdict.step,'trace':verdict.trace,
                     'edits':edits,'edit_count':2}
            split=source.stem+'_double'
            new.update(id=row['id']+'_double',parent_id=row['id'],split=split)
            paired.append(new)
        assert len(paired)==200 and sum(not r['valid'] for r in paired)==100
        assert [(r['grid'],r['layout_id']) for r in rows]==[(r['grid'],r['layout_id']) for r in paired]
        datasets[split]=paired
    assert before=={str(f):sha(f) for f in sources}
    output.mkdir(parents=True)
    for split,rows in datasets.items():write_jsonl(output/f'{split}.jsonl',rows)
    write_json(output/'manifest.json',{'seed':seed,'purpose':'optional_generalization_test_only',
        'cpu_construction_seconds':time.perf_counter()-started,'gpu_seconds':0,
        'source_sha256':before,'generator_sha256':sha(__file__),
        'files_sha256':{f'{s}.jsonl':sha(output/f'{s}.jsonl') for s in datasets},
        'counts':{s:{'rows':200,'valid':100,'invalid_double_edit':100,'unique_layouts':200} for s in datasets},
        'rules':'same held-out layouts/start/goal/valid controls as v1; two distinct original edit sites and token edit distance2; two edits do not imply two observable failures; targets used only for construction/scoring; no train/dev changes'})
    return datasets


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.output)
