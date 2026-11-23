"""Random sampling, strict quota checks, no hand-authored maps or fixed routes."""
import argparse,collections,hashlib,json,random,secrets,sys
from dataclasses import asdict
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'nlp-stepverify-1119'))
from ppnl.core import Grid,ACTIONS,execute,OfficialOracle,shortest_path
from ppnl.data import read_jsonl,write_json,write_jsonl

KINDS=('obstacle_collision','out_of_bounds','invalid_action_token','goal_not_reached')
WORDS=('jump','wait','north','south','east','west','forward','backward','turn_left','turn_right','stay','teleport','upward','downward','leftward','rightward','UP','DOWN','LEFT','RIGHT')

def random_route(rng):
    cells=[(r,c) for r in range(6) for c in range(6)]
    density=rng.uniform(.12,.32)
    walls={p for p in cells if rng.random()<density}
    free=[p for p in cells if p not in walls]
    if len(walls)<3 or len(free)<8:return None
    pos=rng.choice(free);visited={pos};positions=[pos];actions=[]
    limit=rng.randint(4,25)
    for _ in range(limit):
        candidates=[]
        for a,(dr,dc) in ACTIONS.items():
            q=pos[0]+dr,pos[1]+dc
            if q in free and q not in visited:candidates.append((a,q))
        if not candidates:break
        a,pos=rng.choice(candidates);actions.append(a);positions.append(pos);visited.add(pos)
    if len(actions)<3:return None
    stop=rng.randint(3,len(actions));actions=actions[:stop];positions=positions[:stop+1]
    g=Grid(6,6,tuple(sorted(walls)),positions[0],positions[-1])
    return g,actions,positions

def inserted_error(g,positions,kind,rng,avoid=()):
    choices=[]
    for index,pos in enumerate(positions):
        if index in avoid:continue
        if kind=='invalid_action_token':
            choices.append((index,rng.choice(WORDS)))
        else:
            for a,(dr,dc) in ACTIONS.items():
                q=pos[0]+dr,pos[1]+dc
                if (kind=='out_of_bounds' and not g.inside(q)) or (kind=='obstacle_collision' and q in g.obstacles):choices.append((index,a))
    return rng.choice(choices) if choices else None

def scan_noop(g,plan):
    """Audit-only replay: reject illegal moves and stay put; original executor unchanged."""
    pos=g.start;events=[]
    for step,a in enumerate(plan.split(),1):
        q=pos;reason=None
        if a not in ACTIONS:reason='invalid_action_token'
        else:
            dr,dc=ACTIONS[a];q=pos[0]+dr,pos[1]+dc
            if not g.inside(q):reason='out_of_bounds'
            elif q in g.obstacles:reason='obstacle_collision'
        if reason:events.append({'step':step,'reason':reason,'position':list(pos),'action':a})
        else:pos=q
    if pos!=g.goal:events.append({'step':len(plan.split())+1,'reason':'goal_not_reached','position':list(pos),'action':'<eos>'})
    return events

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int)
    a=p.parse_args();seed=a.seed if a.seed is not None else secrets.randbits(128);rng=random.Random(seed)
    if a.out.exists():raise FileExistsError(a.out)
    excluded=set();files={}
    for root in (BASE/'nlp-stepverify-1119/data',BASE/'nlp-newpart1-update/part23/data'):
        for f in root.rglob('*.jsonl'):
            if f.name not in ('train.jsonl','dev.jsonl','test_id.jsonl','test_ood.jsonl'):continue
            files[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
            for r in read_jsonl(f):
                if 'grid' in r:excluded.add(Grid.from_dict(r['grid']).layout_id)
                elif 'layout_id' in r:excluded.add(r['layout_id'])
    oracle=OfficialOracle(BASE.parent/'output/NLP-Project-review/executor-point-sg.py')
    used=set(excluded);rows=[];attempts=collections.Counter()
    for kind in KINDS:
        accepted=0
        while accepted<50:
            attempts[kind]+=1
            if attempts[kind]>100000:raise RuntimeError('Quota sampling exhausted')
            sampled=random_route(rng)
            if sampled is None:continue
            g,base,positions=sampled
            if g.layout_id in used:continue
            if kind=='goal_not_reached':
                cut=rng.randint(1,len(base)-1);tokens=base[:cut]
                edit={'kind':kind,'operation':'truncate','cut_after':cut,'removed':base[cut:]}
            else:
                choice=inserted_error(g,positions,kind,rng)
                if choice is None:continue
                index,word=choice;tokens=base[:index]+[word]+base[index:]
                edit={'kind':kind,'operation':'insert_rejected_action','base_index':index,'action':word}
            candidate=' '.join(tokens);v=oracle.check(g,candidate);events=scan_noop(g,candidate)
            if [e['reason'] for e in events]!=[kind] or v.reason!=kind:continue
            assert execute(g,' '.join(base)).valid
            used.add(g.layout_id);accepted+=1
            rows.append({'id':'random-'+f'{rng.getrandbits(96):024x}','split':'random_single','task':'verification','grid':asdict(g),'layout_id':g.layout_id,'candidate':candidate,'label':v.label(),'valid':False,'reason':v.reason,'step':v.step,'trace':v.trace,'reference_path':' '.join(base),'shortest_path_length':len(shortest_path(g)),'edit_count':1,'edits':[edit],'audit_events':events})
    rng.shuffle(rows)
    assert len(rows)==200 and len({r['layout_id'] for r in rows})==200
    assert collections.Counter(r['reason'] for r in rows)==collections.Counter(dict.fromkeys(KINDS,50))
    assert not ({r['layout_id'] for r in rows}&excluded)
    a.out.mkdir(parents=True)
    write_jsonl(a.out/'single_error_200.jsonl',rows)
    write_json(a.out/'manifest.json',{'seed':seed,'seed_source':'user' if a.seed is not None else 'OS entropy via secrets','grid':'6x6','counts':dict(collections.Counter(r['reason'] for r in rows)),'valid_controls':0,'randomized':['Bernoulli obstacle grid with random density','start','self-avoiding route and endpoint','error location','illegal word','final row order'],'route_generator':'random walk; BFS only checks reachability/shortest length, not constructing fixed reference routes','exclusion_files':files,'excluded_layouts':len(excluded),'attempts':dict(attempts),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((a.out/'single_error_200.jsonl').read_bytes()).hexdigest(),'double_status':'Awaiting definition of goal_not_reached + goal_not_reached','audit_semantics':'Standard executor labels first error. Separate rejected-move no-op scan audits exact number of error events; no change to project executor.'})
    print(json.dumps({'output':str(a.out),'counts':dict(collections.Counter(r['reason'] for r in rows)),'excluded_layouts':len(excluded)},ensure_ascii=False))

if __name__=='__main__':main()
