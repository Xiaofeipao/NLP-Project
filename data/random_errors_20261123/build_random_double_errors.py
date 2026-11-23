"""Nine supported pairs, exactly two audit-visible events, no double EOS label."""
import argparse,collections,hashlib,itertools,json,random,secrets,sys
from dataclasses import asdict
from pathlib import Path
from build_random_error_sets import BASE,KINDS,random_route,inserted_error,scan_noop
from ppnl.core import execute,OfficialOracle,shortest_path
from ppnl.data import read_jsonl,write_json,write_jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if (a.out/'double_error_200.jsonl').exists():raise FileExistsError(a.out)
 seed=secrets.randbits(128);rng=random.Random(seed)
 manifest=json.loads((a.out/'manifest.json').read_text());single=read_jsonl(a.out/'single_error_200.jsonl')
 used={r['layout_id'] for r in single}
 from ppnl.core import Grid
 for f in manifest['exclusion_files']:
  for r in read_jsonl(f):
   if 'grid' in r:used.add(Grid.from_dict(r['grid']).layout_id)
   elif 'layout_id' in r:used.add(r['layout_id'])
 excluded=set(used)
 pairs=[p for p in itertools.combinations_with_replacement(KINDS,2) if p!=('goal_not_reached','goal_not_reached')]
 quotas={pair:22 for pair in pairs}
 for pair in rng.sample(pairs,2):quotas[pair]+=1
 oracle=OfficialOracle(BASE.parent/'output/NLP-Project-review/executor-point-sg.py')
 rows=[];attempts={}
 for pair,count in quotas.items():
  n=0;trials=0
  while n<count:
   trials+=1
   if trials>100000:raise RuntimeError(('Sampling exhausted',pair))
   sampled=random_route(rng)
   if sampled is None:continue
   g,reference,positions=sampled
   if g.layout_id in used:continue
   route=list(reference);edits=[]
   if 'goal_not_reached' in pair:
    cut=rng.randint(2,len(route)-1);route=route[:cut];positions=positions[:cut+1]
    edits.append({'kind':'goal_not_reached','operation':'truncate','cut_after':cut,'removed':reference[cut:]})
   physical=[k for k in pair if k!='goal_not_reached'];rng.shuffle(physical)
   operations={};failed=False
   for kind in physical:
    choice=inserted_error(g,positions,kind,rng,avoid=operations)
    if choice is None:failed=True;break
    index,word=choice;operations[index]=(kind,word)
    edits.append({'kind':kind,'operation':'insert_rejected_action','base_index':index,'action':word})
   if failed:continue
   tokens=[]
   for i in range(len(route)+1):
    if i in operations:tokens.append(operations[i][1])
    if i<len(route):tokens.append(route[i])
   candidate=' '.join(tokens);events=scan_noop(g,candidate)
   if len(events)!=2 or collections.Counter(e['reason'] for e in events)!=collections.Counter(pair):continue
   v=oracle.check(g,candidate)
   assert not v.valid and (v.step,v.reason)==(events[0]['step'],events[0]['reason'])
   assert execute(g,' '.join(reference)).valid
   used.add(g.layout_id);n+=1
   rows.append({'id':'random-'+f'{rng.getrandbits(96):024x}','split':'random_double','task':'verification','grid':asdict(g),'layout_id':g.layout_id,'candidate':candidate,'label':v.label(),'valid':False,'reason':v.reason,'step':v.step,'trace':v.trace,'reference_path':' '.join(reference),'shortest_path_length':len(shortest_path(g)),'error_pair':list(pair),'edit_count':2,'edits':edits,'audit_events':events})
  attempts['+'.join(pair)]=trials
 rng.shuffle(rows)
 assert len(rows)==200 and len({r['layout_id'] for r in rows})==200
 assert not ({r['layout_id'] for r in rows}&excluded)
 counts=collections.Counter('+'.join(r['error_pair']) for r in rows)
 assert counts==collections.Counter({'+'.join(k):v for k,v in quotas.items()})
 write_jsonl(a.out/'double_error_200.jsonl',rows)
 write_json(a.out/'double_manifest.json',{'seed':seed,'seed_source':'OS entropy via secrets','counts':dict(counts),'excluded_pair':['goal_not_reached','goal_not_reached'],'allocation':'22 per pair plus two randomly selected pairs each receive one extra','first_error_counts':dict(collections.Counter(r['reason'] for r in rows)),'valid_controls':0,'unique_layouts':200,'cross_single_overlap':0,'old_layout_overlap':0,'attempts':attempts,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((a.out/'double_error_200.jsonl').read_bytes()).hexdigest(),'semantics':'Original project execute stops at first error, unchanged. audit_events uses rejected illegal moves as no-op and continues, then evaluates EOS once. Exactly two distinct event steps; physical errors at different base indices. Reference route reaches goal. No hidden second terminal label.'})
 manifest['double_status']='Complete: 200 records, nine supported pairs; see double_manifest.json'
 write_json(a.out/'manifest.json',manifest)
 print(json.dumps({'counts':dict(counts),'first_error_counts':dict(collections.Counter(r['reason'] for r in rows))},ensure_ascii=False))

if __name__=='__main__':main()
