"""Independent audit: no generator or rejected-move scanner imported."""
import collections,hashlib,json,sys,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'tmp/nlp-stepverify-1119'))
from ppnl.core import Grid,execute,OfficialOracle,ACTIONS
out=root/'output/random-error-datasets-20261123'
oracle=OfficialOracle(root/'output/NLP-Project-review/executor-point-sg.py')
known={json.loads(x)['layout_id'] for x in (root/'tmp/latest_paired_states.jsonl').read_text(encoding='utf-8-sig').splitlines()}
layouts=set();ids=set();report={}
for name,wanted in [('single_error_200',1),('double_error_200',2)]:
 rows=[json.loads(x) for x in (out/f'{name}.jsonl').read_text().splitlines()]
 offsets=[]
 for r in rows:
  g=Grid.from_dict(r['grid']);assert g.layout_id==r['layout_id']
  assert g.layout_id not in layouts and g.layout_id not in known and r['id'] not in ids
  layouts.add(g.layout_id);ids.add(r['id'])
  v=oracle.check(g,r['candidate']);assert not v.valid and v.label()==r['label']
  assert (v.reason,v.step)==(r['reason'],r['step'])
  assert execute(g,r['reference_path']).valid
  pos=g.start;events=[]
  for step,token in enumerate(r['candidate'].split(),1):
   if token not in ACTIONS:reason='invalid_action_token'
   else:
    dr,dc=ACTIONS[token];q=(pos[0]+dr,pos[1]+dc)
    reason='out_of_bounds' if not g.inside(q) else 'obstacle_collision' if q in g.obstacles else None
   if reason:events.append((step,reason))
   else:pos=q
  if pos!=g.goal:events.append((len(r['candidate'].split())+1,'goal_not_reached'))
  assert len(events)==wanted and len({e[0] for e in events})==wanted
  assert events==[(e['step'],e['reason']) for e in r['audit_events']]
  assert events[0]==(v.step,v.reason)
  if wanted==2:assert collections.Counter(r['error_pair'])==collections.Counter(e[1] for e in events)
  offsets.append(v.step)
 report[name]={'rows':len(rows),'independent_event_audits_passed':len(rows),'strict_project_and_official_adapter_passed':len(rows),'first_error_counts':dict(collections.Counter(r['reason'] for r in rows)),'candidate_lengths':dict(sorted(collections.Counter(len(r['candidate'].split()) for r in rows).items())),'first_error_steps':dict(sorted(collections.Counter(offsets).items()))}
assert len(layouts)==400
report.update(unique_layouts=400,latest_paired_layout_overlap=0,latest_paired_states_sha256=hashlib.sha256((root/'tmp/latest_paired_states.jsonl').read_bytes()).hexdigest(),unknown_token_note='Strict project extension; upstream executor ignores unknown tokens, so unknown-token labels come from project semantics.')
(out/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for n in ('build_random_error_sets.py','build_random_double_errors.py','audit_random_error_sets.py'):
 (out/n).write_bytes((root/'tmp'/n).read_bytes())
with zipfile.ZipFile(out.parent/'random-error-datasets-20261123.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in out.iterdir():
  if f.is_file():z.write(f,arcname=f.name)
print(json.dumps({'rows':400,'independent_audit':'passed','zip':str(out.parent/'random-error-datasets-20261123.zip')}))
