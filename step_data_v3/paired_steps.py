"""Build individual last-move examples, reusing G states and preserving five labels."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import random
from shutil import copyfile
from tempfile import TemporaryDirectory

from ppnl.core import ACTIONS, Grid, OfficialOracle, execute, shortest_path
from ppnl.data import read_jsonl
from ppnl.experiment_freeze import file_hash
from ppnl.policy_v3_data import distances_to_goal
from . import last_move, single
from .audit import check_g, require
from .build import SOURCE_SHA, V2_COMMIT, write_lines, write_object
from .model_io import read_split
from .schema import State, render
from .shortcut_audit import inspect_dataset


TRAIN_QUOTA = dict(valid=400, goal_not_reached=700, obstacle_collision=450, out_of_bounds=250, invalid_action_token=200)
DEV_QUOTA = dict(valid=20, goal_not_reached=35, obstacle_collision=22, out_of_bounds=13, invalid_action_token=10)
COMMON_WORDS = ("jump","wait","north","turn","spin")
TRAIN_WORDS = COMMON_WORDS + ("stay","walk","east","rigt","sidestep","upward","DOWN","leftwards",
                              "backward","dash","hop","idle","leap","slide","back")
DEV_WORDS = COMMON_WORDS + ("pivot","rght","climb","west","forwards")


@dataclass(frozen=True)
class Observation:
    state: State
    action: str
    donor: str
    parent: str
    step: int | None = None


def delta(p,a):
    dr,dc = ACTIONS[a]
    return p[0]+dr,p[1]+dc


def old_controls(old, states):
    groups, families = defaultdict(list), defaultdict(list)
    for r in old:
        src = r.get("source",{})
        if src.get("kind")=="mixed_step" and r["candidate"] in ACTIONS:
            groups[src["pair"]].append(Observation(states[r["state_id"]],r["candidate"],r["donor_id"],r["id"]))
    for group in groups.values():
        if len(group)!=2 or group[0].action!=group[1].action:
            continue
        x,y = (item.state for item in group)
        labels = {last_move.transition(o.state,o.action)[1] for o in group}
        if x.current==y.current and x.obstacles==y.obstacles and x.goal!=y.goal and labels=={"valid","goal_not_reached"}:
            families["change_goal"].append(group)
        elif x.current==y.current and x.goal==y.goal and set(x.obstacles)^set(y.obstacles)=={delta(x.current,group[0].action)}:
            families["change_wall"].append(group)
        elif x.obstacles==y.obstacles and x.goal==y.goal and labels=={"out_of_bounds","goal_not_reached"}:
            families["change_edge"].append(group)
    return families


class Builder:
    def __init__(self, split, old, states, raw):
        self.split, self.old, self.states, self.raw = split,old,states,raw
        self.units = 20 if split=="train" else 1
        self.quota = TRAIN_QUOTA if split=="train" else DEV_QUOTA
        self.rng = random.Random(42 if split=="train" else 43)
        self.gs = {r["state_id"]:dict(r) for r in old if r["mode"]=="g_next"}
        self.used = {sid:states[sid] for sid in self.gs}
        self.rows, self.seen, self.vstates = [], set(), set()
        self.counts, self.directions, self.group_counts = Counter(), Counter(), Counter()

    def add(self, observations, kind):
        keys = [(o.state.id,o.action) for o in observations]
        labels = [last_move.transition(o.state,o.action)[1] for o in observations]
        if len(keys)!=len(set(keys)) or not self.seen.isdisjoint(keys):
            return False
        if any(self.counts[k]+v>self.quota[k] for k,v in Counter(labels).items()):
            return False
        self.group_counts[kind] += 1
        group = f"{self.split}-{kind}-{self.group_counts[kind]}"
        for o,label,(sid,a) in zip(observations,labels,keys,strict=True):
            source = {"kind":kind,"group":group,"parent":o.parent}
            if o.step is not None:
                source["step"] = o.step
            self.rows.append(dict(id=f"{self.split}-last-{len(self.rows)+1:04}",mode="v_step",task="verification",
                             state_id=sid,candidate=a,input=last_move.step_input(o.state,a),target=label,
                             donor_id=o.donor,shared_g_id=self.gs[sid]["id"] if sid in self.gs else None,source=source))
            self.used[sid] = o.state
            self.vstates.add(sid)
            self.counts[label] += 1
            self.directions[label,a] += 1
        self.seen.update(keys)
        return True

    def controls(self):
        families = old_controls(self.old,self.states)
        for family,number in (("change_goal",3),("change_wall",3),("change_edge",2)):
            pool = families[family]
            self.rng.shuffle(pool)
            for group in pool:
                self.add(group,family)
                if self.group_counts[family]==number*self.units:
                    break
            require(self.group_counts[family]==number*self.units,family)

    def wall_swaps(self):
        """Same position, goal, action and number of walls. Swap the blocked destination."""
        layouts = {r["layout_id"]:r for r in sorted(self.raw,key=lambda x:x["id"])}
        shared_walls = defaultdict(list)
        for lid,r in layouts.items():
            walls = frozenset(map(tuple,r["grid"]["obstacles"]))
            for p in sorted(walls):
                shared_walls[walls-{p}].append((lid,p))
        g_goals = defaultdict(list)
        for sid in self.gs:
            s = self.states[sid]
            g_goals[s.record()["layout_id"],s.current].append(s.goal)
        possibilities = []
        for common,options in shared_walls.items():
            for (l,p),(m,q) in combinations(sorted(options),2):
                for a,(dr,dc) in ACTIONS.items():
                    cur = (p[0]-dr,p[1]-dc)
                    if cur in common or not (0<=cur[0]<6 and 0<=cur[1]<6):
                        continue
                    b = next((k for k in ACTIONS if delta(cur,k)==q),None)
                    if b is not None and a!=b:
                        possibilities.append((l,m,cur,p,q,a,b))
        self.rng.shuffle(possibilities)
        possibilities.sort(key=lambda z:not(g_goals[z[0],z[2]] or g_goals[z[1],z[2]]))
        for l,m,cur,p,q,a,b in possibilities:
            left,right = (Grid.from_dict(layouts[k]["grid"]) for k in (l,m))
            anchors = sorted(set(g_goals[l,cur]+g_goals[m,cur]))
            self.rng.shuffle(anchors)
            free = [(r,c) for r in range(left.rows) for c in range(left.cols)]
            self.rng.shuffle(free)
            for goal in dict.fromkeys(anchors+free):
                if goal==cur or goal in left.obstacles or goal in right.obstacles:
                    continue
                ss = [State.from_grid(g,cur,goal) for g in (left,right)]
                if any(cur not in distances_to_goal(s.grid()) for s in ss):
                    continue
                obs = [Observation(s,act,layouts[k]["id"],layouts[k]["id"]) for s,k in zip(ss,(l,m))
                       for act in (a,b)]
                if self.add(obs,"swap_walls"):
                    break
            if self.group_counts["swap_walls"]==2*self.units:
                return
        raise ValueError("Insufficient same-count wall swap groups")

    def four_directions(self):
        candidates = list(self.gs.items())
        self.rng.shuffle(candidates)
        for family,count in (("adjacent",3),("detour",2),("tie",2),("open",1)):
            n = 0
            for sid,g in candidates:
                if sid in self.vstates:
                    continue
                s = self.states[sid]
                mm = lambda pos:sum(abs(x-y) for x,y in zip(pos,s.goal))
                observations = [Observation(s,a,g["donor_id"],g["id"]) for a in ACTIONS]
                legal = {a for a in ACTIONS if last_move.transition(s,a)[0] is not None}
                applicable = {"adjacent":mm(s.current)==1,
                    "detour":all(mm(delta(s.current,a))>mm(s.current) for a in g["optimal_actions"]),
                    "tie":len(g["optimal_actions"])>1 and 1<len(legal)<4,
                    "open":len(legal)==4 and mm(s.current)>1}
                if applicable[family] and self.add(observations,"four_"+family):
                    n += 1
                    if n==count*self.units:
                        break
            require(n==count*self.units,(self.split,family,n))

    def cover_layouts(self):
        covered = {s.record()["layout_id"] for s in self.used.values()}
        for r in self.raw:
            if r["layout_id"] in covered:
                continue
            grid = Grid.from_dict(r["grid"])
            path = shortest_path(grid)
            for tick in execute(grid," ".join(path)).trace:
                obs = Observation(State.from_grid(grid,tick["from"]),tick["action"],r["id"],r["id"],tick["step"])
                if self.add([obs],"shortest_prefix"):
                    covered.add(r["layout_id"])
                    break
        require(covered=={r["layout_id"] for r in self.raw},"layout coverage")

    def fill_moves(self):
        sources = {"G_move":defaultdict(list),"prefix":defaultdict(list),"prior_move":defaultdict(list)}
        for sid,g in self.gs.items():
            for a in ACTIONS:
                s = self.states[sid]
                label = last_move.transition(s,a)[1]
                sources["G_move"][label,a].append(Observation(s,a,g["donor_id"],g["id"]))
        for r in self.raw:
            if "candidate" not in r:
                continue
            grid = Grid.from_dict(r["grid"])
            for tick in execute(grid,r["candidate"]).trace:
                a = tick["action"]
                if a in ACTIONS:
                    s = State.from_grid(grid,tick["from"])
                    label = last_move.transition(s,a)[1]
                    sources["prefix"][label,a].append(Observation(s,a,r["id"],r["id"],tick["step"]))
        for r in self.old:
            a = r.get("candidate")
            if a in ACTIONS:
                s = self.states[r["state_id"]]
                label = last_move.transition(s,a)[1]
                sources["prior_move"][label,a].append(Observation(s,a,r["donor_id"],r["id"]))
        for buckets in sources.values():
            for pool in buckets.values():
                self.rng.shuffle(pool)
        for label in ("valid","goal_not_reached","obstacle_collision","out_of_bounds"):
            while self.counts[label] < self.quota[label]:
                actions = list(ACTIONS)
                self.rng.shuffle(actions)
                actions.sort(key=lambda a:self.directions[label,a])
                source_order = ["G_move","prefix","prior_move"] if self.counts[label]%3 else ["prefix","G_move","prior_move"]
                added = False
                for a in actions:
                    for kind in source_order:
                        pool = sources[kind][label,a]
                        while pool and not added:
                            added = self.add([pool.pop()],kind)
                        if added:
                            break
                    if added:
                        break
                require(added,("no available move",self.split,label,self.counts))

    def lexical_cases(self):
        words = TRAIN_WORDS if self.split=="train" else DEV_WORDS
        gs = list(self.gs.values())
        self.rng.shuffle(gs)
        # Half share a state that already has a physical candidate. Half sample additional G states.
        old = [g for g in gs if g["state_id"] in self.vstates]
        new = [g for g in gs if g["state_id"] not in self.vstates]
        for i in range(self.quota["invalid_action_token"]):
            g = (old if i%2==0 else new).pop()
            a = words[i%len(words)]
            require(self.add([Observation(self.states[g["state_id"]],a,g["donor_id"],g["id"])],"lexical"),g["id"])

    def run(self):
        self.controls()
        self.wall_swaps()
        self.four_directions()
        # Keep actual departures from an already reached goal, with the last-move semantics.
        for r in self.old:
            if r.get("source",{}).get("kind")=="depart_goal":
                self.add([Observation(self.states[r["state_id"]],r["candidate"],r["donor_id"],r["id"])],"depart_goal")
        self.cover_layouts()
        self.fill_moves()
        self.lexical_cases()
        by_state = defaultdict(list)
        for r in self.rows:
            by_state[r["state_id"]].append(r["id"])
        for sid,g in self.gs.items():
            g["anchor_v_id"] = next(iter(by_state[sid]),None)
        require(self.counts==self.quota,(self.split,self.counts))
        rows = list(self.gs.values())+self.rows
        self.rng.shuffle(rows)
        return rows,self.used


def verify_group(items, states):
    kind = items[0]["source"]["kind"]
    ss = {r["state_id"]:states[r["state_id"]] for r in items}
    actions, targets = {r["candidate"] for r in items},{r["target"] for r in items}
    key = items[0]["source"]["group"]
    if kind.startswith("four_"):
        require(len(items)==4 and len(ss)==1 and actions==set(ACTIONS) and
                all(r["shared_g_id"]==r["source"]["parent"] for r in items),key)
    elif kind=="same_state_arrival":
        require(len(items)==2 and len(ss)==1 and len(actions)==2 and actions.issubset(ACTIONS)
                and targets=={"valid","goal_not_reached"},key)
    elif kind=="swap_walls":
        require(len(items)==4 and len(ss)==2 and len(actions)==2,key)
        a,b = ss.values()
        dests = {delta(a.current,move) for move in actions}
        require((a.current,a.goal,len(a.obstacles))==(b.current,b.goal,len(b.obstacles)) and
                set(a.obstacles)^set(b.obstacles)==dests,key)
        for field in ("state_id","candidate"):
            for value in {x[field] for x in items}:
                require({x["target"] for x in items if x[field]==value}=={"goal_not_reached","obstacle_collision"},key)
    elif kind in ("change_goal","change_wall","change_edge"):
        require(len(items)==2 and len(ss)==2 and len(actions)==1,key)
        a,b = ss.values()
        move = next(iter(actions))
        if kind=="change_goal":
            require(a.current==b.current and a.obstacles==b.obstacles and a.goal!=b.goal and
                    targets=={"valid","goal_not_reached"},key)
        elif kind=="change_wall":
            require(a.current==b.current and a.goal==b.goal and set(a.obstacles)^set(b.obstacles)=={delta(a.current,move)} and
                    targets=={"goal_not_reached","obstacle_collision"},key)
        else:
            require(a.goal==b.goal and a.obstacles==b.obstacles and
                    (delta(a.current,move)==b.current or delta(b.current,move)==a.current) and
                    targets=={"goal_not_reached","out_of_bounds"},key)
    else:
        require(len(items)==1,key)


def audit_split(rows, states, old, raw, oracle, split):
    prior = {r["id"]:r for r in old}
    donors = {r["id"]:r for r in raw}
    gs = {r["state_id"]:r for r in rows if r["mode"]=="g_next"}
    vs = {r["id"]:r for r in rows if r["mode"]=="v_step"}
    quota = TRAIN_QUOTA if split=="train" else DEV_QUOTA
    require(len(gs)==len(vs)==sum(quota.values()) and len(rows)==2*len(vs),split)
    require(len({r["input"] for r in rows})==len({r["id"] for r in rows})==len(rows),split)
    groups = defaultdict(list)
    for r in rows:
        s = states[r["state_id"]]
        require(donors[r["donor_id"]]["layout_id"]==s.record()["layout_id"],r["id"])
        if r["mode"]=="g_next":
            require(all(r[k]==prior[r["id"]][k] for k in ("input","target","state_id","optimal_actions","distance")) and
                    r["input"]==render("g_next",s),r["id"])
            check_g(r,s,oracle)
            if r["anchor_v_id"]:
                require(vs[r["anchor_v_id"]]["state_id"]==s.id,r["id"])
            continue
        a = r["candidate"]
        last_move.check_token(a)
        grid,prefix = s.witness()
        truth = oracle.check(grid," ".join(prefix+[a]))
        require(r["target"]==(truth.reason or "valid") and r["input"]==last_move.step_input(s,a) and
                r["shared_g_id"]==(gs[s.id]["id"] if s.id in gs else None),r["id"])
        origin, parent = r["source"],r["source"]["parent"]
        kind = origin["kind"]
        groups[origin["group"]].append(r)
        if kind in ("prefix","shortest_prefix"):
            original = donors[parent]
            grid = Grid.from_dict(original["grid"])
            tokens = shortest_path(grid) if kind=="shortest_prefix" else original["candidate"].split()
            step = origin["step"]
            trace = oracle.check(grid," ".join(tokens[:step-1]))
            pos = grid.start if step==1 else trace.trace[step-2]["to"]
            require(trace.feasible and tokens[step-1]==a and s==State.from_grid(grid,pos),r["id"])
        elif kind in ("G_move","lexical") or kind.startswith("four_"):
            require(parent==r["shared_g_id"] and prior[parent]["state_id"]==s.id,r["id"])
        elif kind=="swap_walls":
            require(parent==r["donor_id"],r["id"])
        else:
            require(prior[parent]["state_id"]==s.id and prior[parent]["candidate"]==a,r["id"])
    for items in groups.values():
        verify_group(items,states)
    counts = Counter(r["target"] for r in vs.values())
    layouts = {s.record()["layout_id"] for s in states.values()}
    require(counts==quota and layouts=={r["layout_id"] for r in raw},split)
    swap = [r for r in vs.values() if r["source"]["kind"]=="swap_walls"]
    return {"G":len(gs),"V":len(vs),"labels":counts,"layouts":len(layouts),"empty_or_multitoken_candidates":0,
            "groups":Counter(v[0]["source"]["kind"] for v in groups.values()),
            "matched_wall_swap_rows":len(swap),"matched_wall_swap_wall_blind_correct":
            sum(r["target"]=="goal_not_reached" for r in swap)}


def build(source, previous, executor, output):
    if output.exists():
        raise FileExistsError(output)
    oracle = OfficialOracle(executor)
    require(all(file_hash(source/name)==h for name,h in SOURCE_SHA.items()),"frozen v2 files")
    require(oracle.sha256==json.loads((source/"manifest.json").read_bytes())["executor_sha256"],"executor")
    parent = json.loads((previous/"manifest.json").read_bytes())
    require(file_hash(previous/"states.jsonl")==parent["sha256"]["states.jsonl"],"previous state registry")
    registry = {r["id"]:State.from_record(r) for r in read_jsonl(previous/"states.jsonl")}
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/"dataset"
        stage.mkdir()
        used,layouts,report = {},{},{}
        for split in ("train","dev"):
            old = read_split(previous,split,version=single.VERSION,fingerprint=single.template_hash())
            raw = read_jsonl(source/f"{split}.jsonl")
            rows,states = Builder(split,old,registry,raw).run()
            write_lines(stage/f"{split}.jsonl",rows)
            report[split] = audit_split(read_jsonl(stage/f"{split}.jsonl"),states,old,raw,oracle,split)
            used.update(states)
            layouts[split] = {s.record()["layout_id"] for s in states.values()}
        write_lines(stage/"states.jsonl",(s.record() for sid,s in sorted(used.items())))
        for split in ("test_id","test_ood"):
            name = split+".jsonl"
            copyfile(source/name,stage/name)
            cases = read_jsonl(stage/name)
            grids = [Grid.from_dict(r["grid"]) for r in cases]
            candidates = [r["candidate"] for r in cases]
            predicted = last_move.scan(grids,candidates,lambda items:[last_move.transition(s,a)[1] for s,a in items])
            require(len(cases)==200 and file_hash(stage/name)==SOURCE_SHA[name] and predicted==[r["label"] for r in cases]
                    ==[oracle.check(g,c).label() for g,c in zip(grids,candidates,strict=True)],split)
            layouts[split] = {g.layout_id for g in grids}
            report[split] = {"frozen_cases":len(cases),"reference_step_verdicts_aggregated_correctly":len(cases),
                             "empty_cases_with_no_model_call":sum(not c.split() for c in candidates)}
        require(layouts["train"].isdisjoint(layouts["dev"]|layouts["test_id"]|layouts["test_ood"])
                and layouts["dev"].isdisjoint(layouts["test_id"]|layouts["test_ood"]),"environment overlap")
        write_object(stage/"audit.json",report)
        write_object(stage/"shortcut_audit.json",{"previous":inspect_dataset(previous,False),"last_move":inspect_dataset(stage,True)})
        write_object(stage/"manifest.json",{"schema":last_move.VERSION,"template_sha256":last_move.template_hash(),
                     "base_commit":V2_COMMIT,"source_sha256":SOURCE_SHA,"previous_manifest_sha256":file_hash(previous/"manifest.json"),
                     "executor_sha256":oracle.sha256,"seed":{"train":42,"dev":43},
                     "sha256":{p.name:file_hash(p) for p in sorted(stage.iterdir())}})
        stage.rename(output)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("data/derived"))
    parser.add_argument("--previous",type=Path,default=Path("data/step_semantic_single"))
    parser.add_argument("--executor",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source,args.previous,args.executor,args.out),indent=2))
