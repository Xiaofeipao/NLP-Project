"""Select balanced supervised steps and export exhaustive G-state diagnostics."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import random
from shutil import copyfile
from tempfile import TemporaryDirectory

from ppnl.core import ACTIONS, Grid, OfficialOracle, shortest_path
from ppnl.data import read_jsonl
from ppnl.experiment_freeze import file_hash
from ppnl.policy_v3_data import distances_to_goal
from . import g_probes, last_move
from .audit import check_g, require
from .build import SOURCE_SHA, V2_COMMIT, write_lines, write_object
from .model_io import read_split
from .paired_steps import TRAIN_WORDS, DEV_WORDS, verify_group
from .schema import State, render
from .sampling import longer_path, random_path, positions
from .shortcut_audit import geometry_blind, inspect_dataset
from .single import LABELS


def slot(row):
    return row["target"], row["candidate"]


@lru_cache(maxsize=12000)
def shortest_final_moves(grid):
    """All directions that can finish a shortest route from this original starting point."""
    d = distances_to_goal(replace(grid,start=grid.goal,goal=grid.start))
    last = d[grid.goal]-1
    return {a:(grid.goal[0]-dr,grid.goal[1]-dc) for a,(dr,dc) in ACTIONS.items()
            if d.get((grid.goal[0]-dr,grid.goal[1]-dc))==last}


def dual_axis_arrival(grid,state,action):
    if grid.goal!=state.goal:
        return False
    options = shortest_final_moves(grid)
    return (options.get(action)==state.current and len({a in ("left","right") for a in options})==2)


def arrival_axis_report(rows, states, donors):
    directions, categories, double, seen = Counter(), Counter(), Counter(), defaultdict(set)
    for row in rows:
        if row["mode"]!="v_step" or row["target"]!="valid":
            continue
        action = row["candidate"]
        state = states[row["state_id"]]
        grid = Grid.from_dict(donors[row["donor_id"]]["grid"])
        directions[action] += 1
        axis = "horizontal" if action in ("left","right") else "vertical"
        options = shortest_final_moves(grid)
        if grid.goal!=state.goal:
            categories["different_goal"] += 1
        elif options.get(action)!=state.current:
            categories["outside_donor_shortest_approaches"] += 1
        elif dual_axis_arrival(grid,state,action):
            categories["both_axes_"+axis] += 1
            double[action] += 1
            seen[grid].add(axis)
        else:
            categories["one_axis_"+axis] += 1
    return {"directions":directions,"eligibility":categories,"dual_axis_directions":double,
            "dual_axis_original_grids_observed_on_both_axes":sum(len(axes)==2 for axes in seen.values())}


class Selection:
    def __init__(self, split, prior, states, raw):
        self.split, self.states = split, dict(states)
        self.raw = {r["id"]:r for r in raw}
        self.gs = {r["state_id"]:dict(r) for r in prior if r["mode"]=="g_next"}
        self.g_id = {sid:g["id"] for sid,g in self.gs.items()}
        self.rng = random.Random(42 if split=="train" else 43)
        self.units = 20 if split=="train" else 1
        words = TRAIN_WORDS if split=="train" else DEV_WORDS
        per_class = 20 * self.units
        self.quota = {(label,a):per_class//4 for label in LABELS if label!="invalid_action_token" for a in ACTIONS}
        self.quota.update({("invalid_action_token",word):per_class//len(words) for word in words})
        self.pool, self.old_groups = {}, defaultdict(list)
        for row in prior:
            if row["mode"]=="v_step":
                key = self.pool_row(row["state_id"],row["candidate"],row["donor_id"],"previous_V",row["id"])
                self.old_groups[row["source"]["kind"],row["source"]["group"]].append(key)
        for probe in g_probes.g_examples(prior,states):
            self.pool_row(probe["state_id"],probe["candidate"],probe["donor_id"],"G_direction",probe["g_id"])
        # Use both possible approach axes in the original donor geometry, including
        # alternatives missed by the archived BFS tie-breaking order.
        for r in raw:
            grid = Grid.from_dict(r["grid"])
            for a,pos in shortest_final_moves(grid).items():
                state = State.from_grid(grid,pos)
                self.states[state.id] = state
                self.pool_row(state.id,a,r["id"],"source_shortest_arrival",r["id"])
        shuffled = list(self.gs.values())
        self.rng.shuffle(shuffled)
        for i,g in enumerate(shuffled):
            self.pool_row(g["state_id"],words[i%len(words)],g["donor_id"],"G_word",g["id"])
        self.kept, self.groups = {}, []
        self.counts, self.usage = Counter(), Counter()

    def pool_row(self, sid, action, donor, origin, parent):
        key = sid,action
        if key not in self.pool:
            self.pool[key] = dict(g_probes.example(self.states[sid],action),mode="v_step",task="verification",
                                  donor_id=donor,shared_g_id=self.g_id.get(sid),origin=origin,parent_id=parent)
        return key

    def take(self, keys, kind=None):
        require(len(keys)==len(set(keys)),"duplicate group member")
        fresh = [key for key in keys if key not in self.kept]
        cost = Counter(slot(self.pool[key]) for key in fresh)
        if any(self.counts[k]+n>self.quota[k] for k,n in cost.items()):
            return False
        for key in fresh:
            row = dict(self.pool[key],id=f"{self.split}-balanced-V-{len(self.kept)+1:04}")
            self.kept[key] = row
            self.usage[key[0]] += 1
        self.counts.update(cost)
        if kind:
            self.groups.append({"id":f"{self.split}-control-{len(self.groups)+1:04}","kind":kind,
                                "rows":[self.kept[k]["id"] for k in keys]})
        return True

    def cover_layouts(self):
        layout = {sid:self.states[sid].record()["layout_id"] for sid in {sid for sid,_ in self.pool}}
        covered = {layout[sid] for sid in self.gs.keys()|self.usage.keys()}
        missing = {r["layout_id"] for r in self.raw.values()}-covered
        for lid in sorted(missing):
            choices = [k for k,r in self.pool.items() if layout[k[0]]==lid]
            self.rng.shuffle(choices)
            choices.sort(key=lambda k:self.counts[slot(self.pool[k])]/self.quota[slot(self.pool[k])])
            if not any(self.take([k]) for k in choices):
                parent = next(self.pool[k] for k in choices if self.pool[k]["origin"]=="previous_V")
                other = [self.pool_row(parent["state_id"],a,parent["donor_id"],"layout_move",parent["parent_id"])
                         for a in ACTIONS]
                require(any(self.take([k]) for k in other),("uncovered source layout",lid))

    def original_controls(self):
        for kind, wanted in (("swap_walls",2),("change_goal",1),("change_wall",1),("change_edge",1)):
            groups = [keys for (k,_),keys in self.old_groups.items() if k==kind]
            self.rng.shuffle(groups)
            kept = 0
            for keys in groups:
                if self.take(keys,kind):
                    kept += 1
                if kept==wanted*self.units:
                    break
            require(kept>0,(self.split,kind))
        # Also keep available examples of moving again after a legal intermediate arrival.
        departures = [v[0] for (k,_),v in self.old_groups.items() if k=="depart_goal"]
        self.rng.shuffle(departures)
        for key in departures[:max(1,self.units//2)]:
            self.take([key])

    def full_direction_groups(self):
        options = []
        for sid,g in self.gs.items():
            s = self.states[sid]
            distance = lambda pos:sum(abs(x-y) for x,y in zip(pos,s.goal))
            keys = [(sid,a) for a in ACTIONS]
            outcomes = [self.pool[k]["target"] for k in keys]
            if "valid" in outcomes:
                bucket = "adjacent"
            elif all(distance(last_move.transition(s,a)[0])>distance(s.current) for a in g["optimal_actions"]):
                bucket = "detour"
            elif set(outcomes)=={"goal_not_reached"}:
                bucket = "open"
            elif len(g["optimal_actions"])>1:
                bucket = "tie"
            else:
                bucket = "ordinary"
            legal = {a for (_,a),label in zip(keys,outcomes) if label in ("valid","goal_not_reached")}
            added_unfinished = sum(self.pool[k]["target"]=="goal_not_reached" and k not in self.kept for k in keys)
            # At equal cost, favor observing both an optimal and a legal nonoptimal move.
            contrast = bool(legal-set(g["optimal_actions"]))
            options.append((added_unfinished-int(contrast),self.rng.random(),bucket,sid,keys))
        options.sort()
        selected = set()
        for name in ("open","adjacent","detour","tie","ordinary"):
            minimum = 1 if self.units==1 else (4 if name=="open" else 12)
            n = 0
            for _,_,bucket,sid,keys in options:
                if bucket==name and self.take(keys,"four_"+name):
                    selected.add(sid)
                    n += 1
                    if n==minimum:
                        break
        for _,_,bucket,sid,keys in options:
            if len(selected)>=8*self.units:
                break
            if sid not in selected and self.take(keys,"four_"+bucket):
                selected.add(sid)

    def reinforce_action_pairs(self):
        # A legal but nonoptimal move matters to V even when G would choose a different direction.
        options = []
        for sid,g in self.gs.items():
            best = set(g["optimal_actions"])
            other = [a for a in ACTIONS if a not in best and self.pool[sid,a]["target"] in ("valid","goal_not_reached")]
            if not other or (any((sid,a) in self.kept for a in best) and any((sid,a) in self.kept for a in other)):
                continue
            candidates = [[(sid,a),(sid,b)] for a in ACTIONS if a in best for b in other]
            self.rng.shuffle(candidates)
            candidates.sort(key=lambda ks:sum(k not in self.kept for k in ks))
            options.append((sum(k not in self.kept for k in candidates[0]),self.rng.random(),candidates))
        for _,_,candidates in sorted(options):
            for keys in candidates:
                if self.take(keys):
                    break

    def balance_arrivals(self):
        available, chosen = defaultdict(list), Counter()
        grid_axes, paired = defaultdict(lambda:defaultdict(list)), defaultdict(set)
        for key,r in self.pool.items():
            if r["target"]!="valid":
                continue
            a = key[1]
            grid = Grid.from_dict(self.raw[r["donor_id"]]["grid"])
            both = dual_axis_arrival(grid,self.states[key[0]],a)
            axis = "H" if a in ("left","right") else "V"
            if key in self.kept:
                chosen[a,both] += 1
                if both:
                    paired[grid].add(axis)
            else:
                available[a,both].append(key)
                if both:
                    grid_axes[grid][axis].append(key)
        caps = {a:self.quota["valid",a] for a in ACTIONS}
        bounds = {}
        for a,cap in caps.items():
            low = max(chosen[a,True], cap-chosen[a,False]-len(available[a,False]))
            high = min(cap-chosen[a,False], chosen[a,True]+len(available[a,True]))
            require(low<=high,(self.split,"arrival capacity",a))
            bounds[a] = low,high
        axes = (("left","right"),("up","down"))
        low = max(sum(bounds[a][0] for a in axis) for axis in axes)
        high = min(sum(bounds[a][1] for a in axis) for axis in axes)
        require(low<=high,(self.split,"arrival-axis quotas",bounds))
        axis_total = min(high,max(low,caps["up"]))
        targets = {}
        for a,b in axes:
            lo = max(bounds[a][0],axis_total-bounds[b][1])
            hi = min(bounds[a][1],axis_total-bounds[b][0])
            first = min(hi,max(lo,axis_total//2))
            for direction,n in ((a,first),(b,axis_total-first)):
                targets[direction,True] = n
                targets[direction,False] = caps[direction]-n
        # Prefer concrete matched examples starting from the same original grid and endpoints.
        grids = list(grid_axes)
        self.rng.shuffle(grids)
        grids.sort(key=lambda grid:len(paired[grid])!=1)
        for grid in grids:
            missing = [axis for axis in ("H","V") if axis not in paired[grid]]
            picks = []
            for axis in missing:
                choices = [key for key in grid_axes[grid][axis] if chosen[key[1],True]<targets[key[1],True]]
                if choices:
                    picks.append(self.rng.choice(choices))
            if picks and len(picks)==len(missing):
                require(self.take(picks),"matched shortest arrivals")
                chosen.update((key[1],True) for key in picks)
        for spec,target in targets.items():
            bag = [key for key in available[spec] if key not in self.kept]
            self.rng.shuffle(bag)
            bag.sort(key=lambda k:self.pool[k]["origin"]!="previous_V")
            for key in bag[:target-chosen[spec]]:
                require(self.take([key]),key)

    def fill(self):
        bags = defaultdict(list)
        order = list(self.pool)
        self.rng.shuffle(order)
        for key in order:
            if key not in self.kept:
                bags[slot(self.pool[key])].append(key)
        for k in self.quota:
            bag = bags[k]
            while self.counts[k]<self.quota[k]:
                require(bool(bag),("insufficient unique inputs",self.split,k))
                # Sample a previously unrepresented G state first. Within ties retain parent examples.
                i = min(range(len(bag)),key=lambda i:(bag[i][0] not in self.gs,
                        self.usage[bag[i][0]]>0,self.pool[bag[i]]["origin"]!="previous_V"))
                require(self.take([bag.pop(i)]),k)

    def run(self):
        for sid in self.gs:
            for a in ACTIONS:
                if self.pool[sid,a]["target"]=="valid":
                    require(self.take([(sid,a)]),"every available G arrival")
        self.cover_layouts()
        self.original_controls()
        self.full_direction_groups()
        self.reinforce_action_pairs()
        self.balance_arrivals()
        self.fill()
        require(self.counts==self.quota,self.split)
        by_state = defaultdict(list)
        for r in self.kept.values():
            by_state[r["state_id"]].append(r["id"])
        for sid,g in self.gs.items():
            g["anchor_v_id"] = next(iter(by_state[sid]),None)
        rows = list(self.gs.values())+list(self.kept.values())
        self.rng.shuffle(rows)
        return rows,self.groups,{sid:self.states[sid] for sid in self.gs.keys()|self.usage.keys()}


def audit_split(rows, groups, states, prior, raw, oracle, split, *, parent_verification=None):
    prior_ids = {r["id"]:r for r in prior}
    donors = {r["id"]:r for r in raw}
    ids = {r["id"]:r for r in rows}
    gs = {r["state_id"]:r for r in rows if r["mode"]=="g_next"}
    vs = [r for r in rows if r["mode"]=="v_step"]
    per_class = 400 if split=="train" else 20
    require(len(gs)==len(vs)==5*per_class and len(rows)==2*len(vs),split)
    require(len(rows)==len(ids)==len({r["input"] for r in rows}),f"{split} duplicates/conflicts")
    require(set(gs)=={r["state_id"] for r in prior if r["mode"]=="g_next"},"all previous G states")
    g_sets, g_rep, g_members, local_counts = Counter(), Counter(), Counter(), Counter()
    physical, word_counts, coverage, origin = defaultdict(Counter), Counter(), Counter(), Counter()
    rng = random.Random(4201)
    for r in rows:
        s = states[r["state_id"]]
        require(donors[r["donor_id"]]["layout_id"]==s.record()["layout_id"],r["id"])
        require(not any(c in r["input"] for c in ("\r","\t")) and
                all(line==line.rstrip() for line in r["input"].split("\n")),r["id"])
        require(r["id"] not in r["input"] and r["state_id"] not in r["input"] and r["donor_id"] not in r["input"],r["id"])
        if r["mode"]=="g_next":
            require(r["task"]=="generation" and all(r[k]==prior_ids[r["id"]][k] for k in
                    ("input","target","optimal_actions","state_id","distance")) and r["input"]==render("g_next",s),r["id"])
            if r["anchor_v_id"]:
                anchor = ids[r["anchor_v_id"]]
                require(anchor["mode"]=="v_step" and anchor["state_id"]==s.id,r["id"])
            forced = check_g(r,s,oracle)
            coverage["G_forced_to_increase_Manhattan"] += forced
            g_sets[len(r["optimal_actions"])] += 1
            g_rep[r["target"]] += 1
            g_members.update(r["optimal_actions"])
            grid = s.grid()
            route = random_path(grid,rng)
            points = positions(grid,route)
            require(oracle.check(grid," ".join(route)).valid and len(route)==r["distance"]
                    and len(set(points))==len(points),r["id"])
            coverage["G_reference_rollouts_shortest_and_cycle_free"] += 1
            continue
        a = r["candidate"]
        require(r["mode"]=="v_step" and r["task"]=="verification" and
                r["shared_g_id"]==(gs[s.id]["id"] if s.id in gs else None),r["id"])
        require(r["input"]==g_probes.step_input(s,a) and r["target"]==g_probes.reference(s,a,oracle),r["id"])
        require(r["input"].splitlines()[5:7]==["Allowed moves: "+" ".join(ACTIONS),"Last move: "+json.dumps(a)],r["id"])
        require(r["input"].count("Last move:")==1 and r["input"].count("Allowed moves:")==1,r["id"])
        if r["origin"]=="source_shortest_arrival":
            grid = Grid.from_dict(donors[r["donor_id"]]["grid"])
            require(r["parent_id"]==r["donor_id"] and s==State.from_grid(grid,s.current)
                    and r["target"]=="valid" and shortest_final_moves(grid).get(a)==s.current,r["id"])
            prefix = [] if s.current==grid.start else shortest_path(replace(grid,goal=s.current))
            require(prefix is not None and len(prefix)+1==len(shortest_path(grid))
                    and oracle.check(grid," ".join(prefix+[a])).valid,r["id"])
        elif r["origin"]=="V_direction":
            parent = parent_verification[r["parent_id"]]
            require(parent["mode"]=="v_step" and parent["state_id"]==s.id and
                    parent["donor_id"]==r["donor_id"] and a in ACTIONS and
                    r["shared_g_id"] is None,r["id"])
        else:
            parent = prior_ids[r["parent_id"]]
            require(parent["state_id"]==s.id and r["donor_id"]==parent["donor_id"],r["id"])
            if r["origin"]=="previous_V":
                require(r["target"]==parent["target"] and a==parent["candidate"],r["id"])
            elif r["origin"]=="layout_move":
                require(parent["mode"]=="v_step" and a in ACTIONS,r["id"])
            else:
                require(parent["mode"]=="g_next" and ((r["origin"]=="G_direction" and a in ACTIONS)
                        or (r["origin"]=="G_word" and a not in ACTIONS)),r["id"])
        origin[r["origin"]] += 1
        local_counts[r["target"]] += 1
        if a in ACTIONS:
            physical[r["target"]][a] += 1
        else:
            word_counts[a] += 1
        coverage["V_real_nonempty_single_moves"] += 1
        coverage["V_moves_from_goal"] += s.current==s.goal
        coverage["V_shared_G"] += s.id in gs
    words = TRAIN_WORDS if split=="train" else DEV_WORDS
    require(local_counts=={c:per_class for c in LABELS},split)
    require(all(physical[c]=={a:per_class//4 for a in ACTIONS} for c in LABELS if c!="invalid_action_token"),split)
    require(word_counts=={w:per_class//len(words) for w in words},split)
    require({s.record()["layout_id"] for s in states.values()}=={r["layout_id"] for r in raw},split)
    group_scores = defaultdict(Counter)
    for g in groups:
        items = [dict(ids[i],source={"kind":g["kind"],"group":g["id"],"parent":ids[i]["shared_g_id"]})
                 for i in g["rows"]]
        verify_group(items,states)
        score = group_scores[g["kind"]]
        score["total"] += len(items)
        score["correct_without_walls"] += sum(geometry_blind(states[r["state_id"]],r["candidate"],True)
                                              ==r["target"] for r in items)
    arrival = arrival_axis_report(rows,states,donors)
    counts = arrival["eligibility"]
    require(counts["both_axes_horizontal"]==counts["both_axes_vertical"]>0,split)
    return {"generation":len(gs),"verification":len(vs),"unique_inputs":len(rows),"labels":local_counts,
            "per_direction":dict(physical),"invalid_words":word_counts,"coverage":coverage,"V_origins":origin,
            "shortest_arrival_axes":arrival,
            "paired_controls_without_walls":dict(group_scores),
            "G_optimal_set_sizes":g_sets,"G_representative_actions":g_rep,"G_optimal_action_memberships":g_members,
            "control_groups":Counter(g["kind"] for g in groups),"layouts":len({r["layout_id"] for r in raw}),
            "duplicate_inputs":0,"conflicting_identical_inputs":0,"wrong_oracle_targets":0,"malformed_prompts":0}


def check_route_shapes(rows, states, oracle):
    """Reference-only regression cases. No extra rollout appears in a training loader."""
    rng = random.Random(207)
    gs = [r for r in rows if r["mode"]=="g_next" and r["distance"]>=3]
    rng.shuffle(gs)
    cases, original_lengths = [], []
    for kind in ("backtrack","square","simple_detour"):
        n = 0
        for g in gs:
            grid = states[g["state_id"]].grid()
            base = random_path(grid,rng)
            plan = longer_path(grid,base,kind,rng)
            if not plan:
                continue
            text = " ".join(plan)
            require(len(plan)>len(base) and oracle.check(grid,text).valid,kind)
            cases.append((grid,text,kind))
            original_lengths.append(len(base))
            n += 1
            if n==20:
                break
        require(n==20,kind)
    preds = last_move.scan([g for g,_,_ in cases],[p for _,p,_ in cases],
                           lambda obs:[last_move.transition(s,a)[1] for s,a in obs])
    require(preds==["valid"]*len(cases),"valid nonoptimal paths")
    return {"reference_valid_by_shape":Counter(k for _,_,k in cases),"shortest_route_length_range":
            [min(original_lengths),max(original_lengths)]}


def token_check(root, tokenizer_path, states):
    from ppnl.policy import state_prompt
    from state_baselines.step_verification import step_prompt as v2_step_prompt
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    unknown = tokenizer.token_to_id("<unk>")
    stats = {}
    paths = [root/"train.jsonl",root/"dev.jsonl",*sorted((root/"diagnostics").glob("*.jsonl"))]
    for path in paths:
        rows = read_jsonl(path)
        n, m, total = 0, 0, 0
        for start in range(0,len(rows),128):
            part = rows[start:start+128]
            xs = tokenizer.encode_batch([r["input"] for r in part])
            ys = tokenizer.encode_batch([r["target"] for r in part])
            require(all(unknown not in e.ids and len(e.ids)<=512 for e in xs),path.name)
            require(all(e.ids[-1]==tokenizer.token_to_id("</s>") and len(e.ids)<=128 for e in ys),path.name)
            require(tokenizer.decode_batch([e.ids for e in ys])==[r["target"] for r in part],path.name)
            n = max(n,max(len(e.ids) for e in xs))
            m = max(m,max(len(e.ids) for e in ys))
            total += sum(len(e.ids) for e in xs)
        stats[str(path.relative_to(root).as_posix())] = {"rows":len(rows),"max_input_tokens":n,"max_target_tokens":m,
                                                        "mean_input_tokens":round(total/len(rows),2)}
    by_mode = defaultdict(lambda:defaultdict(list))
    for row in read_jsonl(root/"train.jsonl"):
        state = states[row["state_id"]]
        grid,_ = state.witness()
        mode = row["mode"]
        if mode=="g_next":
            by_mode[mode]["v2"].append(state_prompt(grid,state.current,local_status=True))
            by_mode[mode]["previous_single"].append(render(mode,state))
        else:
            by_mode[mode]["v2"].append(v2_step_prompt(grid,state.current,row["candidate"]))
            by_mode[mode]["previous_single"].append(last_move.step_input(state,row["candidate"]))
        by_mode[mode]["current"].append(row["input"])
    comparison = {mode:{name:round(sum(len(e.ids) for e in tokenizer.encode_batch(texts))/len(texts),2)
                        for name,texts in versions.items()} for mode,versions in by_mode.items()}
    return {"tokenizer_sha256":file_hash(tokenizer_path),"all_targets_roundtrip":True,
            "mean_train_prompt_tokens_same_states":comparison,
            "input_unknown_tokens":0,"truncated_rows":0,"files":stats}


def build(previous, source, executor, tokenizer, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    oracle = OfficialOracle(executor)
    require(all(file_hash(source/name)==h for name,h in SOURCE_SHA.items()),"frozen v2 inputs")
    original_manifest = json.loads((source/"manifest.json").read_bytes())
    require(oracle.sha256==original_manifest["executor_sha256"],"original executor")
    pm = json.loads((previous/"manifest.json").read_bytes())
    require(file_hash(previous/"states.jsonl")==pm["sha256"]["states.jsonl"],"previous state registry")
    registry = {r["id"]:State.from_record(r) for r in read_jsonl(previous/"states.jsonl")}
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp)/"dataset"
        diagnostics = stage/"diagnostics"
        diagnostics.mkdir(parents=True)
        used, layouts, inputs, tests = {}, {}, {}, {}
        report = {"supervised":{},"g_diagnostics":{},"heldout":{}}
        groups_by_split = {}
        for sp,seed in (("train",42),("dev",43)):
            old = read_split(previous,sp,version=last_move.VERSION,fingerprint=last_move.template_hash())
            raw = read_jsonl(source/f"{sp}.jsonl")
            rows,groups,states = Selection(sp,old,registry,raw).run()
            write_lines(stage/f"{sp}.jsonl",rows)
            rows = read_jsonl(stage/f"{sp}.jsonl")
            report["supervised"][sp] = audit_split(rows,groups,states,old,raw,oracle,sp)
            report["supervised"][sp]["previous_shortest_arrival_axes"] = arrival_axis_report(
                old,registry,{r["id"]:r for r in raw})
            report["g_diagnostics"][sp] = g_probes.write_g_views(diagnostics,sp,rows,states,oracle,seed)
            if sp=="train":
                report["nonoptimal_paths"] = check_route_shapes(rows,states,oracle)
            groups_by_split[sp] = groups
            used.update(states)
            layouts[sp] = {s.record()["layout_id"] for s in states.values()}
            inputs[sp] = {r["input"] for r in rows}
        write_lines(stage/"states.jsonl",(s.record() for sid,s in sorted(used.items())))
        write_object(stage/"control_groups.json",groups_by_split)
        for sp,seed,nedit in (("test_id",44,1),("test_ood",45,2)):
            name = sp+".jsonl"
            copyfile(source/name,stage/name)
            cases = read_jsonl(stage/name)
            require(file_hash(stage/name)==SOURCE_SHA[name] and len(cases)==200,sp)
            require(all(r["edit_count"]==(0 if r["valid"] else nedit) for r in cases),sp)
            require(all(len({e["original_index"] for e in r["edits"]})==len(r["edits"]) for r in cases),sp)
            report["heldout"][sp] = g_probes.write_heldout_view(diagnostics,sp,cases,oracle,seed)
            report["heldout"][sp]["original_edit_counts"] = Counter(r["edit_count"] for r in cases)
            by_operation = defaultdict(Counter)
            for case in cases:
                kinds = "+".join(e["kind"] for e in case["edits"]) or "unchanged"
                by_operation[kinds][case["reason"] or "valid"] += 1
            report["heldout"][sp]["source_edit_kinds_vs_first_failure"] = dict(by_operation)
            layouts[sp] = {r["layout_id"] for r in cases}
            held = read_jsonl(diagnostics/f"{sp}_local.jsonl")
            inputs[sp] = {r["input"] for r in held}
            tests[sp] = {(r["layout_id"],str(r["grid"]["start"]),str(r["grid"]["goal"]),r["candidate"]) for r in cases}
        pairs = (("train","dev"),("train","test_id"),("train","test_ood"),("dev","test_id"),("dev","test_ood"))
        for a,b in pairs:
            require(layouts[a].isdisjoint(layouts[b]) and inputs[a].isdisjoint(inputs[b]),(a,b))
        report["overlaps"] = {"learning_environment_or_input_overlap_with_other_partitions":0,
                              "fixed_ID_OOD_shared_layouts":len(layouts["test_id"]&layouts["test_ood"]),
                              "fixed_ID_OOD_identical_grid_and_candidate":len(tests["test_id"]&tests["test_ood"])}
        report["train_grid_sizes"] = sorted({(r["grid"]["rows"],r["grid"]["cols"]) for r in read_jsonl(source/"train.jsonl")})
        walls = [len(r["grid"]["obstacles"]) for r in read_jsonl(source/"train.jsonl")]
        report["train_wall_count_range"] = [min(walls),max(walls)]
        report["tokenization"] = token_check(stage,tokenizer,used)
        write_object(stage/"audit.json",report)
        write_object(stage/"shortcut_audit.json",inspect_dataset(stage,True))
        write_object(stage/"manifest.json",{"schema":g_probes.VERSION,"template_sha256":g_probes.template_hash(),
                     "base_commit":V2_COMMIT,"previous_manifest_sha256":file_hash(previous/"manifest.json"),
                     "source_sha256":SOURCE_SHA,"executor_sha256":oracle.sha256,"tokenizer_sha256":file_hash(tokenizer),
                     "seeds":{"train":42,"dev":43,"test_id_view":44,"test_ood_view":45},
                     "sha256":{p.relative_to(stage).as_posix():file_hash(p) for p in sorted(stage.rglob("*")) if p.is_file()}})
        stage.rename(output)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous",type=Path,default=Path("data/step_last_move"))
    parser.add_argument("--source",type=Path,default=Path("data/derived"))
    parser.add_argument("--executor",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    results = build(args.previous,args.source,args.executor,args.tokenizer,args.out)
    print({s:d["labels"] for s,d in results["supervised"].items()})
