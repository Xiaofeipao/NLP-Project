"""Coverage and leakage checks for adding alternative moves to frozen training states."""
from collections import Counter, defaultdict
from functools import lru_cache

from ppnl.core import ACTIONS, Grid
from . import last_move
from .audit import require
from .paired_actions import ARRIVED, UNFINISHED, adjacent, row_key
from .schema import State


def coverage(rows, states, previous_v):
    gs = {r["state_id"] for r in rows if r["mode"]=="g_next"}
    vs = [r for r in rows if r["mode"]=="v_step"]
    by_state = defaultdict(dict)
    for r in vs:
        if r["candidate"] in ACTIONS:
            by_state[r["state_id"]][r["candidate"]] = r["target"]
    full = {s for s,v in by_state.items() if len(v)==4}
    paired = {s for s,v in by_state.items() if {ARRIVED,UNFINISHED}.issubset(v.values())}
    cardinality = Counter(len(set(by_state[s].values())) for s in full)
    legal = [r for r in vs if r["target"] in (ARRIVED,UNFINISHED)]
    correct = sum(adjacent(states[r["state_id"]])==(r["target"]==ARRIVED) for r in legal)
    far = sum(r["target"]==UNFINISHED and not adjacent(states[r["state_id"]]) for r in legal)
    old_keys = {row_key(r) for r in previous_v}
    old_states = {s for s,_ in old_keys}
    fresh = {row_key(r) for r in vs}-old_keys
    return {
        "V_classes":Counter(r["target"] for r in vs),
        "states_by_supervised_physical_action_count":Counter(len(v) for v in by_state.values()),
        "arrival_paired_with_legal_nonterminal":len(paired),"paired_G_states":len(paired&gs),
        "full_four_states":len(full),"full_four_G_states":len(full&gs),
        "full_four_previous_V_only_states":len((full & old_states)-gs),
        "full_four_adjacent":sum(adjacent(states[s]) for s in full),
        "full_four_outcome_count_histogram":cardinality,
        "full_four_with_at_least_three_outcomes":cardinality[3]+cardinality[4],
        "old_V_states_receiving_new_actions":len({s for s,a in fresh}&old_states),
        "old_V_only_states_receiving_new_actions":len(({s for s,a in fresh}&old_states)-gs),
        "new_supervised_V_inputs":len(fresh),"removed_supervised_V_inputs":len(old_keys-{row_key(r) for r in vs}),
        "unfinished_far_from_goal":far,"distance_only_correct_on_two_legal_classes":[correct,len(legal)],
    }


def verify_replacements(old, new):
    require(len(old)==len(new),"Changed training budget")
    updated_anchors = 0
    for a,b in zip(old,new,strict=True):
        require(a["mode"]==b["mode"],"Training task moved")
        if a["mode"]=="g_next":
            require(dict(a,anchor_v_id=b["anchor_v_id"])==b,"Changed G supervision")
            updated_anchors += a["anchor_v_id"]!=b["anchor_v_id"]
        else:
            require((a["candidate"],a["target"])==(b["candidate"],b["target"]),"Changed class or action quota")
            if a["id"]==b["id"] or a["target"] in (ARRIVED,"invalid_action_token"):
                require(a==b,"Changed source row without giving the case a new ID")
    return updated_anchors


@lru_cache(maxsize=32000)
def square_orbits(state, action):
    """Canonical wall layout and (state, move), transforming the move together with the grid."""
    n = state.rows
    require(n==state.cols,"Expected square benchmark maps")
    mid = (n-1)//2
    dr,dc = ACTIONS.get(action,(0,0))
    forms, layouts = [], []
    for mirror in (False,True):
        for turns in range(4):
            def transform(point):
                r,c = point
                if mirror:
                    c = n-1-c
                for _ in range(turns):
                    r,c = c,n-1-r
                return r,c
            walls = tuple(sorted(map(transform,state.obstacles)))
            start = transform((mid,mid))
            end = transform((mid+dr,mid+dc))
            direction = (end[0]-start[0],end[1]-start[1]) if action in ACTIONS else action
            # Give valid compass moves and arbitrary lexical tokens distinct comparable keys.
            token = ("physical",direction) if action in ACTIONS else ("word",direction)
            forms.append((n,walls,transform(state.current),transform(state.goal),token))
            layouts.append((n,walls))
    return min(layouts),min(forms)


def leakage_report(train, dev, tests, states):
    train_v = [r for r in train if r["mode"]=="v_step"]
    train_keys = {row_key(r) for r in train_v}
    train_inputs = {r["input"] for r in train}
    train_states = {r["state_id"] for r in train}
    train_layouts = {states[s].record()["layout_id"] for s in train_states}
    canon_v = {square_orbits(states[s],a)[1] for s,a in train_keys}
    canon_layouts = {square_orbits(states[s],"")[0] for s in train_states}
    dev_v = [r for r in dev if r["mode"]=="v_step"]
    result = {}

    def compare(name, cases, partitions):
        observed = {(s.id,a) for s,a in cases}
        canon = {square_orbits(s,a)[1] for s,a in cases}
        layouts = {s.record()["layout_id"] for s in partitions}
        symmetry = {square_orbits(s,"")[0] for s in partitions}
        require(not (layouts & train_layouts) and not (observed & train_keys),f"Exact leakage: {name}")
        result[name] = {"observed_moves":len(cases),"distinct_state_actions":len(observed),
                        "layout_overlap":len(layouts&train_layouts),"state_action_overlap":len(observed&train_keys),
                        "symmetric_layout_classes_in_common":len(symmetry&canon_layouts),
                        "symmetric_state_action_classes_in_common":len(canon&canon_v)}

    require(train_inputs.isdisjoint(r["input"] for r in dev),"Training/development prompt overlap")
    compare("dev",[(states[r["state_id"]],r["candidate"]) for r in dev_v],{states[r["state_id"]] for r in dev})
    for name,rows in tests.items():
        cases, layouts = [], []
        for r in rows:
            grid = Grid.from_dict(r["grid"])
            layouts.append(State.from_grid(grid))
            for state,action in last_move.path_steps(grid,r["candidate"]):
                cases.append((state,action))
                if last_move.transition(state,action)[0] is None:
                    break
        compare(name,cases,layouts)
    return result
