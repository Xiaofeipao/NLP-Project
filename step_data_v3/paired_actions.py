"""Choose independent single-action V examples from the frozen training partition."""
from collections import Counter, defaultdict
import random

from ppnl.core import ACTIONS
from . import g_probes, last_move
from .audit import require

ARRIVED = "valid"
UNFINISHED = "goal_not_reached"
WORD = "invalid_action_token"
CONTROL_MIN = {"swap_walls": 12, "change_wall": 6, "change_goal": 6, "change_edge": 6,
               "four_open": 4, "four_tie": 6, "four_detour": 12, "four_ordinary": 6}


def adjacent(state):
    return sum(abs(a-b) for a,b in zip(state.current,state.goal))==1


def row_key(row):
    return row["state_id"], row["candidate"]


def decision_sets(states, state_ids):
    return {s: {a:last_move.transition(states[s],a)[1] for a in ACTIONS} for s in sorted(state_ids)}


def select_keys(train, states, groups):
    """Minimize replacements under class, direction, coverage and control-group constraints."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    gs = {r["state_id"]:r for r in train if r["mode"]=="g_next"}
    old = {row_key(r):r for r in train if r["mode"]=="v_step"}
    ids = {r["id"]:k for k,r in old.items()}
    anchors = {ids[g["anchor_v_id"]] for g in gs.values() if g["anchor_v_id"]}
    words = {k for k in old if k[1] not in ACTIONS}
    positive = {s for (s,a),r in old.items() if r["target"]==ARRIVED}
    labels = decision_sets(states,gs.keys() | {s for s,_ in old})
    # All previous V states are candidates. Also allow previously unused G states
    # with at most one unfinished direction, which can represent narrow passages.
    pool = {s for s,_ in old} | {s for s in gs if list(labels[s].values()).count(UNFINISHED)<=1}
    old_four = {s for s in pool if all((s,a) in old for a in ACTIONS)}
    full_candidates = sorted(s for s in pool if (s in positive or ARRIVED not in labels[s].values())
                             and (len(set(labels[s].values()))>=2 or s in old_four))
    keys = sorted((old.keys()-words) | {(s,a) for s in full_candidates for a in ACTIONS})
    xi = {k:i for i,k in enumerate(keys)}
    yi = {s:len(xi)+i for i,s in enumerate(full_candidates)}
    old_control = [g for g in groups if not g["kind"].startswith("four_")]
    zi = {g["id"]:len(xi)+len(yi)+i for i,g in enumerate(old_control)}
    size = len(xi)+len(yi)+len(zi)
    lower, upper, cost = np.zeros(size), np.ones(size), np.zeros(size)
    row_index, col_index, coefficients, lo, hi = [], [], [], [], []

    def constraint(indices, minimum=0, maximum=np.inf, weights=None):
        indices = list(indices)
        row_index.extend([len(lo)]*len(indices))
        col_index.extend(indices)
        coefficients.extend([1]*len(indices) if weights is None else weights)
        lo.append(minimum)
        hi.append(maximum)

    slots, layouts = defaultdict(list), defaultdict(list)
    layout_ids = {s:states[s].record()["layout_id"] for s in labels}
    fixed_layouts = {layout_ids[s] for s in gs} | {layout_ids[s] for s,_ in words}
    rng = random.Random(42)
    for (s,a),i in xi.items():
        target = labels[s][a]
        # Tie-breakers together cost less than one replacement for the frozen set.
        cost[i] = int((s,a) not in old) - 0.0002*int((s,a) in anchors) + 0.00003*rng.random()
        if target==ARRIVED:
            lower[i] = upper[i] = int((s,a) in old)
        if (s,a) in old and states[s].current==states[s].goal:
            lower[i] = 1
        slots[target,a].append(i)
        if layout_ids[s] not in fixed_layouts:
            layouts[layout_ids[s]].append(i)
    quota = Counter((r["target"],a) for (s,a),r in old.items() if a in ACTIONS)
    for k,n in quota.items():
        constraint(slots[k], n, n)
    for indices in layouts.values():
        constraint(indices,1)

    # Pair every G arrival for which another physical action is legal and unfinished.
    for s in sorted(positive & gs.keys()):
        alternatives = [xi[s,a] for a in ACTIONS if labels[s][a]==UNFINISHED]
        if alternatives:
            constraint(alternatives,1)
    near_unfinished = [i for (s,a),i in xi.items() if labels[s][a]==UNFINISHED and adjacent(states[s])]
    constraint(near_unfinished,217,255)

    # A quartet counts only when all four independent prompts are supervised.
    for s,j in yi.items():
        for a in ACTIONS:
            constraint([xi[s,a],j],weights=[1,-1])
    rich = {s for s in positive if len(set(labels[s].values()))==4}
    for s in rich:
        lower[yi[s]] = 1
    constraint((j for s,j in yi.items() if s in gs),155)
    constraint((j for s,j in yi.items() if s not in gs),18)
    constraint((j for s,j in yi.items() if s in positive),90)
    constraint((j for s,j in yi.items() if len(set(labels[s].values()))>=3),145)

    families = defaultdict(list)
    for g in groups:
        if g["kind"].startswith("four_"):
            families[g["kind"]].append(yi[ids[g["rows"][0]][0]])
        else:
            j = zi[g["id"]]
            families[g["kind"]].append(j)
            for v_id in g["rows"]:
                constraint([xi[ids[v_id]],j],weights=[1,-1])
    for kind,n in CONTROL_MIN.items():
        constraint(families[kind],n)
    matrix = coo_matrix((coefficients,(row_index,col_index)),shape=(len(lo),size)).tocsc()
    result = milp(cost,integrality=np.ones(size),bounds=Bounds(lower,upper),
                  constraints=LinearConstraint(matrix,lo,hi),options={"mip_rel_gap":0.0001})
    require(result.success, f"Paired-action selection: {result.message}")
    picked = np.rint(result.x)
    actual = matrix @ picked
    require(all(actual >= lo) and all(actual <= hi) and all(lower <= picked) and all(picked <= upper),
            "Rounded selection violates its quotas")
    return {k for k,i in xi.items() if picked[i]} | words


def make_rows(train, selected, states):
    old = {row_key(r):r for r in train if r["mode"]=="v_step"}
    gs = {r["state_id"]:r for r in train if r["mode"]=="g_next"}
    parents = defaultdict(list)
    for (s,a),r in old.items():
        parents[s].append(r)
    new = {k:old[k] for k in selected & old.keys()}
    additions = defaultdict(list)
    for index,(s,a) in enumerate(sorted(selected-old.keys()),1):
        if s in gs:
            parent, origin, shared = gs[s], "G_direction", gs[s]["id"]
        else:
            parent = min(parents[s],key=lambda r:(r["target"]!=ARRIVED,r["id"]))
            origin, shared = "V_direction", None
        r = dict(g_probes.example(states[s],a), id=f"train-paired-V-{index:04}", mode="v_step",
                 task="verification", donor_id=parent["donor_id"], origin=origin,
                 shared_g_id=shared, parent_id=parent["id"])
        new[s,a] = r
        additions[r["target"],a].append(r)
    by_state = defaultdict(list)
    for (s,_),r in new.items():
        by_state[s].append(r["id"])
    rng = random.Random(42)
    for bag in additions.values():
        rng.shuffle(bag)
    rows = []
    for r in train:
        if r["mode"]=="g_next":
            options = by_state[r["state_id"]]
            anchor = r["anchor_v_id"] if r["anchor_v_id"] in options else min(options,default=None)
            rows.append(dict(r,anchor_v_id=anchor))
        elif row_key(r) in selected:
            rows.append(r)
        else:
            # Preserve even the class and action at each position in the original shuffled file.
            rows.append(additions[r["target"],r["candidate"]].pop())
    require(not any(additions.values()),"Unused replacement rows")
    return rows


def control_groups(train, parent_groups, states):
    vs = {row_key(r):r for r in train if r["mode"]=="v_step"}
    ids = {r["id"]:r for r in vs.values()}
    gs = {r["state_id"] for r in train if r["mode"]=="g_next"}
    result = [g for g in parent_groups if all(i in ids for i in g["rows"])]
    marked = {ids[g["rows"][0]]["state_id"] for g in result if g["kind"].startswith("four_")}
    for s in sorted({s for s,_ in vs}):
        actions = [a for a in ACTIONS if (s,a) in vs]
        labels = {a:vs[s,a]["target"] for a in actions}
        if len(actions)==4 and s not in marked:
            kind = "four_v_state" if s not in gs else ("four_adjacent" if adjacent(states[s]) else "four_actions")
            result.append({"id":f"train-four-{s}","kind":kind,"rows":[vs[s,a]["id"] for a in actions]})
        if ARRIVED in labels.values() and UNFINISHED in labels.values():
            ids_pair = [next(vs[s,a]["id"] for a in actions if labels[a]==target)
                        for target in (ARRIVED, UNFINISHED)]
            result.append({"id":f"train-pair-{s}","kind":"same_state_arrival","rows":ids_pair})
    return result
