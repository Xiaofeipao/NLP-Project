"""Test restricted-information rules on held-out layouts and count exact state/action controls."""
from collections import Counter, defaultdict

from ppnl.core import ACTIONS
from ppnl.data import read_jsonl
from .last_move import LABELS, transition
from .schema import State


def geometry_blind(state, action, last_move):
    """Keep vocabulary, coordinates and goal; ignore every obstacle."""
    if action is None:
        return "valid" if state.current==state.goal else "goal_not_reached"
    if action not in ACTIONS:
        return "invalid_action_token"
    dr,dc = ACTIONS[action]
    dest = state.current[0]+dr, state.current[1]+dc
    if not (0 <= dest[0] < state.rows and 0 <= dest[1] < state.cols):
        return "out_of_bounds"
    return "valid" if not last_move or dest==state.goal else "goal_not_reached"


def metrics(gold, predictions):
    counts = Counter(zip(gold,predictions,strict=True))
    recall, precision, macro = {}, {}, []
    for cls in LABELS:
        hit = counts[cls,cls]
        actual, predicted = gold.count(cls),predictions.count(cls)
        recall[cls] = [hit, actual]
        precision[cls] = [hit, predicted]
        macro.append(2*hit/(actual+predicted) if actual+predicted else 0)
    return {"correct":sum(counts[c,c] for c in LABELS),"n":len(gold),
            "macro_f1":round(sum(macro)/len(macro),4),"recall_counts":recall,"precision_counts":precision}


def features(row,state):
    a = row["candidate"]
    fields = ("Current:","Goal:","Walls:","Action:","Last move:")
    skeleton = tuple(line.split(":",1)[0] if line.startswith(fields) else line for line in row["input"].splitlines())
    return {"constant_template":skeleton, "action_only":a,
            "format_only":(len(state.obstacles),len(a or ""),len(row["input"])),
            "action_and_wall_count":(a,len(state.obstacles))}


def lookup(counts):
    return max(LABELS,key=lambda c:counts[c])


def inspect_coverage(rows,states):
    gs = {r["state_id"]:r for r in rows if r["mode"]=="g_next"}
    vs = defaultdict(list)
    for r in rows:
        if r["mode"]=="v_step":
            vs[r["state_id"]].append(r)
    counts = Counter()
    direction_labels = defaultdict(Counter)
    for sid,group in vs.items():
        real = {r["candidate"]:r["target"] for r in group if r["candidate"] in ACTIONS}
        for a,label in real.items():
            direction_labels[a][label] += 1
        if sid not in gs:
            continue
        counts["shared_G_states"] += 1
        counts["shared_V_rows"] += len(group)
        counts["G_with_multiple_compass_actions"] += len(real)>=2
        counts["G_with_all_four_compass_actions"] += len(real)==4
        counts["G_with_distinct_V_targets_across_compass_actions"] += len(set(real.values()))>1
        legal = {a for a in real if transition(states[sid],a)[0] is not None}
        best = set(gs[sid]["optimal_actions"])
        counts["G_showing_optimal_and_legal_nonoptimal_moves"] += bool(legal & best and legal-best)
    horizontal_first = 0
    for sid,g in gs.items():
        s = states[sid]
        dr,dc = s.goal[0]-s.current[0],s.goal[1]-s.current[1]
        a = ("right" if dc>0 else "left") if dc else ("down" if dr>0 else "up")
        horizontal_first += a in g["optimal_actions"]
    return {**counts,"compass_action_by_label":dict(direction_labels),
            "G_horizontal_first_without_walls":[horizontal_first,len(gs)]}


def inspect_dataset(root, last_move):
    states = {r["id"]:State.from_record(r) for r in read_jsonl(root/"states.jsonl")}
    rows = {s:read_jsonl(root/f"{s}.jsonl") for s in ("train","dev")}
    train = [r for r in rows["train"] if r["mode"]=="v_step"]
    prior = lookup(Counter(r["target"] for r in train))
    learned = defaultdict(lambda:defaultdict(Counter))
    for r in train:
        for name,key in features(r,states[r["state_id"]]).items():
            learned[name][key][r["target"]] += 1
    train_words = {r["candidate"] for r in train if r["target"]=="invalid_action_token"}
    report = {}
    for split,all_rows in rows.items():
        ver = [r for r in all_rows if r["mode"]=="v_step"]
        gold = [r["target"] for r in ver]
        preds = defaultdict(list)
        for r in ver:
            s, a = states[r["state_id"]],r["candidate"]
            for name,key in features(r,s).items():
                cell = learned[name].get(key)
                preds[name].append(lookup(cell) if cell else prior)
            preds["coordinates_and_goal_without_walls"].append(geometry_blind(s,a,last_move))
        words = Counter(r["candidate"] for r in ver if r["target"]=="invalid_action_token")
        report[split] = {"labels":Counter(gold),"shortcut_scores":{k:metrics(gold,p) for k,p in preds.items()},
                         "reuse":inspect_coverage(all_rows,states),"invalid_words":words,
                         "template_skeletons":len({features(r,states[r["state_id"]])["constant_template"] for r in ver}),
                         "invalid_rows_with_word_unseen_in_train":sum(n for k,n in words.items() if k not in train_words),
                         "stop_inputs":sum(r["candidate"] is None for r in ver),
                         "unfinished_from_stop":sum(r["candidate"] is None and r["target"]=="goal_not_reached" for r in ver)}
    return report
