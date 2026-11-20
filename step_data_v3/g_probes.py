"""One verification prompt for sampling, training, online inference and diagnostic views."""
from collections import Counter, defaultdict
from pathlib import Path
import random

from ppnl.core import ACTIONS, Grid
from ppnl.data import read_jsonl
from . import last_move, model_io
from .audit import require
from .build import write_lines
from .paired_steps import TRAIN_WORDS, DEV_WORDS
from .schema import State
from .single import LABELS

VERSION = "ppnl-g-state-balanced-v1"
VERIFIER_LOSS = "0.5 * first_token_CE + 0.5 * mean_remaining_token_CE"


def step_input(state, action):
    return last_move.step_input(state,action,explicit_vocabulary=True)


def template_hash():
    return last_move.template_hash(step_input)


def prepare_training(runner, dataset):
    if hasattr(runner.model, "verifier_head"):
        return runner.prepare_training(dataset)
    label_tokens = runner.tokenizer(text_target=list(LABELS), add_special_tokens=False)["input_ids"]
    if not all(label_tokens) or len({tokens[0] for tokens in label_tokens}) != len(LABELS):
        raise ValueError("Each V class must have a distinct first target token")
    features = model_io.prepare_training(runner,dataset,version=VERSION,fingerprint=template_hash())
    runner.model.config.ppnl_verifier_loss = VERIFIER_LOSS
    return features


def batch_loss(runner, batch):
    if hasattr(runner.model, "verifier_head"):
        return runner.batch_loss(batch)
    return model_io.batch_loss(runner, batch)


def generation_policy(runner, check=lambda: None):
    return model_io.StatePolicy(runner,check,fingerprint=template_hash())


def verify_moves(runner, observations, check=lambda: None):
    return last_move.verify_moves(runner,observations,check,renderer=step_input)


def verify_last(runner, state_before_last, last_action, check=lambda: None):
    label, = verify_moves(runner,[(state_before_last,last_action)],check)
    return label


def verify_plans(runner, grids, plans, check=lambda: None):
    return last_move.scan(grids,plans,lambda obs:verify_moves(runner,obs,check))


def example(state, action):
    return {"state_id":state.id,"candidate":action,"input":step_input(state,action),
            "target":last_move.transition(state,action)[1]}


def g_examples(rows, states):
    for g in rows:
        if g["mode"]=="g_next":
            for action in ACTIONS:
                yield dict(example(states[g["state_id"]],action),id=f"{g['id']}-probe-{action}",
                           g_id=g["id"],donor_id=g["donor_id"],is_G_representative=(action==g["target"]),
                           is_G_optimal=(action in g["optimal_actions"]))


def reference(state, action, oracle):
    grid, prefix = state.witness()
    return oracle.check(grid," ".join(prefix+[action])).reason or "valid"


def balanced_rows(rows, seed):
    """Equal class counts, unique prompts. Break ties deterministically before seeing predictions."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    by_class, used = defaultdict(list), {}
    for r in shuffled:
        prompt, label = r["input"], r["target"]
        if prompt in used:
            require(used[prompt]==label,"conflicting targets for the same diagnostic input")
            continue
        used[prompt] = label
        by_class[label].append(r)
    n = min(len(by_class[c]) for c in LABELS)
    require(n>0,"all five labels must be present for a balanced view")
    chosen = [r for c in LABELS for r in by_class[c][:n]]
    rng.shuffle(chosen)
    return chosen


def write_g_views(directory, split, rows, states, oracle, seed):
    gs = [r for r in rows if r["mode"]=="g_next"]
    supervised = {r["input"] for r in rows if r["mode"]=="v_step"}
    physical = list(g_examples(gs,states))
    require(len(physical)==4*len(gs)==len({r["input"] for r in physical}),split)
    require(sum(r["is_G_representative"] for r in physical)==len(gs) and
            all(not r["is_G_representative"] or r["is_G_optimal"] for r in physical),split)
    require(all(r["target"] in ("valid","goal_not_reached") for r in physical if r["is_G_optimal"]),split)
    lexical_n = sum(r["target"]=="valid" for r in physical)
    words = TRAIN_WORDS if split=="train" else DEV_WORDS
    rng = random.Random(seed)
    shuffled = list(gs)
    rng.shuffle(shuffled)
    lexical = [dict(example(states[g["state_id"]],words[i%len(words)]),id=f"{g['id']}-probe-word",
                    g_id=g["id"],donor_id=g["donor_id"],is_G_representative=False,is_G_optimal=False)
               for i,g in enumerate(shuffled[:lexical_n])]
    for r in physical+lexical:
        s = states[r["state_id"]]
        require(r["input"]==step_input(s,r["candidate"]) and r["target"]==reference(s,r["candidate"],oracle),r["id"])
        r["in_same_split_supervised_V"] = r["input"] in supervised
    balanced = balanced_rows(physical+lexical,seed)
    require(len(balanced)==5*lexical_n and sum(r["target"]=="valid" and r["in_same_split_supervised_V"]
            for r in physical)==lexical_n,split)
    write_lines(directory/f"g_{split}_four.jsonl",physical)
    write_lines(directory/f"g_{split}_balanced.jsonl",balanced)
    return {"G_states":len(gs),"four_prompts":len(physical),"natural_labels":Counter(r["target"] for r in physical),
            "G_representative_action_V_labels":Counter(r["target"] for r in physical if r["is_G_representative"]),
            "G_all_optimal_actions_V_labels":Counter(r["target"] for r in physical if r["is_G_optimal"]),
            "natural_also_in_same_split_V":sum(r["in_same_split_supervised_V"] for r in physical),
            "balanced_labels":Counter(r["target"] for r in balanced),
            "balanced_also_in_same_split_V":sum(r["in_same_split_supervised_V"] for r in balanced)}


def write_heldout_view(directory, split, cases, oracle, seed):
    """One reachable real action per original nonempty candidate. Leave the original test untouched."""
    observations, skipped, prompts = [], [], {}
    checked = last_move.scan([Grid.from_dict(r["grid"]) for r in cases],[r["candidate"] for r in cases],
                             lambda items:[last_move.transition(s,a)[1] for s,a in items])
    require(checked==[r["label"] for r in cases],split)
    for r in cases:
        grid, candidate = Grid.from_dict(r["grid"]),r["candidate"]
        truth = oracle.check(grid,candidate)
        require(truth.label()==r["label"],r["id"])
        if not candidate.split():
            skipped.append(r["id"])
            continue
        step = len(candidate.split()) if truth.feasible else truth.step
        state, action = list(last_move.path_steps(grid,candidate))[step-1]
        record = dict(example(state,action),id=r["id"],observed_move_number=step,original_error_step=truth.step,
                      layout_id=r["layout_id"],state=state.record())
        require(record["target"]==(truth.reason or "valid")==reference(state,action,oracle)
                and State.from_record(record["state"])==state and record["input"]==step_input(state,action),r["id"])
        require(prompts.setdefault(record["input"],record["target"])==record["target"],r["id"])
        observations.append(record)
    selection = {r["id"] for r in balanced_rows(observations,seed)}
    for r in observations:
        r["in_balanced_subset"] = r["id"] in selection
    write_lines(directory/f"{split}_local.jsonl",observations)
    return {"original_cases":len(cases),"reference_scan_exact":len(checked),"real_action_rows":len(observations),
            "unique_local_inputs":len(prompts),"no_move_case_ids":skipped,"labels":Counter(r["target"] for r in observations),
            "balanced_labels":Counter(r["target"] for r in observations if r["in_balanced_subset"])}


def read_heldout_subset(dataset, split):
    if split not in ("test_id","test_ood"):
        raise ValueError(split)
    return [r for r in read_jsonl(Path(dataset)/"diagnostics"/f"{split}_local.jsonl") if r["in_balanced_subset"]]
