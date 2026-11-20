"""Independently replay decisions, counterfactual pairs and split membership."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path

from ppnl.core import ACTIONS, Grid, OfficialOracle, parse_verdict, shortest_path
from ppnl.data import edit_distance, read_jsonl
from ppnl.experiment_freeze import file_hash
from .schema import VERSION, TASK, State, render, template_hash


def require(condition, detail):
    if not condition:
        raise ValueError(f"Audit failed: {detail}")


def shifted(p, action):
    dr, dc = ACTIONS[action]
    return p[0] + dr, p[1] + dc


def v2_shortcut(candidate):
    tokens = candidate.split()
    if not tokens:
        return "invalid | step=1 | reason=goal_not_reached"
    if "jump" in tokens:
        return f"invalid | step={tokens.index('jump')+1} | reason=invalid_action_token"
    return "valid"


def check_v(row, state, oracle):
    grid, prefix = state.witness()
    tokens = row["candidate"].split()
    local = row["mode"] == "v_step"
    require(local or not prefix, row["id"])
    require(not local or len(tokens) <= 1, row["id"])
    verdict = oracle.check(grid, " ".join(prefix + tokens))
    step = None if verdict.valid else verdict.step - len(prefix)
    require(parse_verdict(row["target"]) == (verdict.valid, step, verdict.reason), row["id"])
    expected = {"valid": verdict.valid, "step": step, "reason": verdict.reason}
    if local:
        pos = state.current
        if tokens and verdict.feasible:
            pos = shifted(pos, tokens[0])
        expected.update(action_legal=verdict.feasible if tokens else None,
                        at_goal_after=pos == state.goal if verdict.feasible else None,
                        position_after=list(pos) if verdict.feasible else None)
    require(expected == row["labels"], row["id"])


def check_g(row, state, oracle):
    grid = state.grid()
    reference = shortest_path(grid)
    require(reference and row["distance"] == len(reference), row["id"])
    best = []
    for action in ACTIONS:
        pos = shifted(grid.start, action)
        if not grid.inside(pos) or pos in grid.obstacles:
            continue
        tail = [] if pos == grid.goal else shortest_path(replace(grid, start=pos))
        if tail is not None and 1 + len(tail) == len(reference):
            require(oracle.check(grid, " ".join([action] + tail)).valid, row["id"])
            best.append(action)
    require(best == row["optimal_actions"] and row["target"] in best, row["id"])
    before = sum(abs(a-b) for a,b in zip(grid.start, grid.goal))
    forced = all(sum(abs(a-b) for a,b in zip(shifted(grid.start, act), grid.goal)) > before for act in best)
    return forced


def check_edit(row, state, oracle):
    original, damaged = row["reference"].split(), row["candidate"].split()
    require(oracle.check(state.grid(), row["reference"]).valid, row["id"])
    require(len(shortest_path(state.grid())) == len(original), row["id"])
    edit = row["edit"]
    i, op = edit["offset"], edit["operation"]
    require(0 <= i < len(original) + (op == "insert"), row["id"])
    distance = edit_distance(original, damaged)
    if op == "truncate":
        require(damaged == original[:i] and distance == len(original)-i, row["id"])
        return
    require(distance == 1, row["id"])
    if op == "delete":
        expected = original[:i] + original[i+1:]
    elif op == "replace":
        expected = original[:i] + [edit["token"]] + original[i+1:]
    elif op == "insert":
        expected = original[:i] + [edit["token"]] + original[i:]
    else:
        raise ValueError(op)
    require(damaged == expected, row["id"])


def check_group(rows, states):
    family = rows[0]["group_kind"]
    paired = family in ("move_goal", "stop_goal", "wall", "edge", "token", "plan_pair")
    require(len(rows) == (2 if paired else 1), rows[0]["group_id"])
    if not paired:
        return
    a, b = rows
    x, y = states[a["state_id"]], states[b["state_id"]]
    causes = {r["labels"]["reason"] or "valid" for r in rows}
    actions = [r["candidate"] for r in rows]
    require((x.rows, x.cols) == (y.rows, y.cols), a["group_id"])
    if family in ("move_goal", "stop_goal"):
        good = (x.current == y.current and x.obstacles == y.obstacles and x.goal != y.goal
                and actions[0] == actions[1] and bool(actions[0]) == (family == "move_goal")
                and causes == {"valid", "goal_not_reached"})
    elif family == "wall":
        dest = shifted(x.current, actions[0])
        good = (x.current == y.current and x.goal == y.goal and dest != x.goal
                and actions[0] == actions[1] and set(x.obstacles) ^ set(y.obstacles) == {dest}
                and causes == {"goal_not_reached", "obstacle_collision"})
    elif family == "edge":
        good = (x.goal == y.goal and x.obstacles == y.obstacles and actions[0] == actions[1]
                and (shifted(x.current, actions[0]) == y.current or shifted(y.current, actions[0]) == x.current)
                and causes == {"goal_not_reached", "out_of_bounds"})
    elif family == "token":
        good = x == y and causes == {"valid", "invalid_action_token"}
    else:
        good = x == y and "valid" in causes and len(causes) == 2
    require(good, a["group_id"])


def shortcut_summary(rows):
    targets_per_action = defaultdict(Counter)
    for row in rows:
        targets_per_action[row["candidate"]][row["target"]] += 1
    return {"rows": len(rows),
            "v2_action_only_rule_exact": sum(v2_shortcut(r["candidate"]) == r["target"] for r in rows),
            "best_in_sample_lookup_by_action_exact": sum(max(c.values()) for c in targets_per_action.values())}


def audit_learning_split(rows, states, donors, oracle, split):
    units = 20 if split == "train" else 1
    groups, gstates = defaultdict(list), {}
    by_id = {r["id"]: r for r in rows}
    modes = Counter(r["mode"] for r in rows)
    require(modes == {"g_next": 100*units, "v_step": 60*units, "v_plan": 40*units}, split)
    require(len(by_id) == len(rows) == len({r["input"] for r in rows}), f"{split} duplicates")
    hist, anchors, kinds, done, directions = defaultdict(Counter), Counter(), Counter(), Counter(), Counter()
    step_rows = []
    for row in rows:
        s, mode = states[row["state_id"]], row["mode"]
        require(row["task"] == TASK[mode] and donors[row["donor_id"]]["layout_id"] == s.record()["layout_id"], row["id"])
        require(row["input"] == render(mode, s, row.get("candidate")), row["id"])
        require("\r" not in row["input"] and "\t" not in row["input"], row["id"])
        groups[row["group_id"]].append(row)
        if mode == "g_next":
            forced = check_g(row, s, oracle)
            gstates[s.id] = row["id"]
            done["must_increase_manhattan"] += forced
            done["multiple_optimal_actions"] += len(row["optimal_actions"]) > 1
            require(row["group_kind"] != "forced" or forced, row["id"])
            require(row["group_kind"] != "tie" or len(row["optimal_actions"]) > 1, row["id"])
            if row["anchor_v_id"]:
                anchor = by_id[row["anchor_v_id"]]
                require(anchor["mode"].startswith("v_") and anchor["state_id"] == s.id, row["id"])
                anchors[anchor["labels"]["reason"] or "valid"] += 1
            continue
        check_v(row, s, oracle)
        reason = row["labels"]["reason"] or "valid"
        hist[mode][reason] += 1
        if mode == "v_step":
            step_rows.append(row)
            if "source_observation" in row:
                original = donors[row["donor_id"]]
                grid = Grid.from_dict(original["grid"])
                link = row["source_observation"]
                position = link["step"] - 1
                attempt = original["candidate"].split()
                require(0 <= position < len(attempt) and attempt[position] == row["candidate"], row["id"])
                prior = oracle.check(grid, " ".join(attempt[:position]))
                require(prior.feasible, row["id"])
                origin = grid.start if position == 0 else prior.trace[position-1]["to"]
                goal = shifted(origin, row["candidate"]) if link["goal_changed"] else grid.goal
                require(s == State.from_grid(grid, origin, goal), row["id"])
                done["original_trace_changed_goal" if link["goal_changed"] else "original_trace_factual"] += 1
            key = ("move" if row["candidate"] else "stop") + "/" + reason
            done[key] += 1
            if row["candidate"] in ACTIONS:
                directions[row["candidate"] + "/" + reason] += 1
        elif reason != "valid":
            check_edit(row, s, oracle)
            kinds["edit/"+row["edit"]["operation"]] += 1
            done["full_nonempty_unfinished"] += reason == "goal_not_reached" and bool(row["candidate"])
            done["full_errors_after_first_step"] += row["labels"]["step"] > 1
        else:
            base, plan = row["reference"].split(), row["candidate"].split()
            require(oracle.check(s.grid(), row["reference"]).valid and len(shortest_path(s.grid())) == len(base), row["id"])
            kind = row["valid_kind"]
            require((len(plan) == len(base)) if kind == "shortest" else (len(plan) > len(base)), row["id"])
            kinds["valid/"+kind] += 1
            if kind == "simple_detour":
                path = oracle.check(s.grid(), row["candidate"]).trace
                require(len({s.current, *(p["to"] for p in path)}) == len(path)+1, row["id"])
    for group in groups.values():
        check_group(group, states)
    for row in rows:
        if row["task"] == "verification":
            require(row["shared_g_id"] == gstates.get(row["state_id"]), row["id"])
    require(len(anchors) == 5 and set(anchors.values()) == {7*units}, f"{split} G/V anchor mix")
    require(hist["v_plan"] == {"valid": 20*units, **{r:5*units for r in
            ("goal_not_reached", "obstacle_collision", "out_of_bounds", "invalid_action_token")}}, split)
    require(hist["v_step"] == {"valid": 17*units, "goal_not_reached": 19*units,
            "obstacle_collision": 8*units, "out_of_bounds": 8*units, "invalid_action_token": 8*units}, split)
    for action in ACTIONS:
        for reason in ("valid", "goal_not_reached"):
            require(directions[action+"/"+reason] > 0, f"{split}/{action}/{reason}")
    vrows = [r for r in rows if r["task"] == "verification"]
    shared = [r for r in vrows if r["shared_g_id"]]
    shared_moves = [r for r in shared if r["mode"] == "v_step" and r["candidate"]]
    return {"modes": modes, "labels": dict(hist), "groups": Counter(v[0]["group_kind"] for v in groups.values()),
            "coverage": done, "directions": directions, "full_path_kinds": kinds,
            "explicit_shared_G_anchors_by_V_label": anchors, "G_states_also_in_V": len({r["state_id"] for r in shared}),
            "V_rows_also_in_G": len(shared), "all_shared_V_labels": Counter(r["labels"]["reason"] or "valid" for r in shared),
            "shared_step_move_is_G_optimal": sum(r["candidate"] in by_id[r["shared_g_id"]]["optimal_actions"] for r in shared_moves),
            "shared_nonempty_step_attempts": len(shared_moves), "step_shortcut": shortcut_summary(step_rows),
            "unique_model_inputs": len({r["input"] for r in rows})}


def audit(directory, source, oracle):
    from .build import SOURCE_SHA
    require(all(file_hash(source / f) == h for f, h in SOURCE_SHA.items()), "frozen source hashes")
    original_manifest = json.loads((source / "manifest.json").read_bytes())
    require(original_manifest["executor_sha256"] == oracle.sha256, "upstream executor")
    raw_states = read_jsonl(directory / "states.jsonl")
    states = {r["id"]: State.from_record(r) for r in raw_states}
    require(len(states) == len(raw_states), "duplicate state")
    for raw in raw_states:
        s = states[raw["id"]]
        require(s.id == raw["id"] and tuple(sorted(set(s.obstacles))) == s.obstacles
                and s.record()["layout_id"] == raw["layout_id"], raw["id"])
    splits, layouts, input_sets, used_states = {}, {}, {}, set()
    for split in ("train", "dev", "test_id", "test_ood"):
        rows = read_jsonl(directory / f"{split}.jsonl")
        originals = read_jsonl(source / f"{split}.jsonl")
        used_states.update(r["state_id"] for r in rows)
        layouts[split] = {states[r["state_id"]].record()["layout_id"] for r in rows}
        allowed = original_manifest["layout_splits"]["test" if split.startswith("test") else split]
        require(layouts[split] <= set(allowed), f"{split} environments")
        input_sets[split] = {r["input"] for r in rows}
        if split in ("train", "dev"):
            splits[split] = audit_learning_split(rows, states, {r["id"]: r for r in originals}, oracle, split)
            require(layouts[split] == {r["layout_id"] for r in originals}, f"{split} source layout coverage")
        else:
            require(len(rows) == len(originals) == 200, split)
            for row, old in zip(rows, originals, strict=True):
                s = states[row["state_id"]]
                check_v(row, s, oracle)
                require(s == State.from_grid(Grid.from_dict(old["grid"])) and row["mode"] == "v_plan"
                        and row["task"] == "verification" and row["input"] == render("v_plan", s, old["candidate"])
                        and row["target"] == old["label"] and row["reference"] == old["target_plan"]
                        and all(row[k] == old[k] for k in ("id", "candidate", "source_id", "edits", "edit_count")),
                        row["id"])
            splits[split] = {"modes": {"v_plan": len(rows)}, "source_records_matched_in_order": len(rows),
                             "labels": Counter(r["labels"]["reason"] or "valid" for r in rows)}
        splits[split]["layouts"] = len(layouts[split])
    require(used_states == set(states), "unreferenced state")
    independent = [("train","dev"), ("train","test_id"), ("train","test_ood"),
                   ("dev","test_id"), ("dev","test_ood")]
    for a,b in independent:
        require(layouts[a].isdisjoint(layouts[b]) and input_sets[a].isdisjoint(input_sets[b]), f"{a}/{b} overlap")
    return {"schema": VERSION, "splits": splits, "unique_state_count": len(states),
            "learning_split_overlap_with_other_partitions": 0,
            "frozen_id_ood_shared_layouts": len(layouts["test_id"] & layouts["test_ood"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("data/derived"))
    parser.add_argument("--executor", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_bytes())
    require(manifest["schema"] == VERSION and manifest["template_sha256"] == template_hash(), "schema")
    require(all(file_hash(args.data / name) == h for name,h in manifest["sha256"].items()), "dataset hashes")
    result = audit(args.data, args.source, OfficialOracle(args.executor))
    require(result == json.loads((args.data / "audit.json").read_bytes()), "published metrics")
    print("All split, label, shortest-action, counterfactual, prompt and hash checks passed.")
