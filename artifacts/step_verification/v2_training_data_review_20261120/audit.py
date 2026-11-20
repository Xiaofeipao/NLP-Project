"""Reproduce the v2 training-data review from frozen Git blobs; Python standard library."""
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
V2 = "9eeb494fe1f775ad033ce2f668e22517cd194aee"
SHORTCUT = "ba1620f01fdc255d0f590180b3b4192e75a6cf67"
PART23 = "5f06e80b2302c67fcafac3b0920c36ed1d4c3c61"
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
HORIZONTAL = {"left", "right"}
VERTICAL = {"up", "down"}


def source(commit, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{path}"])


def rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def layout(grid):
    return grid["rows"], grid["cols"], tuple(sorted(map(tuple, grid["obstacles"])))


def transition(shape, position, action):
    height, width, walls = shape
    if action not in MOVES:
        return position, "invalid_action_token"
    dr, dc = MOVES[action]
    dest = position[0] + dr, position[1] + dc
    if not (0 <= dest[0] < height and 0 <= dest[1] < width):
        return position, "out_of_bounds"
    if dest in walls:
        return position, "obstacle_collision"
    return dest, None


def local_target(shape, goal, position, action):
    if action is None:
        return "valid" if position == goal else "invalid | step=1 | reason=goal_not_reached"
    dest, error = transition(shape, position, action)
    if error:
        return f"invalid | step=1 | reason={error}"
    return "valid" if dest == goal else "invalid | step=2 | reason=goal_not_reached"


def replay(grid, plan):
    shape, goal = layout(grid), tuple(grid["goal"])
    position = tuple(grid["start"])
    positions, labels = [position], []
    actions = plan.split()
    for i, action in enumerate(actions, 1):
        labels.append(local_target(shape, goal, position, action))
        position, error = transition(shape, position, action)
        if error:
            return f"invalid | step={i} | reason={error}", positions, labels
        positions.append(position)
    if not actions:
        labels.append(local_target(shape, goal, position, None))
    result = "valid" if position == goal else f"invalid | step={len(actions) + 1} | reason=goal_not_reached"
    return result, positions, labels


@lru_cache(maxsize=None)
def shortest(shape, origin, order=tuple(MOVES)):
    depth, last_action = {origin: 0}, {}
    queue = deque([origin])
    while queue:
        current = queue.popleft()
        for action in order:
            nxt, error = transition(shape, current, action)
            if error is None and nxt not in depth:
                depth[nxt] = depth[current] + 1
                last_action[nxt] = action
                queue.append(nxt)
    return depth, last_action


def optimal_moves(shape, goal, position):
    depth, _ = shortest(shape, goal)
    optimal = set()
    for action in MOVES:
        dest, error = transition(shape, position, action)
        if error is None and depth[dest] == depth[position] - 1:
            optimal.add(action)
    return optimal


def must_move_away(shape, goal, position):
    def distance(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
    optimal = optimal_moves(shape, goal, position)
    return bool(optimal) and all(distance(transition(shape, position, a)[0]) > distance(position) for a in optimal)


def verification_examples(training):
    for row in training:
        if "state_verification" in row:
            label = row["state_verification"]["local_stop_target"]
            yield row["grid"], tuple(row["policy_position"]), None, label
        if "verification_position" in row:
            yield row["grid"], tuple(row["verification_position"]), row["action"], row["target"]


def check_text_inputs(training):
    """Re-render the archived inputs with the original commit's own text functions."""
    modules = (
        "ppnl/__init__.py", "ppnl/core.py", "ppnl/policy.py",
        "state_baselines/step_verification.py",
    )
    counts = Counter()
    targets_by_input = defaultdict(set)
    with tempfile.TemporaryDirectory() as directory:
        for name in modules:
            path = Path(directory, name)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(source(V2, name))
        sys.path.insert(0, directory)
        try:
            from ppnl.core import Grid
            from ppnl.policy import state_prompt
            from state_baselines.step_verification import step_prompt

            for row in training:
                grid = Grid.from_dict(row["grid"])
                text = row["input"]
                if row["task"] == "generation":
                    expected = state_prompt(grid, tuple(row["policy_position"]), local_status=True)
                else:
                    expected = step_prompt(grid, row["verification_position"], row["action"])
                    targets_by_input[text].add(row["target"])
                counts[row["task"] + "_exact_render"] += text == expected
                counts["literal_record_id_in_input"] += row["id"] in text
                counts["carriage_return_or_tab_in_input"] += "\r" in text or "\t" in text
            counts["direct_V_distinct_inputs"] = len(targets_by_input)
            counts["direct_V_conflicting_identical_inputs"] = sum(len(v) > 1 for v in targets_by_input.values())
        finally:
            sys.path.remove(directory)
    return dict(counts)


def paired_coverage(training):
    # A group, not a row count. The V prompt does not contain the original route's start.
    changed_goal, changed_wall = defaultdict(set), defaultdict(set)
    for grid, position, action, target in verification_examples(training):
        if action not in MOVES:
            continue
        shape, goal = layout(grid), tuple(grid["goal"])
        label = target.split("reason=")[-1]
        if label in {"valid", "goal_not_reached"}:
            changed_goal[shape, position, action].add(label)
        dr, dc = MOVES[action]
        cell = position[0] + dr, position[1] + dc
        if label in {"goal_not_reached", "obstacle_collision"}:
            # Same action and position, goal, size and all other walls.
            other_walls = tuple(p for p in shape[2] if p != cell)
            changed_wall[shape[:2], other_walls, goal, position, action].add(label)
    return {
        "move_only_goal_changed_both_arrival_labels": sum(len(v) == 2 for v in changed_goal.values()),
        "move_only_destination_wall_changed_free_nonterminal_and_collision": sum(len(v) == 2 for v in changed_wall.values()),
    }


def positive_paths(verification, parents):
    stats = Counter()
    for row in verification:
        if not row["valid"]:
            continue
        grid = row["grid"]
        start, goal = tuple(grid["start"]), tuple(grid["goal"])
        shape = layout(grid)
        depth, original = shortest(shape, start)
        assert original[goal] == row["action"]
        possible = set()
        for action, (dr, dc) in MOVES.items():
            if depth.get((goal[0] - dr, goal[1] - dc)) == depth[goal] - 1:
                possible.add(action)
        stats["valid_rows"] += 1
        stats["horizontal_last_move"] += row["action"] in HORIZONTAL
        if possible & HORIZONTAL and possible & VERTICAL:
            stats["both_final_axes_shortest"] += 1
            stats["published_horizontal_among_those"] += row["action"] in HORIZONTAL
            _, alternative = shortest(shape, start, ("left", "right", "up", "down"))
            stats["horizontal_with_reversed_axis_priority"] += alternative[goal] in HORIZONTAL
        parent = parents[row["parent_id"]]
        if parent["valid_kind"] == "nonoptimal":
            actions, reference = parent["candidate"].split(), parent["target_plan"].split()
            stats["nonoptimal_positive_rows"] += 1
            _, positions, labels = replay(grid, parent["target_plan"])
            dr, dc = MOVES[actions[0]]
            stats["nonoptimal_only_add_two_inverse_initial_actions"] += (
                MOVES[actions[1]] == (-dr, -dc) and actions[2:] == reference
            )
            stats["same_last_input_and_target_as_shortest"] += (
                tuple(row["verification_position"]) == positions[-2]
                and row["action"] == reference[-1] and row["target"] == labels[-1]
            )
    return dict(stats)


def parent_coverage(parent_rows):
    candidates = [r for r in parent_rows if "candidate" in r]
    counts = Counter()
    for row in candidates:
        result, _, labels = replay(row["grid"], row["candidate"])
        assert row["label"] == result
        counts.update(labels)
    return {
        "candidate_count": len(candidates),
        "max_actions": max(len(r["candidate"].split()) for r in candidates),
        "local_observations_to_first_failure_or_end": sum(counts.values()),
        "legal_nonterminal_real_action_observations": counts["invalid | step=2 | reason=goal_not_reached"],
    }


def planning_failures(records):
    counts = Counter()
    for row in records:
        result, positions, _ = replay(row["grid"], row["direct"])
        assert row["gold"] == result
        if result == "valid":
            continue
        counts["failed"] += 1
        counts["exhausted_36"] += len(row["direct"].split()) == 36
        final_two = positions[-2:]
        if len(positions) >= 7 and final_two[0] != final_two[1] and positions[-7:] == [positions[-1]] + final_two * 3:
            counts["alternating_two_cells_for_last_six_moves"] += 1
            grid = row["grid"]
            counts["such_cycles_with_a_state_requiring_moving_away"] += any(
                must_move_away(layout(grid), tuple(grid["goal"]), p) for p in final_two
            )
    return dict(counts)


def token_only_prediction(plan):
    actions = plan.split()
    if not actions:
        return "invalid | step=1 | reason=goal_not_reached"
    if "jump" in actions:
        return f"invalid | step={actions.index('jump') + 1} | reason=invalid_action_token"
    return "valid"


def main():
    evidence = zipfile.ZipFile(io.BytesIO(source(V2, "deliverables/step_verification_v1_v2_evidence.zip")))
    raw = evidence.read("v2/train.jsonl")
    selection = json.loads(evidence.read("v2/results/selection.json"))
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == selection["identity"]["dual_verification"]["train_sha256"]
    training = rows(raw)
    assert len({r["id"] for r in training}) == len(training)
    direct = [r for r in training if r["task"] == "verification"]
    generation = [r for r in training if r["task"] == "generation"]
    for grid, position, action, target in verification_examples(training):
        assert local_target(layout(grid), tuple(grid["goal"]), position, action) == target
    for r in generation:
        assert optimal_moves(layout(r["grid"]), tuple(r["grid"]["goal"]), tuple(r["policy_position"])) == set(r["optimal_actions"])
    splits = {s: rows(source(V2, f"data/policy_v3/{s}.jsonl")) for s in ("train", "dev", "test_id", "test_ood")}
    layouts = {s: {layout(r["grid"]) for r in records} for s, records in splits.items()}
    parents = {r["id"]: r for r in splits["train"] if "candidate" in r}
    report = {
        "commits": {"v2": V2, "new_text_v": SHORTCUT, "part2_part3": PART23},
        "training_sha256": digest,
        "records": {"G": len(generation), "direct_V": len(direct)},
        "text_format": check_text_inputs(training),
        "verified_training_targets": {"G_optimal_action_sets": len(generation), "V_direct": len(direct), "V_auxiliary_stop": len(generation)},
        "G_parent_task": Counter(r["parent_task"] for r in generation),
        "direct_V_by_class": Counter(r["reason"] or "valid" for r in direct),
        "original_candidates_when_replayed": {s: parent_coverage(records) for s, records in splits.items()},
        "direct_V_illegal_word_tokens": Counter(r["action"] for r in direct if r["reason"] == "invalid_action_token"),
        "positive_path_sampling": positive_paths(direct, parents),
        "paired_training_groups": {"v2": paired_coverage(training)},
        "train_grid_shapes": sorted({str((r["grid"]["rows"], r["grid"]["cols"])) for r in training}),
        "train_obstacles": Counter(len(r["grid"]["obstacles"]) for r in training),
        "distinct_layouts": {s: len(v) for s, v in layouts.items()},
        "train_layouts_shared_with_dev_or_test": len(layouts["train"] & (layouts["dev"] | layouts["test_id"] | layouts["test_ood"])),
        "G_training_states_where_all_shortest_next_moves_increase_Manhattan": sum(
            must_move_away(layout(r["grid"]), tuple(r["grid"]["goal"]), tuple(r["policy_position"])) for r in generation
        ),
        "G_rollout_failures": {s: planning_failures(rows(evidence.read(f"v2/results/{s}/flan_t5_base/planning.jsonl"))) for s in ("dev", "test_id")},
    }
    tests = {}
    for s in ("test_id", "test_ood"):
        predictions = rows(evidence.read(f"v2/results/{s}/flan_t5_base/candidates.jsonl"))
        truth = Counter(r["reason"] or "valid" for r in predictions)
        correct = Counter(r["reason"] or "valid" for r in predictions if r["self"]["feedback"].split("reason=")[-1] == r["gold"].split("reason=")[-1])
        tests[s] = {"truth": truth, "correct_failure_type_including_valid": correct}
    report["fixed_candidate_predictions"] = tests
    report["three_rule_prediction_exact_agreement"] = {}
    for split in ("dev", "test_id", "test_ood"):
        candidates = rows(evidence.read(f"v2/results/{split}/flan_t5_base/candidates.jsonl"))
        report["three_rule_prediction_exact_agreement"][split] = {
            "n": len(candidates),
            "same_as_archived_model_feedback": sum(
                token_only_prediction(r["direct"]) == r["self"]["feedback"] for r in candidates
            ),
        }
    lexical = json.loads(source(SHORTCUT, "deliverables/newpart1_text_verifier/shortcut/lexical_baseline.json"))
    report["V2_direct_only_action_lookup_F1"] = {s: lexical["v2"]["candidate_metrics"][s]["macro_f1"] for s in tests}
    report["V2_learned_F1"] = {
        s: json.loads(evidence.read(f"v2/results/{s}/flan_t5_base/metrics.json"))["fixed_candidates"]["verification"]["macro_f1"]
        for s in tests
    }
    for arm in ("shortcut", "hierarchical"):
        packed = source(SHORTCUT, f"deliverables/newpart1_text_verifier/{arm}/evidence.tar.gz")
        with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as files:
            report["paired_training_groups"][arm] = paired_coverage(rows(files.extractfile("./balanced.jsonl").read()))
    for arm in ("7x7", "dense_6x6"):
        held_out = rows(source(PART23, f"data/generalization_v1/test_{arm}.jsonl"))
        report.setdefault("part2_part3_held_out", {})[arm] = {
            "n": len(held_out), "obstacle_counts": Counter(len(r["grid"]["obstacles"]) for r in held_out)
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
