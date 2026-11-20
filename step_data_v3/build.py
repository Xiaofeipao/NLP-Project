"""Build the training/development sets and fixed-candidate evaluation views."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from ppnl.core import Grid, OfficialOracle, REASONS, execute
from ppnl.data import read_jsonl
from ppnl.experiment_freeze import file_hash
from ppnl.policy_v3_data import optimal_moves
from .schema import VERSION, TASK, State, render, step_labels, template_hash
from .sampling import (BAD_WORDS, DIST, Donors, forced_detour, longer_path, move_pair,
                       random_path, single_error)


V2_COMMIT = "9eeb494fe1f775ad033ce2f668e22517cd194aee"
SOURCE_SHA = {
    "train.jsonl": "d1212c424d5e1dcdc7d99fd0520ce9d28ab7ef292c7c5121a3dd4cc54d5842a2",
    "dev.jsonl": "6b60713aa88053bacbe2a009c68baf69d51a1f3121390d0aea1c7f256a92bb3a",
    "test_id.jsonl": "26b5f561b9fd7b3bd5e963b0186abdf0b42cb7da726f20833b229969900220c1",
    "test_ood.jsonl": "c3e5b77f488931c15708ffdd32e88f725cda710f6eb3df517767fe1e390ba80c",
    "manifest.json": "b2b1a83ff6f62d0d056fd50b7b2a8ab3c1b824bbe5356000589243343a83e766",
}
STEP_GROUPS = {"move_goal": 6, "stop_goal": 4, "wall": 4, "edge": 4, "token": 4,
               "arrive": 3, "continue": 1, "collision": 4, "bounds": 4, "unknown": 4}
OPERATIONS = {"obstacle_collision": ("insert", "replace", "delete"),
              "out_of_bounds": ("insert", "replace"),
              "invalid_action_token": ("replace", "insert"),
              "goal_not_reached": ("truncate", "delete", "replace", "insert")}
LONG_KINDS = ("backtrack", "square", "simple_detour")


def write_lines(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_object(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def plan_row(state, plan, donor):
    text = " ".join(plan)
    verdict = execute(state.grid(), text)
    labels = {"valid": verdict.valid, "step": verdict.step, "reason": verdict.reason}
    return dict(mode="v_plan", state=state, candidate=text, donor_id=donor,
                target=verdict.label(), labels=labels)


class SplitBuilder:
    def __init__(self, rows, split, seed, units):
        self.pool, self.split, self.units = Donors(rows), split, units
        self.rng = random.Random(seed)
        self.words = BAD_WORDS if split == "train" else (*BAD_WORDS, "hop", "pivot")
        self.rows, self.states, self.inputs = [], {}, set()
        self.groups = 0
        self.remaining_layouts = set(self.pool.layouts)

    def add(self, examples, group_kind):
        if not examples:
            return False
        texts = [render(e["mode"], e["state"], e.get("candidate")) for e in examples]
        if len(set(texts)) != len(texts) or not self.inputs.isdisjoint(texts):
            return False
        self.groups += 1
        for example, prompt in zip(examples, texts, strict=True):
            state = example.pop("state")
            self.states[state.id] = state
            self.remaining_layouts.discard(state.record()["layout_id"])
            record = dict(id=f"{self.split}-{len(self.rows)+1:04}", task=TASK[example["mode"]],
                          state_id=state.id, group_id=f"{self.split}-group-{self.groups:04}",
                          group_kind=group_kind, input=prompt, **example)
            self.rows.append(record)
        self.inputs.update(texts)
        return True

    def take(self, count, kind, make):
        added = 0
        for _ in range(count * 600 + 2000):
            if self.add(make(added), kind):
                added += 1
                if added == count:
                    return
        raise ValueError(f"{self.split}/{kind}: generated {added}/{count} groups")

    def step_examples(self, family):
        result = []
        if family in ("move_goal", "arrive", "continue", "collision", "bounds"):
            attempts = self.pool.observed(family, self.rng)
        else:
            attempts = [(*a, None) for a in move_pair(family, self.pool, self.rng, self.words)]
        for state, action, donor, observation in attempts:
            target, labels = step_labels(state, action)
            row = dict(mode="v_step", state=state, candidate=action, donor_id=donor,
                       target=target, labels=labels)
            if observation is not None:
                row["source_observation"] = observation
            result.append(row)
        return result

    def plan_examples(self, reason, index):
        grid, donor = self.pool.sample(self.rng)
        state = State.from_grid(grid)
        base = random_path(grid, self.rng)
        ops = OPERATIONS[reason]
        damaged = single_error(grid, base, reason, ops[index % len(ops)], self.rng, self.words)
        if damaged is None:
            return []
        changed, edit = damaged
        # Alternate the positive's length class within each reason, across the whole split.
        ordinal = index * len(REASONS) + REASONS.index(reason)
        use_long = (index + REASONS.index(reason)) % 2 == 1
        kind = LONG_KINDS[(ordinal // 2) % len(LONG_KINDS)] if use_long else "shortest"
        positive = longer_path(grid, base, kind, self.rng) if use_long else base
        if positive is None:
            return []
        return [dict(plan_row(state, positive, donor), valid_kind=kind, reference=" ".join(base)),
                dict(plan_row(state, changed, donor), edit=edit, reference=" ".join(base))]

    def generation(self, state, donor, anchor=None):
        grid = state.grid()
        moves = optimal_moves(grid, grid.start, DIST(grid))
        return dict(mode="g_next", state=state, donor_id=donor, target=self.rng.choice(moves),
                    optimal_actions=moves, distance=DIST(grid)[grid.start], anchor_v_id=anchor)

    def new_generation(self, kind):
        layout = None
        if self.remaining_layouts and (kind == "ordinary" or (kind == "tie" and self.rng.random() < 0.8)):
            layout = self.rng.choice(sorted(self.remaining_layouts))
        grid, donor = self.pool.sample(self.rng, layout)
        distances = DIST(grid)
        points = []
        for position, distance in distances.items():
            if distance == 0:
                continue
            moves = optimal_moves(grid, position, distances)
            if kind == "forced" and not forced_detour(grid, position, moves):
                continue
            if kind == "tie" and len(moves) < 2:
                continue
            points.append(position)
        if not points:
            return []
        return [self.generation(State.from_grid(grid, self.rng.choice(points)), donor)]

    def shared_generation(self):
        by_reason = defaultdict(list)
        for row in self.rows:
            state = self.states[row["state_id"]]
            if state.current != state.goal:
                by_reason[row["labels"]["reason"] or "valid"].append(row)
        for bag in by_reason.values():
            self.rng.shuffle(bag)
        # The five anchor quotas apply to distinct G states. One state may also serve other V rows.
        for _ in range(7 * self.units):
            for reason in ("valid", *REASONS):
                bag = by_reason[reason]
                while bag:
                    v = bag.pop()
                    g = self.generation(self.states[v["state_id"]], v["donor_id"], v["id"])
                    if self.add([g], "shared"):
                        break
                else:
                    raise ValueError(f"Insufficient distinct G/V anchors for {reason}")

    def build(self):
        for family, count in STEP_GROUPS.items():
            self.take(count * self.units, family, lambda _, f=family: self.step_examples(f))
        for reason in REASONS:
            self.take(5 * self.units, "plan_pair", lambda i, r=reason: self.plan_examples(r, i))
        self.shared_generation()
        for kind, count in (("forced", 25), ("tie", 20), ("ordinary", 20)):
            self.take(count * self.units, kind, lambda _, k=kind: self.new_generation(k))
        generations = {r["state_id"]: r["id"] for r in self.rows if r["mode"] == "g_next"}
        for row in self.rows:
            if row["mode"] != "g_next":
                row["shared_g_id"] = generations.get(row["state_id"])
        self.rng.shuffle(self.rows)
        return self.rows, self.states


def fixed_view(source_rows, states):
    rows = []
    for original in source_rows:
        state = State.from_grid(Grid.from_dict(original["grid"]))
        states[state.id] = state
        row = plan_row(state, original["candidate"].split(), original["id"])
        row.pop("state")
        rows.append(dict(id=original["id"], task="verification", state_id=state.id,
                         source_id=original["source_id"], edit_count=original["edit_count"],
                         edits=original["edits"], reference=original["target_plan"],
                         input=render("v_plan", state, original["candidate"]), **row))
        if rows[-1]["target"] != original["label"] or rows[-1]["candidate"] != original["candidate"]:
            raise ValueError(original["id"])
    return rows


def build(source, executor, output, seed=42):
    from .audit import audit
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if {f: file_hash(source / f) for f in SOURCE_SHA} != SOURCE_SHA:
        raise ValueError("Expected the frozen StepVerification-v1-v2 source files")
    oracle = OfficialOracle(executor)
    if oracle.sha256 != json.loads((source / "manifest.json").read_bytes())["executor_sha256"]:
        raise ValueError("Unexpected official executor")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        states = {}
        for split, units in (("train", 20), ("dev", 1)):
            rows, registry = SplitBuilder(read_jsonl(source / f"{split}.jsonl"), split,
                                          seed if split == "train" else seed + 1, units).build()
            write_lines(stage / f"{split}.jsonl", rows)
            states.update(registry)
        for split in ("test_id", "test_ood"):
            frozen = fixed_view(read_jsonl(source / f"{split}.jsonl"), states)
            write_lines(stage / f"{split}.jsonl", frozen)
        write_lines(stage / "states.jsonl", (s.record() for _, s in sorted(states.items())))
        report = audit(stage, source, oracle)
        write_object(stage / "audit.json", report)
        write_object(stage / "manifest.json", {"schema": VERSION, "template_sha256": template_hash(), "base_commit": V2_COMMIT,
                     "seed": seed, "dev_seed": seed + 1, "source_sha256": SOURCE_SHA,
                     "executor_sha256": oracle.sha256, "sha256": {
                         p.name: file_hash(p) for p in sorted(stage.iterdir())}})
        stage.rename(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/derived"))
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = build(args.source, args.executor, args.out, args.seed)
    print({split: stats["modes"] for split, stats in result["splits"].items()})
