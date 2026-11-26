"""Freeze the Part 1 comparison using the original coordinates, prompts and PPNL cases."""

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "experiments/path_text_gv/prompts/coordinate_gcv"
LABELS = ("valid", "obstacle_collision", "out_of_bounds", "invalid_action_token", "goal_not_reached")
MOVES = ("up", "down", "left", "right")
ERROR_SETS = ("single_error", "double_error")
SPLIT_ROLES = {"dev": "model_selection", "single_error": "test", "double_error": "validation"}
SOURCES = {
    "dev": ROOT / "data/derived/dev.jsonl",
    **{s: ROOT / "data/random_errors_20261123" / (s + "_200.jsonl") for s in ERROR_SETS},
}
DATASET_REVISION = "d88f85b872560f9313ac4319cdd5fa86151c0780"
REPLACEMENT = SOURCES["single_error"].with_name("illegal_word_replacement_20261126.json")
METRIC = "exact_line_by_gold_supported_type_v1"
SEED = 42
UPDATES = 125
MODELS = {
    "flan_t5_base": (
        "google/flan-t5-base",
        "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
        "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8",
        "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a",
    ),
    "bart_base": (
        "facebook/bart-base",
        "aadd2ab0ae0c8268c7c9693540e9904811f36177",
        "1bd3ac8e5b3ac71c77cf18fbcd8b113e0bfceced94c5cfbbb9a2b4bb4781190b",
        "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
    ),
}


def read(path):
    return json.loads(path.read_bytes())


def rows_at(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def put(path, value, lines=False):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_name(path.name + ".pending")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        if lines:
            for row in value:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
    tmp.replace(path)


def templates():
    return {t: (PROMPTS / (t + ".txt")).read_text(encoding="utf-8").rstrip("\r\n") for t in "GCV"}


def point(x):
    return f"({x[0]},{x[1]})"


def gold_rows(split):
    return [r for r in rows_at(SOURCES[split]) if r.get("task", "verification") == "verification"]


def corrected_error_inputs(part, examples, corpus, source, replacement):
    path = SOURCES[part]
    original = corpus / (part.split("_")[0] + "_inputs.jsonl")
    assert digest(original) == source["files"][original.name]
    manifest = replacement["files"][path.name]
    assert source["sources"][path.relative_to(ROOT).as_posix()] == manifest["before_sha256"]
    assert digest(path) == manifest["after_sha256"]
    original_rows = rows_at(original)
    by_id = defaultdict(list)
    for edit in manifest["changes"]:
        by_id[edit["id"]].append(edit)
    assert len(by_id) == manifest["changed_rows"]
    assert len(manifest["changes"]) == manifest["replaced_action_occurrences"]
    blind = []
    for base, gold in zip(original_rows, examples, strict=True):
        assert (base["id"], base["grid"]) == (gold["id"], gold["grid"])
        actions = base["candidate"].split()
        for edit in by_id.pop(base["id"], []):
            index = edit["step"] - 1
            assert actions[index] == edit["old"]
            assert edit["new"] not in MOVES and edit["new"] == replacement["mapping"][edit["old"].upper()]
            actions[index] = edit["new"]
        candidate = " ".join(actions)
        assert candidate == gold["candidate"]
        assert all(action in MOVES or action.lower() not in MOVES for action in actions)
        blind.append(dict(base, candidate=candidate))
    assert not by_id and len(blind) == manifest["rows"]
    return blind, manifest


def prepare(out):
    from ppnl.core import ACTIONS, Grid, OfficialOracle, shortest_path
    from ppnl.data import read_jsonl
    from ppnl.policy_v3_data import distances_to_goal, optimal_moves

    start = time.perf_counter()
    corpus = ROOT / "data/coordinate_gcv_review"
    source = read(corpus / "manifest.json")
    for name, h in source["prompts_sha256"].items():
        assert digest(PROMPTS / (name + ".txt")) == h
    assert digest(corpus / "train.jsonl") == source["files"]["train.jsonl"]
    step_data = ROOT / "data/step_g_shared_4000"
    assert digest(step_data / "states.jsonl") == source["sources"]["data/step_g_shared_4000/states.jsonl"]
    states = {r["id"]: r for r in read_jsonl(step_data / "states.jsonl")}
    parents = read_jsonl(step_data / "train.jsonl")
    full = read_jsonl(corpus / "train.jsonl")
    assert len({r["id"] for r in full}) == len(full) == 36800
    oracle_path = ROOT / "vendor/ppnl/executor-point-sg.py"
    oracle = OfficialOracle(oracle_path)
    assert oracle.sha256 == read(ROOT / "data/derived/manifest.json")["executor_sha256"]
    prompts = templates()
    by_case = defaultdict(dict)
    for r in full:
        by_case[r["case"]][r["id"]] = r
    counts = Counter(r["task"] for r in full)
    if (
        len(by_case) != 4000
        or set(by_case) != {p["id"] for p in parents}
        or counts != {"G": 4000, "C": 16400, "V": 16400}
    ):
        raise ValueError("Expected all 4,000 original states and their registered G/C/V views")
    assert sum(len(p["compass"]) for p in parents) == 16000
    assert sum(len(p["lexical"]) for p in parents) == 400
    for p in parents:
        state = states[p["id"]]
        g = Grid.from_dict(dict(state, start=state["current"]))
        group = by_case[p["id"]]
        fields = dict(
            current=point(g.start),
            goal=point(g.goal),
            walls="[" + ", ".join(map(point, sorted(g.obstacles))) + "]",
        )
        row = group[p["generation"]["id"]]
        assert row["text"] == prompts["G"].format(**fields)
        assert set(row["answers"]) == set(optimal_moves(g, g.start, distances_to_goal(g)))
        assert sorted(x["candidate"] for x in p["compass"]) == sorted(ACTIONS)
        for item in p["compass"] + p["lexical"]:
            action = item["candidate"]
            result = oracle.check(g, action)
            cell = point(result.trace[0]["to"]) if action in ACTIONS else "no_coordinate"
            c, v = group[item["id"] + "/C"], group[item["id"]]
            assert c["text"] == prompts["C"].format(current=fields["current"], action=action) and c[
                "answers"
            ] == [cell]
            assert v["text"] == prompts["V"].format(**dict(fields, current=cell))
            assert v["answers"] == ["valid" if result.valid else result.reason]
    train_layouts = {states[k]["layout_id"] for k in by_case}
    profiles, split_hashes = {}, {}
    replacement = read(REPLACEMENT)
    used_layouts = set(train_layouts)
    for part, path in SOURCES.items():
        split_hashes[part] = digest(path)
        examples = gold_rows(part)
        assert len(examples) == (100 if part == "dev" else 200)
        assert len({r["id"] for r in examples}) == len(examples)
        layouts = {r["layout_id"] for r in examples}
        assert not layouts & used_layouts
        used_layouts.update(layouts)
        blind = [{k: r[k] for k in ("id", "grid", "candidate")} for r in examples]
        if part in ERROR_SETS:
            blind, corrections = corrected_error_inputs(part, examples, corpus, source, replacement)
        for r in examples:
            grid = Grid.from_dict(r["grid"])
            assert grid.layout_id == r["layout_id"] and (grid.rows, grid.cols) == (6, 6)
            result = oracle.check(grid, r["candidate"])
            assert (result.label(), result.valid, result.step, result.reason) == (
                r["label"],
                r["valid"],
                r["step"],
                r["reason"],
            )
            reference = r.get("target_plan", r.get("reference_path"))
            assert oracle.check(grid, reference).valid
            shortest = len(shortest_path(grid))
            if part == "dev":
                assert len(reference.split()) == shortest
                assert len(r["edits"]) == (0 if result.valid else 1)
            else:
                assert shortest == r["shortest_path_length"] and not result.valid
                events = r["audit_events"]
                expected = 1 if part == "single_error" else 2
                assert r["edit_count"] == len(r["edits"]) == len(events) == expected
                steps = [e["step"] for e in events]
                assert sorted(set(steps)) == steps
                assert (steps[0], events[0]["reason"]) == (result.step, result.reason)
        kinds = Counter(r["reason"] or "valid" for r in examples)
        profiles[part] = dict(
            role=SPLIT_ROLES[part], n=len(examples), layouts=len(layouts), classes=dict(kinds)
        )
        put(out / (part + ".jsonl"), blind, True)
        if part in ERROR_SETS:
            profiles[part].update(
                changed_candidates=corrections["changed_rows"],
                replaced_action_occurrences=corrections["replaced_action_occurrences"],
            )
    put(out / "state_ids.json", sorted(by_case))
    frozen = dict(
        seed=SEED,
        models=MODELS,
        checkpoint_epoch=3,
        order="C GV GV",
        updates=375,
        quota=dict(
            unit="source state",
            states=len(by_case),
            supervised_rows=len(full),
            interpretation="Instructor accepted the 4,000 original states, as relayed by the user on 2026-11-25.",
        ),
        rows_by_task=counts,
        candidate_views=dict(compass=16000, lexical=400),
        states_sha256=digest(out / "state_ids.json"),
        metric=METRIC,
        model_comparison="dev_only",
        dev_selection="mean(direct environment success, structured exact-line macro-F1, self-repair success on gold-invalid candidates)",
        family_tie_order=["structured_macro_f1", "repair_success", "flan_t5_base"],
        reference="original uniform Flan epoch 3",
        training_sources=dict(
            full=digest(corpus / "train.jsonl"),
            parents=digest(step_data / "train.jsonl"),
        ),
        corrected_error_source=dict(
            ref="hope_dataset",
            commit=DATASET_REVISION,
            replacement_manifest_sha256=digest(REPLACEMENT),
        ),
        source_splits=split_hashes,
        prompts=source["prompts_sha256"],
        inputs={p: digest(out / (p + ".jsonl")) for p in profiles},
        held_out=profiles,
        train_layouts=len(train_layouts),
        V_labels=dict(Counter(r["answers"][0] for r in full if r["task"] == "V")),
        official_executor=oracle.sha256,
    )
    put(out / "protocol.json", frozen)
    put(out / "preparation_time.json", dict(cpu_wall_seconds=time.perf_counter() - start))
    return frozen


def schedule(rows):
    groups = {s: defaultdict(list) for s in ("C", "GV")}
    actions = {}
    for i, r in enumerate(rows):
        groups["C" if r["task"] == "C" else "GV"][r["case"]].append(i)
        if r["task"] == "C":
            actions[r["id"][:-2]] = r["text"].splitlines()[1].removeprefix("Action: ")
    cases = sorted(groups["C"])
    assert set(cases) == set(groups["GV"]) and len(cases) >= UPDATES
    cells = Counter((r["answers"][0], actions[r["id"]]) for r in rows if r["task"] == "V")
    words = Counter(label for label, action in cells)
    assert set(words) == set(LABELS)
    weights = []
    for r in rows:
        if r["task"] == "C":
            w = Fraction(len(cases), len(actions))
        elif r["task"] == "G":
            w = Fraction(1, 2)
        else:
            label = r["answers"][0]
            w = Fraction(len(cases), 10 * words[label] * cells[label, actions[r["id"]]])
        weights.append(float(w))
    streams = {}
    for name, passes in (("C", 1), ("GV", 2)):
        stream = []
        for cycle in range(passes):
            rng = random.Random(SEED + (0 if name == "C" else 100) + cycle)
            order = cases.copy()
            rng.shuffle(order)
            for j in range(UPDATES):
                batch = [
                    i
                    for case in order[j * len(cases) // UPDATES : (j + 1) * len(cases) // UPDATES]
                    for i in groups[name][case]
                ]
                rng.shuffle(batch)
                stream.append(batch)
        streams[name] = stream
    batches = []
    for j in range(UPDATES):
        batches.extend(
            [
                ("C", j, streams["C"][j]),
                ("GV", 2 * j, streams["GV"][2 * j]),
                ("GV", 2 * j + 1, streams["GV"][2 * j + 1]),
            ]
        )
    return batches, weights, len(cases) / UPDATES


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/part1_baseline/data")
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), indent=2))
