"""Recompute the labels and check the actual five-class training features on CPU."""
import argparse
from collections import Counter, deque
from pathlib import Path

from transformers import AutoTokenizer
from ppnl.core import ACTIONS, Grid, execute
from ppnl.data import read_jsonl, write_json
from ppnl.experiment_freeze import file_hash
from ppnl.policy import NAMES
from step_data_v3 import g_probes
from step_data_v3.five_class import CLASS_LABELS, FiveClassRunner
from step_data_v3.schema import State, render
from step_data_v3.train_gpu import split, validate_inputs


def shortest_set(s):
    walls = set(s.obstacles)
    dist, queue = {s.goal: 0}, deque([s.goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTIONS.values():
            p = (r + dr, c + dc)
            if 0 <= p[0] < s.rows and 0 <= p[1] < s.cols and p not in walls and p not in dist:
                dist[p] = 1 + dist[(r, c)]
                queue.append(p)
    r, c = s.current
    return {a for a, (dr, dc) in ACTIONS.items() if dist.get((r+dr, c+dc)) == dist[s.current] - 1}


def step_label(s, a):
    # The executor starts from a legal witness and also handles states already at the goal.
    grid, prefix = s.witness()
    verdict = execute(grid, " ".join(prefix + [a]))
    return "valid" if verdict.valid else verdict.reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "pretrained", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    states = validate_inputs(args.data, args.pretrained)
    runner = FiveClassRunner.__new__(FiveClassRunner)
    runner.tokenizer = AutoTokenizer.from_pretrained(args.pretrained, local_files_only=True)
    runner.input_limit = 128
    report = {"manifest_sha256": file_hash(args.data / "manifest.json"), "prompt_sha256": g_probes.template_hash(),
              "tokenizer_sha256": file_hash(args.pretrained / "tokenizer.json"), "class_order": list(CLASS_LABELS)}
    prompts, layouts = {}, {}
    for name in ("train", "dev"):
        rows = split(args.data, name)
        features = runner.encode_rows(rows)
        prompts[name] = {r["input"] for r in rows}
        layouts[name] = {states[r["state_id"]].record()["layout_id"] for r in rows}
        v_classes, n_multi_g, n_g = Counter(), 0, 0
        for row, feature in zip(rows, features, strict=True):
            state = states[row["state_id"]]
            if row["mode"] == "v_step":
                a = row["candidate"]
                assert a.split() == [a] and row["input"] == g_probes.step_input(state, a)
                assert CLASS_LABELS[feature["labels"]] == row["target"] == step_label(state, a)
                assert not any(feature["policy_actions"])
                v_classes[row["target"]] += 1
            else:
                assert row["input"] == render("g_next", state) and feature["labels"] == -100
                actions = shortest_set(state)
                assert actions == set(row["optimal_actions"]) == {a for a, allowed in
                    zip(NAMES, feature["policy_actions"], strict=True) if allowed}
                n_g += 1
                n_multi_g += len(actions) > 1
        report[name] = {"V_label_and_index_match": dict(v_classes), "G_shortest_sets_match": n_g,
                        "G_multi_optimal": n_multi_g, "max_encoded_input_tokens": max(len(f["input_ids"]) for f in features)}
    assert not (prompts["train"] & prompts["dev"])
    report["exact_train_dev_input_overlap"] = len(prompts["train"] & prompts["dev"])
    for name in ("test_id", "test_ood"):
        rows = split(args.data, name)
        layouts[name] = {r["layout_id"] for r in rows}
        assert all(execute(Grid.from_dict(r["grid"]), r["candidate"]).label() == r["label"] for r in rows)
        report[name] = {"full_path_oracle_matches": len(rows)}
    for name in ("dev", "test_id", "test_ood"):
        assert not (layouts[name] & layouts["train"])
    report["exact_layout_overlap_with_train"] = {name: len(layouts[name] & layouts["train"]) for name in
                                                ("dev", "test_id", "test_ood")}
    probes = read_jsonl(args.data / "diagnostics/g_dev_four.jsonl")
    assert all(step_label(states[r["state_id"]], r["candidate"]) == r["target"] for r in probes)
    report["G_dev_probe_reference_moves"] = len(probes)
    write_json(args.out, report)


if __name__ == "__main__":
    main()
