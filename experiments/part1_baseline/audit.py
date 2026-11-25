"""Offline provenance of the original source states and actual prompt reuse."""

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from ppnl.core import Grid, OfficialOracle
from .protocol import ERROR_SETS, METRIC, MODELS, ROOT, SOURCES, digest, gold_rows, put, read, rows_at


def audit(work):
    start = perf_counter()
    inputs = ROOT / "data/coordinate_gcv_review/train.jsonl"
    rows = rows_at(inputs)
    base = ROOT / "data/derived/train.jsonl"
    origins = {r["id"]: r for r in rows_at(base)}
    parent_file = ROOT / "data/step_g_shared_4000/train.jsonl"
    parents = {r["id"]: r for r in rows_at(parent_file)}
    oracle = OfficialOracle(ROOT / "vendor/ppnl/executor-point-sg.py")
    ids, edits, source_types = [], Counter(), Counter()
    for ident in sorted({r["case"] for r in rows}):
        p = parents[ident]
        ref = p["source"]
        key = ref.get("donor_id", ref.get("row_id"))
        donor = origins[key]
        assert donor["layout_id"] == p["layout_id"]
        grid = Grid.from_dict(donor["grid"])
        parent_plan = donor["target"] if donor["task"] == "generation" else donor["target_plan"]
        assert oracle.check(grid, parent_plan).valid
        n = donor.get("edit_count")
        kind = "generation"
        if n is not None:
            answer = oracle.check(grid, donor["candidate"])
            assert answer.label() == donor["label"] and n == len(donor["edits"]) and n in (0, 1)
            kind = donor["valid_kind"] if n == 0 else donor["edits"][0]["kind"]
            assert answer.valid == (n == 0)
            if n:
                edits[kind] += 1
        source_types["generation" if n is None else ("single_edit" if n else "valid_" + kind)] += 1
        ids.append(
            dict(
                case=ident,
                layout_id=p["layout_id"],
                source=ref,
                official_training_row=key,
                source_id=donor["source_id"],
                source_kind=kind,
                source_edit_count=n,
            )
        )
    by = defaultdict(dict)
    tallies = Counter()
    for r in rows:
        t, s = r["task"], r["text"]
        gold = tuple(sorted(r["answers"]))
        if s in by[t] and by[t][s] != gold:
            raise ValueError("Conflicting targets for identical input")
        by[t][s] = gold
        tallies[t] += 1
    lineage = work / "data/state_lineage.jsonl"
    put(lineage, ids, True)
    report = dict(
        training=digest(inputs),
        parents=digest(parent_file),
        official_train=digest(base),
        records=dict(tallies),
        distinct_text={t: len(v) for t, v in by.items()},
        multi_answer_G=sum(len(x) > 1 for x in by["G"].values()),
        cases=len(ids),
        distinct_original_training_rows=len({r["official_training_row"] for r in ids}),
        cases_by_original_source=source_types,
        single_edits=edits,
        lineage=digest(lineage),
        cpu_wall_seconds=perf_counter() - start,
    )
    put(work / "data/state_audit.json", report)
    return report


def audit_results(work, history=None):
    from ppnl.evaluation import verification_metrics
    from .evaluate import canonical_verdict, env_key, grade
    from .study import LEGACY_WEIGHTS, development, winner

    start = perf_counter()
    completion = read(work / "study_done.json")
    selection = read(work / "selection.json")
    protocol = read(work / "data/protocol.json")
    assert completion["selection_sha256"] == digest(work / "selection.json")
    assert selection["protocol"] == digest(work / "data/protocol.json")
    assert selection["dev_inputs"] == protocol["inputs"]["dev"] and protocol["metric"] == METRIC
    assert set(completion["test_receipts"]) == set(ERROR_SETS)
    oracle = OfficialOracle(ROOT / "vendor/ppnl/executor-point-sg.py")
    primary = selection["primary_model"]
    checked, replay, project_metrics = {}, {}, {}
    for family in MODELS:
        model = work / "models/expanded" / family
        wanted = selection["scores"][family]
        assert digest(model / "model.safetensors") == wanted["weights"]
        assert digest(model / "tokenizer.json") == MODELS[family][3]
        splits = ["dev", *ERROR_SETS] if family == primary else ["dev"]
        family_dir = work / "scores" / family
        assert {p.name for p in family_dir.iterdir() if p.is_dir()} == set(splits)
        for split in splits:
            output = family_dir / split
            receipt = read(output / "done.json")
            assert receipt["weights"] == wanted["weights"] and receipt["inputs"] == protocol["inputs"][split]
            assert receipt["protocol"] == selection["protocol"] and receipt["metric"] == METRIC
            assert all(digest(output / name) == h for name, h in receipt["files"].items())
            assert digest(SOURCES[split]) == protocol["source_splits"][split]
            gold = gold_rows(split)
            raw = rows_at(output / "candidate_predictions.jsonl")
            env = rows_at(output / "generated_predictions.jsonl")
            source_by_id = {c["id"]: c for c in gold}
            assert len(env) == len({env_key(c) for c in gold})
            assert {env_key(source_by_id[c["id"]]) for c in env} == {env_key(c) for c in gold}
            full = split in ERROR_SETS
            assert receipt["full"] == full
            assert receipt["selection"] == (completion["selection_sha256"] if full else None)
            metrics, cases, environments, _ = grade(oracle, gold, raw, env, full)
            assert metrics == read(output / "metrics.json") and cases == rows_at(
                output / "candidate_scores.jsonl"
            )
            assert environments == rows_at(output / "generated_scores.jsonl")
            verification = metrics["verification"]
            assert verification["exact"] == metrics["joint"] + metrics["valid_correct"]
            assert verification["exact"] == sum(z["gold"] == z["prediction"] for z in cases)
            verdicts = [z["prediction"] if canonical_verdict(z["prediction"]) else "malformed" for z in cases]
            binary = verification_metrics(gold, verdicts)
            project_metrics[f"{family}/{split}"] = dict(
                candidates=len(cases),
                gold_valid=metrics["valid_count"],
                gold_invalid=metrics["invalid_count"],
                binary_macro_f1=binary["macro_f1"],
                invalid_detected=sum(
                    not g["valid"] and v.startswith("invalid |") for g, v in zip(gold, verdicts, strict=True)
                ),
                first_error_localization_accuracy=binary["first_error_localization_accuracy"],
                structured_macro_f1=verification["structured_macro_f1"],
                exact_line_accuracy=verification["exact_accuracy"],
                reason_correct=metrics["reason_correct"],
                step_correct=metrics["step_correct"],
                candidate_repair=metrics["candidate_repair"],
                environment_pipelines=metrics["pipeline"],
            )
            checked[f"{family}/{split}"] = dict(
                candidates=len(cases),
                environments=len(environments),
                exact=verification["exact"],
                supported_types=verification["supported_types"],
            )
            if full:
                assert completion["test_receipts"][split] == digest(output / "done.json")
                if history is not None and wanted["weights"] == LEGACY_WEIGHTS:
                    old = history / (split.split("_")[0] + "_predictions.jsonl")
                    old_rows = rows_at(old)
                    assert [r["id"] for r in old_rows] == [r["id"] for r in raw]
                    replay[split] = dict(
                        identical=sum(a == b for a, b in zip(raw, old_rows, strict=True)),
                        total=len(raw),
                        original_sha256=digest(old),
                    )
            else:
                assert receipt["files"]["metrics.json"] == wanted["metrics_sha256"]
                assert development(metrics) == {k: wanted[k] for k in development(metrics)}
    assert primary == winner(selection["scores"])
    result = dict(
        primary=primary,
        weights_verified=list(MODELS),
        scores_recalculated=checked,
        identical_historical_records=replay,
        cpu_wall_seconds=perf_counter() - start,
    )
    put(work / "project_metrics.json", project_metrics)
    result["project_metrics_sha256"] = digest(work / "project_metrics.json")
    confirmation = work / "rerun_confirmation.json"
    if confirmation.exists():
        repeat = read(confirmation)
        assert repeat["identical"] and repeat["selection"] == completion["selection_sha256"]
        assert set(repeat["runs"]) == set(checked)
        for key, second in repeat["runs"].items():
            first = read(work / "scores" / key / "done.json")
            assert first["files"] == second["files"] and first["weights"] == second["weights"]
            assert second["started_unix"] > first["completed_unix"] and second["device"].startswith("cuda")
            assert set(second["same_files"]) == set(first["files"]) and all(second["same_files"].values())
        result["independent_gpu_confirmation_sha256"] = digest(confirmation)
    put(work / "validation.json", result)
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", action="store_true", help="Regrade the saved predictions and verify both models"
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="Optional directory containing the first uniform run's archived predictions",
    )
    parser.add_argument("--work", type=Path, default=ROOT / "outputs/part1_baseline")
    args = parser.parse_args()
    work = args.work.resolve()
    print(json.dumps(audit_results(work, args.history) if args.results else audit(work), indent=2))
