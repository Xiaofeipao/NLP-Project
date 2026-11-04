"""Evaluation harness (separate from the agent runtime in ppnl.harness).

Design rules
------------
1. Reuse, never reimplement. Every planner is the function already used by
   ``scripts/evaluate_policy.py`` and every number comes from
   ``ppnl.evaluation.plan_metrics``, so harness output is directly comparable with the
   frozen ``outputs/planning_v3`` results.
2. Write to a fresh directory. The harness refuses to write into the baseline
   experiment root, so frozen ``eval_*`` artefacts are never overwritten.
3. No silent fallback. A bounded search that fails returns an empty plan and is counted
   as a failure, exactly like the frozen pipeline.
4. Record the comparison, do not assert it away. Differences against the frozen metrics
   are reported per field instead of crashing the run, because a deliberate change of
   method is a legitimate reason for a difference.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import html
import json
from pathlib import Path
import time

from .core import Grid, OfficialOracle, execute
from .data import read_jsonl, write_json, write_jsonl
from .evaluation import plan_metrics
from .modeling import gpu_budget
from .policy import greedy_plans, optimal_search_plans, search_plans

SPLITS = ("dev", "test_id", "test_ood")
TOLERANCE = 1e-9
HEADLINE_FIELDS = ("success_rate", "feasibility", "optimality", "path_efficiency_on_success")


@dataclass(frozen=True)
class Method:
    """One planner under test. ``runner`` returns (plans, expansions-or-None)."""

    name: str
    needs_model: bool
    description: str
    runner: object


def _greedy_unmasked(grids, scorer):
    return greedy_plans(grids, scorer, masked=False), None


def _greedy_masked(grids, scorer):
    return greedy_plans(grids, scorer, masked=True), None


def _best_first_neural(grids, scorer):
    return search_plans(grids, scorer, weight=0.25)


def _best_first_symbolic(grids, scorer):
    return search_plans(grids, None, weight=0.0)


def _astar_neural_tiebreak(grids, scorer):
    return optimal_search_plans(grids, scorer, goal_tiebreak=False)


def _astar_goal_tiebreak(grids, scorer):
    return optimal_search_plans(grids, None, goal_tiebreak=True)


METHODS = (
    Method("state_policy", True,
           "learned one-action policy with explicit state observations; no mask, no search",
           _greedy_unmasked),
    Method("legal_mask_no_search", True,
           "same policy plus legal unvisited-action masking",
           _greedy_masked),
    Method("neural_best_first", True,
           "best-first graph search ranked by learned action likelihood and Manhattan distance",
           _best_first_neural),
    Method("symbolic_astar_no_model", False,
           "A* without any neural scores; necessary search ablation",
           _best_first_symbolic),
    Method("neural_optimal_astar", True,
           "unit-cost A*; neural likelihood only breaks equal g+Manhattan ties",
           _astar_neural_tiebreak),
    Method("symbolic_astar_goal_tiebreak", False,
           "strong pure A* ablation: lower Manhattan breaks equal f ties",
           _astar_goal_tiebreak),
)
METHOD_BY_NAME = {method.name: method for method in METHODS}


def resolve_methods(spec, symbolic_only=False):
    """Accept ``all`` or a comma-separated subset; reject unknown names loudly."""
    if spec in (None, "", "all"):
        names = [method.name for method in METHODS]
    else:
        names = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [name for name in names if name not in METHOD_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Known: {sorted(METHOD_BY_NAME)}")
    chosen = [METHOD_BY_NAME[name] for name in names]
    if symbolic_only:
        chosen = [method for method in chosen if not method.needs_model]
    if not chosen:
        raise ValueError("No method left to run")
    return chosen


def _key(grid):
    return grid.layout_id, grid.start, grid.goal


def split_rows(config, split):
    """Same rows and same de-duplication rule as ``scripts/evaluate_policy.py``."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split}; expected one of {SPLITS}")
    rows = read_jsonl(Path(config["policy_source_dataset"]) / f"{split}.jsonl")
    if split == "dev":
        rows = [row for row in rows if row["task"] == "verification"]
    unique = {}
    for row in rows:
        grid = Grid.from_dict(row["grid"])
        unique.setdefault(_key(grid), grid)
    return rows, list(unique.values())


def repair_prefix(grid, candidate):
    """Keep only physically executed actions; drop the failing action and suffix.

    Byte-for-byte the routine in ``scripts/evaluate_policy.py``; a regression test keeps
    the two implementations identical.
    """
    verdict = execute(grid, candidate)
    prefix, position = [], grid.start
    for event in verdict.trace:
        if event["reason"]:
            break
        prefix.append(event["action"])
        position = tuple(event["to"])
        if position == grid.goal:
            return prefix, None
    return prefix, replace(grid, start=position)


def weights_dir(config):
    root = Path(config["output_root"])
    return Path(config.get("policy_checkpoint", str(root / "flan_t5_small" / "train" / "best")))


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def frozen_metrics(config, split, baseline=None):
    directory = Path(baseline or config["output_root"]) / f"eval_{split}" / "metrics.json"
    return json.loads(directory.read_text(encoding="utf-8")) if directory.exists() else None


def baseline_direct_plans(config, split, grids):
    """Stored whole-sequence plans from the original baseline run; no model call."""
    directory = Path(config["baseline_root"]) / ("eval_dev" if split == "dev" else f"eval_{split}_full")
    path = directory / "pipeline_traces.jsonl"
    if not path.exists():
        return None, f"missing {path}"
    recorded = {_key(Grid.from_dict(row["grid"])): row["direct"]["plan"]
                for row in read_jsonl(path)}
    missing = [_key(grid) for grid in grids if _key(grid) not in recorded]
    if missing:
        return None, f"{len(missing)} grid(s) absent from {path.name}"
    return [recorded[_key(grid)] for grid in grids], None


def _compare(actual, expected, baseline_label):
    """Field-by-field delta between a harness run and the frozen metrics."""
    if expected is None:
        return {"baseline": None, "status": "no_frozen_result", "differences": []}
    differences = []
    for section in ("planning", "prefix_repair"):
        for method, values in (actual.get(section) or {}).items():
            reference = (expected.get(section) or {}).get(method)
            if reference is None:
                differences.append({"section": section, "method": method, "field": None,
                                    "expected": None, "actual": values.get("n"),
                                    "note": "method absent from frozen result"})
                continue
            for field in ("n",) + HEADLINE_FIELDS:
                want, got = reference.get(field), values.get(field)
                if want is None and got is None:
                    continue
                if want is None or got is None or abs(float(got) - float(want)) > TOLERANCE:
                    differences.append({"section": section, "method": method, "field": field,
                                        "expected": want, "actual": got})
    return {"baseline": baseline_label,
            "status": "match" if not differences else "differs",
            "differences": differences}


def _evaluate_split(config, split, methods, scorer, check, baseline, with_direct):
    rows, grids = split_rows(config, split)
    entry = {"n_grids": len(grids), "n_rows": len(rows),
             "n_invalid": sum(1 for row in rows if not row["valid"]),
             "planning": {}, "prefix_repair": {}, "seconds_by_planner": {}, "mean_expansions": {}}
    plans_by_method = {}
    if with_direct:
        direct, reason = baseline_direct_plans(config, split, grids)
        if direct is None:
            entry["direct_generation_skipped"] = reason
        else:
            plans_by_method["original_whole_plan"] = direct
    for method in methods:
        check()
        started = time.perf_counter()
        plans, expansions = method.runner(grids, scorer)
        entry["seconds_by_planner"][method.name] = time.perf_counter() - started
        if expansions is not None:
            entry["mean_expansions"][method.name] = sum(expansions) / len(expansions)
        plans_by_method[method.name] = plans
    entry["planning"] = {name: plan_metrics(grids, plans) for name, plans in plans_by_method.items()}

    invalid = [row for row in rows if not row["valid"]]
    prefixes, repair_grids, indices = {}, [], []
    for index, row in enumerate(invalid):
        grid = Grid.from_dict(row["grid"])
        prefix, remaining = repair_prefix(grid, row["candidate"])
        prefixes[index] = prefix
        if remaining is not None:
            repair_grids.append(remaining)
            indices.append(index)
    repair_targets = [Grid.from_dict(row["grid"]) for row in invalid]
    repair_plans = {}
    for method in methods:
        check()
        suffixes, _ = method.runner(repair_grids, scorer)
        by_index = dict(zip(indices, suffixes))
        repair_plans[method.name] = [" ".join(prefixes[i] + by_index.get(i, "").split())
                                     for i in range(len(invalid))]
        entry["prefix_repair"][method.name] = plan_metrics(repair_targets, repair_plans[method.name])
    return entry, grids, rows, plans_by_method, invalid, prefixes, repair_plans


def run(config, splits=SPLITS, methods="all", out=None, baseline=None,
        allow_new_weights=False, symbolic_only=False, with_direct=True,
        write_html=True, config_path=None, require_official=True):
    """Run the harness end to end and return the full result dictionary."""
    chosen = resolve_methods(methods, symbolic_only)
    if not splits or any(split not in SPLITS for split in splits) or len(set(splits)) != len(splits):
        raise ValueError("Choose distinct supported splits")
    weights = weights_dir(config)
    weight_file = weights / "model.safetensors"
    weight_hash = sha256_file(weight_file) if weight_file.exists() else None
    baseline_root = Path(baseline or config["output_root"])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    destination = Path(out or Path("outputs/harness") / stamp)
    if destination.resolve() == baseline_root.resolve() or baseline_root.resolve() in destination.resolve().parents:
        raise ValueError("Refusing to write inside the baseline experiment root")
    if destination.exists():
        raise ValueError("Evaluation output directory already exists; choose a fresh directory")

    selection_path = baseline_root / "policy_selection.json"
    weights_match = None
    if selection_path.exists() and weight_hash:
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        weights_match = frozen.get("weights_sha256") == weight_hash
        if weights_match is False and not allow_new_weights:
            raise ValueError("Weights differ from the development-selected checkpoint; "
                             "pass --allow-new-weights to evaluate anyway")

    manifest_path = Path(config["dataset_root"]) / "manifest.json"
    oracle, executor_sha = None, None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = OfficialOracle(manifest["executor_path"])
        if candidate.sha256 != manifest["executor_sha256"]:
            raise ValueError("Official executor changed since data construction")
        oracle, executor_sha = candidate, candidate.sha256
    elif require_official:
        raise ValueError("Formal evaluation requires the data manifest and official executor")

    model_methods = [method for method in chosen if method.needs_model]
    result = {"generated_unix": time.time(), "config": str(config_path) if config_path else None,
              "weights": str(weights), "weights_sha256": weight_hash,
              "weights_match_selection": weights_match,
              "baseline_root": str(baseline_root), "tolerance": TOLERANCE,
              "executor_sha256": executor_sha, "symbolic_only": symbolic_only,
              "ground_truth": "official" if oracle else "local_fixture_only",
              "methods": [{"name": method.name, "needs_model": method.needs_model,
                           "description": method.description} for method in chosen],
              "splits": {}}

    budget = (gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"],
                         f"harness:{'+'.join(splits)}") if model_methods else nullcontext(lambda: None))
    scorer = None
    destination.mkdir(parents=True, exist_ok=False)
    started_all = time.perf_counter()
    with budget as check:
        if model_methods:
            from .modeling import Seq2Seq, require_cuda
            from .policy import NeuralPolicy
            require_cuda()
            runner = Seq2Seq(str(weights), config)
            scorer = NeuralPolicy(runner, check)
        for split in splits:
            check()
            started_split = time.perf_counter()
            entry, grids, rows, plans, invalid, prefixes, repair_plans = _evaluate_split(
                config, split, chosen, scorer, check, baseline_root, with_direct)
            entry["comparison"] = _compare(entry, frozen_metrics(config, split, baseline_root),
                                           str(baseline_root / f"eval_{split}" / "metrics.json"))
            entry["split"] = split
            result["splits"][split] = entry
            detail = destination / "detail"
            traces = []
            for index, grid in enumerate(grids):
                record = {"split": split, "grid": asdict(grid), "plans": {}}
                for name, plan_list in plans.items():
                    verdict = oracle.check(grid, plan_list[index]) if oracle else execute(grid, plan_list[index])
                    record["plans"][name] = {"plan": plan_list[index], "verdict": verdict.label()}
                traces.append(record)
            write_jsonl(detail / f"{split}_planning_traces.jsonl", traces)
            repair_traces = []
            for index, row in enumerate(invalid):
                grid = Grid.from_dict(row["grid"])
                record = {"id": row["id"], "candidate": row["candidate"],
                          "kept_prefix": " ".join(prefixes[index]), "grid": row["grid"], "plans": {}}
                for name, candidates in repair_plans.items():
                    verdict = oracle.check(grid, candidates[index]) if oracle else execute(grid, candidates[index])
                    record["plans"][name] = {"plan": candidates[index], "verdict": verdict.label()}
                repair_traces.append(record)
            write_jsonl(detail / f"{split}_repair_traces.jsonl", repair_traces)
            result["splits"][split]["elapsed_seconds"] = time.perf_counter() - started_split
    result["elapsed_seconds_total"] = time.perf_counter() - started_all
    if scorer is not None:
        result["cached_policy_states"] = len(scorer.cache)
    write_json(destination / "results.json", result)
    (destination / "report.md").write_text(render_markdown(result), encoding="utf-8")
    if write_html:
        (destination / "report.html").write_text(render_html(result), encoding="utf-8")
    result["output_dir"] = str(destination)
    print(console_table(result), flush=True)
    print(f"\nArtifacts: {destination}", flush=True)
    return result


def _pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _delta(actual, expected):
    if actual is None or expected is None:
        return ""
    change = 100 * (actual - expected)
    return "" if abs(change) < 0.05 else f"{change:+.1f}pt"


def console_table(result):
    lines, width = [], 30
    for split, entry in result["splits"].items():
        lines.append(f"\n== {split} == n={entry['n_grids']} (invalid rows {entry['n_invalid']})")
        header = f"{'method':<{width}}{'success':>9}{'feasible':>10}{'optimal':>9}{'efficiency':>12}{'expand':>8}{'sec':>8}{'Δsuccess':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        reference = {d["method"]: d for d in entry["comparison"]["differences"]
                     if d["section"] == "planning" and d["field"] == "success_rate"}
        for name, values in entry["planning"].items():
            frozen = None
            diff = reference.get(name)
            if diff and diff["field"] == "success_rate":
                frozen = diff["expected"]
            expansions = entry["mean_expansions"].get(name)
            lines.append(
                f"{name:<{width}}{_pct(values['success_rate']):>9}{_pct(values['feasibility']):>10}"
                f"{_pct(values['optimality']):>9}{_pct(values['path_efficiency_on_success']):>12}"
                f"{'n/a' if expansions is None else f'{expansions:.2f}':>8}"
                f"{entry['seconds_by_planner'].get(name, 0):>8.2f}"
                f"{_delta(values['success_rate'], frozen):>10}")
        lines.append(f"frozen comparison: {entry['comparison']['status']}"
                     f" ({len(entry['comparison']['differences'])} differing field(s))")
    return "\n".join(lines)


def render_markdown(result):
    lines = ["# Evaluation harness report", "",
             f"- weights: `{result['weights']}`",
             f"- weights sha256: `{result['weights_sha256']}`",
             f"- matches development selection: {result['weights_match_selection']}",
             f"- frozen baseline: `{result['baseline_root']}`",
             f"- total wall clock: {result['elapsed_seconds_total']:.1f} s", ""]
    for split, entry in result["splits"].items():
        lines += [f"## {split}", "",
                  f"{entry['n_grids']} unique grids, {entry['n_invalid']} invalid candidate rows.", "",
                  "| method | success | feasibility | optimality | path efficiency | mean expansions | seconds |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, values in entry["planning"].items():
            expansions = entry["mean_expansions"].get(name)
            lines.append(f"| {name} | {_pct(values['success_rate'])} | {_pct(values['feasibility'])} | "
                         f"{_pct(values['optimality'])} | {_pct(values['path_efficiency_on_success'])} | "
                         f"{'n/a' if expansions is None else f'{expansions:.2f}'} | "
                         f"{entry['seconds_by_planner'].get(name, 0):.2f} |")
        lines += ["", "### Prefix repair (invalid candidates only)", "",
                  "| method | success | feasibility | optimality | path efficiency |",
                  "| --- | --- | --- | --- | --- |"]
        for name, values in entry["prefix_repair"].items():
            lines.append(f"| {name} | {_pct(values['success_rate'])} | {_pct(values['feasibility'])} | "
                         f"{_pct(values['optimality'])} | {_pct(values['path_efficiency_on_success'])} |")
        comparison = entry["comparison"]
        lines += ["", f"Frozen comparison: **{comparison['status']}**."]
        if comparison["differences"]:
            lines += ["", "| section | method | field | frozen | harness |", "| --- | --- | --- | --- | --- |"]
            for difference in comparison["differences"]:
                lines.append(f"| {difference['section']} | {difference['method']} | {difference['field']} | "
                             f"{difference['expected']} | {difference['actual']} |")
        lines.append("")
    lines += ["## Method definitions", ""]
    for method in result["methods"]:
        lines.append(f"- `{method['name']}`: {method['description']}")
    lines += ["", "Timings share one state-score cache and are incremental, not cold-start "
                  "comparisons. Search success still depends on symbolic legality and Manhattan "
                  "heuristics, so it is not evidence of standalone language-model reasoning."]
    return "\n".join(lines)


def render_html(result):
    """Self-contained report; no external assets so it opens offline."""
    escape = html.escape
    blocks = []
    for split, entry in result["splits"].items():
        rows = []
        for name, values in entry["planning"].items():
            expansions = entry["mean_expansions"].get(name)
            rows.append(
                "<tr><td class='name'>{}</td><td><div class='bar'><span style='width:{:.1f}%'></span></div>"
                "<b>{}</b></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    escape(name), 100 * (values["success_rate"] or 0), _pct(values["success_rate"]),
                    _pct(values["feasibility"]), _pct(values["optimality"]),
                    "n/a" if expansions is None else f"{expansions:.2f}"))
        blocks.append(
            f"<section><h2>{escape(split)}</h2>"
            f"<p class='meta'>{entry['n_grids']} unique grids · {entry['n_invalid']} invalid candidate rows · "
            f"frozen comparison <b>{escape(entry['comparison']['status'])}</b></p>"
            "<table><thead><tr><th>method</th><th>success</th><th>feasibility</th>"
            "<th>optimality</th><th>mean expansions</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></section>")
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>PPNL harness report</title>
<style>
:root{{color-scheme:light}}
body{{margin:0;padding:32px;font:15px/1.6 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
background:#f6f7f9;color:#1a1d21}}
h1{{font-size:22px;margin:0 0 4px}}
h2{{font-size:17px;margin:0 0 8px;text-transform:none}}
.meta{{color:#5b6270;font-size:13px;margin:0 0 12px}}
section{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:20px 22px;margin:18px 0;
box-shadow:0 1px 2px rgba(16,24,40,.04)}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{text-align:left;font-weight:600;color:#5b6270;border-bottom:1px solid #e3e6ea;padding:8px 10px}}
td{{padding:8px 10px;border-bottom:1px solid #f0f2f5}}
td.name{{font-family:ui-monospace,Consolas,monospace;font-size:13px;color:#1a1d21}}
tr:last-child td{{border-bottom:none}}
.bar{{position:relative;display:inline-block;width:110px;height:8px;background:#eef1f5;border-radius:4px;
margin-right:10px;vertical-align:middle;overflow:hidden}}
.bar span{{display:block;height:100%;background:#2f6feb;border-radius:4px}}
footer{{color:#5b6270;font-size:12px;margin-top:24px}}
</style></head><body>
<h1>PPNL evaluation harness</h1>
<p class="meta">weights <code>{escape(str(result['weights_sha256']))}</code> ·
selection match <b>{escape(str(result['weights_match_selection']))}</b> ·
{result['elapsed_seconds_total']:.1f}s wall clock</p>
{''.join(blocks)}
<footer>Timings share one state-score cache and are incremental. Search results rely on symbolic
legality and Manhattan heuristics; they are not standalone language-model reasoning.</footer>
</body></html>"""
