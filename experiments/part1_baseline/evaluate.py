"""Blind plan prediction and separate official-executor grading on frozen IDs."""

from collections import Counter, defaultdict
import json
import time

from ppnl.core import Grid, OfficialOracle, parse_verdict, shortest_path
from .policy import DELIVERED, direct, episode, line, run
from .protocol import (
    ERROR_SETS,
    LABELS,
    METRIC,
    ROOT,
    SOURCES,
    digest,
    gold_rows,
    point,
    put,
    read,
    rows_at,
    templates,
)


def env_key(case):
    grid = case["grid"]
    return (
        grid["rows"],
        grid["cols"],
        tuple(sorted(map(tuple, grid["obstacles"]))),
        tuple(grid["start"]),
        tuple(grid["goal"]),
    )


class ExecutorFeedback:
    """Only invoked by the explicitly named oracle-feedback pipeline."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.seconds = 0.0
        self.calls = 0

    def __call__(self, grid, prefix, before, action):
        start = time.perf_counter()
        result = self.oracle.check(Grid.from_dict(grid), " ".join([*prefix, action]))
        step = result.trace[len(prefix)]
        assert point(step["from"]) == before and step["action"] == action
        cell = "no_coordinate" if result.reason == "invalid_action_token" else point(step["to"])
        self.seconds += time.perf_counter() - start
        self.calls += 1
        return dict(
            before=before, action=action, cell=cell, verdict="valid" if result.valid else result.reason
        )


def predictions(blind, predict, full, oracle, separate_generation_batch=False):
    if any(set(r) != {"id", "grid", "candidate"} for r in blind):
        raise ValueError("Model input files must contain only blind fields")
    ps, cache = templates(), {}
    callback = ExecutorFeedback(oracle)
    start = time.perf_counter()
    candidate = run([episode(c, ps) for c in blind], predict, cache)
    unique = {}
    for c in blind:
        unique.setdefault(env_key(c), c)
    # Score each environment once, with a stable independent batch for the generation arms.
    envs = [unique[key] for key in sorted(unique)] if separate_generation_batch else list(unique.values())
    env_cache = {} if separate_generation_batch else cache
    generated = run([direct(c, ps) for c in envs], predict, env_cache)
    if full:
        pending = [
            dict(id=c["id"], grid=c["grid"], candidate=g["plan"])
            for c, g in zip(envs, generated, strict=True)
        ]
        self_fix = run([episode(c, ps) for c in pending], predict, env_cache)
        ext_fix = run([episode(c, ps, feedback=callback) for c in pending], predict, env_cache)
        for g, ours, external in zip(generated, self_fix, ext_fix, strict=True):
            assert g["id"] == ours["id"] == external["id"]
            g.update(self_repair=ours, executor_repair=external)
    return (
        candidate,
        generated,
        dict(
            inference_wall_s=time.perf_counter() - start,
            distinct_model_prompts=len(cache.keys() | env_cache.keys()),
            model_inputs_evaluated=len(cache) + (len(env_cache) if separate_generation_batch else 0),
            executor_feedback_cpu_wall_s=callback.seconds,
            executor_feedback_calls=callback.calls,
        ),
    )


def canonical_verdict(text):
    parsed = parse_verdict(text)
    if parsed is None:
        return None
    valid, step, reason = parsed
    expected = "valid" if valid else f"invalid | step={step} | reason={reason}"
    return parsed if text == expected else None


def structured_verification(gold, lines):
    """A type earns a true positive only for its complete canonical verdict, including the index."""
    support, predicted, matches = Counter(), Counter(), Counter()
    for row, output in zip(gold, lines, strict=True):
        expected = row["reason"] or "valid"
        parsed = canonical_verdict(output)
        kind = "malformed" if parsed is None else ("valid" if parsed[0] else parsed[2])
        support[expected] += 1
        predicted[kind] += 1
        if output == row["label"]:
            matches[expected] += 1
    per_type = {}
    for kind in LABELS:
        n, p, tp = support[kind], predicted[kind], matches[kind]
        per_type[kind] = dict(
            gold=n, predicted=p, tp=tp, fp=p - tp, fn=n - tp, f1=2 * tp / (n + p) if n else None
        )
    supported = [s["f1"] for s in per_type.values() if s["gold"]]
    exact = sum(matches.values())
    return dict(
        metric=METRIC,
        n=len(gold),
        exact=exact,
        exact_accuracy=exact / len(gold),
        structured_macro_f1=sum(supported) / len(supported),
        supported_types=len(supported),
        false_valid=predicted["valid"] - matches["valid"],
        malformed=predicted["malformed"],
        per_type=per_type,
    )


def grade(official, gold, candidate, generated, full):
    begin = time.perf_counter()
    if [r["id"] for r in gold] != [r["id"] for r in candidate]:
        raise ValueError("Gold ID order mismatch")
    records = {r["id"]: r for r in gold}
    lines = [line(a["verification"]) for a in candidate]
    verification = structured_verification(gold, lines)

    def plan(source, text, delivered):
        grid = Grid.from_dict(source["grid"])
        result = official.check(grid, text)
        best = shortest_path(grid)
        return dict(
            success=bool(delivered and result.valid),
            feasible=bool(delivered and result.feasible),
            optimal=bool(delivered and result.valid and len(text.split()) == len(best)),
            delivered=delivered,
            actual_verdict=result.label(),
        )

    def aggregate(vals):
        n = len(vals)
        counts = {k: sum(x[k] for x in vals) for k in ("success", "feasible", "optimal", "delivered")}
        return dict(n=n, counts=counts, **{k: v / n if n else None for k, v in counts.items()})

    by = defaultdict(list)
    scored = []
    matrix = Counter()
    for r, a, raw in zip(gold, candidate, lines, strict=True):
        true = official.check(Grid.from_dict(r["grid"]), r["candidate"])
        assert true.label() == r["label"]
        result = canonical_verdict(raw)
        pred = "malformed" if result is None else ("valid" if result[0] else result[2])
        family = r["reason"] or "valid"
        matrix[family, pred] += 1
        z = dict(
            id=r["id"],
            gold=r["label"],
            prediction=raw,
            first_step_correct=False,
            reason_correct=False,
            joint=False,
            repair=plan(r, a["repaired_path"], a["repair_status"] in DELIVERED),
        )
        if not r["valid"]:
            z.update(
                first_step_correct=result is not None and not result[0] and result[1] == r["step"],
                reason_correct=result is not None and not result[0] and result[2] == r["reason"],
                joint=raw == r["label"],
            )
        by[family].append(z)
        scored.append(z)
    invalid = [z for label, items in by.items() if label != "valid" for z in items]
    by_reason = {}
    for label, items in sorted(by.items()):
        predicted = {p: n for (y, p), n in matrix.items() if y == label}
        by_reason[label] = dict(
            n=len(items), prediction_counts=predicted, exact=sum(s["prediction"] == s["gold"] for s in items)
        )
        if label != "valid":
            by_reason[label].update(
                step_correct=sum(s["first_step_correct"] for s in items),
                reason_correct=sum(s["reason_correct"] for s in items),
                joint=sum(s["joint"] for s in items),
                repair=aggregate([s["repair"] for s in items]),
            )
    env_rows = []
    for g in generated:
        r = records[g["id"]]
        q = dict(id=g["id"], direct=plan(r, g["plan"], g["status"] != "malformed_C"))
        if full:
            for arm in ("self_repair", "executor_repair"):
                out = g[arm]
                q[arm] = plan(r, out["repaired_path"], out["repair_status"] in DELIVERED)
        env_rows.append(q)
    names = ("direct", "self_repair", "executor_repair") if full else ("direct",)
    pipeline = {name: aggregate([r[name] for r in env_rows]) for name in names}
    metrics = dict(
        verification=verification,
        candidate_repair=aggregate([r["repair"] for r in invalid]),
        invalid_count=len(invalid),
        step_correct=sum(s["first_step_correct"] for s in invalid),
        reason_correct=sum(s["reason_correct"] for s in invalid),
        joint=sum(s["joint"] for s in invalid),
        valid_count=len(by["valid"]),
        valid_correct=sum(z["prediction"] == "valid" for z in by["valid"]),
        repair_stops=dict(Counter(a["repair_status"] for a in candidate)),
        by_reason=by_reason,
        pipeline=pipeline,
    )
    return metrics, scored, env_rows, time.perf_counter() - begin


def evaluate(net, work, split, family, model_hash, full=False):
    freeze = read(work / "data/protocol.json")
    blind_path = work / "data" / (split + ".jsonl")
    source = SOURCES[split]
    if freeze["metric"] != METRIC:
        raise ValueError("Scoring protocol differs from the frozen specification")
    protocol_hash = digest(work / "data/protocol.json")
    selection_hash = None
    if split in ERROR_SETS:
        selection_path = work / "selection.json"
        selection = read(selection_path)
        if (
            selection["protocol"] != protocol_hash
            or selection["primary_model"] != family
            or selection["scores"][family]["weights"] != model_hash
        ):
            raise ValueError("Only the checkpoint selected on development may open the test inputs")
        selection_hash = digest(selection_path)
    if digest(blind_path) != freeze["inputs"][split] or digest(source) != freeze["source_splits"][split]:
        raise ValueError("Evaluation split changed")
    official = OfficialOracle(ROOT / "vendor/ppnl/executor-point-sg.py")
    assert official.sha256 == freeze["official_executor"]
    out = work / "scores" / family / split
    if (out / "done.json").exists():
        old = read(out / "done.json")
        if (
            old["protocol"] != protocol_hash
            or old["selection"] != selection_hash
            or old["weights"] != model_hash
            or old["full"] != full
            or old["inputs"] != digest(blind_path)
            or any(digest(out / name) != h for name, h in old["files"].items())
        ):
            raise ValueError("Evaluation receipt mismatch")
        return read(out / "metrics.json")
    started = time.time()
    blind = rows_at(blind_path)
    candidate, generated, runtime = predictions(
        blind, net.predict, full, official, separate_generation_batch=split != "dev"
    )
    put(out / "candidate_predictions.jsonl", candidate, True)
    put(out / "generated_predictions.jsonl", generated, True)
    gold = gold_rows(split)
    metrics, details, env, elapsed = grade(official, gold, candidate, generated, full)
    put(out / "candidate_scores.jsonl", details, True)
    put(out / "generated_scores.jsonl", env, True)
    put(out / "metrics.json", metrics)
    files = [
        "metrics.json",
        "candidate_predictions.jsonl",
        "generated_predictions.jsonl",
        "candidate_scores.jsonl",
        "generated_scores.jsonl",
    ]
    put(
        out / "done.json",
        dict(
            protocol=protocol_hash,
            selection=selection_hash,
            metric=METRIC,
            device=str(net.device),
            started_unix=started,
            completed_unix=time.time(),
            weights=model_hash,
            inputs=digest(blind_path),
            full=full,
            **runtime,
            grading_cpu_wall_s=elapsed,
            files={f: digest(out / f) for f in files},
        ),
    )
    print(
        json.dumps(
            dict(
                split=split,
                model=model_hash[:10],
                full=full,
                exact=metrics["verification"]["exact"],
                structured_macro_f1=metrics["verification"]["structured_macro_f1"],
                repair=metrics["candidate_repair"]["success"],
                direct=metrics["pipeline"]["direct"]["success"],
            ),
            ensure_ascii=True,
        ),
        flush=True,
    )
    return metrics
