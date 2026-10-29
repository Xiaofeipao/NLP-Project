"""Official-world import, layout-level splits, and reproducible corruptions."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from .core import ACTIONS, Grid, OfficialOracle, UPSTREAM_REVISION, shortest_path
from .prompts import prompt


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def corrupt(base, count, rng, kind=None):
    """Edit distinct original positions. OOD uses two independent local edits.

    Retry filtering ensures token edit distance equals the requested edit count.
    Two injected errors need not produce two observable executor failures.
    """
    if count not in (1, 2) or len(base) < count:
        raise ValueError("Insufficient source actions")
    positions = sorted(rng.sample(range(len(base)), count))
    operations = {}
    kinds = ("substitution", "deletion", "insertion", "invalid_token")
    for index in positions:
        choice = kind if count == 1 and kind else rng.choice(kinds)
        if choice == "premature_stopping":
            # This is one truncation operation, possibly deleting several tokens.
            return base[:index], [{"kind": choice, "original_index": index + 1,
                                   "removed": base[index:]}]
        if choice == "substitution":
            replacement = rng.choice([a for a in ACTIONS if a != base[index]])
        elif choice == "invalid_token":
            replacement = "jump"
        elif choice == "insertion":
            replacement = rng.choice(list(ACTIONS))
        else:
            replacement = None
        operations[index] = {"kind": choice, "original_index": index + 1,
                             "before": base[index], "after": replacement}
    output = []
    for i, action in enumerate(base):
        op = operations.get(i)
        if not op:
            output.append(action)
        elif op["kind"] == "insertion":
            output.extend([op["after"], action])
        elif op["kind"] != "deletion":
            output.append(op["after"])
    return output, list(operations.values())


def nonoptimal(grid, base):
    # A reversible free-cell detour preserves feasibility and adds exactly 2 moves.
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    for token, (dr, dc) in ACTIONS.items():
        nxt = grid.start[0] + dr, grid.start[1] + dc
        if grid.inside(nxt) and nxt not in grid.obstacles:
            return [token, opposite[token]] + base
    raise ValueError("A reachable distinct goal must have a free neighbor")


def load_official(root):
    sources, seen, records = [], set(), []
    # Import all 6x6 official SG files, then rebuild stricter layout-level splits.
    for path in sorted(Path(root).glob("*6x6*.json")):
        if "more_obstacles" in path.name:
            continue
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for i, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            grid = Grid.from_world(row["world"])
            if (grid.rows, grid.cols) != (6, 6):
                continue
            key = (grid.layout_id, grid.start, grid.goal)
            if key in seen:
                continue
            seen.add(key)
            solution = shortest_path(grid)
            if solution is not None and len(solution) >= 2:
                records.append((grid, solution, f"{path.name}:{i}"))
    if not records:
        raise ValueError("No reachable official 6x6 single-goal records found")
    return records, sources


def make_candidate(grid, base, rng, oracle, valid, errors, long_valid=False, kind=None):
    if valid:
        tokens = nonoptimal(grid, base) if long_valid else base
        plan = " ".join(tokens)
        verdict = oracle.check(grid, plan)
        if not verdict.valid:
            raise AssertionError("Invalid positive example")
        return plan, verdict, []
    for _ in range(1000):
        tokens, edits = corrupt(base, errors, rng, kind)
        if edits[0]["kind"] != "premature_stopping" and edit_distance(base, tokens) != errors:
            continue
        plan = " ".join(tokens)
        verdict = oracle.check(grid, plan)
        if not verdict.valid:
            return plan, verdict, edits
    raise RuntimeError(f"Cannot construct requested corruption for {grid.layout_id}")


def build_dataset(config, root, executor, output):
    start = time.perf_counter()
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new dataset output directory: {out}")
    rng = random.Random(config["seed"])
    oracle = OfficialOracle(executor)
    records, sources = load_official(root)
    layouts = sorted({g.layout_id for g, _, _ in records})
    rng.shuffle(layouts)
    if len(layouts) < 10:
        raise ValueError("At least 10 unique official layouts are required")
    ndev, ntest = max(1, len(layouts) // 10), max(1, len(layouts) // 10)
    ids = {"dev": set(layouts[:ndev]), "test": set(layouts[ndev:ndev + ntest]),
           "train": set(layouts[ndev + ntest:])}
    groups = {name: [x for x in records if x[0].layout_id in split] for name, split in ids.items()}
    limits = config["data"]
    if sum(limits["train"].values()) > 4000 or sum(limits["dev"].values()) > 200:
        raise ValueError("Document budget exceeded")
    if not 1 <= limits["test_per_set"] <= 200:
        raise ValueError("Test budget must be 1..200")
    counts, reason_counts = {}, {}
    corruption_kinds = ("substitution", "deletion", "insertion", "premature_stopping", "invalid_token")

    def row_for(record, row_id, valid, errors=1, long_valid=False, kind=None):
        grid, base, source = record
        candidate, verdict, edits = make_candidate(grid, base, rng, oracle, valid, errors, long_valid, kind)
        return {"id": row_id, "layout_id": grid.layout_id, "grid": asdict(grid), "source_id": source,
                "candidate": candidate, "label": verdict.label(), "valid": verdict.valid,
                "reason": verdict.reason, "step": verdict.step, "target_plan": " ".join(base),
                "edits": edits, "edit_count": len(edits), "trace": verdict.trace,
                "valid_kind": ("nonoptimal" if long_valid else "optimal") if valid else None}

    for split in ("train", "dev"):
        rows = []
        for task, count in limits[split].items():
            for i in range(count):
                record = rng.choice(groups[split])
                grid, base, source = record
                if task == "generation":
                    oracle.check(grid, " ".join(base))
                    row = {"id": f"{split}-{task}-{i}", "grid": asdict(grid), "layout_id": grid.layout_id,
                           "source_id": source, "task": task, "input": prompt(task, grid),
                           "target": " ".join(base)}
                else:
                    valid = task == "verification" and i % 2 == 0
                    row = row_for(record, f"{split}-{task}-{i}", valid,
                                  long_valid=valid and i % 4 == 0,
                                  kind=corruption_kinds[(i // 2) % len(corruption_kinds)])
                    row.update(task=task, input=prompt(task, grid, row["candidate"], row["label"]),
                               target=row["label"] if task == "verification" else row["target_plan"])
                rows.append(row)
        rng.shuffle(rows)
        write_jsonl(out / f"{split}.jsonl", rows)
        counts[split] = dict(Counter(r["task"] for r in rows))
        reason_counts[split] = dict(Counter(r.get("reason") or "valid" for r in rows if r["task"] != "generation"))

    # Paired test environments isolate the corruption shift; both are held out from train/dev.
    test_records = list(groups["test"])
    rng.shuffle(test_records)
    test_records = test_records[:limits["test_per_set"]]
    if len(test_records) < limits["test_per_set"]:
        raise ValueError("Not enough unique held-out start/goal environments for requested test size")
    for split, errors in (("test_id", 1), ("test_ood", 2)):
        rows = []
        for i, record in enumerate(test_records):
            valid = i % 2 == 0
            row = row_for(record, f"{split}-{i}", valid, errors, valid and i % 4 == 0,
                          corruption_kinds[(i // 2) % 5] if errors == 1 else None)
            row["split"] = split
            rows.append(row)
        write_jsonl(out / f"{split}.jsonl", rows)
        counts[split] = len(rows)
        reason_counts[split] = dict(Counter(r["reason"] or "valid" for r in rows))
    write_json(out / "manifest.json", {
        "seed": config["seed"], "upstream_revision": UPSTREAM_REVISION,
        "executor_sha256": oracle.sha256, "executor_path": str(Path(executor).resolve()), "official_sources": sources,
        "layout_splits": {k: sorted(v) for k, v in ids.items()}, "counts": counts,
        "reason_counts": reason_counts, "cpu_construction_seconds": time.perf_counter() - start,
        "rules": {"failure_step": "1-based; EOS = action_count + 1",
                  "ood": "two distinct original edit sites, token edit distance 2; valid controls retained",
                  "invalid_token": "strict vocabulary extension; official ignores unknown tokens",
                  "split": "obstacle-layout grouping; paired ID/OOD test environments",
                  "source_path": str(Path(root).resolve())}})


def audit_dataset(directory):
    directory = Path(directory)
    splits = {s: read_jsonl(directory / f"{s}.jsonl") for s in ("train", "dev", "test_id", "test_ood")}
    layout_sets = {s: {r["layout_id"] for r in rows} for s, rows in splits.items()}
    for left, right in (("train", "dev"), ("train", "test_id"), ("train", "test_ood"),
                        ("dev", "test_id"), ("dev", "test_ood")):
        if layout_sets[left] & layout_sets[right]:
            raise ValueError(f"Layout leakage between {left} and {right}")
    for s, cap in (("train", 4000), ("dev", 200), ("test_id", 200), ("test_ood", 200)):
        if not 0 < len(splits[s]) <= cap:
            raise ValueError(f"Invalid {s} count")
    for s, rows in splits.items():
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate example IDs in {s}")
        for row in rows:
            if Grid.from_dict(row["grid"]).layout_id != row["layout_id"]:
                raise ValueError("Mismatched layout ID")
    return {s: len(rows) for s, rows in splits.items()}


def task_index(directory, split, task):
    """Select the frozen task subset without generating or replacing examples."""
    if split not in ("train", "dev") or task not in ("generation", "verification", "repair"):
        raise ValueError("Expected a train/dev split and a known task")
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(root / f"{split}.jsonl")
    selected = [row for row in rows if row["task"] == task]
    if len(selected) != manifest["counts"][split][task]:
        raise ValueError("Task count disagrees with frozen manifest")
    if len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate task example IDs")
    if any(row["layout_id"] not in manifest["layout_splits"][split] for row in selected):
        raise ValueError("Task subset crosses layout split")
    return selected


def task_index_manifest(directory):
    """Record source bytes and exact IDs for the first fixed 4000/200 protocol."""
    root = Path(directory)
    result = {"source_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
              "splits": {}}
    for split in ("train", "dev"):
        result["splits"][split] = {"source_sha256": hashlib.sha256((root / f"{split}.jsonl").read_bytes()).hexdigest(),
                                   "tasks": {}}
        for task in ("generation", "verification", "repair"):
            rows = task_index(root, split, task)
            result["splits"][split]["tasks"][task] = {"count": len(rows), "ids": [r["id"] for r in rows]}
    return result
