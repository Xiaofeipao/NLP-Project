"""Fixed task schedules and state-grouped minibatches shared by all three arms."""
from collections import Counter, defaultdict
import hashlib
import math
import random

from texts import ACTIONS, CLASSES


SEED = 42
CASES_PER_UPDATE = 32
UPDATES_PER_EPOCH = 125
SCHEDULES = {
    "uniform": ("C", "GV", "GV") * 125,
    "short_warmup": ("C",) * 25 + ("C", "GV", "GV") * 100 + ("GV",) * 50,
    "c_first": ("C",) * 125 + ("GV",) * 250,
}


def make_streams(rows):
    groups = {"C": defaultdict(list), "GV": defaultdict(list)}
    actions = {}
    for i, row in enumerate(rows):
        task = row["task"]
        if task not in ("C", "G", "V"):
            raise ValueError(f"Unknown task in {row['id']}")
        stream = "C" if task == "C" else "GV"
        groups[stream][row["case"]].append(i)
        if task == "C":
            line = row["text"].splitlines()[1]
            if not line.startswith("Action: ") or not row["id"].endswith("/C"):
                raise ValueError(f"Missing matched C action: {row['id']}")
            actions[row["id"][:-2]] = (row["case"], line.removeprefix("Action: "))

    cases = sorted(groups["C"])
    if len(cases) != 4000 or set(groups["GV"]) != set(cases) or len(actions) != 16400:
        raise ValueError("Expected all 4,000 reviewed training states and their C/V pairs")
    for case in cases:
        c_ids, gv_ids = groups["C"][case], groups["GV"][case]
        ordinary = Counter(actions[rows[i]["id"][:-2]][1] for i in c_ids)
        if any(ordinary[a] != 1 for a in ACTIONS) or sum(rows[i]["task"] == "G" for i in gv_ids) != 1:
            raise ValueError(f"Incomplete action group for {case}")
    cells = Counter()
    for row in rows:
        if row["task"] == "V":
            case, action = actions[row["id"]]
            if case != row["case"]:
                raise ValueError(f"Crossed C/V pair: {row['id']}")
            cells[row["answers"][0], action] += 1
    directions = Counter(label for label, action in cells)
    if set(directions) != set(CLASSES) or sum(cells.values()) != len(actions):
        raise ValueError("Incomplete V class groups")
    weights, masses = [], Counter()
    for row in rows:
        task = row["task"]
        if task == "C":
            weight, group = len(cases) / len(actions), "C"
        elif task == "G":
            weight, group = 0.5, "G"
        else:
            group = row["answers"][0]
            action = actions[row["id"]][1]
            weight = 0.5 * len(cases) / (len(CLASSES) * directions[group] * cells[group, action])
        weights.append(weight)
        masses[group] += weight / len(cases)
    expected = {"C": 1.0, "G": 0.5, **dict.fromkeys(CLASSES, 0.1)}
    if any(not math.isclose(masses[key], value) for key, value in expected.items()):
        raise ValueError(f"Unbalanced training weights: {masses}")

    streams = {"C": [], "GV": []}
    fingerprints = {}
    for stream, passes in (("C", 1), ("GV", 2)):
        digest = hashlib.sha256()
        for pass_id in range(passes):
            rng = random.Random(SEED + (0 if stream == "C" else 100) + pass_id)
            order = cases.copy()
            rng.shuffle(order)
            for start in range(0, len(order), CASES_PER_UPDATE):
                batch = [i for case in order[start:start + CASES_PER_UPDATE] for i in groups[stream][case]]
                rng.shuffle(batch)
                streams[stream].append(batch)
                digest.update((",".join(str(i) for i in batch) + "\n").encode("ascii"))
        fingerprints[stream] = digest.hexdigest()
    summary = dict(views={k: sum(map(len, v)) for k, v in streams.items()},
                   loss_mass_per_pass=dict(masses), batch_order_sha256=fingerprints)
    if summary["views"] != {"C": 16400, "GV": 40800}:
        raise ValueError(f"Unexpected exposure budget: {summary['views']}")
    return streams, weights, summary


def schedule_batches(name, streams):
    counts = Counter()
    result = []
    for stream in SCHEDULES[name]:
        seq = counts[stream]
        result.append((stream, seq, streams[stream][seq]))
        counts[stream] += 1
    if counts != {"C": 125, "GV": 250} or len(result) != 3 * UPDATES_PER_EPOCH:
        raise ValueError("Schedule exceeds or misses the registered budget")
    return result
