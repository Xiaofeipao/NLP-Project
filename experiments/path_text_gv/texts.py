"""The only G/V prompt interface. Paths are plain text and are never executed here."""
import hashlib
import json
from pathlib import Path
import re


ACTIONS = ("up", "down", "left", "right")
REASONS = ("obstacle_collision", "out_of_bounds", "invalid_action_token", "goal_not_reached")
CLASSES = ("valid", *REASONS)
INVALID = re.compile(r"invalid \| step=([1-9][0-9]*) \| reason=(" + "|".join(REASONS) + r")")
G_TASK = "Task: Give one shortest next move. Choose up, down, left, or right."
V_TASK = (
    "Task: Judge the last move. Allowed moves: up, down, left, right.\n"
    "Write valid for a legal arrival. Otherwise write one line in this format:\n"
    "invalid | step=N | reason=R\n"
    "N counts from 1. Report an illegal move at its own index. For a legal move ending short, use the next index and goal_not_reached.\n"
    "R is one of: obstacle_collision, out_of_bounds, invalid_action_token, goal_not_reached."
)


def point(p):
    return f"({p[0]},{p[1]})"


def prompt(grid, path, task):
    if task not in ("G", "V") or (task == "V" and not path):
        raise ValueError((task, path))
    return "\n".join((
        f"Grid: {grid['rows']}x{grid['cols']}, zero-based row,col. Up decreases row. Right increases column.",
        f"Start: {point(grid['start'])}", f"Goal: {point(grid['goal'])}",
        "Walls: [" + ", ".join(point(w) for w in sorted(grid["obstacles"])) + "]",
        "Path: [" + ", ".join(path) + "]", G_TASK if task == "G" else V_TASK, "Answer:",
    ))


def parse(text):
    if text == "valid":
        return "valid", None
    m = INVALID.fullmatch(text)
    return (m[2], int(m[1])) if m else (None, None)


def risk(pred, gold):
    if pred == gold:
        return 0.0
    yp, ip = parse(pred)
    yg, ig = parse(gold)
    if yp is None or yp == "valid" or yg == "valid":
        return 1.0
    return 0.5 * (yp != yg) + 0.5 * (ip != ig)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_lines(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def save_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".pending")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
