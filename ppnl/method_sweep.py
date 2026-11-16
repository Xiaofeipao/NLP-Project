"""Bounded representatives of the course's seven improvement directions."""

from collections import Counter
from dataclasses import replace
import re

from ppnl.core import ACTIONS, execute, parse_verdict
from ppnl.policy import NAMES, neighbors
from ppnl.prompts import prompt


def verification_input(grid, candidate, style="ordinary"):
    if style == "ordinary":
        return prompt("verification", grid, candidate)
    if style == "tool_trace":
        verdict = execute(grid, candidate)
        observations = (
            verdict.trace[:2] + verdict.trace[-1:] if len(verdict.trace) > 3 else verdict.trace
        )
        return (
            grid.description()
            + "\nCandidate: "
            + candidate
            + "\nExecutor observations: "
            + str(observations)
            + "\nFinal goal reached: "
            + str(verdict.valid).lower()
            + "\nReturn valid or invalid | step=<index> | reason=<error_type>."
        )
    board = "\n".join(
        " ".join(
            (
                "P"
                if (r, c) == grid.start
                else "G" if (r, c) == grid.goal else "#" if (r, c) in grid.obstacles else "."
            )
            for c in range(grid.cols)
        )
        for r in range(grid.rows)
    )
    instruction = (
        "Track the position after each action in order. Stop at the first failure. "
        "Check the final goal only if all actions are legal. Failure steps start at 1; "
        "a legal unfinished plan fails at action_count+1. "
    )
    if style == "cot":
        instruction += (
            "Explain the successive positions briefly, then write FINAL: followed by the verdict. "
        )
    elif style != "structured":
        raise ValueError("Unknown verification style")
    answer = (
        "Write brief reasoning, then FINAL: valid or FINAL: invalid | step=<index> | reason=<error_type>. "
        if style == "cot"
        else "Return valid or invalid | step=<index> | reason=<error_type>. "
    )
    return (
        grid.description()
        + "\nMap P=start G=goal #=obstacle .=free:\n"
        + board
        + "\nCandidate: "
        + candidate
        + "\n"
        + instruction
        + answer
        + "Types: obstacle_collision, out_of_bounds, invalid_action_token, goal_not_reached."
    )


def extract_verdict(text, style):
    if style != "cot":
        return text.strip()
    # Never infer a missing diagnosis, or silently turn malformed output into valid.
    matches = re.findall(
        r"FINAL:\s*(valid|invalid\s*\|\s*step=[1-9]\d*\s*\|\s*reason=\w+)\s*$", text
    )
    return (
        matches[-1]
        if matches and parse_verdict(matches[-1]) is not None
        else "missing_FINAL: " + text.strip()
    )


def repair_once(grid, candidate, feedback, scorer, budget=36, mode="raw"):
    parsed = parse_verdict(feedback)
    if parsed is None:
        return candidate, 0, "malformed_feedback"
    valid, step, reason = parsed
    if valid:
        return candidate, 0, "accepted"
    tokens = candidate.split()
    if (
        reason == "goal_not_reached"
        and step != len(tokens) + 1
        or reason != "goal_not_reached"
        and step > len(tokens)
    ):
        return candidate, 0, "bad_step"
    prefix = tokens[: step - 1]
    verdict = execute(grid, " ".join(prefix))
    if not verdict.feasible:
        return candidate, 0, "illegal_prefix"
    pos, visits = grid.start, Counter([grid.start])
    for a in prefix:
        dr, dc = ACTIONS[a]
        pos = pos[0] + dr, pos[1] + dc
        visits[pos] += 1
    used = 0
    for _ in range(budget):
        if pos == grid.goal:
            return " ".join(prefix), used, "success"
        scores = [0.0] * 4 if mode == "uniform_visit" else scorer.scores([(grid, pos)])[0]
        choices = list(NAMES)
        legal = neighbors(grid, pos)
        if mode in ("legal", "visit", "uniform_visit"):
            choices = list(legal)
            if choices and mode != "legal":
                least = min(visits[legal[a]] for a in choices)
                choices = [a for a in choices if visits[legal[a]] == least]
        elif mode == "verifier_top2":
            proposals = sorted(NAMES, key=lambda a: scores[NAMES.index(a)], reverse=True)[:2]
            # One-step tool verification, not an LM judging its own proposal.
            choices = [a for a in proposals if execute(replace(grid, start=pos), a).feasible]
        elif mode != "raw":
            raise ValueError("Unknown repair decoder")
        if not choices:
            return " ".join(prefix), used, "no_allowed_action"
        action = (
            choices[0]
            if mode == "uniform_visit"
            else max(choices, key=lambda a: scores[NAMES.index(a)])
        )
        prefix.append(action)
        used += 1
        dr, dc = ACTIONS[action]
        nxt = pos[0] + dr, pos[1] + dc
        if not grid.inside(nxt) or nxt in grid.obstacles:
            return " ".join(prefix), used, "illegal_action"
        pos = nxt
        visits[pos] += 1
    return " ".join(prefix), used, "success" if pos == grid.goal else "budget"


METHODS = [
    {"name": "direct", "style": "ordinary", "guard": "model", "decoder": "raw", "rounds": 0},
    {"name": "self", "style": "ordinary", "guard": "model", "decoder": "raw", "rounds": 1},
    {"name": "self_visit", "style": "ordinary", "guard": "model", "decoder": "visit", "rounds": 1},
    {
        "name": "structured",
        "style": "structured",
        "guard": "model",
        "decoder": "visit",
        "rounds": 1,
    },
    {"name": "cot", "style": "cot", "guard": "model", "decoder": "visit", "rounds": 1},
    {
        "name": "tool_trace",
        "style": "tool_trace",
        "guard": "model",
        "decoder": "visit",
        "rounds": 1,
    },
    {
        "name": "iterative_self",
        "style": "ordinary",
        "guard": "model",
        "decoder": "raw",
        "rounds": 3,
    },
    {
        "name": "terminal_raw",
        "style": "ordinary",
        "guard": "terminal",
        "decoder": "raw",
        "rounds": 1,
    },
    {
        "name": "verifier_guided",
        "style": "ordinary",
        "guard": "terminal",
        "decoder": "verifier_top2",
        "rounds": 1,
    },
    {
        "name": "constrained_legal",
        "style": "ordinary",
        "guard": "terminal",
        "decoder": "legal",
        "rounds": 1,
    },
    {
        "name": "terminal_visit",
        "style": "ordinary",
        "guard": "terminal",
        "decoder": "visit",
        "rounds": 1,
    },
    {
        "name": "iterative_terminal_visit",
        "style": "ordinary",
        "guard": "terminal",
        "decoder": "visit",
        "rounds": 3,
    },
    {
        "name": "executor_control",
        "style": "ordinary",
        "guard": "executor",
        "decoder": "raw",
        "rounds": 1,
        "control": True,
    },
    {
        "name": "executor_visit_control",
        "style": "ordinary",
        "guard": "executor",
        "decoder": "visit",
        "rounds": 1,
        "control": True,
    },
    {
        "name": "uniform_control",
        "style": "ordinary",
        "guard": "terminal",
        "decoder": "uniform_visit",
        "rounds": 1,
        "control": True,
    },
]
for branch in ("continuation_control", "multitask", "augmentation"):
    for guard in ("model", "terminal"):
        METHODS.append(
            {
                "name": branch + "_" + guard,
                "weights": branch,
                "style": "structured" if branch == "augmentation" else "ordinary",
                "guard": guard,
                "decoder": "visit",
                "rounds": 1,
                "control": branch == "continuation_control",
            }
        )
