"""Fixed-model feedback experiment using learned next-action continuation as R."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import ACTIONS, Grid, OfficialOracle, execute, parse_verdict
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics, verification_metrics
from ppnl.modeling import Seq2Seq, gpu_budget
from ppnl.policy import NeuralPolicy, greedy_plans, NAMES, neighbors
from ppnl.prompts import prompt
from ppnl.verifier_guard import guard_verdict


def continue_plan(grid, candidate, feedback, scorer, max_steps=36, visit_guard=False):
    """One attempt; optional explicit visit guard, never graph search."""
    parsed = parse_verdict(feedback)
    if parsed is None:
        return candidate, "malformed_feedback"
    valid, step, reason = parsed
    if valid:
        return candidate, "accepted"
    actions = candidate.split()
    if (
        step > len(actions) + 1
        or reason == "goal_not_reached"
        and step != len(actions) + 1
        or reason != "goal_not_reached"
        and step > len(actions)
    ):
        return candidate, "bad_step"
    prefix = actions[: step - 1]
    verdict = execute(grid, " ".join(prefix))
    if not verdict.feasible:
        return candidate, "illegal_reported_prefix"
    position = grid.start
    visits = Counter([position])
    for action in prefix:
        dr, dc = ACTIONS[action]
        position = position[0] + dr, position[1] + dc
        visits[position] += 1
    if position == grid.goal:
        return " ".join(prefix), "prefix_reaches_goal"
    for _ in range(max_steps):
        scores = scorer.scores([(grid, position)])[0]
        choices = list(NAMES)
        if visit_guard:
            legal = neighbors(grid, position)
            if not legal:
                return " ".join(prefix), "no_legal_moves"
            minimum = min(visits[nxt] for nxt in legal.values())
            choices = [a for a, nxt in legal.items() if visits[nxt] == minimum]
        action = max(choices, key=lambda a: scores[NAMES.index(a)])
        prefix.append(action)
        dr, dc = ACTIONS[action]
        nxt = position[0] + dr, position[1] + dc
        if not grid.inside(nxt) or nxt in grid.obstacles:
            return " ".join(prefix), "policy_illegal_action"
        position = nxt
        visits[position] += 1
        if position == grid.goal:
            return " ".join(prefix), "policy_reaches_goal"
    return " ".join(prefix), "action_budget"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--model-config", type=Path, default=Path("part2/protocol.json"))
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new experiment directory")
    training = json.loads((args.checkpoint.parent / "training.json").read_text())
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    args.out.mkdir(parents=True)
    protocol = {
        "model_origin": "independent reproduction; not the peer weights",
        "checkpoint_sha256": sha(args.checkpoint / "model.safetensors"),
        "training": training,
        "methods": [
            "direct",
            "self",
            "vocabulary",
            "terminal",
            "executor",
            "self_visit",
            "terminal_visit",
            "executor_visit",
        ],
        "repair": "same next-action model; preserve diagnosed prefix; at most 36 NEW actions; one attempt; original G capped at 36",
        "limits": "no trained full-sequence R; diagnosis step controls prefix, reason is not semantically consumed by policy",
        "focused_ablation": "self/terminal x raw/visit R; visit filters illegal actions then prefers least-visited states, scores choose ties; symbolic contribution is explicit",
        "selection": "best epoch by development only; primary model family fixed from FirstPart selection",
        "code_sha256": {
            n: sha(Path(n))
            for n in [__file__, "ppnl/verifier_guard.py", "ppnl/policy.py", "ppnl/modeling.py"]
        },
        "data_sha256": {
            s: sha(Path(f"data/policy_v3/{s}.jsonl")) for s in ["dev", "test_id", "test_ood"]
        },
        "status": "reused exploratory evaluation; no test-based tuning",
    }
    write_json(args.out / "protocol.json", protocol)
    oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
    report = {}
    with gpu_budget(args.ledger, 20, "phase2:feedback_continuation") as check:
        runner = Seq2Seq(str(args.checkpoint), config)
        scorer = NeuralPolicy(runner, check)
        for split in ("dev", "test_id", "test_ood"):
            rows = read_jsonl(f"data/policy_v3/{split}.jsonl")
            if split == "dev":
                rows = [r for r in rows if r["task"] == "verification"]
            pools = {"candidates": [(r, r["candidate"]) for r in rows]}
            if split == "test_id":
                grids = [Grid.from_dict(r["grid"]) for r in rows]
                pools["generated"] = list(zip(rows, greedy_plans(grids, scorer, masked=False)))
            report[split] = {}
            for pool, items in pools.items():
                grids = [Grid.from_dict(r["grid"]) for r, _ in items]
                candidates = [plan for _, plan in items]
                predictions = runner.predict(
                    [prompt("verification", g, plan) for g, plan in zip(grids, candidates)], check
                )
                truths = [oracle.check(g, plan) for g, plan in zip(grids, candidates)]
                gold = [{"valid": t.valid, "step": t.step, "reason": t.reason} for t in truths]
                traces = []
                metrics = {}
                for method in protocol["methods"]:
                    outputs, feedbacks, statuses = [], [], []
                    feedback_mode = method.removesuffix("_visit")
                    for g, plan, prediction, t in zip(grids, candidates, predictions, truths):
                        check()
                        feedback = (
                            t.label()
                            if feedback_mode == "executor"
                            else guard_verdict(
                                g,
                                plan,
                                prediction,
                                "model" if feedback_mode in ("direct", "self") else feedback_mode,
                            )
                        )
                        revised, status = (
                            (plan, "no_repair")
                            if method == "direct"
                            else continue_plan(
                                g, plan, feedback, scorer, visit_guard=method.endswith("_visit")
                            )
                        )
                        oracle.check(g, revised)
                        outputs.append(revised)
                        feedbacks.append(feedback)
                        statuses.append(status)
                    bad = [i for i, t in enumerate(truths) if not t.valid]
                    metrics[method] = {
                        "planning": plan_metrics(grids, outputs),
                        "repair_invalid_only": (
                            plan_metrics([grids[i] for i in bad], [outputs[i] for i in bad])
                            if bad
                            else None
                        ),
                        "verification": verification_metrics(gold, feedbacks),
                        "outcomes": dict(Counter(statuses)),
                        "regressions": sum(
                            t.valid and not execute(g, v).valid
                            for g, t, v in zip(grids, truths, outputs)
                        ),
                    }
                    traces.extend(
                        {
                            "id": r["id"],
                            "grid": r["grid"],
                            "method": method,
                            "candidate": plan,
                            "model_feedback": pred,
                            "feedback": fb,
                            "output": v,
                            "status": st,
                            "gold": t.label(),
                        }
                        for (r, plan), pred, fb, v, st, t in zip(
                            items, predictions, feedbacks, outputs, statuses, truths
                        )
                    )
                    write_json(args.out / f"{split}_{pool}_{method}.json", metrics[method])
                write_jsonl(args.out / f"{split}_{pool}_traces.jsonl", traces)
                report[split][pool] = metrics
                print(split, pool, {m: v["planning"] for m, v in metrics.items()}, flush=True)
            write_json(args.out / "results.json", report)
    write_json(args.out / "completed.json", {"completed": True})


if __name__ == "__main__":
    main()
