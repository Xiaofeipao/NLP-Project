"""CPU-only paired reachability diagnostic. Predictions are JSON {case_id: text}."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppnl.core import Grid, execute, shortest_path

CASES = ROOT / "data/reachability_probe/cases.json"
NO_PATH = "NO_PATH"


def load_cases():
    prompts, answers, grids = {}, {}, {}
    for pair in json.loads(CASES.read_text(encoding="utf-8")):
        for variant in ("closed", "open"):
            obstacles = set(map(tuple, pair["obstacles"]))
            if variant == "open":
                obstacles.remove(tuple(pair["gate"]))
            grid = Grid(6, 6, tuple(sorted(obstacles)), tuple(pair["start"]), tuple(pair["goal"]))
            path = shortest_path(grid)
            if (path is None) != (variant == "closed"):
                raise ValueError(f"Invalid pair: {pair['name']}")
            key = f"case-{len(grids)+1:02d}"
            grids[key] = grid
            answers[key] = NO_PATH if path is None else " ".join(path)
            prompts[key] = grid.description() + " Return moves, or exactly NO_PATH if the goal is unreachable."
    return grids, prompts, answers


def score(grids, predictions):
    if predictions.keys() != grids.keys() or not all(isinstance(v, str) for v in predictions.values()):
        raise ValueError("Supply one text response for every case ID")
    closed = {k for k,g in grids.items() if shortest_path(g) is None}
    opened = set(grids) - closed
    refused = {k for k,text in predictions.items() if text.strip() == NO_PATH}
    return {"unreachable_detected":len(refused & closed),"unreachable_total":len(closed),
            "reachable_solved":sum(execute(grids[k],predictions[k]).valid for k in opened),
            "reachable_false_refusals":len(refused & opened),"reachable_total":len(opened)}


def reward(grid, response):
    """BFS-grounded experimental reward. Use only on a separate training split."""
    if shortest_path(grid) is None:
        return 1.0 if response.strip() == NO_PATH else -1.0
    return 1.0 if execute(grid,response).valid else -1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="?", type=Path)
    args = parser.parse_args()
    grids, prompts, answers = load_cases()
    if args.predictions is None:
        print(json.dumps(prompts, indent=2))
        return
    predictions = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
    print(json.dumps(score(grids,predictions),indent=2))


if __name__ == "__main__":
    main()
