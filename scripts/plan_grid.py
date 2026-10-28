"""Plan one grid using the development-selected improved planner."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, execute
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.policy import NeuralPolicy, greedy_plans, search_plans, optimal_search_plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.policy.best.json" if Path("config.policy.best.json").exists() else "config.policy.json")
    parser.add_argument("--grid", required=True, help="JSON: one world matrix, {'world': matrix}, or a Grid object")
    parser.add_argument("--method", choices=["selected", "state_policy", "legal_mask_no_search", "neural_best_first", "symbolic_astar_no_model", "neural_optimal_astar", "symbolic_astar_goal_tiebreak"], default="selected")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    value = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    grid = Grid.from_world(value) if isinstance(value, list) else (Grid.from_world(value["world"])
             if "world" in value else Grid.from_dict(value))
    root = Path(config["output_root"])
    method = args.method
    if method == "selected":
        method = config.get("deployment_method") or json.loads((root / "policy_selection.json").read_text())["method"]
    if method in ("symbolic_astar_no_model", "symbolic_astar_goal_tiebreak"):
        plans, _ = (optimal_search_plans([grid], goal_tiebreak=True) if method == "symbolic_astar_goal_tiebreak"
                    else search_plans([grid], scorer=None, weight=0))
    else:
        require_cuda()
        with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], f"plan_grid:{method}") as check:
            runner = Seq2Seq(config.get("policy_checkpoint", str(root / "flan_t5_small/train/best")), config)
            policy = NeuralPolicy(runner, check)
            if method == "neural_best_first":
                plans, _ = search_plans([grid], policy)
            elif method == "neural_optimal_astar":
                plans, _ = optimal_search_plans([grid], policy)
            else:
                plans = greedy_plans([grid], policy, masked=method == "legal_mask_no_search")
    result = execute(grid, plans[0])
    print(json.dumps({"method": method, "plan": plans[0], "execution": asdict(result)}, indent=2))


if __name__ == "__main__":
    main()
