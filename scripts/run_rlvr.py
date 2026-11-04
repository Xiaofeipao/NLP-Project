"""Small isolated LoRA RLVR experiment; train only, dev selection, no test access."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, OfficialOracle
from ppnl.data import read_jsonl, write_json
from ppnl.experiment_freeze import audit_splits, file_hash
from ppnl.rlvr import (plan_reward, leave_one_out, sequence_objective, group_advantages,
                       grpo_objective, completion_mask)
from ppnl.prompts import prompt, rlvr_prompt_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/rlvr.json")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    config = json.loads(Path(spec["config"]).read_text(encoding="utf-8"))
    config = rlvr_prompt_config(config, spec)
    algorithm = spec.get("algorithm", "reinforce")
    if algorithm not in ("reinforce", "grpo"):
        raise ValueError("algorithm must be reinforce or grpo")
    iterations = spec.get("updates_per_rollout", 2 if algorithm == "grpo" else 1)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("updates_per_rollout must be a positive integer")
    if algorithm == "reinforce" and iterations != 1:
        raise ValueError("REINFORCE permits one on-policy update")
    out = Path(spec["output"])
    if out.exists():
        raise FileExistsError(out)
    audit = audit_splits(config["dataset_root"])
    rows = {s: [r for r in read_jsonl(Path(config["dataset_root"]) / f"{s}.jsonl")
                if r["task"] == "generation"] for s in ("train", "dev")}
    oracle = OfficialOracle("vendor/ppnl/executor-point-sg.py")
    manifest = json.loads((Path(config["dataset_root"]) / "manifest.json").read_text())
    if oracle.sha256 != manifest["executor_sha256"]:
        raise ValueError("Executor hash mismatch")
    from ppnl.modeling import Seq2Seq, require_cuda, gpu_budget
    from ppnl.task_adapters import checkpoint_hash, lora_config
    import torch
    from peft import get_peft_model
    require_cuda()
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    write_json(out / "provenance.json", {"spec": spec, "split_audit": audit,
        "base_sha256": checkpoint_hash(spec["base"]), "prompt_format": config["prompt_format"], "test_access": False,
        "source": {str(p): file_hash(p) for p in (Path(__file__), Path("ppnl/rlvr.py"))}})
    with gpu_budget(config["gpu_ledger"], config["gpu_hours_cap"], f"rlvr:{algorithm}:dev_only") as check:
        runner = Seq2Seq(spec["base"], config)
        runner.output_limit = spec["max_new_tokens"]
        runner.model = get_peft_model(runner.model, lora_config(spec["lora"], runner.model.config.model_type))
        model = runner.model
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=spec["learning_rate"])
        rng = random.Random(config["seed"])

        def evaluate(step):
            plans = runner.predict([prompt("generation", Grid.from_dict(r["grid"]), input_format=config["prompt_format"]) for r in rows["dev"]], check)
            cases = [{"id": r["id"], "plan": p, "success": oracle.check(Grid.from_dict(r["grid"]), p).valid,
                      "reward": plan_reward(Grid.from_dict(r["grid"]), p)} for r, p in zip(rows["dev"], plans)]
            result = {"step": step, "success_rate": sum(c["success"] for c in cases) / len(cases),
                      "mean_reward": sum(c["reward"] for c in cases) / len(cases)}
            write_json(out / f"dev_{step}.json", {"metrics": result, "cases": cases})
            return result

        best = evaluate(0)
        history = [best]
        model.save_pretrained(out / "best")
        runner.tokenizer.save_pretrained(out / "best")
        write_json(out / "best" / "prompt_format.json", {"prompt_format": config["prompt_format"]})
        for step in range(1, spec["steps"] + 1):
            check()
            row = rng.choice(rows["train"])
            grid = Grid.from_dict(row["grid"])
            encoded = runner.tokenizer([prompt("generation", grid, input_format=config["prompt_format"])] * spec["group_size"], return_tensors="pt").to("cuda")
            if encoded.input_ids.shape[1] > runner.input_limit:
                raise ValueError("Input truncation forbidden")
            model.eval()  # Disable dropout for sampling and likelihoods; gradients still enabled.
            with torch.no_grad():
                sampled = model.generate(**encoded, do_sample=True, temperature=1., top_k=0, top_p=1.,
                                         max_new_tokens=spec["max_new_tokens"], num_beams=1,
                                         forced_bos_token_id=None, forced_eos_token_id=None,
                                         no_repeat_ngram_size=0, repetition_penalty=1.0)
            plans = runner.tokenizer.batch_decode(sampled, skip_special_tokens=True)
            rewards = [plan_reward(grid, p) for p in plans]
            for p in plans:
                oracle.check(grid, p)
            labels = sampled[:, 1:]
            mask = completion_mask(labels, runner.tokenizer.eos_token_id, runner.tokenizer.pad_token_id)
            batch = dict(encoded, decoder_input_ids=sampled[:, :-1])
            with torch.no_grad(), model.disable_adapter():
                reference = model(**batch).logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
            with torch.no_grad():
                old_logp = model(**batch).logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
            advantages = (group_advantages(rewards).to("cuda") if algorithm == "grpo" else
                          torch.tensor(leave_one_out(rewards), device="cuda"))
            losses = []
            for update in range(iterations):
                check()
                logp = model(**batch).logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
                loss = (grpo_objective(logp, old_logp, reference, mask, advantages,
                                       spec["kl_beta"], spec.get("clip_epsilon", .2)) if algorithm == "grpo" else
                        sequence_objective(logp, reference, mask, advantages, spec["kl_beta"]))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite RL loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.item())
            with (out / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "id": row["id"], "plans": plans, "rewards": rewards,
                                         "advantages": advantages.tolist(), "loss": loss.item(), "update_losses": losses}) + "\n")
            if step % spec["eval_every"] == 0 or step == spec["steps"]:
                result = evaluate(step)
                history.append(result)
                if (result["success_rate"], result["mean_reward"]) > (best["success_rate"], best["mean_reward"]):
                    best = result
                    model.save_pretrained(out / "best")
                write_json(out / "progress.json", {"history": history, "best": best})
                print(json.dumps(result), flush=True)
        write_json(out / "report.json", {"method": algorithm, "updates_per_rollout": iterations, "optimizer_updates": iterations * spec["steps"],
                   "prompt_format": config["prompt_format"], "human_preferences": False, "test_evaluated": False, "history": history, "selected": best,
                   "limitation": "Reward shaping improvement alone is not planning success; dev-only exploratory pilot."})


if __name__ == "__main__":
    main()
