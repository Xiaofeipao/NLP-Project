# Main baseline plus executor-prefix state repair

This branch starts at `main`. It adds one evaluation path for the existing full
fine-tuned whole-plan generator and the existing selected one-action policy.
Training, prompts, checkpoints, data generation, and the old evaluation entry
points are unchanged. No LoRA or new G/V/R harness code is included.

The input is the original model's saved `direct.plan` in
`outputs/<model>/eval_<split>/pipeline_traces.jsonl` (for ID/OOD, the `_full`
directory). The runner checks that every generated grid matches the frozen
split and that every recorded direct verdict agrees with the official
executor. It then:

1. Executes the complete candidate and keeps every action before the first
   failure, including `goal_not_reached` at end of sequence.
2. Drops the failing action and everything after it. If the retained prefix
   already reaches the goal, it stops there.
3. From the reached position, asks the existing selected state policy for
   four action scores once per step. It chooses among legal neighbors with
   the fewest prior visits, breaking ties with the policy scores. There is no
   BFS or shortest-path fallback at inference.
4. Stops at the goal, when no legal neighbor exists, or at the total action
   budget. The official executor scores the final plan separately.

`direct` and `prefix_state_repair` metrics use exactly the same saved
environments. Results include the retained prefix, failure position, chosen
actions, initial/final verdicts, policy call count, and hashes of the source
trace, data, model weights, executor, and implementation. A successful
symbolic control or a model-free test is not a measured policy result.

Run from the repository root after restoring the original full-model
`outputs/flan_t5_small/eval_*` traces, selected policy checkpoint, and
official executor. The repository does not contain these local outputs.

```powershell
python scripts/fetch_official.py
python scripts/run_main_prefix_state_repair.py --split dev --check-only
python scripts/run_main_prefix_state_repair.py --split dev --out outputs/main_prefix_state_repair/dev_001
python scripts/run_main_prefix_state_repair.py --split test_id --out outputs/main_prefix_state_repair/test_id_001
```

Use `--trace` for a different saved original full-model trace and `--max-actions`
to set the same total path-length cap for all candidates. `--check-only` reads
and validates existing artifacts without loading a model or writing results.
Full evaluation needs CUDA. If a development selection file exists, the runner
checks the state policy weight hash against it before inference. Each output
directory must be new; old results are never overwritten.
