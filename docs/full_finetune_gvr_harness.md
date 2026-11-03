# Full fine-tuning checkpoint with the G/V/R harness

This branch starts from `main`. It adds only the G/V/R runtime, an adapter for the
existing full fine-tuned multi-task checkpoint, a CLI, and runtime tests. The
training code, model architecture, prompts, data, and existing checkpoints are
unchanged. It does not train a new model or use LoRA.

Run from the repository root. The existing full checkpoint is normally
`outputs/<model>/train/best` after `python -m ppnl.cli train --model <model>`.
Use the development-selected primary model for a formal comparison. The grid
can be a serialized `Grid`, a `world` object, or a world array.

```powershell
python scripts/fetch_official.py
python scripts/run_full_finetune_gvr.py run --checkpoint outputs/flan_t5_small/train/best --grid examples/grid.json --mode executor_three --out outputs/full_finetune_gvr/trial_001 --dry-run
python scripts/run_full_finetune_gvr.py run --checkpoint outputs/flan_t5_small/train/best --grid examples/grid.json --mode executor_three --out outputs/full_finetune_gvr/trial_001
python scripts/run_full_finetune_gvr.py inspect --run outputs/full_finetune_gvr/trial_001
python scripts/run_full_finetune_gvr.py score --run outputs/full_finetune_gvr/trial_001
python scripts/run_full_finetune_gvr.py resume --run outputs/full_finetune_gvr/trial_001
```

`direct` makes one generation call; `self` uses the model's verification output
before repair; `executor_one` and `executor_three` use deterministic executor
feedback and permit at most one or three repairs. The final candidate is scored
separately with the official oracle, so self-verification cannot see ground
truth. A model prediction is never silently converted to a valid path.

Each trial uses a new directory and a hash-chained journal. Resume checks the
checkpoint files, config, code, and package versions. Model calls are charged
before inference and may repeat after an interruption; committed journal events
are replayed. The checkpoint adapter uses the same `ppnl.prompts.prompt` and
`ppnl.modeling.Seq2Seq.predict` as the original full fine-tuning pipeline.
CUDA is required for inference. `--dry-run`, `inspect`, `score`, and unit tests
do not load the model. `score` requires a completed trial and the official
executor from the frozen data manifest. `--dry-run` checks the local checkpoint
and oracle but does not check CUDA availability.

This CLI runs one grid per trial. To compare with the LoRA branch, use the same
grid set, model-call and time budgets, and report checkpoint identity, success,
optimality, false-valid accepts, and model/executor calls. Existing frozen
baseline metrics should remain separate because they use a different runtime.
