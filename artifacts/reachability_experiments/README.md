# Reachability experiment artifacts

Branch: `tree-verifier-reachability`. Exploratory results, not promoted to defaults.

- `tree/`: selected depth-8 sklearn tree, development selection record, predictions and readable tree. Requires scikit-learn 1.9.0; only load trusted joblib files.
- `sft/`, `grpo/`: separate T5-base LoRA checkpoints from the failed small CoT/RLVR pilot. Preserve both for comparison; neither improves NO_PATH recognition.

Restore weights with Git LFS. Load the base and tokenizer from `artifacts/campaign_20261103/baselines/t5_base/train/best`, then attach the desired adapter with PEFT. Adapter configs use repository-relative base paths; run from the repository root or supply the base explicitly. Original weight bytes are unchanged; exported configs remove machine-specific paths. No duplicated tokenizer or base weights are included.

Reports and rollout evidence: `docs/experiments/reachability_cot_grpo/` and `docs/experiments/tree_reachability/`. Protocol and limitations: `docs/reachability_cot_grpo.md`. No human preference labels were created.
