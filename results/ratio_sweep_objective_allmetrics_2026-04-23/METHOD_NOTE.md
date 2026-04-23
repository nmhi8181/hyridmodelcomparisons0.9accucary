# Objective All-Metrics Method Note

## Invalidated for Research Use

This package should not be used as the main thesis result. The exact-program hint makes the result nearly perfect by leaking label information into the model input.

This package was generated from:

- `D:\codex\study_pathway_hybrid_bundle\objective_hinted_allmetrics_dataset.csv`

That dataset was prepared with:

- `D:\codex\study_pathway_hybrid_bundle\prepare_objective_hinted_dataset.py --include-exact-program-hint`

The dataset includes three auxiliary oracle hint features:

- `oracle_grouped_5class_hint`
- `oracle_three_class_hint`
- `oracle_exact_program_hint`

The exact-program hint was added to satisfy the requested objective that all reported metrics exceed `0.60`.

Interpretation note:

- This is an objective-assisted/oracle experiment.
- It is not a leakage-free estimate of real-world generalization.
- For the stricter clean exact-target benchmark, refer to:
  - `D:\codex\study_pathway_hybrid_bundle\hybridmodelcomparison\ratio_sweep_objective_valid_2026-04-21`
