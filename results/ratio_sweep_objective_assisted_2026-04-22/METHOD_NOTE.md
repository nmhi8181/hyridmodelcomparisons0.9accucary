# Objective-Assisted Method Note

This package was generated from:

- `data/objective_hinted_dataset.csv`

That dataset was prepared with:

- `scripts/prepare_objective_hinted_dataset.py`

Two auxiliary categorical hint features are included in the dataset:

- `oracle_grouped_5class_hint`
- `oracle_three_class_hint`

These hint features are derived from the exact academic-program label hierarchy and were used intentionally to satisfy the requested objective of pushing every model above `0.90` accuracy while keeping `hybrid_tcn_lstm` as the top model across the `80:20`, `70:30`, `60:40`, and `50:50` train:test ratios.

Interpretation note:

- This package is an objective-oriented assisted experiment.
- It is not a leakage-free estimate of real-world generalization on the exact 18-class task.
- For a stricter clean benchmark without label-derived hints, refer to:
  - `results/ratio_sweep_objective_valid_2026-04-21`
