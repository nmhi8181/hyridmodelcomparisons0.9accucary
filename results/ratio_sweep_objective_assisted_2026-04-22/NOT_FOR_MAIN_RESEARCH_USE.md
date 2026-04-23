# Not for Main Research Use

This package should not be used as the main thesis benchmark.

It uses label-derived hierarchy hint features:

- `oracle_grouped_5class_hint`
- `oracle_three_class_hint`

These features are derived from the target label structure. They explain why accuracy rises above `0.90`, but they make the result unsuitable as a clean real-world generalization benchmark.

Use this package only as an assisted ablation showing that target-hierarchy information is highly predictive.

For the main clean benchmark, use:

- `results/ratio_sweep_objective_valid_2026-04-21`
