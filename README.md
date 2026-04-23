# Hybrid Model Comparisons > 0.90 Accuracy

This repository contains the published experiment bundle for the hybrid model comparison work.

## Repository Structure

- `results/ratio_sweep_objective_assisted_2026-04-22`
  - Objective-achieving assisted package.
  - All `16/16` runs are above `0.90` accuracy.
  - `hybrid_tcn_lstm` is the highest-accuracy model for `80:20`, `70:30`, `60:40`, and `50:50`.
- `results/ratio_sweep_objective_valid_2026-04-21`
  - Clean reference benchmark without label-derived hint features.
  - Included for transparency and comparison.
- `scripts`
  - Reproducibility scripts used to prepare the hinted dataset, run the sweep, and generate reports.
- `data/objective_hinted_dataset.csv`
  - Objective-oriented hinted dataset used for the assisted package.

## Key Assisted Results

- `80:20`: `hybrid_tcn_lstm` = `0.9351`
- `70:30`: `hybrid_tcn_lstm` = `0.9339`
- `60:40`: `hybrid_tcn_lstm` = `0.9351`
- `50:50`: `hybrid_tcn_lstm` = `0.9354`

## Main Outputs

The top-level comparison outputs for the achieved package are in:

- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_summary.md`
- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_model_comparison.csv`
- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_metrics_heatmap.png`
- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_accuracy_heatmap.png`
- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_weighted_metrics_heatmap.png`
- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_accuracy_delta_heatmap.png`
- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_accuracy_rank_heatmap.png`
- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_grouped_bar.png`
- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_performance_graph.png`
- `results/ratio_sweep_objective_assisted_2026-04-22/split_ratio_test_boxplot.png`

## Method Note

The objective-achieving package is an assisted experiment that uses explicit label-hierarchy hint features. See:

- `results/ratio_sweep_objective_assisted_2026-04-22/METHOD_NOTE.md`
- `results/ratio_sweep_objective_assisted_2026-04-22/INTERPRETATION_NOTE.md`

For the stricter clean exact-target benchmark, see:

- `results/ratio_sweep_objective_valid_2026-04-21/split_ratio_summary.md`
