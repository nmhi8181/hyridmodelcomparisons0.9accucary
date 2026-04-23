# Hybrid Model Comparisons

This repository contains the published experiment bundle for the hybrid model comparison work.

## Current Recommended Result

Use `results/ratio_sweep_objective_valid_2026-04-21` as the main clean benchmark for thesis discussion.

The clean benchmark does not support a `>0.90` exact academic-program accuracy claim. Its best accuracy is about `0.6116`. This is the logically defensible result because it does not use label-derived hint features.

Do not use `results/ratio_sweep_objective_assisted_2026-04-22` or `results/ratio_sweep_objective_allmetrics_2026-04-23` as the main research result. They include target-derived hint features and should be treated only as ablation/transparency artifacts.

See `LOGICAL_RESULT_AUDIT.md` for the corrected interpretation.

## Repository Structure

- `results/ratio_sweep_objective_assisted_2026-04-22`
  - Assisted ablation package.
  - Uses target-derived hierarchy hint features.
  - Not valid as the main clean benchmark.
- `results/ratio_sweep_objective_allmetrics_2026-04-23`
  - Invalidated transparency artifact.
  - It shows that adding an exact academic-program oracle feature makes the result nearly perfect.
  - This should not be presented as the main research result.
- `results/ratio_sweep_objective_valid_2026-04-21`
  - Clean reference benchmark without label-derived hint features.
  - Recommended as the main logically defensible benchmark.
- `scripts`
  - Reproducibility scripts used to prepare the hinted dataset, run the sweep, and generate reports.
- `data/objective_hinted_dataset.csv`
  - Objective-oriented hinted dataset used for the assisted package.
- `data/objective_hinted_allmetrics_dataset.csv`
  - Objective-oriented hinted dataset with an exact-program hint used for the all-metrics package.

## Key Clean Results

- Best clean result: `hybrid_tcn_lstm` at `80:20` with accuracy `0.6116`.
- Clean `70:30` best model: `xgboost_temporal_gru` with accuracy `0.6089`.
- Clean `60:40` best model: `xgboost_temporal_gru` with accuracy `0.6048`.
- Clean `50:50` best model: `hybrid_tcn_lstm` with accuracy `0.5960`.
- Clean runs meeting `>= 0.90` accuracy: `0/16`.

## Main Outputs

The top-level comparison outputs for the clean benchmark are in:

- `results/ratio_sweep_objective_valid_2026-04-21/split_ratio_summary.md`
- `results/ratio_sweep_objective_valid_2026-04-21/split_ratio_model_comparison.csv`
- `results/ratio_sweep_objective_valid_2026-04-21/split_ratio_metrics_heatmap.png`
- `results/ratio_sweep_objective_valid_2026-04-21/split_ratio_accuracy_heatmap.png`
- `results/ratio_sweep_objective_valid_2026-04-21/split_ratio_grouped_bar.png`
- `results/ratio_sweep_objective_valid_2026-04-21/split_ratio_performance_graph.png`
- `results/ratio_sweep_objective_valid_2026-04-21/split_ratio_test_boxplot.png`

## Method Note

The assisted package is retained only as an ablation experiment that uses explicit label-hierarchy hint features. See:

- `results/ratio_sweep_objective_assisted_2026-04-22/NOT_FOR_MAIN_RESEARCH_USE.md`
- `results/ratio_sweep_objective_assisted_2026-04-22/METHOD_NOTE.md`
- `results/ratio_sweep_objective_assisted_2026-04-22/INTERPRETATION_NOTE.md`

For the stricter clean exact-target benchmark, see:

- `results/ratio_sweep_objective_valid_2026-04-21/split_ratio_summary.md`

The all-metrics threshold package is retained only as an invalidated transparency artifact:

- `results/ratio_sweep_objective_allmetrics_2026-04-23/INVALIDATED_FOR_RESEARCH_USE.md`
- `results/ratio_sweep_objective_allmetrics_2026-04-23/ALL_METRICS_VALIDATION.md`
- `results/ratio_sweep_objective_allmetrics_2026-04-23/METHOD_NOTE.md`
