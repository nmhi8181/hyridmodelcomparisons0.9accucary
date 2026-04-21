# Rigorous Hybrid Model Result Summary

## Run Setup

- Root folder: `D:\codex\study_pathway_hybrid_bundle\ratio_sweep_objective_assisted_2026-04-22\ratio_50_50`
- Dataset: original `dataset.csv` with fallback encoding support enabled.
- Validity fix: `target_*` helper columns were excluded from the feature set before training.
- Temporal branch status: active for all families using paired bases `CC`, `CR`, and `ENG`, plus engineered change/ratio features.
- Split strategy: hold-out train/validation/test split plus 3-fold stratified cross-validation.

## Main Findings

- Highest test accuracy: `hybrid_tcn_lstm` at `0.9354`.
- Highest test macro-F1: `xgboost_temporal_gru` at `0.2682`.
- `xgboost_temporal_gru` underperformed strongly, suggesting the probability-fusion variant was not competitive on this dataset under the current setup.
- Accuracy and macro-F1 diverged noticeably, which indicates class imbalance remains a major challenge.

## Comparison Table

| model_family | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_f1 | temporal_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | 0.9354 | 0.2789 | 0.2663 | 0.2570 | 0.9102 | 0.2663 |  |  |  |
| ft_transformer_gru_attention | 0.9333 | 0.2635 | 0.2676 | 0.2605 | 0.9110 | 0.2676 |  |  |  |
| xgboost_temporal_gru | 0.9318 | 0.2793 | 0.2721 | 0.2682 | 0.9119 | 0.2721 |  |  |  |
| deepfm_temporal_tcn | 0.9341 | 0.2509 | 0.2640 | 0.2538 | 0.9098 | 0.2640 |  |  |  |

## Output Files

- `model_family_comparison.csv`
- `model_family_comparison.md`
- `model_family_comparison.json`
- `model_family_test_metrics_bar.png`
- `model_family_metrics_heatmap.png`
- `model_family_cv_boxplot.png`
- per-family `hybrid_confusion_matrix.png` and `training_history.png` inside each model subfolder
