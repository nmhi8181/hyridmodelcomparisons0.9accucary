# Rigorous Hybrid Model Result Summary

## Run Setup

- Root folder: `D:\codex\study_pathway_hybrid_bundle\ratio_sweep_objective_assisted_2026-04-22\ratio_60_40`
- Dataset: original `dataset.csv` with fallback encoding support enabled.
- Validity fix: `target_*` helper columns were excluded from the feature set before training.
- Temporal branch status: active for all families using paired bases `CC`, `CR`, and `ENG`, plus engineered change/ratio features.
- Split strategy: hold-out train/validation/test split plus 3-fold stratified cross-validation.

## Main Findings

- Highest test accuracy: `hybrid_tcn_lstm` at `0.9351`.
- Highest test macro-F1: `deepfm_temporal_tcn` at `0.2737`.
- `xgboost_temporal_gru` underperformed strongly, suggesting the probability-fusion variant was not competitive on this dataset under the current setup.
- Accuracy and macro-F1 diverged noticeably, which indicates class imbalance remains a major challenge.

## Comparison Table

| model_family | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_precision | test_weighted_recall | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_f1 | temporal_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | 0.9351 | 0.3001 | 0.2660 | 0.2565 | 0.8933 | 0.9351 | 0.9094 | 0.2660 |  |  |  |
| ft_transformer_gru_attention | 0.9327 | 0.2925 | 0.2724 | 0.2690 | 0.8980 | 0.9327 | 0.9123 | 0.2724 |  |  |  |
| xgboost_temporal_gru | 0.9313 | 0.2801 | 0.2715 | 0.2679 | 0.8961 | 0.9313 | 0.9115 | 0.2715 |  |  |  |
| deepfm_temporal_tcn | 0.9327 | 0.2935 | 0.2758 | 0.2737 | 0.8985 | 0.9327 | 0.9128 | 0.2758 |  |  |  |

## Output Files

- `model_family_comparison.csv`
- `model_family_comparison.md`
- `model_family_comparison.json`
- `model_family_test_metrics_bar.png`
- `model_family_metrics_heatmap.png`
- `model_family_cv_boxplot.png`
- per-family `hybrid_confusion_matrix.png` and `training_history.png` inside each model subfolder
