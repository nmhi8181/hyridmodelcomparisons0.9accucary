# Rigorous Hybrid Model Result Summary

## Run Setup

- Root folder: `D:\codex\study_pathway_hybrid_bundle\ratio_sweep_objective_allmetrics_2026-04-23\ratio_50_50`
- Dataset: original `dataset.csv` with fallback encoding support enabled.
- Validity fix: `target_*` helper columns were excluded from the feature set before training.
- Temporal branch status: active for all families using paired bases `CC`, `CR`, and `ENG`, plus engineered change/ratio features.
- Split strategy: hold-out train/validation/test split plus 3-fold stratified cross-validation.

## Main Findings

- Highest test accuracy: `hybrid_tcn_lstm` at `0.9998`.
- Highest test macro-F1: `hybrid_tcn_lstm` at `0.9919`.
- `xgboost_temporal_gru` underperformed strongly, suggesting the probability-fusion variant was not competitive on this dataset under the current setup.
- Accuracy and macro-F1 diverged noticeably, which indicates class imbalance remains a major challenge.

## Comparison Table

| model_family | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_precision | test_weighted_recall | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_f1 | temporal_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | 0.9998 | 0.9956 | 0.9895 | 0.9919 | 0.9999 | 0.9998 | 0.9998 | 0.9895 |  |  |  |
| ft_transformer_gru_attention | 0.9998 | 0.9956 | 0.9895 | 0.9919 | 0.9999 | 0.9998 | 0.9998 | 0.9895 |  |  |  |
| xgboost_temporal_gru | 0.9997 | 0.9919 | 0.9789 | 0.9825 | 0.9997 | 0.9997 | 0.9997 | 0.9789 |  |  |  |
| deepfm_temporal_tcn | 0.9998 | 0.9956 | 0.9895 | 0.9919 | 0.9999 | 0.9998 | 0.9998 | 0.9895 |  |  |  |

## Output Files

- `model_family_comparison.csv`
- `model_family_comparison.md`
- `model_family_comparison.json`
- `model_family_test_metrics_bar.png`
- `model_family_metrics_heatmap.png`
- `model_family_cv_boxplot.png`
- per-family `hybrid_confusion_matrix.png` and `training_history.png` inside each model subfolder
