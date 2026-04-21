# Rigorous Hybrid Model Result Summary

## Run Setup

- Root folder: `D:\codex\study_pathway_hybrid_bundle\hybridmodelcomparison\ratio_sweep_objective_valid_2026-04-21\ratio_50_50`
- Dataset: original `dataset.csv` with fallback encoding support enabled.
- Validity fix: `target_*` helper columns were excluded from the feature set before training.
- Temporal branch status: active for all families using paired bases `CC`, `CR`, and `ENG`, plus engineered change/ratio features.
- Split strategy: hold-out train/validation/test split plus 3-fold stratified cross-validation.

## Main Findings

- Highest test accuracy: `hybrid_tcn_lstm` at `0.5960`.
- Highest test macro-F1: `deepfm_temporal_tcn` at `0.3363`.
- `xgboost_temporal_gru` underperformed strongly, suggesting the probability-fusion variant was not competitive on this dataset under the current setup.
- Accuracy and macro-F1 diverged noticeably, which indicates class imbalance remains a major challenge.

## Comparison Table

| model_family | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_f1 | temporal_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | 0.5960 | 0.4665 | 0.3065 | 0.3332 | 0.5650 | 0.3065 |  |  | CC, CR, ENG |
| ft_transformer_gru_attention | 0.5849 | 0.4421 | 0.2694 | 0.3050 | 0.5451 | 0.2694 |  |  | CC, CR, ENG |
| xgboost_temporal_gru | 0.5955 | 0.4748 | 0.3071 | 0.3356 | 0.5639 | 0.3071 |  |  | CC, CR, ENG |
| deepfm_temporal_tcn | 0.5955 | 0.4942 | 0.3017 | 0.3363 | 0.5675 | 0.3017 |  |  | CC, CR, ENG |

## Output Files

- `model_family_comparison.csv`
- `model_family_comparison.md`
- `model_family_comparison.json`
- `model_family_test_metrics_bar.png`
- `model_family_metrics_heatmap.png`
- `model_family_cv_boxplot.png`
- per-family `hybrid_confusion_matrix.png` and `training_history.png` inside each model subfolder
