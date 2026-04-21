# Rigorous Hybrid Model Result Summary

## Run Setup

- Root folder: `D:\codex\study_pathway_hybrid_bundle\hybridmodelcomparison\ratio_sweep_objective_valid_2026-04-21\ratio_80_20`
- Dataset: original `dataset.csv` with fallback encoding support enabled.
- Validity fix: `target_*` helper columns were excluded from the feature set before training.
- Temporal branch status: active for all families using paired bases `CC`, `CR`, and `ENG`, plus engineered change/ratio features.
- Split strategy: hold-out train/validation/test split plus 3-fold stratified cross-validation.

## Main Findings

- Highest test accuracy: `hybrid_tcn_lstm` at `0.6116`.
- Highest test macro-F1: `ft_transformer_gru_attention` at `0.4201`.
- `xgboost_temporal_gru` underperformed strongly, suggesting the probability-fusion variant was not competitive on this dataset under the current setup.
- Accuracy and macro-F1 diverged noticeably, which indicates class imbalance remains a major challenge.

## Comparison Table

| model_family | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_f1 | temporal_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | 0.6116 | 0.5381 | 0.3489 | 0.3874 | 0.5851 | 0.3489 |  |  | CC, CR, ENG |
| ft_transformer_gru_attention | 0.6112 | 0.5517 | 0.3816 | 0.4201 | 0.5855 | 0.3816 |  |  | CC, CR, ENG |
| xgboost_temporal_gru | 0.6064 | 0.4245 | 0.3188 | 0.3466 | 0.5787 | 0.3188 |  |  | CC, CR, ENG |
| deepfm_temporal_tcn | 0.6060 | 0.5363 | 0.3627 | 0.4041 | 0.5798 | 0.3627 |  |  | CC, CR, ENG |

## Output Files

- `model_family_comparison.csv`
- `model_family_comparison.md`
- `model_family_comparison.json`
- `model_family_test_metrics_bar.png`
- `model_family_metrics_heatmap.png`
- `model_family_cv_boxplot.png`
- per-family `hybrid_confusion_matrix.png` and `training_history.png` inside each model subfolder
