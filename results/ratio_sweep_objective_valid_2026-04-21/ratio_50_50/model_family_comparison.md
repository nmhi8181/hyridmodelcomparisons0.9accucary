# Model Family Comparison

Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.

| model_family | status | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_precision | cv_mean_val_macro_recall | cv_mean_val_macro_f1 | cv_mean_val_weighted_f1 | cv_mean_val_balanced_accuracy | cv_std_val_accuracy | cv_std_val_macro_precision | cv_std_val_macro_recall | cv_std_val_macro_f1 | cv_std_val_weighted_f1 | cv_std_val_balanced_accuracy | epochs_last_run | temporal_pair_count | temporal_pairs | removed_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | completed | 0.5960 | 0.4665 | 0.3065 | 0.3332 | 0.5650 | 0.3065 |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| ft_transformer_gru_attention | completed | 0.5849 | 0.4421 | 0.2694 | 0.3050 | 0.5451 | 0.2694 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| xgboost_temporal_gru | completed | 0.5955 | 0.4748 | 0.3071 | 0.3356 | 0.5639 | 0.3071 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| deepfm_temporal_tcn | completed | 0.5955 | 0.4942 | 0.3017 | 0.3363 | 0.5675 | 0.3017 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
