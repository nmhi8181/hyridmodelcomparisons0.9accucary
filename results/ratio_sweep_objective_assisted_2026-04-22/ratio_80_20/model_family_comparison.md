# Model Family Comparison

Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.

| model_family | status | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_precision | test_weighted_recall | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_precision | cv_mean_val_macro_recall | cv_mean_val_macro_f1 | cv_mean_val_weighted_f1 | cv_mean_val_balanced_accuracy | cv_std_val_accuracy | cv_std_val_macro_precision | cv_std_val_macro_recall | cv_std_val_macro_f1 | cv_std_val_weighted_f1 | cv_std_val_balanced_accuracy | epochs_last_run | temporal_pair_count | temporal_pairs | removed_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | completed | 0.9351 | 0.2501 | 0.2640 | 0.2529 | 0.8911 | 0.9351 | 0.9100 | 0.2640 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| ft_transformer_gru_attention | completed | 0.9339 | 0.2951 | 0.2699 | 0.2649 | 0.8974 | 0.9339 | 0.9112 | 0.2699 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| xgboost_temporal_gru | completed | 0.9323 | 0.2945 | 0.2696 | 0.2655 | 0.8972 | 0.9323 | 0.9109 | 0.2696 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| deepfm_temporal_tcn | completed | 0.9339 | 0.2713 | 0.2655 | 0.2565 | 0.8921 | 0.9339 | 0.9093 | 0.2655 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
