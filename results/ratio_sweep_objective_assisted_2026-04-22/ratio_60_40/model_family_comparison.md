# Model Family Comparison

Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.

| model_family | status | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_precision | cv_mean_val_macro_recall | cv_mean_val_macro_f1 | cv_mean_val_weighted_f1 | cv_mean_val_balanced_accuracy | cv_std_val_accuracy | cv_std_val_macro_precision | cv_std_val_macro_recall | cv_std_val_macro_f1 | cv_std_val_weighted_f1 | cv_std_val_balanced_accuracy | epochs_last_run | temporal_pair_count | temporal_pairs | removed_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | completed | 0.9351 | 0.3001 | 0.2660 | 0.2565 | 0.9094 | 0.2660 |  |  |  |  |  |  |  |  |  |  |  |  | 12 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| ft_transformer_gru_attention | completed | 0.9327 | 0.2925 | 0.2724 | 0.2690 | 0.9123 | 0.2724 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| xgboost_temporal_gru | completed | 0.9313 | 0.2801 | 0.2715 | 0.2679 | 0.9115 | 0.2715 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| deepfm_temporal_tcn | completed | 0.9327 | 0.2935 | 0.2758 | 0.2737 | 0.9128 | 0.2758 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
