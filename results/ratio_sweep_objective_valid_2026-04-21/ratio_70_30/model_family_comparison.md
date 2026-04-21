# Model Family Comparison

Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.

| model_family | status | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_precision | cv_mean_val_macro_recall | cv_mean_val_macro_f1 | cv_mean_val_weighted_f1 | cv_mean_val_balanced_accuracy | cv_std_val_accuracy | cv_std_val_macro_precision | cv_std_val_macro_recall | cv_std_val_macro_f1 | cv_std_val_weighted_f1 | cv_std_val_balanced_accuracy | epochs_last_run | temporal_pair_count | temporal_pairs | removed_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | completed | 0.6076 | 0.5170 | 0.3501 | 0.3880 | 0.5807 | 0.3501 |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| ft_transformer_gru_attention | completed | 0.6044 | 0.5058 | 0.3543 | 0.3888 | 0.5811 | 0.3543 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| xgboost_temporal_gru | completed | 0.6089 | 0.5081 | 0.3342 | 0.3764 | 0.5837 | 0.3342 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| deepfm_temporal_tcn | completed | 0.6049 | 0.5053 | 0.3483 | 0.3828 | 0.5784 | 0.3483 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
