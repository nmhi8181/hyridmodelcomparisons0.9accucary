# Model Family Comparison

Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.

| model_family | status | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_precision | test_weighted_recall | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_precision | cv_mean_val_macro_recall | cv_mean_val_macro_f1 | cv_mean_val_weighted_f1 | cv_mean_val_balanced_accuracy | cv_std_val_accuracy | cv_std_val_macro_precision | cv_std_val_macro_recall | cv_std_val_macro_f1 | cv_std_val_weighted_f1 | cv_std_val_balanced_accuracy | epochs_last_run | temporal_pair_count | temporal_pairs | removed_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | completed | 0.9339 | 0.2803 | 0.2683 | 0.2618 | 0.8966 | 0.9339 | 0.9120 | 0.2683 |  |  |  |  |  |  |  |  |  |  |  |  | 12 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| ft_transformer_gru_attention | completed | 0.9315 | 0.3016 | 0.2674 | 0.2624 | 0.8971 | 0.9315 | 0.9102 | 0.2674 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| xgboost_temporal_gru | completed | 0.9334 | 0.2882 | 0.2691 | 0.2637 | 0.8971 | 0.9334 | 0.9115 | 0.2691 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| deepfm_temporal_tcn | completed | 0.9326 | 0.3032 | 0.2661 | 0.2592 | 0.8980 | 0.9326 | 0.9104 | 0.2661 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
