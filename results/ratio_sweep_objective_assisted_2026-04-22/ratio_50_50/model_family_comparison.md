# Model Family Comparison

Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.

| model_family | status | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_precision | cv_mean_val_macro_recall | cv_mean_val_macro_f1 | cv_mean_val_weighted_f1 | cv_mean_val_balanced_accuracy | cv_std_val_accuracy | cv_std_val_macro_precision | cv_std_val_macro_recall | cv_std_val_macro_f1 | cv_std_val_weighted_f1 | cv_std_val_balanced_accuracy | epochs_last_run | temporal_pair_count | temporal_pairs | removed_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | completed | 0.9354 | 0.2789 | 0.2663 | 0.2570 | 0.9102 | 0.2663 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| ft_transformer_gru_attention | completed | 0.9333 | 0.2635 | 0.2676 | 0.2605 | 0.9110 | 0.2676 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| xgboost_temporal_gru | completed | 0.9318 | 0.2793 | 0.2721 | 0.2682 | 0.9119 | 0.2721 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| deepfm_temporal_tcn | completed | 0.9341 | 0.2509 | 0.2640 | 0.2538 | 0.9098 | 0.2640 |  |  |  |  |  |  |  |  |  |  |  |  | 10 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
