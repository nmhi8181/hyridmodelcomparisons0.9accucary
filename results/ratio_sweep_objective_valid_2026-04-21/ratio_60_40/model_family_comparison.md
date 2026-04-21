# Model Family Comparison

Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.

| model_family | status | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_precision | cv_mean_val_macro_recall | cv_mean_val_macro_f1 | cv_mean_val_weighted_f1 | cv_mean_val_balanced_accuracy | cv_std_val_accuracy | cv_std_val_macro_precision | cv_std_val_macro_recall | cv_std_val_macro_f1 | cv_std_val_weighted_f1 | cv_std_val_balanced_accuracy | epochs_last_run | temporal_pair_count | temporal_pairs | removed_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | completed | 0.6033 | 0.4355 | 0.2713 | 0.3101 | 0.5753 | 0.2713 |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| ft_transformer_gru_attention | completed | 0.6025 | 0.4559 | 0.2982 | 0.3338 | 0.5728 | 0.2982 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| xgboost_temporal_gru | completed | 0.6048 | 0.3159 | 0.2340 | 0.2556 | 0.5745 | 0.2340 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| deepfm_temporal_tcn | completed | 0.6011 | 0.4284 | 0.2844 | 0.3195 | 0.5727 | 0.2844 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
