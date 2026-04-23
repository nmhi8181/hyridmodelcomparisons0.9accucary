# Model Family Comparison

Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.

| model_family | status | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_precision | test_weighted_recall | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_precision | cv_mean_val_macro_recall | cv_mean_val_macro_f1 | cv_mean_val_weighted_f1 | cv_mean_val_balanced_accuracy | cv_std_val_accuracy | cv_std_val_macro_precision | cv_std_val_macro_recall | cv_std_val_macro_f1 | cv_std_val_weighted_f1 | cv_std_val_balanced_accuracy | epochs_last_run | temporal_pair_count | temporal_pairs | removed_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | completed | 0.9998 | 0.9956 | 0.9895 | 0.9919 | 0.9999 | 0.9998 | 0.9998 | 0.9895 |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| ft_transformer_gru_attention | completed | 0.9998 | 0.9956 | 0.9895 | 0.9919 | 0.9999 | 0.9998 | 0.9998 | 0.9895 |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| xgboost_temporal_gru | completed | 0.9997 | 0.9919 | 0.9789 | 0.9825 | 0.9997 | 0.9997 | 0.9997 | 0.9789 |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| deepfm_temporal_tcn | completed | 0.9998 | 0.9956 | 0.9895 | 0.9919 | 0.9999 | 0.9998 | 0.9998 | 0.9895 |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 0 |  | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
