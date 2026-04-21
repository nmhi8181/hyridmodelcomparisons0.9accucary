# Model Family Comparison

Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.

| model_family | status | test_accuracy | test_macro_precision | test_macro_recall | test_macro_f1 | test_weighted_f1 | test_balanced_accuracy | cv_mean_val_accuracy | cv_mean_val_macro_precision | cv_mean_val_macro_recall | cv_mean_val_macro_f1 | cv_mean_val_weighted_f1 | cv_mean_val_balanced_accuracy | cv_std_val_accuracy | cv_std_val_macro_precision | cv_std_val_macro_recall | cv_std_val_macro_f1 | cv_std_val_weighted_f1 | cv_std_val_balanced_accuracy | epochs_last_run | temporal_pair_count | temporal_pairs | removed_classes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_tcn_lstm | completed | 0.6116 | 0.5381 | 0.3489 | 0.3874 | 0.5851 | 0.3489 |  |  |  |  |  |  |  |  |  |  |  |  | 8 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| ft_transformer_gru_attention | completed | 0.6112 | 0.5517 | 0.3816 | 0.4201 | 0.5855 | 0.3816 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| xgboost_temporal_gru | completed | 0.6064 | 0.4245 | 0.3188 | 0.3466 | 0.5787 | 0.3188 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
| deepfm_temporal_tcn | completed | 0.6060 | 0.5363 | 0.3627 | 0.4041 | 0.5798 | 0.3627 |  |  |  |  |  |  |  |  |  |  |  |  | 6 | 3 | CC, CR, ENG | INDUSTRIAL CONTROL AND AUTOMATION ENGINEERING, TEXTILE ENGINEERING |
