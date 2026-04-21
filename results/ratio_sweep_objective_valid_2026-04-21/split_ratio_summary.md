# Split Ratio Summary

- Best overall accuracy: `hybrid_tcn_lstm` at ratio `80:20` with accuracy `0.6116`.
- Runs meeting `>= 0.60` accuracy: `12` out of `16`.
- Runs meeting `>= 0.90` accuracy: `0` out of `16`.

## Best Model Per Ratio

| split_ratio | best_model | test_accuracy | test_macro_f1 |
| --- | --- | --- | --- |
| 80:20 | hybrid_tcn_lstm | 0.6116 | 0.3874 |
| 70:30 | xgboost_temporal_gru | 0.6089 | 0.3764 |
| 60:40 | xgboost_temporal_gru | 0.6048 | 0.2556 |
| 50:50 | hybrid_tcn_lstm | 0.5960 | 0.3332 |
