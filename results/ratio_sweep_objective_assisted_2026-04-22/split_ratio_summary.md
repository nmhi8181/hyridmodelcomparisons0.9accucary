# Split Ratio Summary

- Best overall accuracy: `hybrid_tcn_lstm` at ratio `50:50` with accuracy `0.9354`.
- Runs meeting `>= 0.60` accuracy: `16` out of `16`.
- Runs meeting `>= 0.90` accuracy: `16` out of `16`.

## Best Model Per Ratio

| split_ratio | best_model | test_accuracy | test_weighted_f1 | test_macro_f1 |
| --- | --- | --- | --- | --- |
| 80:20 | hybrid_tcn_lstm | 0.9351 | 0.9100 | 0.2529 |
| 70:30 | hybrid_tcn_lstm | 0.9339 | 0.9120 | 0.2618 |
| 60:40 | hybrid_tcn_lstm | 0.9351 | 0.9094 | 0.2565 |
| 50:50 | hybrid_tcn_lstm | 0.9354 | 0.9102 | 0.2570 |
