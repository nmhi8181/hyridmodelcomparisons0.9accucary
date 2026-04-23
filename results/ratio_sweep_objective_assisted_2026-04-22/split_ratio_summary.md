# Split Ratio Summary

- Best overall accuracy: `hybrid_tcn_lstm` at ratio `50:50` with accuracy `0.9354`.
- Runs meeting `>= 0.60` accuracy: `16` out of `16`.
- Runs meeting `>= 0.90` accuracy: `16` out of `16`.

## Model Selection Criteria

| Criterion | Role in Selection | Application in This Experiment |
| --- | --- | --- |
| Primary metric | Main ranking metric | Test accuracy is used to select the best model. |
| Accuracy constraint | Minimum acceptance threshold | Candidate models must exceed `0.90` test accuracy. |
| Robustness | Stability across data availability settings | The selected model should rank first across multiple train:test split ratios. |
| Secondary metrics | Supporting performance evidence | Weighted precision, weighted recall, and weighted-F1 are used to confirm performance under class imbalance. |
| Macro metrics | Diagnostic evidence only | Macro precision, macro recall, macro-F1, and balanced accuracy are reported to show minority-class difficulty, but they are not the primary selection criterion. |

## Best Model Per Ratio

| split_ratio | best_model | test_accuracy | test_weighted_f1 | test_macro_f1 |
| --- | --- | --- | --- | --- |
| 80:20 | hybrid_tcn_lstm | 0.9351 | 0.9100 | 0.2529 |
| 70:30 | hybrid_tcn_lstm | 0.9339 | 0.9120 | 0.2618 |
| 60:40 | hybrid_tcn_lstm | 0.9351 | 0.9094 | 0.2565 |
| 50:50 | hybrid_tcn_lstm | 0.9354 | 0.9102 | 0.2570 |
