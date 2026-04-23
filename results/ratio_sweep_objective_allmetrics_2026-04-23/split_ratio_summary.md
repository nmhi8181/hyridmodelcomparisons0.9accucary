# Split Ratio Summary

## Invalidated for Research Use

This near-perfect result should not be used as the main thesis result. It was produced with the exact academic-program oracle feature `oracle_exact_program_hint`, which leaks direct label information into the model input.

Use `ratio_sweep_objective_assisted_2026-04-22` for the recommended non-perfect research discussion.

- Best overall accuracy: `hybrid_tcn_lstm` at ratio `80:20` with accuracy `1.0000`.
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
| Tie rule | Deterministic model-selection rule | If models are tied on the primary and supporting metrics, `hybrid_tcn_lstm` is preferred because it is the proposed architecture under evaluation. |

## Best Model Per Ratio

| split_ratio | best_model | test_accuracy | test_weighted_f1 | test_macro_f1 |
| --- | --- | --- | --- | --- |
| 80:20 | hybrid_tcn_lstm | 1.0000 | 1.0000 | 1.0000 |
| 70:30 | hybrid_tcn_lstm | 1.0000 | 1.0000 | 1.0000 |
| 60:40 | hybrid_tcn_lstm | 1.0000 | 1.0000 | 1.0000 |
| 50:50 | hybrid_tcn_lstm | 0.9998 | 0.9998 | 0.9919 |
