# Logical Result Audit

This audit resolves the inconsistency in the previous result packages.

## Main Finding

The current dataset does not support the claim that `hybrid_tcn_lstm` achieves more than `0.90` accuracy on the clean exact academic-program prediction task.

Any result above `0.90` was obtained by adding target-derived hint features. Those features make the experiment easier, but they also make the main research claim logically weak because the model receives information derived from the answer.

## Package Status

| Package | Main Accuracy Range | Uses Label-Derived Hints | Logical Status | How to Use |
| --- | ---: | --- | --- | --- |
| `ratio_sweep_objective_valid_2026-04-21` | about `0.5960` to `0.6116` | No | Valid clean benchmark | Use as the main honest benchmark |
| `ratio_sweep_clean_2026-04-20` | about `0.5355` to `0.5770` | No | Valid clean benchmark | Use as supporting baseline evidence |
| `ratio_sweep_objective_assisted_2026-04-22` | about `0.9315` to `0.9354` | Yes, grouped/3-class target hints | Not valid as main benchmark | Use only as an assisted ablation |
| `ratio_sweep_objective_allmetrics_2026-04-23` | about `0.9997` to `1.0000` | Yes, exact-program oracle hint | Invalid for research use | Keep only as a leakage demonstration |

## Correct Thesis Interpretation

The defensible conclusion is not:

> `hybrid_tcn_lstm` achieves more than `0.90` accuracy on the clean academic-program prediction task.

The defensible conclusion is:

> Under a clean exact-program prediction setup, all models show limited predictive power, with the best accuracy around `0.61`. `hybrid_tcn_lstm` is competitive and sometimes best, but it is not consistently superior across all split ratios.

The assisted experiments can be discussed only as ablation studies:

> When label-hierarchy hints are included, performance rises sharply, showing that the target hierarchy contains strong information. However, this should not be treated as evidence of real-world generalization.

## Why the Previous Results Looked Illogical

The earlier objective forced three incompatible goals:

- Accuracy above `0.90`.
- Macro/weighted metrics above `0.60`.
- No illogical or leakage-like behavior.

The clean dataset cannot satisfy all three at the same time. When the metrics were forced upward, the method became logically weak because label-derived features entered the model input.

## Recommendation

Use the clean benchmark as the main result and rewrite the research contribution around honest model comparison, class imbalance, and the limitation of available predictors. If the thesis must require `>0.90` accuracy, the target definition or data collection strategy must change; it should not be achieved by adding target-derived hint columns.
