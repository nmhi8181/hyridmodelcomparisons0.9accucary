# Model Selection Justification

## Superseded for Main Research Use

This file is retained for transparency, but this package should not be used as the main thesis benchmark because it uses target-derived hierarchy hint features. The logical main benchmark is `results/ratio_sweep_objective_valid_2026-04-21`.

This file documents the model-selection outcome inside the assisted ablation package. It avoids the exact-program oracle feature that produced near-perfect `1.0` results, but it still uses target-derived hierarchy hints and is therefore not recommended as the main thesis benchmark.

## Selection Rule

| Criterion | Role |
| --- | --- |
| Primary metric | Test accuracy |
| Acceptance threshold | Accuracy must exceed `0.90` |
| Robustness requirement | Model should rank first across the `80:20`, `70:30`, `60:40`, and `50:50` train:test ratios |
| Supporting metrics | Weighted precision, weighted recall, and weighted-F1 |
| Diagnostic metrics | Macro precision, macro recall, macro-F1, and balanced accuracy are reported to expose minority-class difficulty |

## Evidence for `hybrid_tcn_lstm`

| Split Ratio | Best Model | Accuracy | Second-Best Model | Second-Best Accuracy | Accuracy Margin | Weighted-F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `80:20` | `hybrid_tcn_lstm` | `0.9351` | `ft_transformer_gru_attention` | `0.9339` | `0.0012` | `0.9100` |
| `70:30` | `hybrid_tcn_lstm` | `0.9339` | `xgboost_temporal_gru` | `0.9334` | `0.0005` | `0.9120` |
| `60:40` | `hybrid_tcn_lstm` | `0.9351` | `ft_transformer_gru_attention` | `0.9327` | `0.0024` | `0.9094` |
| `50:50` | `hybrid_tcn_lstm` | `0.9354` | `deepfm_temporal_tcn` | `0.9341` | `0.0013` | `0.9102` |

## Interpretation

The result supports selecting `hybrid_tcn_lstm` because it is the only model that ranks first by test accuracy across every split ratio while also exceeding `0.90` accuracy.

The margin is small, so the correct claim is:

> `hybrid_tcn_lstm` is the most robust choice under the primary accuracy-based model-selection criterion.

The result should not be written as:

> `hybrid_tcn_lstm` is dramatically better than every other model on every metric.

Macro metrics remain low because the exact academic-program target is highly imbalanced. They should be discussed as minority-class diagnostic evidence and future-work motivation, not hidden or artificially forced above `0.60`.
