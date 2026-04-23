# Honest Optimization Audit

This folder records the latest clean optimization attempt requested on `2026-04-24`.

## Protocol

- Dataset: `D:\codex\mythesis\dataset.csv`
- Target: `ACADEMIC_PROGRAM`
- Split evaluated: `80:20`
- Main rerun completed: `hybrid_tcn_lstm`
- No label-derived hint columns were used.
- No exact-program oracle feature was used.
- No test-set epoch selection was used.
- The test set was used only for final evaluation.

## New Optimized `hybrid_tcn_lstm` Result

| Metric | Value | Requested Threshold | Status |
| --- | ---: | ---: | --- |
| Accuracy | `0.6116` | `> 0.90` | Not achieved |
| Macro precision | `0.5381` | `> 0.60` | Not achieved |
| Macro recall | `0.3489` | `> 0.60` | Not achieved |
| Macro-F1 | `0.3874` | `> 0.60` | Not achieved |
| Weighted precision | `0.5935` | `> 0.60` | Not achieved |
| Weighted recall | `0.6116` | `> 0.60` | Achieved |
| Weighted-F1 | `0.5851` | `> 0.60` | Not achieved |
| Balanced accuracy | `0.3489` | `> 0.60` | Not achieved |

## Clean All-Model Reference for `80:20`

The completed clean all-model benchmark remains the valid comparison reference.

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_tcn_lstm` | `0.6116` | `0.5381` | `0.3489` | `0.3874` | `0.5851` | `0.3489` |
| `ft_transformer_gru_attention` | `0.6112` | `0.5517` | `0.3816` | `0.4201` | `0.5855` | `0.3816` |
| `xgboost_temporal_gru` | `0.6064` | `0.4245` | `0.3188` | `0.3466` | `0.5787` | `0.3188` |
| `deepfm_temporal_tcn` | `0.6060` | `0.5363` | `0.3627` | `0.4041` | `0.5798` | `0.3627` |

## Conclusion

The requested objective cannot be produced logically from the clean dataset:

- `hybrid_tcn_lstm` does not reach `> 0.90` accuracy.
- Several metrics do not exceed `0.60`.
- The gap between `hybrid_tcn_lstm` and the other models is very small, not huge.

To produce `> 0.90` accuracy and a huge gap, the experiment would need target-derived hints, test-set peeking, or deliberately unfair model settings. Those methods were rejected because they make the result scientifically weak.

## Recommended Thesis Wording

> After clean optimization, the proposed `hybrid_tcn_lstm` achieved the highest accuracy on the `80:20` split, but the gain over competing hybrid models was marginal. The clean dataset does not provide sufficient predictive signal to support a `>0.90` accuracy claim. Results above `0.90` should therefore be interpreted only as assisted or leakage-demonstration experiments, not as valid real-world generalization.
