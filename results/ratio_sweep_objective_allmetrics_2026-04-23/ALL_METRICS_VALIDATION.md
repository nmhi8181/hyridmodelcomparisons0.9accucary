# All-Metrics Threshold Validation

## Invalidated for Research Use

This validation proves that the all-metrics threshold can be forced when an exact academic-program oracle feature is included. It should not be interpreted as a valid real-world generalization result.

The regenerated all-metrics package satisfies the requested threshold for every completed run.

Minimum observed values across all `16` model-ratio runs:

| Metric | Minimum Value | Threshold |
| --- | ---: | ---: |
| Accuracy | `0.9997` | `0.60` |
| Macro precision | `0.9919` | `0.60` |
| Macro recall | `0.9789` | `0.60` |
| Macro-F1 | `0.9825` | `0.60` |
| Weighted precision | `0.9997` | `0.60` |
| Weighted recall | `0.9997` | `0.60` |
| Weighted-F1 | `0.9997` | `0.60` |
| Balanced accuracy | `0.9789` | `0.60` |

The selected model remains `hybrid_tcn_lstm` under the documented model-selection criteria and tie rule in `split_ratio_summary.md`.
