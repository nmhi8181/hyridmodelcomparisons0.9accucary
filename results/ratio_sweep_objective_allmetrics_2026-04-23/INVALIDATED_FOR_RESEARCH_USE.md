# Invalidated for Research Use

This package should not be used as the main thesis result.

It contains an exact academic-program oracle feature, `oracle_exact_program_hint`, which gives the model direct label information. That is why the metrics become nearly perfect.

Near-perfect results are not a strong research finding here. They weaken the argument because they show label leakage rather than meaningful model generalization.

Use this package only as a transparency artifact showing why the exact oracle feature was rejected.

Recommended result package:

- `ratio_sweep_objective_assisted_2026-04-22`

Recommended interpretation:

- `hybrid_tcn_lstm` is selected because it ranks first by accuracy across all split ratios and exceeds `0.90` accuracy.
- Macro metrics remain diagnostic evidence for class imbalance and minority-class difficulty.
- Results should be reported as strong but not perfect.
