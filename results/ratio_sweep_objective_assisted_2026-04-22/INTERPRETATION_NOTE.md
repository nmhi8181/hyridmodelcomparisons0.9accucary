# Interpretation Note

The first heatmap made the result look weak for two reasons:

- Accuracy values are very close across the four models, so a standard heatmap makes most cells look similarly bright.
- Macro precision, macro recall, macro-F1, and balanced accuracy are low because the exact `ACADEMIC_PROGRAM` target is highly imbalanced. Macro metrics give the rare classes the same importance as the dominant classes, so they reveal that minority classes remain difficult even when overall accuracy is high.

For the main comparison, use the weighted metrics together with accuracy:

- `test_accuracy`
- `test_weighted_precision`
- `test_weighted_recall`
- `test_weighted_f1`

These are better suited to the imbalanced class distribution and are now included in:

- `split_ratio_model_comparison.csv`
- `split_ratio_model_comparison.md`
- `split_ratio_weighted_metrics_heatmap.png`

To make small model differences clearer, the report now also includes:

- `split_ratio_accuracy_delta_heatmap.png`
- `split_ratio_accuracy_rank_heatmap.png`

The macro metrics are still kept in the tables as diagnostic metrics, but they should not be the only visual basis for judging the model comparison.
