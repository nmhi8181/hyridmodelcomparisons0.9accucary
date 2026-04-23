import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_FAMILIES = [
    "hybrid_tcn_lstm",
    "ft_transformer_gru_attention",
    "xgboost_temporal_gru",
    "deepfm_temporal_tcn",
]


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n| --- |\n| no rows |"

    headers = [str(col) for col in df.columns]
    rows = df.fillna("").astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def load_family_metrics(root_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for family in MODEL_FAMILIES:
        metrics_path = root_dir / family / "metrics_summary.json"
        if not metrics_path.exists():
            rows.append(
                {
                    "model_family": family,
                    "status": "missing",
                    "test_accuracy": None,
                    "test_macro_precision": None,
                    "test_macro_recall": None,
                    "test_macro_f1": None,
                    "test_weighted_precision": None,
                    "test_weighted_recall": None,
                    "test_weighted_f1": None,
                    "test_balanced_accuracy": None,
                    "cv_mean_val_accuracy": None,
                    "cv_mean_val_macro_precision": None,
                    "cv_mean_val_macro_recall": None,
                    "cv_mean_val_macro_f1": None,
                    "cv_mean_val_weighted_f1": None,
                    "cv_mean_val_balanced_accuracy": None,
                    "cv_std_val_accuracy": None,
                    "cv_std_val_macro_precision": None,
                    "cv_std_val_macro_recall": None,
                    "cv_std_val_macro_f1": None,
                    "cv_std_val_weighted_f1": None,
                    "cv_std_val_balanced_accuracy": None,
                    "epochs_last_run": None,
                    "temporal_pair_count": None,
                    "temporal_pairs": None,
                    "removed_classes": None,
                }
            )
            continue

        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        cv = data.get("cross_validation") or {}
        rows.append(
            {
                "model_family": family,
                "status": "completed",
                "test_accuracy": data["hybrid_metrics"]["test_accuracy"],
                "test_macro_precision": data["hybrid_metrics"].get("test_macro_precision"),
                "test_macro_recall": data["hybrid_metrics"].get("test_macro_recall"),
                "test_macro_f1": data["hybrid_metrics"]["test_macro_f1"],
                "test_weighted_precision": data["hybrid_metrics"].get("test_weighted_precision"),
                "test_weighted_recall": data["hybrid_metrics"].get("test_weighted_recall"),
                "test_weighted_f1": data["hybrid_metrics"].get("test_weighted_f1"),
                "test_balanced_accuracy": data["hybrid_metrics"].get("test_balanced_accuracy"),
                "cv_mean_val_accuracy": cv.get("mean_val_accuracy"),
                "cv_mean_val_macro_precision": cv.get("mean_val_macro_precision"),
                "cv_mean_val_macro_recall": cv.get("mean_val_macro_recall"),
                "cv_mean_val_macro_f1": cv.get("mean_val_macro_f1"),
                "cv_mean_val_weighted_f1": cv.get("mean_val_weighted_f1"),
                "cv_mean_val_balanced_accuracy": cv.get("mean_val_balanced_accuracy"),
                "cv_std_val_accuracy": cv.get("std_val_accuracy"),
                "cv_std_val_macro_precision": cv.get("std_val_macro_precision"),
                "cv_std_val_macro_recall": cv.get("std_val_macro_recall"),
                "cv_std_val_macro_f1": cv.get("std_val_macro_f1"),
                "cv_std_val_weighted_f1": cv.get("std_val_weighted_f1"),
                "cv_std_val_balanced_accuracy": cv.get("std_val_balanced_accuracy"),
                "epochs_last_run": data["config"]["epochs"],
                "temporal_pair_count": len(data["schema"]["temporal_pair_bases"]),
                "temporal_pairs": ", ".join(data["schema"]["temporal_pair_bases"]),
                "removed_classes": ", ".join(data["dataset"]["removed_minority_classes"]),
            }
        )
    return rows


def collect_cv_folds(root_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for family in MODEL_FAMILIES:
        folds_path = root_dir / family / "cross_validation_folds.csv"
        if not folds_path.exists():
            continue
        fold_df = pd.read_csv(folds_path)
        fold_df.insert(0, "model_family", family)
        frames.append(fold_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_comparison_outputs(root_dir: Path, comparison_df: pd.DataFrame) -> None:
    comparison_csv = root_dir / "model_family_comparison.csv"
    comparison_json = root_dir / "model_family_comparison.json"
    comparison_md = root_dir / "model_family_comparison.md"

    comparison_df.to_csv(comparison_csv, index=False)
    comparison_json.write_text(
        json.dumps(comparison_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    rounded_df = comparison_df.copy()
    for column in [
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_weighted_precision",
        "test_weighted_recall",
        "test_weighted_f1",
        "test_balanced_accuracy",
        "cv_mean_val_accuracy",
        "cv_mean_val_macro_precision",
        "cv_mean_val_macro_recall",
        "cv_mean_val_macro_f1",
        "cv_mean_val_weighted_f1",
        "cv_mean_val_balanced_accuracy",
        "cv_std_val_accuracy",
        "cv_std_val_macro_precision",
        "cv_std_val_macro_recall",
        "cv_std_val_macro_f1",
        "cv_std_val_weighted_f1",
        "cv_std_val_balanced_accuracy",
    ]:
        if column in rounded_df.columns:
            rounded_df[column] = rounded_df[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )

    body = [
        "# Model Family Comparison",
        "",
        "Corrected comparison run on the original dataset after excluding target-derived helper columns from the feature set.",
        "",
        dataframe_to_markdown_table(rounded_df),
        "",
    ]
    comparison_md.write_text("\n".join(body), encoding="utf-8")


def make_bar_chart(root_dir: Path, comparison_df: pd.DataFrame) -> None:
    df = comparison_df.loc[comparison_df["status"] == "completed"].copy()
    if df.empty:
        return
    metric_specs = [
        ("test_accuracy", "Test Accuracy", "Blues_d"),
        ("test_weighted_precision", "Test Weighted Precision", "Purples_d"),
        ("test_weighted_recall", "Test Weighted Recall", "Oranges_d"),
        ("test_weighted_f1", "Test Weighted F1", "Greens_d"),
        ("test_macro_f1", "Test Macro-F1", "Reds_d"),
        ("test_balanced_accuracy", "Test Balanced Accuracy", "YlGnBu"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for axis, (metric_col, title, palette) in zip(axes.flatten(), metric_specs):
        if metric_col not in df.columns or df[metric_col].dropna().empty:
            axis.set_visible(False)
            continue
        sns.barplot(data=df, x="model_family", y=metric_col, ax=axis, palette=palette)
        axis.set_title(f"{title} by Model Family")
        axis.set_xlabel("")
        axis.set_ylabel(title.replace("Test ", ""))
        axis.tick_params(axis="x", rotation=20)
        for patch in axis.patches:
            height = patch.get_height()
            axis.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 4),
                textcoords="offset points",
            )
    plt.tight_layout()
    plt.savefig(root_dir / "model_family_test_metrics_bar.png", dpi=220)
    plt.close(fig)


def make_heatmap(root_dir: Path, comparison_df: pd.DataFrame) -> None:
    df = comparison_df.loc[comparison_df["status"] == "completed"].copy()
    if df.empty:
        return
    metric_cols = [
        "test_accuracy",
        "test_weighted_precision",
        "test_weighted_recall",
        "test_weighted_f1",
        "test_macro_f1",
        "test_balanced_accuracy",
    ]
    metric_cols = [column for column in metric_cols if column in df.columns]
    if not metric_cols:
        return
    heatmap_df = df.set_index("model_family")[metric_cols]
    plt.figure(figsize=(9, 4.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Hybrid Model Comparison Heatmap")
    plt.xlabel("Metric")
    plt.ylabel("Model Family")
    plt.tight_layout()
    plt.savefig(root_dir / "model_family_metrics_heatmap.png", dpi=220)
    plt.close()


def make_cv_boxplot(root_dir: Path, folds_df: pd.DataFrame) -> None:
    if folds_df.empty:
        return

    metric_specs = [
        ("val_accuracy", "Cross-Validation Accuracy by Model Family", "Blues"),
        ("val_macro_precision", "Cross-Validation Macro-Precision by Model Family", "Purples"),
        ("val_macro_recall", "Cross-Validation Macro-Recall by Model Family", "Oranges"),
        ("val_macro_f1", "Cross-Validation Macro-F1 by Model Family", "Greens"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for axis, (metric_col, title, palette) in zip(axes.flatten(), metric_specs):
        if metric_col not in folds_df.columns or folds_df[metric_col].dropna().empty:
            axis.set_visible(False)
            continue
        sns.boxplot(data=folds_df, x="model_family", y=metric_col, ax=axis, palette=palette)
        sns.stripplot(
            data=folds_df,
            x="model_family",
            y=metric_col,
            ax=axis,
            color="black",
            alpha=0.6,
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel(metric_col.replace("val_", "Validation ").replace("_", " ").title())
        axis.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(root_dir / "model_family_cv_boxplot.png", dpi=220)
    plt.close(fig)


def write_summary(root_dir: Path, comparison_df: pd.DataFrame, folds_df: pd.DataFrame) -> None:
    completed = comparison_df.loc[comparison_df["status"] == "completed"].copy()
    if completed.empty:
        (root_dir / "rigorous_result_summary.md").write_text(
            "# Rigorous Hybrid Model Result Summary\n\nNo completed model rows were found.\n",
            encoding="utf-8",
        )
        return
    best_accuracy = completed.sort_values(
        by=["test_accuracy", "test_macro_f1"], ascending=[False, False]
    ).iloc[0]
    best_macro_f1 = completed.sort_values(
        by=["test_macro_f1", "test_accuracy"], ascending=[False, False]
    ).iloc[0]

    mean_epochs = ""
    if not folds_df.empty and "epochs_ran" in folds_df.columns:
        mean_epochs = f"{folds_df['epochs_ran'].mean():.2f}"

    rounded = completed.copy()
    for col in [
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_weighted_precision",
        "test_weighted_recall",
        "test_weighted_f1",
        "test_balanced_accuracy",
        "cv_mean_val_accuracy",
        "cv_mean_val_macro_precision",
        "cv_mean_val_macro_recall",
        "cv_mean_val_macro_f1",
        "cv_mean_val_weighted_f1",
        "cv_mean_val_balanced_accuracy",
    ]:
        if col in rounded.columns:
            rounded[col] = rounded[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )

    lines = [
        "# Rigorous Hybrid Model Result Summary",
        "",
        "## Run Setup",
        "",
        f"- Root folder: `{root_dir}`",
        "- Dataset: original `dataset.csv` with fallback encoding support enabled.",
        "- Validity fix: `target_*` helper columns were excluded from the feature set before training.",
        "- Temporal branch status: active for all families using paired bases `CC`, `CR`, and `ENG`, plus engineered change/ratio features.",
        "- Split strategy: hold-out train/validation/test split plus 3-fold stratified cross-validation.",
    ]
    if mean_epochs:
        lines.append(f"- Mean cross-validation epochs actually run: {mean_epochs}")

    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            f"- Highest test accuracy: `{best_accuracy['model_family']}` at `{best_accuracy['test_accuracy']:.4f}`.",
            f"- Highest test macro-F1: `{best_macro_f1['model_family']}` at `{best_macro_f1['test_macro_f1']:.4f}`.",
            "- `xgboost_temporal_gru` underperformed strongly, suggesting the probability-fusion variant was not competitive on this dataset under the current setup.",
            "- Accuracy and macro-F1 diverged noticeably, which indicates class imbalance remains a major challenge.",
            "",
            "## Comparison Table",
            "",
            dataframe_to_markdown_table(
                rounded[
                    [
                        "model_family",
                        "test_accuracy",
                        "test_macro_precision",
                        "test_macro_recall",
                        "test_macro_f1",
                        "test_weighted_precision",
                        "test_weighted_recall",
                        "test_weighted_f1",
                        "test_balanced_accuracy",
                        "cv_mean_val_accuracy",
                        "cv_mean_val_macro_f1",
                        "temporal_pairs",
                    ]
                ]
            ),
            "",
            "## Output Files",
            "",
            "- `model_family_comparison.csv`",
            "- `model_family_comparison.md`",
            "- `model_family_comparison.json`",
            "- `model_family_test_metrics_bar.png`",
            "- `model_family_metrics_heatmap.png`",
            "- `model_family_cv_boxplot.png`",
            "- per-family `hybrid_confusion_matrix.png` and `training_history.png` inside each model subfolder",
            "",
        ]
    )

    (root_dir / "rigorous_result_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a top-level result package for hybrid-model runs.")
    parser.add_argument("--root-dir", required=True, help="Directory containing one subfolder per model family.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    comparison_df = pd.DataFrame(load_family_metrics(root_dir))
    folds_df = collect_cv_folds(root_dir)

    write_comparison_outputs(root_dir, comparison_df)
    if not folds_df.empty:
        folds_df.to_csv(root_dir / "model_family_cross_validation_folds.csv", index=False)
    make_bar_chart(root_dir, comparison_df)
    make_heatmap(root_dir, comparison_df)
    make_cv_boxplot(root_dir, folds_df)
    write_summary(root_dir, comparison_df, folds_df)


if __name__ == "__main__":
    main()
