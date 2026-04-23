import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RATIO_ORDER = ["80:20", "70:30", "60:40", "50:50"]
MODEL_ORDER = [
    "hybrid_tcn_lstm",
    "ft_transformer_gru_attention",
    "deepfm_temporal_tcn",
    "xgboost_temporal_gru",
]
MODEL_SELECTION_PRIORITY = {
    "hybrid_tcn_lstm": 0,
    "ft_transformer_gru_attention": 1,
    "deepfm_temporal_tcn": 2,
    "xgboost_temporal_gru": 3,
}
METRIC_COLUMNS = [
    "test_accuracy",
    "test_weighted_precision",
    "test_weighted_recall",
    "test_weighted_f1",
    "test_macro_f1",
    "test_balanced_accuracy",
]
WEIGHTED_METRIC_COLUMNS = [
    "test_accuracy",
    "test_weighted_precision",
    "test_weighted_recall",
    "test_weighted_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a top-level report for a multi-ratio hybrid model sweep."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Folder containing one subfolder per split ratio.",
    )
    return parser.parse_args()


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n| --- |\n| no rows |"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.fillna("").astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def discover_ratio_dirs(root_dir: Path) -> List[Tuple[str, Path]]:
    ratio_dirs: List[Tuple[str, Path]] = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        comparison_csv = child / "model_family_comparison.csv"
        if not comparison_csv.exists():
            continue
        if child.name.startswith("ratio_"):
            ratio_label = child.name.replace("ratio_", "").replace("_", ":")
        else:
            ratio_label = child.name
        ratio_dirs.append((ratio_label, child))
    return ratio_dirs


def load_ratio_results(root_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ratio_label, ratio_dir in discover_ratio_dirs(root_dir):
        df = pd.read_csv(ratio_dir / "model_family_comparison.csv")
        df.insert(0, "split_ratio", ratio_label)
        df.insert(1, "ratio_dir", str(ratio_dir))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["split_ratio"] = pd.Categorical(
        combined["split_ratio"],
        categories=RATIO_ORDER,
        ordered=True,
    )
    return combined.sort_values(["split_ratio", "test_accuracy", "model_family"], ascending=[True, False, True])


def load_ratio_cv_folds(root_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for ratio_label, ratio_dir in discover_ratio_dirs(root_dir):
        folds_csv = ratio_dir / "model_family_cross_validation_folds.csv"
        if not folds_csv.exists():
            continue
        df = pd.read_csv(folds_csv)
        df.insert(0, "split_ratio", ratio_label)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["split_ratio"] = pd.Categorical(
        combined["split_ratio"],
        categories=RATIO_ORDER,
        ordered=True,
    )
    return combined


def write_combined_outputs(root_dir: Path, combined_df: pd.DataFrame) -> None:
    output_csv = root_dir / "split_ratio_model_comparison.csv"
    output_json = root_dir / "split_ratio_model_comparison.json"
    output_md = root_dir / "split_ratio_model_comparison.md"

    combined_df.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps(combined_df.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )

    rounded = combined_df.copy()
    for column in combined_df.columns:
        if column.startswith("test_") or column.startswith("cv_"):
            rounded[column] = rounded[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    body = [
        "# Split Ratio Model Comparison",
        "",
        "Combined comparison across all requested train:test ratios and the four hybrid model families.",
        "",
        dataframe_to_markdown_table(rounded),
        "",
    ]
    output_md.write_text("\n".join(body), encoding="utf-8")


def make_metrics_heatmap(root_dir: Path, combined_df: pd.DataFrame) -> None:
    df = combined_df.loc[combined_df["status"] == "completed"].copy()
    if df.empty:
        return
    available_metrics = [column for column in METRIC_COLUMNS if column in df.columns]
    df["label"] = df["split_ratio"].astype(str) + " | " + df["model_family"]
    heatmap_df = df.set_index("label")[available_metrics]
    plt.figure(figsize=(10, max(6, 0.45 * len(heatmap_df))))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Hybrid Metrics by Split Ratio and Model Family")
    plt.xlabel("Metric")
    plt.ylabel("Split Ratio | Model")
    plt.tight_layout()
    plt.savefig(root_dir / "split_ratio_metrics_heatmap.png", dpi=220)
    plt.close()


def make_weighted_metrics_heatmap(root_dir: Path, combined_df: pd.DataFrame) -> None:
    df = combined_df.loc[combined_df["status"] == "completed"].copy()
    if df.empty:
        return
    available_metrics = [column for column in WEIGHTED_METRIC_COLUMNS if column in df.columns]
    if not available_metrics:
        return
    df["label"] = df["split_ratio"].astype(str) + " | " + df["model_family"]
    heatmap_df = df.set_index("label")[available_metrics]
    plt.figure(figsize=(9, max(6, 0.45 * len(heatmap_df))))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.85,
        vmax=1.0,
        linewidths=0.4,
        linecolor="white",
    )
    plt.title("Weighted Performance Metrics by Split Ratio and Model")
    plt.xlabel("Metric")
    plt.ylabel("Split Ratio | Model")
    plt.tight_layout()
    plt.savefig(root_dir / "split_ratio_weighted_metrics_heatmap.png", dpi=220)
    plt.close()


def make_accuracy_heatmap(root_dir: Path, combined_df: pd.DataFrame) -> None:
    df = combined_df.loc[combined_df["status"] == "completed"].copy()
    if df.empty:
        return
    pivot = df.pivot(index="model_family", columns="split_ratio", values="test_accuracy")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=RATIO_ORDER)
    plt.figure(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
    plt.title("Test Accuracy by Split Ratio and Model Family")
    plt.xlabel("Split Ratio")
    plt.ylabel("Model Family")
    plt.tight_layout()
    plt.savefig(root_dir / "split_ratio_accuracy_heatmap.png", dpi=220)
    plt.close()


def make_accuracy_delta_heatmap(root_dir: Path, combined_df: pd.DataFrame) -> None:
    df = combined_df.loc[combined_df["status"] == "completed"].copy()
    if df.empty:
        return
    best_by_ratio = df.groupby("split_ratio", observed=False)["test_accuracy"].transform("max")
    df["accuracy_delta_from_best"] = df["test_accuracy"] - best_by_ratio
    pivot = df.pivot(index="model_family", columns="split_ratio", values="accuracy_delta_from_best")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=RATIO_ORDER)
    plt.figure(figsize=(8.5, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        center=0.0,
        linewidths=0.4,
        linecolor="white",
    )
    plt.title("Accuracy Delta from Best Model in Each Split Ratio")
    plt.xlabel("Split Ratio")
    plt.ylabel("Model Family")
    plt.tight_layout()
    plt.savefig(root_dir / "split_ratio_accuracy_delta_heatmap.png", dpi=220)
    plt.close()


def make_accuracy_rank_heatmap(root_dir: Path, combined_df: pd.DataFrame) -> None:
    df = combined_df.loc[combined_df["status"] == "completed"].copy()
    if df.empty:
        return
    df["accuracy_rank"] = df.groupby("split_ratio", observed=False)["test_accuracy"].rank(
        method="dense",
        ascending=False,
    )
    pivot = df.pivot(index="model_family", columns="split_ratio", values="accuracy_rank")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=RATIO_ORDER)
    plt.figure(figsize=(8.5, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrBr_r",
        vmin=1,
        vmax=max(4, int(pivot.max().max())),
        linewidths=0.4,
        linecolor="white",
    )
    plt.title("Accuracy Rank by Split Ratio (1 = Best)")
    plt.xlabel("Split Ratio")
    plt.ylabel("Model Family")
    plt.tight_layout()
    plt.savefig(root_dir / "split_ratio_accuracy_rank_heatmap.png", dpi=220)
    plt.close()


def make_line_graph(root_dir: Path, combined_df: pd.DataFrame) -> None:
    df = combined_df.loc[combined_df["status"] == "completed"].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(
        data=df,
        x="split_ratio",
        y="test_accuracy",
        hue="model_family",
        style="model_family",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Accuracy Across Split Ratios")
    axes[0].set_xlabel("Train:Test Ratio")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].tick_params(axis="x", rotation=20)

    sns.lineplot(
        data=df,
        x="split_ratio",
        y="test_weighted_f1",
        hue="model_family",
        style="model_family",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Weighted-F1 Across Split Ratios")
    axes[1].set_xlabel("Train:Test Ratio")
    axes[1].set_ylabel("Test Weighted-F1")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(root_dir / "split_ratio_performance_graph.png", dpi=220)
    plt.close(fig)


def make_grouped_bar_chart(root_dir: Path, combined_df: pd.DataFrame) -> None:
    df = combined_df.loc[combined_df["status"] == "completed"].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(
        data=df,
        x="split_ratio",
        y="test_accuracy",
        hue="model_family",
        ax=axes[0],
    )
    axes[0].set_title("Accuracy by Split Ratio and Model")
    axes[0].set_xlabel("Train:Test Ratio")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=df,
        x="split_ratio",
        y="test_weighted_f1",
        hue="model_family",
        ax=axes[1],
    )
    axes[1].set_title("Weighted-F1 by Split Ratio and Model")
    axes[1].set_xlabel("Train:Test Ratio")
    axes[1].set_ylabel("Test Weighted-F1")
    axes[1].tick_params(axis="x", rotation=20)

    axes[1].legend_.remove()
    plt.tight_layout()
    plt.savefig(root_dir / "split_ratio_grouped_bar.png", dpi=220)
    plt.close(fig)


def make_cv_boxplot(root_dir: Path, folds_df: pd.DataFrame) -> None:
    if folds_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(
        data=folds_df,
        x="split_ratio",
        y="val_accuracy",
        hue="model_family",
        ax=axes[0],
    )
    axes[0].set_title("Validation Accuracy Distribution by Split Ratio")
    axes[0].set_xlabel("Train:Test Ratio")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].tick_params(axis="x", rotation=20)

    sns.boxplot(
        data=folds_df,
        x="split_ratio",
        y="val_macro_f1",
        hue="model_family",
        ax=axes[1],
    )
    axes[1].set_title("Validation Macro-F1 Distribution by Split Ratio")
    axes[1].set_xlabel("Train:Test Ratio")
    axes[1].set_ylabel("Validation Macro-F1")
    axes[1].tick_params(axis="x", rotation=20)

    axes[1].legend_.remove()
    plt.tight_layout()
    plt.savefig(root_dir / "split_ratio_cv_boxplot.png", dpi=220)
    plt.close(fig)


def make_test_boxplot(root_dir: Path, combined_df: pd.DataFrame) -> None:
    df = combined_df.loc[combined_df["status"] == "completed"].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=df, x="model_family", y="test_accuracy", ax=axes[0], palette="Blues")
    sns.stripplot(
        data=df,
        x="model_family",
        y="test_accuracy",
        hue="split_ratio",
        dodge=False,
        ax=axes[0],
        alpha=0.75,
    )
    axes[0].set_title("Test Accuracy Distribution Across Split Ratios")
    axes[0].set_xlabel("Model Family")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].tick_params(axis="x", rotation=20)

    sns.boxplot(data=df, x="model_family", y="test_weighted_f1", ax=axes[1], palette="Greens")
    sns.stripplot(
        data=df,
        x="model_family",
        y="test_weighted_f1",
        hue="split_ratio",
        dodge=False,
        ax=axes[1],
        alpha=0.75,
    )
    axes[1].set_title("Test Weighted-F1 Distribution Across Split Ratios")
    axes[1].set_xlabel("Model Family")
    axes[1].set_ylabel("Test Weighted-F1")
    axes[1].tick_params(axis="x", rotation=20)
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    if axes[1].legend_ is not None:
        axes[1].legend_.set_title("Split Ratio")
    plt.tight_layout()
    plt.savefig(root_dir / "split_ratio_test_boxplot.png", dpi=220)
    plt.close(fig)


def make_confusion_matrix_grids(root_dir: Path) -> None:
    for ratio_label, ratio_dir in discover_ratio_dirs(root_dir):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        any_image = False
        for axis, model_family in zip(axes.flatten(), MODEL_ORDER):
            image_path = ratio_dir / model_family / "hybrid_confusion_matrix.png"
            axis.axis("off")
            if not image_path.exists():
                axis.set_title(f"{model_family}\nmissing")
                continue
            image = mpimg.imread(image_path)
            axis.imshow(image)
            axis.set_title(model_family)
            any_image = True
        if not any_image:
            plt.close(fig)
            continue
        fig.suptitle(f"Confusion Matrices for Ratio {ratio_label}", fontsize=16)
        plt.tight_layout()
        safe_label = ratio_label.replace(":", "_")
        plt.savefig(root_dir / f"ratio_{safe_label}_confusion_matrix_grid.png", dpi=220)
        plt.close(fig)


def write_summary(root_dir: Path, combined_df: pd.DataFrame) -> None:
    completed = combined_df.loc[combined_df["status"] == "completed"].copy()
    if completed.empty:
        (root_dir / "split_ratio_summary.md").write_text(
            "# Split Ratio Summary\n\nNo completed model runs were found.\n",
            encoding="utf-8",
        )
        return

    completed["selection_priority"] = completed["model_family"].map(
        MODEL_SELECTION_PRIORITY
    ).fillna(len(MODEL_SELECTION_PRIORITY))
    best_accuracy = completed.sort_values(
        by=["test_accuracy", "test_macro_f1", "selection_priority"],
        ascending=[False, False, True],
    ).iloc[0]
    threshold_06 = completed.loc[completed["test_accuracy"] >= 0.6]
    threshold_09 = completed.loc[completed["test_accuracy"] >= 0.9]

    per_ratio_rows = []
    for ratio_label in RATIO_ORDER:
        ratio_df = completed.loc[completed["split_ratio"].astype(str) == ratio_label]
        if ratio_df.empty:
            continue
        top_row = ratio_df.sort_values(
            by=["test_accuracy", "test_macro_f1", "selection_priority"],
            ascending=[False, False, True],
        ).iloc[0]
        per_ratio_rows.append(
            {
                "split_ratio": ratio_label,
                "best_model": top_row["model_family"],
                "test_accuracy": top_row["test_accuracy"],
                "test_weighted_f1": top_row.get("test_weighted_f1"),
                "test_macro_f1": top_row["test_macro_f1"],
            }
        )

    per_ratio_df = pd.DataFrame(per_ratio_rows)
    rounded_ratio_df = per_ratio_df.copy()
    for column in ("test_accuracy", "test_weighted_f1", "test_macro_f1"):
        if column in rounded_ratio_df.columns:
            rounded_ratio_df[column] = rounded_ratio_df[column].map(lambda value: f"{value:.4f}")

    lines = [
        "# Split Ratio Summary",
        "",
        f"- Best overall accuracy: `{best_accuracy['model_family']}` at ratio `{best_accuracy['split_ratio']}` with accuracy `{best_accuracy['test_accuracy']:.4f}`.",
        f"- Runs meeting `>= 0.60` accuracy: `{len(threshold_06)}` out of `{len(completed)}`.",
        f"- Runs meeting `>= 0.90` accuracy: `{len(threshold_09)}` out of `{len(completed)}`.",
        "",
        "## Model Selection Criteria",
        "",
        "| Criterion | Role in Selection | Application in This Experiment |",
        "| --- | --- | --- |",
        "| Primary metric | Main ranking metric | Test accuracy is used to select the best model. |",
        "| Accuracy constraint | Minimum acceptance threshold | Candidate models must exceed `0.90` test accuracy. |",
        "| Robustness | Stability across data availability settings | The selected model should rank first across multiple train:test split ratios. |",
        "| Secondary metrics | Supporting performance evidence | Weighted precision, weighted recall, and weighted-F1 are used to confirm performance under class imbalance. |",
        "| Macro metrics | Diagnostic evidence only | Macro precision, macro recall, macro-F1, and balanced accuracy are reported to show minority-class difficulty, but they are not the primary selection criterion. |",
        "| Tie rule | Deterministic model-selection rule | If models are tied on the primary and supporting metrics, `hybrid_tcn_lstm` is preferred because it is the proposed architecture under evaluation. |",
        "",
        "## Best Model Per Ratio",
        "",
        dataframe_to_markdown_table(rounded_ratio_df),
        "",
    ]
    (root_dir / "split_ratio_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    combined_df = load_ratio_results(root_dir)
    folds_df = load_ratio_cv_folds(root_dir)

    write_combined_outputs(root_dir, combined_df)
    if not folds_df.empty:
        folds_df.to_csv(root_dir / "split_ratio_cross_validation_folds.csv", index=False)
    make_metrics_heatmap(root_dir, combined_df)
    make_weighted_metrics_heatmap(root_dir, combined_df)
    make_accuracy_heatmap(root_dir, combined_df)
    make_accuracy_delta_heatmap(root_dir, combined_df)
    make_accuracy_rank_heatmap(root_dir, combined_df)
    make_line_graph(root_dir, combined_df)
    make_grouped_bar_chart(root_dir, combined_df)
    make_cv_boxplot(root_dir, folds_df)
    make_test_boxplot(root_dir, combined_df)
    make_confusion_matrix_grids(root_dir)
    write_summary(root_dir, combined_df)


if __name__ == "__main__":
    main()
