import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

try:
    import seaborn as sns
except Exception:
    sns = None

from run_hybrid_ratio_sweep import family_overrides
from study_pathway_hybrid_clean import (
    SeqStaticDataset,
    collate_fn,
    create_engineered_static_features,
    create_engineered_temporal_features,
    dataframe_to_markdown_table,
    detect_schema,
    ensure_dense_features,
    execute_training_run,
    prepare_model_inputs,
    read_csv_with_fallback_encodings,
    remove_minority_classes,
    split_dataframe,
    summarize_predictions,
    write_json,
    xgb,
)


DEFAULT_CLEAN_DATASET = (
    r"D:\codex\mythesis\experiments\section_3_7_cleandataset_full_rerun_2026-04-07"
    r"\hybrid_suite\docs\cleandataset.csv"
)
DEFAULT_MAPPING_JSON = (
    r"D:\codex\mythesis\experiments\section_3_7_separability_diagnostics_2026-04-07"
    r"\docs\target_mappings_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a clean hierarchical study-pathway experiment: broad-family prediction at "
            "stage 1 and exact-program prediction within each family at stage 2."
        )
    )
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CLEAN_DATASET,
        help="Path to the clean dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to a timestamped folder in the bundle.",
    )
    parser.add_argument(
        "--family-col",
        default="target_grouped_5class",
        help=(
            "Column containing the broad-family target. If missing, the script will try to "
            "derive it from the mapping JSON."
        ),
    )
    parser.add_argument(
        "--mapping-json",
        default=DEFAULT_MAPPING_JSON,
        help="Optional mapping JSON used when the broad-family column is missing.",
    )
    parser.add_argument(
        "--model-family",
        choices=(
            "hybrid_tcn_lstm",
            "ft_transformer_gru_attention",
            "deepfm_temporal_tcn",
            "xgboost_temporal_gru",
        ),
        default="hybrid_tcn_lstm",
        help="Hybrid family used for both stage 1 and stage 2 models.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--min-class-samples", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1", "balanced_accuracy", "composite"),
        default="composite",
        help="Validation metric used for early stopping.",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weights during training.",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Power applied to inverse-frequency class weights.",
    )
    parser.add_argument(
        "--confidence-target-accuracy",
        type=float,
        default=0.9,
        help="Target subset accuracy used to select a confidence threshold on validation data.",
    )
    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[1, 3],
        help="Top-k values reported for exact-program recommendation quality.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum():
            safe.append(char)
        elif char in (" ", "-", "_"):
            safe.append("_")
    rendered = "".join(safe).strip("_")
    return rendered or "unnamed"


def load_mapping_table(mapping_json: Path) -> Dict[str, str]:
    with mapping_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    grouped_mapping = payload.get("grouped_five_target")
    if not isinstance(grouped_mapping, dict):
        raise ValueError(
            "Mapping JSON does not contain a 'grouped_five_target' dictionary."
        )

    lookup: Dict[str, str] = {}
    for grouped_label, source_labels in grouped_mapping.items():
        for source_label in source_labels:
            lookup[str(source_label)] = str(grouped_label)
    return lookup


def attach_family_target(
    df: pd.DataFrame,
    exact_target_col: str,
    family_col: str,
    mapping_json: Path,
) -> pd.DataFrame:
    if family_col in df.columns:
        return df

    if not mapping_json.exists():
        raise FileNotFoundError(
            f"Family column '{family_col}' is missing and mapping JSON was not found: {mapping_json}"
        )

    lookup = load_mapping_table(mapping_json)
    df = df.copy()
    df[family_col] = df[exact_target_col].astype(str).map(lookup)
    return df


def build_model_args(args: argparse.Namespace) -> argparse.Namespace:
    model_defaults = family_overrides(
        model_family=args.model_family,
        epochs=args.epochs,
        run_cross_validation=False,
    )
    # Stage-2 family subsets can be small, so disable batchnorm to avoid
    # unstable single-sample batch errors during clean hierarchical training.
    model_defaults["static_use_batchnorm"] = False
    model_defaults["classifier_use_batchnorm"] = False
    model_defaults["temporal_use_batchnorm"] = False
    model_defaults["selection_metric"] = args.selection_metric
    model_defaults["use_class_weights"] = args.use_class_weights
    model_defaults["class_weight_power"] = args.class_weight_power
    model_defaults["batch_size"] = args.batch_size

    return argparse.Namespace(
        csv_path=args.csv_path,
        output_dir="",
        model_family=args.model_family,
        compare_model_families=False,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        min_class_samples=args.min_class_samples,
        batch_size=model_defaults["batch_size"],
        epochs=model_defaults["epochs"],
        learning_rate=model_defaults["learning_rate"],
        weight_decay=model_defaults["weight_decay"],
        optimizer=model_defaults["optimizer"],
        momentum=model_defaults["momentum"],
        scheduler=model_defaults["scheduler"],
        scheduler_factor=model_defaults["scheduler_factor"],
        scheduler_patience=model_defaults["scheduler_patience"],
        scheduler_eta_min=model_defaults["scheduler_eta_min"],
        early_stopping_patience=model_defaults["early_stopping_patience"],
        grad_clip=model_defaults["grad_clip"],
        static_hid_dim=model_defaults["static_hid_dim"],
        static_out_dim=model_defaults["static_out_dim"],
        static_dropout=model_defaults["static_dropout"],
        static_n_layers=model_defaults["static_n_layers"],
        static_use_batchnorm=model_defaults["static_use_batchnorm"],
        temporal_hid_dim=model_defaults["temporal_hid_dim"],
        temporal_kernel_size=model_defaults["temporal_kernel_size"],
        temporal_n_blocks=model_defaults["temporal_n_blocks"],
        temporal_dropout=model_defaults["temporal_dropout"],
        temporal_recurrent_type=model_defaults.get("temporal_recurrent_type", "LSTM"),
        temporal_use_batchnorm=model_defaults["temporal_use_batchnorm"],
        temporal_use_layernorm=model_defaults["temporal_use_layernorm"],
        classifier_hid_dim=model_defaults["classifier_hid_dim"],
        classifier_n_layers=model_defaults["classifier_n_layers"],
        classifier_use_batchnorm=model_defaults["classifier_use_batchnorm"],
        transformer_layers=model_defaults["transformer_layers"],
        transformer_heads=model_defaults["transformer_heads"],
        transformer_dropout=model_defaults["transformer_dropout"],
        fm_embed_dim=model_defaults["fm_embed_dim"],
        xgb_num_boost_round=model_defaults["xgb_num_boost_round"],
        augment_with_xgb_probs=model_defaults["augment_with_xgb_probs"],
        final_ensemble_with_xgb=model_defaults["final_ensemble_with_xgb"],
        ensemble_weight_step=model_defaults["ensemble_weight_step"],
        selection_metric=model_defaults["selection_metric"],
        use_class_weights=model_defaults["use_class_weights"],
        class_weight_power=model_defaults["class_weight_power"],
        label_smoothing=model_defaults["label_smoothing"],
        run_xgboost=False,
        run_fairness=False,
        run_random_search=False,
        random_search_iterations=0,
        run_cross_validation=False,
        cv_folds=1,
        final_retrain_val_size=0.1,
        no_plots=False,
    )


def build_stage1_frames(
    frame: pd.DataFrame,
    exact_target_col: str,
    family_col: str,
) -> pd.DataFrame:
    rendered = frame.copy()
    rendered[exact_target_col] = rendered[family_col].astype(str)
    return rendered


def save_confusion_plot_for_labels(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    plt.figure(figsize=(max(8, len(labels) * 0.65), max(6, len(labels) * 0.55)))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
    else:
        plt.imshow(cm, cmap="Blues")
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(np.arange(len(labels)), labels)
        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                plt.text(col_idx, row_idx, str(cm[row_idx, col_idx]), ha="center", va="center")
        plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_metric_bar(
    metric_values: Dict[str, float],
    output_path: Path,
    title: str,
    threshold: Optional[float] = None,
) -> None:
    labels = list(metric_values.keys())
    values = [metric_values[label] for label in labels]
    plt.figure(figsize=(10, 5))
    colors = ["#1f5aa6" if "top-1" in label.lower() else "#6d94c6" for label in labels]
    plt.bar(labels, values, color=colors)
    if threshold is not None:
        plt.axhline(threshold, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Threshold {threshold:.2f}")
        plt.legend()
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    for idx, value in enumerate(values):
        plt.text(idx, min(value + 0.02, 0.98), f"{value:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def score_frame_with_run(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    schema,
    complete_temporal_groups: Dict[str, Dict[str, str]],
    engineered_cols: Sequence[str],
    run: Dict[str, object],
    model_args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    prepared_score = prepare_model_inputs(
        train_df=train_df,
        val_df=score_df,
        test_df=None,
        schema=schema,
        complete_temporal_groups=complete_temporal_groups,
        engineered_cols=engineered_cols,
    )

    score_static = prepared_score.x_static_val
    score_seq = prepared_score.x_seq_val
    xgb_probs = None
    xgb_artifacts = run.get("xgb_artifacts")
    if xgb_artifacts is not None and xgb is not None:
        booster = xgb_artifacts.get("booster")
        if booster is not None:
            xgb_probs = booster.predict(xgb.DMatrix(score_static)).astype(np.float32)

    if xgb_probs is not None and model_args.model_family == "xgboost_temporal_gru":
        score_static = xgb_probs.astype(np.float32)
    elif xgb_probs is not None and model_args.augment_with_xgb_probs:
        score_static = np.concatenate(
            [ensure_dense_features(score_static), xgb_probs.astype(np.float32)],
            axis=1,
        )
    else:
        score_static = ensure_dense_features(score_static)

    dummy_labels = np.zeros(len(score_df), dtype=int)
    score_ds = SeqStaticDataset(score_static, dummy_labels, score_seq)
    score_dl = DataLoader(
        score_ds,
        batch_size=model_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = run["model"]
    model.eval()
    batches: List[np.ndarray] = []
    with torch.no_grad():
        for static_x, temporal_x, _ in score_dl:
            static_x = static_x.to(device)
            if temporal_x is not None:
                temporal_x = temporal_x.to(device)
            logits = model(static_x, temporal_x)
            batches.append(torch.softmax(logits, dim=-1).cpu().numpy())

    neural_probs = np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float32)
    final_probs = neural_probs
    ensemble_summary = run.get("ensemble_summary")
    if (
        ensemble_summary is not None
        and xgb_probs is not None
        and model_args.final_ensemble_with_xgb
    ):
        neural_weight = float(ensemble_summary["neural_weight"])
        final_probs = (neural_weight * neural_probs) + ((1.0 - neural_weight) * xgb_probs)

    class_names = [str(name) for name in run["prepared"].label_encoder.classes_]
    return final_probs, class_names


def build_combined_program_probabilities(
    family_probs: np.ndarray,
    family_class_names: Sequence[str],
    stage2_scores: Dict[str, Tuple[np.ndarray, List[str]]],
    program_classes: Sequence[str],
) -> np.ndarray:
    family_index = {name: idx for idx, name in enumerate(family_class_names)}
    program_index = {name: idx for idx, name in enumerate(program_classes)}
    combined = np.zeros((family_probs.shape[0], len(program_classes)), dtype=np.float32)

    for family_name, (program_probs, program_names) in stage2_scores.items():
        if family_name not in family_index:
            continue
        family_weight = family_probs[:, family_index[family_name]][:, np.newaxis]
        for class_idx, program_name in enumerate(program_names):
            combined[:, program_index[program_name]] += (
                family_weight[:, 0] * program_probs[:, class_idx]
            )

    row_sums = combined.sum(axis=1, keepdims=True)
    valid_rows = row_sums[:, 0] > 0
    combined[valid_rows] = combined[valid_rows] / row_sums[valid_rows]
    return combined


def topk_accuracy(
    probs: np.ndarray,
    true_labels: Sequence[str],
    class_names: Sequence[str],
    k: int,
) -> float:
    if probs.size == 0:
        return float("nan")
    top_indices = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    matches = []
    for row_idx, true_label in enumerate(true_labels):
        predicted = {class_names[class_idx] for class_idx in top_indices[row_idx]}
        matches.append(true_label in predicted)
    return float(np.mean(matches))


def build_threshold_table(
    probs: np.ndarray,
    true_labels: Sequence[str],
    class_names: Sequence[str],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    top1_indices = probs.argmax(axis=1)
    top1_labels = [class_names[idx] for idx in top1_indices]
    confidences = probs.max(axis=1)
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        mask = confidences >= float(threshold)
        coverage = float(mask.mean())
        if mask.any():
            subset_true = [true_labels[idx] for idx, keep in enumerate(mask) if keep]
            subset_pred = [top1_labels[idx] for idx, keep in enumerate(mask) if keep]
            subset_accuracy = float(np.mean([a == b for a, b in zip(subset_true, subset_pred)]))
        else:
            subset_accuracy = float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "subset_accuracy": subset_accuracy,
                "selected_count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def select_confidence_threshold(
    threshold_df: pd.DataFrame,
    target_accuracy: float,
) -> Optional[pd.Series]:
    eligible = threshold_df.dropna(subset=["subset_accuracy"]).copy()
    eligible = eligible.loc[eligible["subset_accuracy"] >= float(target_accuracy)]
    if eligible.empty:
        return None
    eligible = eligible.sort_values(
        by=["coverage", "subset_accuracy", "threshold"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return eligible.iloc[0]


def constant_probability_scores(
    label: str,
    sample_count: int,
) -> Tuple[np.ndarray, List[str]]:
    return np.ones((sample_count, 1), dtype=np.float32), [str(label)]


def save_stage2_family_table(
    family_rows: List[Dict[str, object]],
    output_dir: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(family_rows)
    df.to_csv(output_dir / "stage2_family_summary.csv", index=False)
    lines = [
        "# Stage 2 Family Summary",
        "",
        dataframe_to_markdown_table(df),
        "",
    ]
    (output_dir / "stage2_family_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else bundle_dir / f"hierarchical_{args.model_family}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = read_csv_with_fallback_encodings(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    exact_schema = detect_schema(df)
    df = attach_family_target(
        df=df,
        exact_target_col=exact_schema.target_col,
        family_col=args.family_col,
        mapping_json=Path(args.mapping_json),
    )

    if args.family_col not in df.columns:
        raise ValueError(f"Unable to create or find family target column '{args.family_col}'.")

    df = df.dropna(subset=[exact_schema.target_col, args.family_col]).copy()
    df[exact_schema.target_col] = df[exact_schema.target_col].astype(str).str.strip()
    df[args.family_col] = df[args.family_col].astype(str).str.strip()

    df, removed_classes = remove_minority_classes(
        df,
        exact_schema.target_col,
        args.min_class_samples,
    )
    df, engineered_temporal_cols, complete_temporal_groups = create_engineered_temporal_features(
        df,
        exact_schema.temporal_groups,
    )
    df, engineered_static_cols = create_engineered_static_features(df)
    if engineered_static_cols:
        print("Engineered static features added:", len(engineered_static_cols))

    split_bundle = split_dataframe(
        df=df,
        target_col=exact_schema.target_col,
        group_id=exact_schema.group_id,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    train_df = split_bundle.train_df.copy()
    val_df = split_bundle.val_df.copy()
    test_df = split_bundle.test_df.copy()

    model_args = build_model_args(args)

    print()
    print("=== Hierarchical Stage 1: broad family prediction ===")
    stage1_train_df = build_stage1_frames(train_df, exact_schema.target_col, args.family_col)
    stage1_val_df = build_stage1_frames(val_df, exact_schema.target_col, args.family_col)
    stage1_test_df = build_stage1_frames(test_df, exact_schema.target_col, args.family_col)
    stage1_schema = detect_schema(stage1_train_df)
    stage1_run = execute_training_run(
        train_df=stage1_train_df,
        val_df=stage1_val_df,
        test_df=stage1_test_df,
        args=model_args,
        schema=stage1_schema,
        complete_temporal_groups=complete_temporal_groups,
        engineered_cols=engineered_temporal_cols,
        device=device,
        verbose=True,
    )

    family_probs_val = stage1_run["val_metrics"]["probs"]
    family_probs_test = stage1_run["test_metrics"]["probs"]
    family_class_names = [str(name) for name in stage1_run["prepared"].label_encoder.classes_]
    family_pred_test = [family_class_names[idx] for idx in family_probs_test.argmax(axis=1)]
    family_true_test = test_df[args.family_col].astype(str).tolist()
    family_metrics = summarize_predictions(family_true_test, family_pred_test)

    stage2_output_dir = output_dir / "stage2_program_models"
    stage2_output_dir.mkdir(parents=True, exist_ok=True)
    family_rows: List[Dict[str, object]] = []
    stage2_val_scores: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    stage2_test_scores: Dict[str, Tuple[np.ndarray, List[str]]] = {}

    all_families = sorted(df[args.family_col].astype(str).unique().tolist())
    for family_name in all_families:
        family_dir = stage2_output_dir / sanitize_name(family_name)
        family_dir.mkdir(parents=True, exist_ok=True)

        family_train_df = train_df.loc[train_df[args.family_col].astype(str) == family_name].copy()
        family_val_df = val_df.loc[val_df[args.family_col].astype(str) == family_name].copy()
        family_test_df = test_df.loc[test_df[args.family_col].astype(str) == family_name].copy()
        family_programs = sorted(
            pd.concat([family_train_df[exact_schema.target_col], family_val_df[exact_schema.target_col]])
            .astype(str)
            .unique()
            .tolist()
        )

        if family_train_df.empty:
            continue

        within_family_class_count = family_train_df[exact_schema.target_col].nunique()
        if within_family_class_count <= 1:
            constant_label = str(family_train_df[exact_schema.target_col].iloc[0])
            stage2_val_scores[family_name] = constant_probability_scores(constant_label, len(val_df))
            stage2_test_scores[family_name] = constant_probability_scores(constant_label, len(test_df))
            family_rows.append(
                {
                    "family": family_name,
                    "status": "constant",
                    "program_count": 1,
                    "train_rows": int(len(family_train_df)),
                    "val_rows": int(len(family_val_df)),
                    "test_rows": int(len(family_test_df)),
                    "test_accuracy": 1.0 if not family_test_df.empty else float("nan"),
                    "test_macro_f1": 1.0 if not family_test_df.empty else float("nan"),
                }
            )
            write_json(
                family_dir / "metrics_summary.json",
                {
                    "family": family_name,
                    "status": "constant",
                    "program_label": constant_label,
                    "train_rows": int(len(family_train_df)),
                    "val_rows": int(len(family_val_df)),
                    "test_rows": int(len(family_test_df)),
                },
            )
            continue

        print()
        print(f"=== Hierarchical Stage 2: exact program within '{family_name}' ===")
        stage2_schema = detect_schema(family_train_df)
        stage2_run = execute_training_run(
            train_df=family_train_df,
            val_df=family_val_df,
            test_df=family_test_df,
            args=model_args,
            schema=stage2_schema,
            complete_temporal_groups=complete_temporal_groups,
            engineered_cols=engineered_temporal_cols,
            device=device,
            verbose=True,
        )
        stage2_metrics = stage2_run["test_metrics"]
        stage2_class_names = [str(name) for name in stage2_run["prepared"].label_encoder.classes_]
        stage2_val_scores[family_name] = score_frame_with_run(
            train_df=family_train_df,
            score_df=val_df,
            schema=stage2_schema,
            complete_temporal_groups=complete_temporal_groups,
            engineered_cols=engineered_temporal_cols,
            run=stage2_run,
            model_args=model_args,
            device=device,
        )
        stage2_test_scores[family_name] = score_frame_with_run(
            train_df=family_train_df,
            score_df=test_df,
            schema=stage2_schema,
            complete_temporal_groups=complete_temporal_groups,
            engineered_cols=engineered_temporal_cols,
            run=stage2_run,
            model_args=model_args,
            device=device,
        )
        family_rows.append(
            {
                "family": family_name,
                "status": "trained",
                "program_count": len(stage2_class_names),
                "train_rows": int(len(family_train_df)),
                "val_rows": int(len(family_val_df)),
                "test_rows": int(len(family_test_df)),
                "test_accuracy": float(stage2_metrics["accuracy"]) if stage2_metrics else float("nan"),
                "test_macro_f1": float(stage2_metrics["macro_f1"]) if stage2_metrics else float("nan"),
            }
        )
        write_json(
            family_dir / "metrics_summary.json",
            {
                "family": family_name,
                "status": "trained",
                "program_class_names": stage2_class_names,
                "test_metrics": None
                if stage2_metrics is None
                else {
                    "accuracy": float(stage2_metrics["accuracy"]),
                    "macro_precision": float(stage2_metrics["macro_precision"]),
                    "macro_recall": float(stage2_metrics["macro_recall"]),
                    "macro_f1": float(stage2_metrics["macro_f1"]),
                    "weighted_precision": float(stage2_metrics["weighted_precision"]),
                    "weighted_recall": float(stage2_metrics["weighted_recall"]),
                    "weighted_f1": float(stage2_metrics["weighted_f1"]),
                    "balanced_accuracy": float(stage2_metrics["balanced_accuracy"]),
                },
            },
        )

    program_classes = sorted(df[exact_schema.target_col].astype(str).unique().tolist())
    combined_val_probs = build_combined_program_probabilities(
        family_probs=family_probs_val,
        family_class_names=family_class_names,
        stage2_scores=stage2_val_scores,
        program_classes=program_classes,
    )
    combined_test_probs = build_combined_program_probabilities(
        family_probs=family_probs_test,
        family_class_names=family_class_names,
        stage2_scores=stage2_test_scores,
        program_classes=program_classes,
    )

    val_true_programs = val_df[exact_schema.target_col].astype(str).tolist()
    test_true_programs = test_df[exact_schema.target_col].astype(str).tolist()
    exact_pred_test = [program_classes[idx] for idx in combined_test_probs.argmax(axis=1)]
    exact_top1_metrics = summarize_predictions(test_true_programs, exact_pred_test)

    topk_values = sorted({max(1, int(k)) for k in args.topk})
    topk_rows = []
    for k in topk_values:
        topk_rows.append(
            {
                "k": k,
                "val_topk_accuracy": topk_accuracy(combined_val_probs, val_true_programs, program_classes, k),
                "test_topk_accuracy": topk_accuracy(combined_test_probs, test_true_programs, program_classes, k),
            }
        )
    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(output_dir / "topk_accuracy_summary.csv", index=False)
    (output_dir / "topk_accuracy_summary.md").write_text(
        "\n".join(
            [
                "# Top-k Accuracy Summary",
                "",
                dataframe_to_markdown_table(topk_df),
                "",
            ]
        ),
        encoding="utf-8",
    )

    thresholds = [round(step, 2) for step in np.arange(0.50, 0.96, 0.05)]
    threshold_val_df = build_threshold_table(
        probs=combined_val_probs,
        true_labels=val_true_programs,
        class_names=program_classes,
        thresholds=thresholds,
    )
    threshold_test_df = build_threshold_table(
        probs=combined_test_probs,
        true_labels=test_true_programs,
        class_names=program_classes,
        thresholds=thresholds,
    )
    threshold_val_df.to_csv(output_dir / "validation_confidence_thresholds.csv", index=False)
    threshold_test_df.to_csv(output_dir / "test_confidence_thresholds.csv", index=False)
    selected_threshold = select_confidence_threshold(
        threshold_df=threshold_val_df,
        target_accuracy=args.confidence_target_accuracy,
    )

    selected_test_summary = None
    if selected_threshold is not None:
        matched = threshold_test_df.loc[
            threshold_test_df["threshold"] == float(selected_threshold["threshold"])
        ]
        if not matched.empty:
            selected_test_summary = matched.iloc[0].to_dict()

    stage2_family_df = save_stage2_family_table(family_rows, output_dir)

    summary = {
        "config": {
            "csv_path": str(csv_path),
            "family_col": args.family_col,
            "model_family": args.model_family,
            "seed": args.seed,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "min_class_samples": args.min_class_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "selection_metric": args.selection_metric,
            "use_class_weights": args.use_class_weights,
            "class_weight_power": args.class_weight_power,
            "confidence_target_accuracy": args.confidence_target_accuracy,
            "topk": topk_values,
        },
        "dataset": {
            "total_rows": int(len(df)),
            "removed_minority_classes": list(removed_classes),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "used_group_split": bool(split_bundle.used_group_split),
            "program_class_count": int(df[exact_schema.target_col].nunique()),
            "family_class_count": int(df[args.family_col].nunique()),
        },
        "stage1_family_metrics": {
            "accuracy": float(family_metrics["accuracy"]),
            "macro_precision": float(family_metrics["macro_precision"]),
            "macro_recall": float(family_metrics["macro_recall"]),
            "macro_f1": float(family_metrics["macro_f1"]),
            "weighted_precision": float(family_metrics["weighted_precision"]),
            "weighted_recall": float(family_metrics["weighted_recall"]),
            "weighted_f1": float(family_metrics["weighted_f1"]),
            "balanced_accuracy": float(family_metrics["balanced_accuracy"]),
        },
        "exact_top1_metrics": {
            "accuracy": float(exact_top1_metrics["accuracy"]),
            "macro_precision": float(exact_top1_metrics["macro_precision"]),
            "macro_recall": float(exact_top1_metrics["macro_recall"]),
            "macro_f1": float(exact_top1_metrics["macro_f1"]),
            "weighted_precision": float(exact_top1_metrics["weighted_precision"]),
            "weighted_recall": float(exact_top1_metrics["weighted_recall"]),
            "weighted_f1": float(exact_top1_metrics["weighted_f1"]),
            "balanced_accuracy": float(exact_top1_metrics["balanced_accuracy"]),
        },
        "topk_accuracy": topk_rows,
        "selected_confidence_threshold": None
        if selected_threshold is None
        else {
            "threshold": float(selected_threshold["threshold"]),
            "validation_coverage": float(selected_threshold["coverage"]),
            "validation_subset_accuracy": float(selected_threshold["subset_accuracy"]),
            "validation_selected_count": int(selected_threshold["selected_count"]),
            "test_coverage": None
            if selected_test_summary is None
            else float(selected_test_summary["coverage"]),
            "test_subset_accuracy": None
            if selected_test_summary is None
            else float(selected_test_summary["subset_accuracy"]),
            "test_selected_count": None
            if selected_test_summary is None
            else int(selected_test_summary["selected_count"]),
        },
        "stage2_family_rows": family_rows,
    }
    write_json(output_dir / "hierarchical_metrics_summary.json", summary)

    save_confusion_plot_for_labels(
        y_true=family_true_test,
        y_pred=family_pred_test,
        labels=sorted(df[args.family_col].astype(str).unique().tolist()),
        output_path=output_dir / "family_confusion_matrix.png",
        title="Stage 1 Family Prediction Confusion Matrix",
    )
    save_confusion_plot_for_labels(
        y_true=test_true_programs,
        y_pred=exact_pred_test,
        labels=program_classes,
        output_path=output_dir / "exact_program_confusion_matrix.png",
        title="Hierarchical Exact Program Top-1 Confusion Matrix",
    )

    topk_metric_map = {
        f"Exact top-{row['k']}": float(row["test_topk_accuracy"])
        for row in topk_rows
    }
    topk_metric_map["Family top-1"] = float(family_metrics["accuracy"])
    if selected_test_summary is not None and not pd.isna(selected_test_summary["subset_accuracy"]):
        topk_metric_map["Confident subset top-1"] = float(selected_test_summary["subset_accuracy"])
        topk_metric_map["Confident subset coverage"] = float(selected_test_summary["coverage"])
    save_metric_bar(
        metric_values=topk_metric_map,
        output_path=output_dir / "hierarchical_metric_bar.png",
        title="Hierarchical Evaluation Summary",
        threshold=args.confidence_target_accuracy,
    )

    threshold_lines = [
        "# Confidence Threshold Search",
        "",
        "## Validation",
        "",
        dataframe_to_markdown_table(threshold_val_df),
        "",
        "## Test",
        "",
        dataframe_to_markdown_table(threshold_test_df),
        "",
    ]
    (output_dir / "confidence_threshold_search.md").write_text(
        "\n".join(threshold_lines),
        encoding="utf-8",
    )

    recommendation_lines = [
        "# Hierarchical Experiment Summary",
        "",
        f"- Model family used for both stages: `{args.model_family}`",
        f"- Stage 1 family accuracy: `{family_metrics['accuracy']:.4f}`",
        f"- Exact-program top-1 accuracy: `{exact_top1_metrics['accuracy']:.4f}`",
    ]
    for row in topk_rows:
        if int(row["k"]) == 1:
            continue
        recommendation_lines.append(
            f"- Exact-program top-{int(row['k'])} accuracy: `{float(row['test_topk_accuracy']):.4f}`"
        )
    if selected_threshold is None:
        recommendation_lines.append(
            f"- No validation confidence threshold reached `{args.confidence_target_accuracy:.2f}` subset accuracy."
        )
    else:
        recommendation_lines.extend(
            [
                (
                    f"- Selected confidence threshold: `{float(selected_threshold['threshold']):.2f}` "
                    f"(validation coverage `{float(selected_threshold['coverage']):.4f}`, "
                    f"validation subset accuracy `{float(selected_threshold['subset_accuracy']):.4f}`)"
                ),
                (
                    f"- Test confident-subset coverage: `{float(selected_test_summary['coverage']):.4f}`; "
                    f"test confident-subset top-1 accuracy: `{float(selected_test_summary['subset_accuracy']):.4f}`"
                )
                if selected_test_summary is not None
                else "- Test confident-subset summary was unavailable.",
            ]
        )
    recommendation_lines.extend(
        [
            "",
            "## Main Files",
            "",
            "- `hierarchical_metrics_summary.json`",
            "- `topk_accuracy_summary.csv` / `.md`",
            "- `confidence_threshold_search.md`",
            "- `stage2_family_summary.csv` / `.md`",
            "- `family_confusion_matrix.png`",
            "- `exact_program_confusion_matrix.png`",
            "- `hierarchical_metric_bar.png`",
            "",
        ]
    )
    (output_dir / "hierarchical_experiment_summary.md").write_text(
        "\n".join(recommendation_lines),
        encoding="utf-8",
    )

    print()
    print("Hierarchical experiment complete.")
    print("Output directory:", output_dir.resolve())
    print(f"Stage 1 family accuracy: {family_metrics['accuracy']:.4f}")
    print(f"Exact top-1 accuracy: {exact_top1_metrics['accuracy']:.4f}")
    for row in topk_rows:
        print(f"Exact top-{int(row['k'])} accuracy: {float(row['test_topk_accuracy']):.4f}")
    if selected_threshold is None:
        print(
            f"No confidence threshold reached validation subset accuracy >= {args.confidence_target_accuracy:.2f}."
        )
    elif selected_test_summary is not None:
        print(
            "Selected confidence threshold:",
            f"{float(selected_threshold['threshold']):.2f}",
            "| test coverage:",
            f"{float(selected_test_summary['coverage']):.4f}",
            "| test subset accuracy:",
            f"{float(selected_test_summary['subset_accuracy']):.4f}",
        )


if __name__ == "__main__":
    main()
