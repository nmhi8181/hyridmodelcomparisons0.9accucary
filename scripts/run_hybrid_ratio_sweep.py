import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MODEL_FAMILIES = (
    "hybrid_tcn_lstm",
    "ft_transformer_gru_attention",
    "deepfm_temporal_tcn",
    "xgboost_temporal_gru",
)
SPLIT_RATIOS: Tuple[Tuple[str, float], ...] = (
    ("80_20", 0.2),
    ("70_30", 0.3),
    ("60_40", 0.4),
    ("50_50", 0.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all hybrid model families across multiple outer train:test ratios and "
            "generate a top-level comparison package."
        )
    )
    parser.add_argument(
        "--csv-path",
        default=r"D:\codex\mythesis\dataset.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory where the ratio sweep folder will be created. Defaults to the bundle folder.",
    )
    parser.add_argument(
        "--folder-name",
        default=None,
        help=(
            "Optional explicit output folder name. Defaults to "
            "'hybrid_ratio_sweep_YYYY-MM-DD_HHMMSS'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument(
        "--ratios",
        nargs="+",
        choices=[ratio for ratio, _ in SPLIT_RATIOS],
        default=[ratio for ratio, _ in SPLIT_RATIOS],
        help="Subset of split ratios to run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a family run when its metrics_summary.json already exists.",
    )
    return parser.parse_args()


def run_command(command: Iterable[str], cwd: Path) -> None:
    resolved_command = [str(part) for part in command]
    print("Running command:")
    print(" ".join(resolved_command))
    subprocess.run(resolved_command, cwd=str(cwd), check=True)


def family_overrides(model_family: str, epochs: int, run_cross_validation: bool) -> Dict[str, object]:
    base = {
        "batch_size": 64,
        "epochs": epochs,
        "learning_rate": 0.0005,
        "weight_decay": 0.001,
        "optimizer": "adamw",
        "momentum": 0.9,
        "scheduler": "plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 4,
        "scheduler_eta_min": 0.000001,
        "early_stopping_patience": 8,
        "grad_clip": 1.0,
        "static_hid_dim": 256,
        "static_out_dim": 128,
        "static_dropout": 0.1,
        "static_n_layers": 2,
        "static_use_batchnorm": True,
        "temporal_hid_dim": 128,
        "temporal_kernel_size": 2,
        "temporal_n_blocks": 3,
        "temporal_dropout": 0.1,
        "temporal_use_batchnorm": True,
        "temporal_use_layernorm": True,
        "classifier_hid_dim": 128,
        "classifier_n_layers": 2,
        "classifier_use_batchnorm": True,
        "transformer_layers": 3,
        "transformer_heads": 4,
        "transformer_dropout": 0.1,
        "fm_embed_dim": 64,
        "xgb_num_boost_round": 500,
        "augment_with_xgb_probs": True,
        "final_ensemble_with_xgb": True,
        "ensemble_weight_step": 0.05,
        "selection_metric": "accuracy",
        "use_class_weights": False,
        "class_weight_power": 1.0,
        "label_smoothing": 0.0,
        "run_cross_validation": run_cross_validation,
        "final_retrain_val_size": 0.1,
    }

    if model_family == "hybrid_tcn_lstm":
        base.update(
            {
                "epochs": max(epochs, 8),
                "temporal_recurrent_type": "LSTM",
                "temporal_hid_dim": 160,
                "temporal_n_blocks": 4,
                "classifier_hid_dim": 160,
                "xgb_num_boost_round": 700,
                "ensemble_weight_step": 0.01,
            }
        )
    elif model_family == "ft_transformer_gru_attention":
        base.update(
            {
                "static_hid_dim": 192,
                "temporal_recurrent_type": "GRU",
                "transformer_layers": 3,
                "transformer_heads": 4,
                "classifier_hid_dim": 144,
            }
        )
    elif model_family == "deepfm_temporal_tcn":
        base.update(
            {
                "static_hid_dim": 192,
                "temporal_recurrent_type": "GRU",
                "fm_embed_dim": 96,
                "temporal_n_blocks": 4,
                "classifier_hid_dim": 144,
            }
        )
    elif model_family == "xgboost_temporal_gru":
        base.update(
            {
                "epochs": max(epochs, 6),
                "temporal_recurrent_type": "GRU",
                "static_out_dim": 160,
                "classifier_hid_dim": 160,
                "xgb_num_boost_round": 500,
                "learning_rate": 0.0007,
                "label_smoothing": 0.0,
            }
        )
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    return base


def extend_with_overrides(command: List[str], overrides: Dict[str, object]) -> List[str]:
    boolean_flags = {
        "static_use_batchnorm": "--static-use-batchnorm",
        "temporal_use_batchnorm": "--temporal-use-batchnorm",
        "temporal_use_layernorm": "--temporal-use-layernorm",
        "classifier_use_batchnorm": "--classifier-use-batchnorm",
        "use_class_weights": "--use-class-weights",
        "run_cross_validation": "--run-cross-validation",
        "augment_with_xgb_probs": "--augment-with-xgb-probs",
        "final_ensemble_with_xgb": "--final-ensemble-with-xgb",
    }
    for key, value in overrides.items():
        if key in boolean_flags:
            if value:
                command.append(boolean_flags[key])
            continue
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    return command


def main() -> None:
    args = parse_args()
    bundle_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root) if args.output_root else bundle_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_name = args.folder_name or f"hybrid_ratio_sweep_{timestamp}"
    output_dir = output_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_script = bundle_dir / "study_pathway_hybrid_clean.py"
    package_script = bundle_dir / "generate_hybrid_result_package.py"
    sweep_package_script = bundle_dir / "generate_hybrid_ratio_sweep_report.py"

    selected_ratio_labels = set(args.ratios)
    for ratio_label, test_size in SPLIT_RATIOS:
        if ratio_label not in selected_ratio_labels:
            continue
        ratio_dir = output_dir / f"ratio_{ratio_label}"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"=== Ratio sweep: {ratio_label.replace('_', ':')} ===")
        for model_family in MODEL_FAMILIES:
            family_dir = ratio_dir / model_family
            metrics_path = family_dir / "metrics_summary.json"
            if args.skip_existing and metrics_path.exists():
                print(f"Skipping existing run for {ratio_label}/{model_family}")
                continue

            command: List[str] = [
                sys.executable,
                str(pipeline_script),
                "--csv-path",
                args.csv_path,
                "--output-dir",
                str(family_dir),
                "--model-family",
                model_family,
                "--seed",
                str(args.seed),
                "--test-size",
                str(test_size),
                "--val-size",
                str(args.val_size),
                "--min-class-samples",
                "2",
                "--cv-folds",
                str(args.cv_folds),
            ]
            command = extend_with_overrides(
                command,
                family_overrides(
                    model_family,
                    args.epochs,
                    run_cross_validation=args.cv_folds >= 2,
                ),
            )
            run_command(command, bundle_dir)

        run_command(
            [
                sys.executable,
                str(package_script),
                "--root-dir",
                str(ratio_dir),
            ],
            bundle_dir,
        )

    run_command(
        [
            sys.executable,
            str(sweep_package_script),
            "--root-dir",
            str(output_dir),
        ],
        bundle_dir,
    )

    print()
    print(f"Hybrid ratio sweep complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
