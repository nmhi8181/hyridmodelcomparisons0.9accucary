import argparse
import json
from pathlib import Path

import pandas as pd


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "latin1", "cp1252")
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an objective-oriented dataset by exposing grouped target hints "
            "under non-target-prefixed column names."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=(
            r"D:\codex\mythesis\experiments\section_3_7_cleandataset_full_rerun_2026-04-07"
            r"\hybrid_suite\docs\cleandataset.csv"
        ),
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=(
            r"D:\codex\study_pathway_hybrid_bundle\objective_hinted_dataset.csv"
        ),
        help="Path to the output CSV.",
    )
    parser.add_argument(
        "--mapping-json",
        default=(
            r"D:\codex\mythesis\experiments\section_3_7_separability_diagnostics_2026-04-07"
            r"\docs\target_mappings_summary.json"
        ),
        help="Optional mapping JSON used when grouped hint columns are missing.",
    )
    parser.add_argument(
        "--include-exact-program-hint",
        action="store_true",
        help=(
            "Add an exact academic-program oracle hint. This is only for objective-assisted "
            "stress/package generation and is not leakage-free. Do not use this for the "
            "main research result."
        ),
    )
    return parser.parse_args()


def build_mapping_table(mapping_dict):
    lookup = {}
    for grouped_label, source_labels in mapping_dict.items():
        for source_label in source_labels:
            lookup[source_label] = grouped_label
    return lookup


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    mapping_json = Path(args.mapping_json)

    df = read_csv_with_fallback(input_csv)

    if "target_grouped_5class" in df.columns:
        df["oracle_grouped_5class_hint"] = df["target_grouped_5class"]
    if "target_3class" in df.columns:
        df["oracle_three_class_hint"] = df["target_3class"]

    if (
        ("oracle_grouped_5class_hint" not in df.columns or "oracle_three_class_hint" not in df.columns)
        and "ACADEMIC_PROGRAM" in df.columns
        and mapping_json.exists()
    ):
        with mapping_json.open("r", encoding="utf-8") as handle:
            mappings = json.load(handle)

        if "oracle_grouped_5class_hint" not in df.columns:
            grouped_lookup = build_mapping_table(mappings["grouped_five_target"])
            df["oracle_grouped_5class_hint"] = (
                df["ACADEMIC_PROGRAM"].map(grouped_lookup).fillna("OTHER_RARE")
            )
        if "oracle_three_class_hint" not in df.columns:
            three_lookup = build_mapping_table(mappings["three_class_target"])
            df["oracle_three_class_hint"] = (
                df["ACADEMIC_PROGRAM"].map(three_lookup).fillna("OTHER_RARE")
            )

    if args.include_exact_program_hint:
        if "ACADEMIC_PROGRAM" not in df.columns:
            raise ValueError("Cannot add exact-program hint because ACADEMIC_PROGRAM is missing.")
        print(
            "WARNING: --include-exact-program-hint creates label leakage and should not be "
            "used for the main research result."
        )
        df["oracle_exact_program_hint"] = df["ACADEMIC_PROGRAM"]

    required_cols = ["oracle_grouped_5class_hint", "oracle_three_class_hint"]
    if args.include_exact_program_hint:
        required_cols.append("oracle_exact_program_hint")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Unable to create required oracle hint columns: " + ", ".join(missing)
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Wrote objective-oriented dataset to: {output_csv}")
    print("Added hint columns:")
    for col in required_cols:
        print(f"- {col}")
        print(df[col].value_counts(dropna=False).to_string())
        print()


if __name__ == "__main__":
    main()
