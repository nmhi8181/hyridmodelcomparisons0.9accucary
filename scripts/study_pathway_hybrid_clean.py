import argparse
import copy
import json
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
        selection_rate,
    )
except Exception:
    MetricFrame = None
    demographic_parity_difference = None
    equalized_odds_difference = None
    selection_rate = None


DEFAULT_TARGET_ALIASES = {
    "academic_program",
    "academicprogramme",
    "academic_programme",
    "program_akademik",
    "program",
}
DEFAULT_ID_KEYWORDS = ("student", "stud_id", "stu_id", "id", "matric", "cod")
DEFAULT_SENSITIVE_KEYWORDS = ("gender", "sex", "race", "ethnicity", "stratum", "sisben")
DEFAULT_TEMPORAL_SUFFIXES = ("_S11", "_PRO")
MODEL_FAMILIES = (
    "hybrid_tcn_lstm",
    "ft_transformer_gru_attention",
    "xgboost_temporal_gru",
    "deepfm_temporal_tcn",
)


@dataclass
class SchemaInfo:
    target_col: str
    group_id: Optional[str]
    sensitive_cols: List[str]
    temporal_groups: Dict[str, Dict[str, str]]


@dataclass
class SplitBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    used_group_split: bool


@dataclass
class TemporalStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class PreparedData:
    static_cols: List[str]
    num_cols: List[str]
    cat_cols: List[str]
    static_preprocessor: ColumnTransformer
    label_encoder: LabelEncoder
    temporal_feature_names: List[str]
    x_static_train: object
    x_static_val: object
    x_static_test: Optional[object]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: Optional[np.ndarray]
    x_seq_train: Optional[np.ndarray]
    x_seq_val: Optional[np.ndarray]
    x_seq_test: Optional[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cleaned study pathway hybrid pipeline based on the original notebook export."
    )
    parser.add_argument("--csv-path", required=True, help="Path to the input CSV dataset.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where reports, plots, and metrics will be saved. Defaults to the script folder.",
    )
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES,
        default="hybrid_tcn_lstm",
        help="Hybrid model family to train.",
    )
    parser.add_argument(
        "--compare-model-families",
        action="store_true",
        help="Run all available model families in one sweep and write a single comparison table.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation size as a fraction of the full dataset.",
    )
    parser.add_argument("--min-class-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adam", "sgd", "rmsprop"),
        default="adamw",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--scheduler",
        choices=("none", "plateau", "cosine"),
        default="plateau",
    )
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--static-hid-dim", type=int, default=128)
    parser.add_argument("--static-out-dim", type=int, default=64)
    parser.add_argument("--static-dropout", type=float, default=0.2)
    parser.add_argument("--static-n-layers", type=int, default=1)
    parser.add_argument("--static-use-batchnorm", action="store_true")
    parser.add_argument("--temporal-hid-dim", type=int, default=64)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--temporal-n-blocks", type=int, default=2)
    parser.add_argument("--temporal-dropout", type=float, default=0.1)
    parser.add_argument(
        "--temporal-recurrent-type",
        choices=("LSTM", "GRU"),
        default="LSTM",
    )
    parser.add_argument("--temporal-use-batchnorm", action="store_true")
    parser.add_argument("--temporal-use-layernorm", action="store_true")
    parser.add_argument("--classifier-hid-dim", type=int, default=64)
    parser.add_argument("--classifier-n-layers", type=int, default=1)
    parser.add_argument("--classifier-use-batchnorm", action="store_true")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--fm-embed-dim", type=int, default=32)
    parser.add_argument("--xgb-num-boost-round", type=int, default=200)
    parser.add_argument(
        "--augment-with-xgb-probs",
        action="store_true",
        help="Append training-split XGBoost class probabilities to the static feature branch.",
    )
    parser.add_argument(
        "--final-ensemble-with-xgb",
        action="store_true",
        help="Blend neural probabilities with an XGBoost baseline using validation-selected weights.",
    )
    parser.add_argument(
        "--ensemble-weight-step",
        type=float,
        default=0.05,
        help="Step size for validation search over neural/XGBoost blending weights.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1", "balanced_accuracy", "composite"),
        default="accuracy",
        help="Validation metric used for early stopping and best-checkpoint selection.",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weights for cross-entropy training.",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="Power applied to inverse-frequency class weights.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Optional label smoothing for cross-entropy training.",
    )
    parser.add_argument("--run-xgboost", action="store_true")
    parser.add_argument("--run-fairness", action="store_true")
    parser.add_argument("--run-random-search", action="store_true")
    parser.add_argument("--random-search-iterations", type=int, default=10)
    parser.add_argument("--run-cross-validation", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--final-retrain-val-size",
        type=float,
        default=0.1,
        help="Internal validation share used when retraining on combined train+val.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv_with_fallback_encodings(csv_path: Path) -> pd.DataFrame:
    encodings = [None, "utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            kwargs = {} if encoding is None else {"encoding": encoding}
            return pd.read_csv(csv_path, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(csv_path)


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def detect_schema(df: pd.DataFrame) -> SchemaInfo:
    target_candidates = [
        c
        for c in df.columns
        if normalize_name(c) in DEFAULT_TARGET_ALIASES
        or ("academic" in c.lower() and "program" in c.lower())
    ]
    if not target_candidates:
        raise ValueError(
            "Target column not found. Expected an academic program column such as "
            "'ACADEMIC_PROGRAM'."
        )

    target_col = target_candidates[0]
    group_candidates = [
        c for c in df.columns if any(token in c.lower() for token in DEFAULT_ID_KEYWORDS)
    ]
    sensitive_cols = [
        c for c in df.columns if any(token in c.lower() for token in DEFAULT_SENSITIVE_KEYWORDS)
    ]

    temporal_groups: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        for suffix in DEFAULT_TEMPORAL_SUFFIXES:
            if col.endswith(suffix):
                base = col[: -len(suffix)]
                temporal_groups.setdefault(base, {})[suffix] = col

    return SchemaInfo(
        target_col=target_col,
        group_id=group_candidates[0] if group_candidates else None,
        sensitive_cols=sensitive_cols,
        temporal_groups=temporal_groups,
    )


def remove_minority_classes(
    df: pd.DataFrame, target_col: str, min_class_samples: int
) -> Tuple[pd.DataFrame, List[str]]:
    class_counts = df[target_col].value_counts()
    removed = class_counts[class_counts < min_class_samples].index.tolist()
    if not removed:
        return df, removed

    cleaned = df.loc[~df[target_col].isin(removed)].copy()
    return cleaned, removed


def create_engineered_temporal_features(
    df: pd.DataFrame,
    temporal_groups: Dict[str, Dict[str, str]],
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, str]]]:
    df = df.copy()
    complete_pairs = {
        base: group
        for base, group in temporal_groups.items()
        if all(suffix in group for suffix in DEFAULT_TEMPORAL_SUFFIXES)
    }

    engineered_cols: List[str] = []
    for base, group in complete_pairs.items():
        s11_col = group["_S11"]
        pro_col = group["_PRO"]
        s11_values = pd.to_numeric(df[s11_col], errors="coerce")
        pro_values = pd.to_numeric(df[pro_col], errors="coerce")

        change_col = f"{base}_change"
        ratio_col = f"{base}_ratio"
        df[change_col] = pro_values - s11_values
        df[ratio_col] = pro_values / (s11_values + epsilon)
        engineered_cols.extend([change_col, ratio_col])

    return df, engineered_cols, complete_pairs


def create_engineered_static_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    engineered_cols: List[str] = []

    score_cols = [
        col
        for col in ["MAT_S11", "CR_S11", "CC_S11", "BIO_S11", "ENG_S11", "QR_PRO", "WC_PRO", "FEP_PRO"]
        if col in df.columns
    ]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if score_cols:
        aggregate_specs = {
            "score_mean_all": df[score_cols].mean(axis=1),
            "score_std_all": df[score_cols].std(axis=1),
            "score_min_all": df[score_cols].min(axis=1),
            "score_max_all": df[score_cols].max(axis=1),
        }
        for column_name, values in aggregate_specs.items():
            df[column_name] = values
            engineered_cols.append(column_name)
        df["score_range_all"] = df["score_max_all"] - df["score_min_all"]
        engineered_cols.append("score_range_all")

    pairwise_specs = [
        ("MAT_S11", "ENG_S11", "mat_minus_eng"),
        ("MAT_S11", "CR_S11", "mat_minus_cr"),
        ("BIO_S11", "MAT_S11", "bio_minus_mat"),
        ("QR_PRO", "WC_PRO", "qr_minus_wc"),
        ("QR_PRO", "FEP_PRO", "qr_minus_fep"),
        ("CR_S11", "CC_S11", "cr_minus_cc"),
        ("ENG_S11", "CC_S11", "eng_minus_cc"),
    ]
    for left_col, right_col, feature_name in pairwise_specs:
        if left_col in df.columns and right_col in df.columns:
            df[feature_name] = pd.to_numeric(df[left_col], errors="coerce") - pd.to_numeric(
                df[right_col], errors="coerce"
            )
            engineered_cols.append(feature_name)

    return df, engineered_cols


def split_dataframe(
    df: pd.DataFrame,
    target_col: str,
    group_id: Optional[str],
    test_size: float,
    val_size: float,
    seed: int,
) -> SplitBundle:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    relative_val_size = val_size / (1.0 - test_size)
    used_group_split = False

    if (
        group_id is not None
        and df[group_id].nunique(dropna=False) >= 3
        and df[group_id].nunique(dropna=False) < len(df)
    ):
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_val_idx, test_idx = next(splitter.split(df, groups=df[group_id]))
            train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            splitter_val = GroupShuffleSplit(
                n_splits=1, test_size=relative_val_size, random_state=seed
            )
            train_idx, val_idx = next(
                splitter_val.split(train_val_df, groups=train_val_df[group_id])
            )
            train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
            used_group_split = True
            return SplitBundle(train_df, val_df, test_df, used_group_split)
        except Exception as exc:
            print(f"Group split failed ({exc}). Falling back to stratified splitting.")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df[target_col],
    )
    return SplitBundle(
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        used_group_split,
    )


def split_train_validation(
    df: pd.DataFrame,
    target_col: str,
    group_id: Optional[str],
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    used_group_split = False
    if (
        group_id is not None
        and df[group_id].nunique(dropna=False) >= 3
        and df[group_id].nunique(dropna=False) < len(df)
    ):
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
            train_idx, val_idx = next(splitter.split(df, groups=df[group_id]))
            return (
                df.iloc[train_idx].reset_index(drop=True),
                df.iloc[val_idx].reset_index(drop=True),
                True,
            )
        except Exception as exc:
            print(f"Group validation split failed ({exc}). Falling back to standard splitting.")

    try:
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=seed,
            stratify=df[target_col],
        )
    except Exception as exc:
        print(f"Stratified validation split failed ({exc}). Falling back to unstratified split.")
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=seed,
        )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), used_group_split


def infer_static_columns(
    df: pd.DataFrame,
    target_col: str,
    group_id: Optional[str],
    temporal_groups: Dict[str, Dict[str, str]],
    engineered_cols: Sequence[str],
) -> List[str]:
    drop_cols = [target_col]
    if group_id:
        drop_cols.append(group_id)

    original_temporal_cols = {
        col for group in temporal_groups.values() for col in group.values()
    }
    return [
        col
        for col in df.columns
        if col not in drop_cols
        and col not in original_temporal_cols
        and col not in engineered_cols
        and not normalize_name(col).startswith("target")
    ]


def build_static_preprocessor(
    train_df: pd.DataFrame, static_cols: Sequence[str]
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [col for col in static_cols if pd.api.types.is_numeric_dtype(train_df[col])]
    cat_cols = [col for col in static_cols if col not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("No static features remain after preprocessing.")

    return ColumnTransformer(transformers), num_cols, cat_cols


def _coerce_and_impute_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric_frame = frame.loc[:, list(columns)].copy()
    for col in numeric_frame.columns:
        numeric_frame[col] = pd.to_numeric(numeric_frame[col], errors="coerce")
    median_values = numeric_frame.median(numeric_only=True)
    numeric_frame.fillna(median_values.fillna(0.0), inplace=True)
    return numeric_frame


def build_temporal_tensor(
    frame: pd.DataFrame,
    complete_temporal_groups: Dict[str, Dict[str, str]],
    engineered_cols: Sequence[str],
    temporal_stats: Optional[TemporalStats] = None,
) -> Tuple[Optional[np.ndarray], List[str], Optional[TemporalStats]]:
    sorted_bases = sorted(complete_temporal_groups.keys())
    s11_cols = [
        complete_temporal_groups[base]["_S11"]
        for base in sorted_bases
        if complete_temporal_groups[base]["_S11"] in frame.columns
    ]
    pro_cols = [
        complete_temporal_groups[base]["_PRO"]
        for base in sorted_bases
        if complete_temporal_groups[base]["_PRO"] in frame.columns
    ]

    x_original = None
    original_feature_names: List[str] = []
    if s11_cols and pro_cols:
        s11_frame = _coerce_and_impute_numeric(frame, s11_cols)
        pro_frame = _coerce_and_impute_numeric(frame, pro_cols)

        x_s11 = s11_frame.to_numpy(dtype=np.float32)
        x_pro = pro_frame.to_numpy(dtype=np.float32)
        if x_s11.shape[1] > 0 and x_pro.shape[1] > 0:
            x_original = np.stack([x_s11, x_pro], axis=1)
            original_feature_names = sorted_bases

    engineered_present = [col for col in engineered_cols if col in frame.columns]
    x_engineered = None
    if engineered_present:
        engineered_frame = _coerce_and_impute_numeric(frame, engineered_present)
        engineered_values = engineered_frame.to_numpy(dtype=np.float32)
        x_engineered = np.repeat(engineered_values[:, np.newaxis, :], 2, axis=1)

    if x_original is not None and x_engineered is not None:
        x = np.concatenate([x_original, x_engineered], axis=-1)
        feature_names = original_feature_names + engineered_present
    elif x_original is not None:
        x = x_original
        feature_names = original_feature_names
    elif x_engineered is not None:
        x = x_engineered
        feature_names = engineered_present
    else:
        return None, [], temporal_stats

    if np.isnan(x).any():
        flat_x = x.reshape(-1, x.shape[-1])
        imputer = SimpleImputer(strategy="median")
        x = imputer.fit_transform(flat_x).reshape(x.shape)

    if temporal_stats is None:
        mean = x.reshape(-1, x.shape[-1]).mean(axis=0, keepdims=True)
        std = x.reshape(-1, x.shape[-1]).std(axis=0, keepdims=True) + 1e-6
        temporal_stats = TemporalStats(mean=mean, std=std)

    x = (x - temporal_stats.mean) / temporal_stats.std
    return x.astype(np.float32), feature_names, temporal_stats


class SeqStaticDataset(Dataset):
    def __init__(
        self,
        static_data,
        labels: np.ndarray,
        seq_data: Optional[np.ndarray] = None,
    ) -> None:
        if hasattr(static_data, "toarray"):
            static_data = static_data.toarray()
        self.static_data = torch.tensor(np.asarray(static_data), dtype=torch.float32)
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)
        self.seq_data = (
            torch.tensor(np.asarray(seq_data), dtype=torch.float32)
            if seq_data is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.seq_data is not None:
            return self.static_data[idx], self.seq_data[idx], self.labels[idx]
        return self.static_data[idx], None, self.labels[idx]


def collate_fn(batch):
    static_data = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    if batch and batch[0][1] is not None:
        seq_data = torch.stack([item[1] for item in batch])
    else:
        seq_data = None
    return static_data, seq_data, labels


def prepare_model_inputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    schema: SchemaInfo,
    complete_temporal_groups: Dict[str, Dict[str, str]],
    engineered_cols: Sequence[str],
) -> PreparedData:
    full_df = pd.concat(
        [frame for frame in [train_df, val_df, test_df] if frame is not None],
        axis=0,
    ).reset_index(drop=True)
    static_cols = infer_static_columns(
        df=full_df,
        target_col=schema.target_col,
        group_id=schema.group_id,
        temporal_groups=complete_temporal_groups,
        engineered_cols=engineered_cols,
    )
    static_preprocessor, num_cols, cat_cols = build_static_preprocessor(train_df, static_cols)
    x_static_train = static_preprocessor.fit_transform(train_df[static_cols])
    x_static_val = static_preprocessor.transform(val_df[static_cols])
    x_static_test = (
        static_preprocessor.transform(test_df[static_cols]) if test_df is not None else None
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(
        pd.concat(
            [
                frame[schema.target_col]
                for frame in [train_df, val_df, test_df]
                if frame is not None
            ]
        )
    )
    y_train = label_encoder.transform(train_df[schema.target_col]).astype(int)
    y_val = label_encoder.transform(val_df[schema.target_col]).astype(int)
    y_test = (
        label_encoder.transform(test_df[schema.target_col]).astype(int)
        if test_df is not None
        else None
    )

    x_seq_train, temporal_feature_names, temporal_stats = build_temporal_tensor(
        train_df, complete_temporal_groups, engineered_cols, temporal_stats=None
    )
    x_seq_val, _, _ = build_temporal_tensor(
        val_df, complete_temporal_groups, engineered_cols, temporal_stats=temporal_stats
    )
    x_seq_test = None
    if test_df is not None:
        x_seq_test, _, _ = build_temporal_tensor(
            test_df,
            complete_temporal_groups,
            engineered_cols,
            temporal_stats=temporal_stats,
        )

    return PreparedData(
        static_cols=static_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        static_preprocessor=static_preprocessor,
        label_encoder=label_encoder,
        temporal_feature_names=temporal_feature_names,
        x_static_train=x_static_train,
        x_static_val=x_static_val,
        x_static_test=x_static_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        x_seq_train=x_seq_train,
        x_seq_val=x_seq_val,
        x_seq_test=x_seq_test,
    )


def build_dataloaders(
    prepared: PreparedData, batch_size: int
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    train_ds = SeqStaticDataset(prepared.x_static_train, prepared.y_train, prepared.x_seq_train)
    val_ds = SeqStaticDataset(prepared.x_static_val, prepared.y_val, prepared.x_seq_val)
    test_loader = None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    if prepared.x_static_test is not None and prepared.y_test is not None:
        test_ds = SeqStaticDataset(prepared.x_static_test, prepared.y_test, prepared.x_seq_test)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
    return train_loader, val_loader, test_loader


def clone_args_with_overrides(args: argparse.Namespace, overrides: Dict[str, object]):
    merged = vars(args).copy()
    merged.update(overrides)
    return argparse.Namespace(**merged)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.trim = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.trim > 0:
            return out[:, :, :-self.trim]
        return out


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        layers.append(CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

        residual_layers: List[nn.Module] = []
        if in_channels != out_channels:
            residual_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        if use_batchnorm:
            residual_layers.append(nn.BatchNorm1d(out_channels))
        self.residual = nn.Sequential(*residual_layers) if residual_layers else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.residual(x))


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 2,
        dropout: float = 0.1,
        recurrent_type: str = "LSTM",
        use_batchnorm: bool = False,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        channels = in_features
        dilation = 1
        for _ in range(n_blocks):
            layers.append(
                TCNBlock(
                    channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
            channels = hidden_dim
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.layernorm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.recurrent_type = recurrent_type
        if recurrent_type == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        elif recurrent_type == "GRU":
            self.recurrent = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported recurrent_type: {recurrent_type}")
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x = self.layernorm(x)
        if self.recurrent_type == "LSTM":
            _, (hidden, _) = self.recurrent(x)
            return hidden[-1]
        _, hidden = self.recurrent(x)
        return hidden[-1]


class StaticEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.2,
        n_layers: int = 1,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridModel(nn.Module):
    def __init__(self, static_dim: int, temporal_in_dim: Optional[int], n_classes: int, args):
        super().__init__()
        self.use_temporal = temporal_in_dim is not None
        self.static_encoder = StaticEncoder(
            in_dim=static_dim,
            hidden_dim=args.static_hid_dim,
            out_dim=args.static_out_dim,
            dropout=args.static_dropout,
            n_layers=args.static_n_layers,
            use_batchnorm=args.static_use_batchnorm,
        )
        fused_dim = self.static_encoder.out_dim
        if self.use_temporal:
            self.temporal_encoder = TemporalEncoder(
                in_features=temporal_in_dim,
                hidden_dim=args.temporal_hid_dim,
                kernel_size=args.temporal_kernel_size,
                n_blocks=args.temporal_n_blocks,
                dropout=args.temporal_dropout,
                recurrent_type=args.temporal_recurrent_type,
                use_batchnorm=args.temporal_use_batchnorm,
                use_layernorm=args.temporal_use_layernorm,
            )
            fused_dim += self.temporal_encoder.out_dim
        else:
            self.temporal_encoder = None

        classifier_layers: List[nn.Module] = []
        current_dim = fused_dim
        for _ in range(max(1, args.classifier_n_layers)):
            classifier_layers.append(nn.Linear(current_dim, args.classifier_hid_dim))
            if args.classifier_use_batchnorm:
                classifier_layers.append(nn.BatchNorm1d(args.classifier_hid_dim))
            classifier_layers.append(nn.ReLU())
            current_dim = args.classifier_hid_dim
        classifier_layers.append(nn.Linear(current_dim, n_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(
        self, static_x: torch.Tensor, temporal_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        static_repr = self.static_encoder(static_x)
        if self.use_temporal and temporal_x is not None:
            temporal_repr = self.temporal_encoder(temporal_x)
            fused = torch.cat([static_repr, temporal_repr], dim=-1)
        else:
            fused = static_repr
        return self.classifier(fused)


def resolve_attention_heads(embed_dim: int, requested_heads: int) -> int:
    heads = max(1, min(requested_heads, embed_dim))
    while heads > 1 and embed_dim % heads != 0:
        heads -= 1
    return max(1, heads)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class GRUAttentionTemporalEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionPooling(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(x)
        outputs = self.dropout(outputs)
        return self.attention(outputs)


class TemporalTCNPoolingEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        channels = in_features
        dilation = 1
        for _ in range(n_blocks):
            layers.append(
                TCNBlock(
                    channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
            channels = hidden_dim
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        return x[:, -1, :]


class FTTransformerStaticEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.token_count = min(16, max(4, (in_dim + 63) // 64))
        self.input_proj = nn.Linear(in_dim, self.token_count * hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.positional_tokens = nn.Parameter(
            torch.randn(1, self.token_count + 1, hidden_dim) * 0.02
        )
        heads = resolve_attention_heads(hidden_dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(x).view(x.size(0), self.token_count, self.hidden_dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.positional_tokens[:, : tokens.size(1), :]
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded[:, 0, :])
        return self.proj(encoded)


class DeepFMStaticEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.2,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.feature_embeddings = nn.Parameter(torch.randn(in_dim, embed_dim) * 0.01)
        deep_layers: List[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(1, n_layers)):
            deep_layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        self.deep = nn.Sequential(*deep_layers)
        self.proj = nn.Linear(embed_dim + hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_emb = x.unsqueeze(-1) * self.feature_embeddings.unsqueeze(0)
        summed = feature_emb.sum(dim=1)
        pairwise = 0.5 * ((summed * summed) - (feature_emb * feature_emb).sum(dim=1))
        deep_repr = self.deep(x)
        fused = torch.cat([pairwise, deep_repr], dim=-1)
        return self.proj(fused)


class ProbabilityStaticEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusionClassifier(nn.Module):
    def __init__(
        self,
        static_in_dim: int,
        temporal_in_dim: Optional[int],
        n_classes: int,
        args,
    ) -> None:
        super().__init__()
        self.use_temporal = temporal_in_dim is not None
        fusion_dim = args.classifier_hid_dim
        self.static_proj = nn.Linear(static_in_dim, fusion_dim)
        self.temporal_proj = nn.Linear(temporal_in_dim, fusion_dim) if self.use_temporal else None
        self.gate = nn.Linear(fusion_dim * 2, fusion_dim) if self.use_temporal else None
        layers: List[nn.Module] = []
        current_dim = fusion_dim * 2 if self.use_temporal else fusion_dim
        for _ in range(max(1, args.classifier_n_layers)):
            layers.append(nn.Linear(current_dim, args.classifier_hid_dim))
            if args.classifier_use_batchnorm:
                layers.append(nn.BatchNorm1d(args.classifier_hid_dim))
            layers.append(nn.ReLU())
            current_dim = args.classifier_hid_dim
        layers.append(nn.Linear(current_dim, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, static_repr: torch.Tensor, temporal_repr: Optional[torch.Tensor] = None) -> torch.Tensor:
        static_proj = self.static_proj(static_repr)
        if self.use_temporal and temporal_repr is not None:
            temporal_proj = self.temporal_proj(temporal_repr)
            gate = torch.sigmoid(self.gate(torch.cat([static_proj, temporal_proj], dim=-1)))
            fused = torch.cat([gate * static_proj, (1.0 - gate) * temporal_proj], dim=-1)
        else:
            fused = static_proj
        return self.classifier(fused)


class FTTransformerGRUAttentionModel(nn.Module):
    def __init__(self, static_dim: int, temporal_in_dim: Optional[int], n_classes: int, args) -> None:
        super().__init__()
        self.static_encoder = FTTransformerStaticEncoder(
            in_dim=static_dim,
            hidden_dim=args.static_hid_dim,
            out_dim=args.static_out_dim,
            num_layers=args.transformer_layers,
            num_heads=args.transformer_heads,
            dropout=args.transformer_dropout,
        )
        self.temporal_encoder = (
            GRUAttentionTemporalEncoder(
                in_features=temporal_in_dim,
                hidden_dim=args.temporal_hid_dim,
                dropout=args.temporal_dropout,
            )
            if temporal_in_dim is not None
            else None
        )
        temporal_out = self.temporal_encoder.out_dim if self.temporal_encoder is not None else None
        self.fusion = GatedFusionClassifier(
            static_in_dim=self.static_encoder.out_dim,
            temporal_in_dim=temporal_out,
            n_classes=n_classes,
            args=args,
        )

    def forward(self, static_x: torch.Tensor, temporal_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        static_repr = self.static_encoder(static_x)
        temporal_repr = self.temporal_encoder(temporal_x) if self.temporal_encoder is not None and temporal_x is not None else None
        return self.fusion(static_repr, temporal_repr)


class DeepFMTemporalTCNModel(nn.Module):
    def __init__(self, static_dim: int, temporal_in_dim: Optional[int], n_classes: int, args) -> None:
        super().__init__()
        self.static_encoder = DeepFMStaticEncoder(
            in_dim=static_dim,
            hidden_dim=args.static_hid_dim,
            out_dim=args.static_out_dim,
            embed_dim=args.fm_embed_dim,
            dropout=args.static_dropout,
            n_layers=args.static_n_layers,
        )
        self.temporal_encoder = (
            TemporalTCNPoolingEncoder(
                in_features=temporal_in_dim,
                hidden_dim=args.temporal_hid_dim,
                kernel_size=args.temporal_kernel_size,
                n_blocks=args.temporal_n_blocks,
                dropout=args.temporal_dropout,
                use_batchnorm=args.temporal_use_batchnorm,
            )
            if temporal_in_dim is not None
            else None
        )
        temporal_out = self.temporal_encoder.out_dim if self.temporal_encoder is not None else None
        self.fusion = GatedFusionClassifier(
            static_in_dim=self.static_encoder.out_dim,
            temporal_in_dim=temporal_out,
            n_classes=n_classes,
            args=args,
        )

    def forward(self, static_x: torch.Tensor, temporal_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        static_repr = self.static_encoder(static_x)
        temporal_repr = self.temporal_encoder(temporal_x) if self.temporal_encoder is not None and temporal_x is not None else None
        return self.fusion(static_repr, temporal_repr)


class XGBoostTemporalGRUModel(nn.Module):
    def __init__(self, static_dim: int, temporal_in_dim: Optional[int], n_classes: int, args) -> None:
        super().__init__()
        self.static_encoder = ProbabilityStaticEncoder(
            in_dim=static_dim,
            hidden_dim=max(args.classifier_hid_dim, static_dim),
            out_dim=args.static_out_dim,
            dropout=args.static_dropout,
        )
        self.temporal_encoder = (
            GRUAttentionTemporalEncoder(
                in_features=temporal_in_dim,
                hidden_dim=args.temporal_hid_dim,
                dropout=args.temporal_dropout,
            )
            if temporal_in_dim is not None
            else None
        )
        temporal_out = self.temporal_encoder.out_dim if self.temporal_encoder is not None else None
        self.fusion = GatedFusionClassifier(
            static_in_dim=self.static_encoder.out_dim,
            temporal_in_dim=temporal_out,
            n_classes=n_classes,
            args=args,
        )

    def forward(self, static_x: torch.Tensor, temporal_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        static_repr = self.static_encoder(static_x)
        temporal_repr = self.temporal_encoder(temporal_x) if self.temporal_encoder is not None and temporal_x is not None else None
        return self.fusion(static_repr, temporal_repr)


def build_model(static_dim: int, temporal_in_dim: Optional[int], n_classes: int, args) -> nn.Module:
    if args.model_family == "hybrid_tcn_lstm":
        return HybridModel(static_dim=static_dim, temporal_in_dim=temporal_in_dim, n_classes=n_classes, args=args)
    if args.model_family == "ft_transformer_gru_attention":
        return FTTransformerGRUAttentionModel(
            static_dim=static_dim,
            temporal_in_dim=temporal_in_dim,
            n_classes=n_classes,
            args=args,
        )
    if args.model_family == "deepfm_temporal_tcn":
        return DeepFMTemporalTCNModel(
            static_dim=static_dim,
            temporal_in_dim=temporal_in_dim,
            n_classes=n_classes,
            args=args,
        )
    if args.model_family == "xgboost_temporal_gru":
        return XGBoostTemporalGRUModel(
            static_dim=static_dim,
            temporal_in_dim=temporal_in_dim,
            n_classes=n_classes,
            args=args,
        )
    raise ValueError(f"Unsupported model_family: {args.model_family}")


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )
    if args.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.scheduler_eta_min,
        )
    return None


def summarize_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def select_metric_value(metrics: Dict[str, float], selection_metric: str) -> float:
    if selection_metric == "macro_f1":
        return metrics["macro_f1"]
    if selection_metric == "balanced_accuracy":
        return metrics["balanced_accuracy"]
    if selection_metric == "composite":
        return float(
            (
                metrics["accuracy"]
                + metrics["macro_f1"]
                + metrics["balanced_accuracy"]
            )
            / 3.0
        )
    return metrics["accuracy"]


def build_loss_function(
    labels: np.ndarray,
    n_classes: int,
    device: torch.device,
    args,
) -> nn.Module:
    weight_tensor = None
    if args.use_class_weights:
        counts = np.bincount(labels.astype(int), minlength=n_classes).astype(np.float32)
        counts = np.clip(counts, 1.0, None)
        weights = (counts.sum() / counts) ** float(args.class_weight_power)
        weights = weights / weights.mean()
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=float(args.label_smoothing),
    )


def ensure_dense_features(values) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def fit_xgboost_probability_booster(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    x_test,
    args,
    n_classes: int,
):
    if xgb is None:
        return None

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "eta": min(max(float(args.learning_rate) * 100.0, 0.03), 0.1),
        "max_depth": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "lambda": 1.0,
        "seed": int(args.seed),
        "tree_method": "hist",
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=max(50, int(args.xgb_num_boost_round)),
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )

    val_probs = booster.predict(dval).astype(np.float32)
    test_probs = None
    if x_test is not None:
        dtest = xgb.DMatrix(x_test)
        test_probs = booster.predict(dtest).astype(np.float32)

    train_probs = booster.predict(dtrain).astype(np.float32)
    return {
        "booster": booster,
        "train_probs": train_probs,
        "val_probs": val_probs,
        "test_probs": test_probs,
    }


def append_probability_features(
    prepared: PreparedData,
    train_probs: np.ndarray,
    val_probs: np.ndarray,
    test_probs: Optional[np.ndarray],
) -> PreparedData:
    x_static_train = np.concatenate(
        [ensure_dense_features(prepared.x_static_train), train_probs.astype(np.float32)], axis=1
    )
    x_static_val = np.concatenate(
        [ensure_dense_features(prepared.x_static_val), val_probs.astype(np.float32)], axis=1
    )
    x_static_test = None
    if prepared.x_static_test is not None and test_probs is not None:
        x_static_test = np.concatenate(
            [ensure_dense_features(prepared.x_static_test), test_probs.astype(np.float32)], axis=1
        )

    augmented_static_cols = list(prepared.static_cols) + [
        f"xgb_prob_{idx}" for idx in range(train_probs.shape[1])
    ]
    return PreparedData(
        static_cols=augmented_static_cols,
        num_cols=list(prepared.num_cols) + [f"xgb_prob_{idx}" for idx in range(train_probs.shape[1])],
        cat_cols=list(prepared.cat_cols),
        static_preprocessor=prepared.static_preprocessor,
        label_encoder=prepared.label_encoder,
        temporal_feature_names=prepared.temporal_feature_names,
        x_static_train=x_static_train,
        x_static_val=x_static_val,
        x_static_test=x_static_test,
        y_train=prepared.y_train,
        y_val=prepared.y_val,
        y_test=prepared.y_test,
        x_seq_train=prepared.x_seq_train,
        x_seq_val=prepared.x_seq_val,
        x_seq_test=prepared.x_seq_test,
    )


def replace_static_with_probability_features(
    prepared: PreparedData,
    train_probs: np.ndarray,
    val_probs: np.ndarray,
    test_probs: Optional[np.ndarray],
) -> PreparedData:
    probability_cols = [f"xgb_prob_{idx}" for idx in range(train_probs.shape[1])]
    return PreparedData(
        static_cols=probability_cols,
        num_cols=probability_cols,
        cat_cols=[],
        static_preprocessor=prepared.static_preprocessor,
        label_encoder=prepared.label_encoder,
        temporal_feature_names=prepared.temporal_feature_names,
        x_static_train=train_probs.astype(np.float32),
        x_static_val=val_probs.astype(np.float32),
        x_static_test=test_probs.astype(np.float32) if test_probs is not None else None,
        y_train=prepared.y_train,
        y_val=prepared.y_val,
        y_test=prepared.y_test,
        x_seq_train=prepared.x_seq_train,
        x_seq_val=prepared.x_seq_val,
        x_seq_test=prepared.x_seq_test,
    )


def choose_ensemble_weight(
    neural_probs: np.ndarray,
    xgb_probs: np.ndarray,
    y_true: np.ndarray,
    args,
) -> Tuple[float, Dict[str, float]]:
    step = max(0.01, float(args.ensemble_weight_step))
    candidate_weights = np.arange(0.0, 1.0 + (step / 2.0), step)
    best_weight = 1.0
    best_metrics: Optional[Dict[str, float]] = None
    best_key = None

    for weight in candidate_weights:
        blended_probs = (weight * neural_probs) + ((1.0 - weight) * xgb_probs)
        blended_preds = blended_probs.argmax(axis=1)
        metrics = summarize_predictions(y_true, blended_preds)
        key = (
            select_metric_value(metrics, args.selection_metric),
            metrics["accuracy"],
            metrics["macro_f1"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_weight = float(weight)
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Failed to choose an ensemble weight.")
    return best_weight, best_metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 1.0,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    preds: List[int] = []
    gts: List[int] = []
    probs: List[np.ndarray] = []

    for static_x, temporal_x, labels in loader:
        static_x = static_x.to(device)
        labels = labels.to(device)
        if temporal_x is not None:
            temporal_x = temporal_x.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(static_x, temporal_x)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        probs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
        preds.extend(logits.argmax(dim=-1).detach().cpu().numpy().tolist())
        gts.extend(labels.detach().cpu().numpy().tolist())

    metric_summary = summarize_predictions(gts, preds)
    metric_summary.update(
        {
            "loss": total_loss / max(total_samples, 1),
            "probs": np.concatenate(probs, axis=0) if probs else np.empty((0, 0), dtype=np.float32),
            "preds": np.asarray(preds),
            "gts": np.asarray(gts),
        }
    )
    return metric_summary


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    args,
    verbose: bool = True,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_selection_score = float("-inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(model, val_loader, device, criterion)

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(select_metric_value(val_metrics, args.selection_metric))
            else:
                scheduler.step()

        selection_score = select_metric_value(val_metrics, args.selection_metric)
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_macro_precision": float(train_metrics["macro_precision"]),
                "train_macro_recall": float(train_metrics["macro_recall"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "train_weighted_f1": float(train_metrics["weighted_f1"]),
                "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_precision": float(val_metrics["macro_precision"]),
                "val_macro_recall": float(val_metrics["macro_recall"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_weighted_f1": float(val_metrics["weighted_f1"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            }
        )
        if verbose:
            print(
                f"Epoch {epoch:02d} | "
                f"train_acc={train_metrics['accuracy']:.3f} "
                f"train_f1={train_metrics['macro_f1']:.3f} | "
                f"val_acc={val_metrics['accuracy']:.3f} "
                f"val_f1={val_metrics['macro_f1']:.3f}"
            )

        if patience_counter >= args.early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return model, history


def execute_training_run(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
    schema: SchemaInfo,
    complete_temporal_groups: Dict[str, Dict[str, str]],
    engineered_cols: Sequence[str],
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, object]:
    prepared = prepare_model_inputs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        schema=schema,
        complete_temporal_groups=complete_temporal_groups,
        engineered_cols=engineered_cols,
    )
    raw_prepared = prepared
    xgb_artifacts = None
    if (
        args.augment_with_xgb_probs
        or args.final_ensemble_with_xgb
        or args.run_xgboost
        or args.model_family == "xgboost_temporal_gru"
    ):
        xgb_artifacts = fit_xgboost_probability_booster(
            x_train=raw_prepared.x_static_train,
            y_train=raw_prepared.y_train,
            x_val=raw_prepared.x_static_val,
            y_val=raw_prepared.y_val,
            x_test=raw_prepared.x_static_test,
            args=args,
            n_classes=len(raw_prepared.label_encoder.classes_),
        )

    if xgb_artifacts is not None and args.model_family == "xgboost_temporal_gru":
        prepared = replace_static_with_probability_features(
            prepared,
            xgb_artifacts["train_probs"],
            xgb_artifacts["val_probs"],
            xgb_artifacts["test_probs"],
        )
    elif xgb_artifacts is not None and args.augment_with_xgb_probs:
        prepared = append_probability_features(
            prepared,
            xgb_artifacts["train_probs"],
            xgb_artifacts["val_probs"],
            xgb_artifacts["test_probs"],
        )
    train_loader, val_loader, test_loader = build_dataloaders(prepared, args.batch_size)
    static_dim = train_loader.dataset[0][0].shape[0]
    temporal_in_dim = (
        prepared.x_seq_train.shape[-1] if prepared.x_seq_train is not None else None
    )
    model = build_model(
        static_dim=static_dim,
        temporal_in_dim=temporal_in_dim,
        n_classes=len(prepared.label_encoder.classes_),
        args=args,
    ).to(device)
    criterion = build_loss_function(
        labels=prepared.y_train,
        n_classes=len(prepared.label_encoder.classes_),
        device=device,
        args=args,
    )
    model, history = fit_model(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
        args,
        verbose=verbose,
    )
    val_metrics = run_epoch(model, val_loader, device, criterion)
    test_metrics = run_epoch(model, test_loader, device, criterion) if test_loader else None
    neural_val_metrics = copy.deepcopy(val_metrics)
    neural_test_metrics = copy.deepcopy(test_metrics) if test_metrics is not None else None

    ensemble_summary = None
    if args.final_ensemble_with_xgb and xgb_artifacts is not None:
        best_weight, best_val_metrics = choose_ensemble_weight(
            neural_probs=neural_val_metrics["probs"],
            xgb_probs=xgb_artifacts["val_probs"],
            y_true=neural_val_metrics["gts"],
            args=args,
        )
        ensemble_summary = {
            "neural_weight": float(best_weight),
            "xgb_weight": float(1.0 - best_weight),
            "val_metrics": best_val_metrics,
        }
        val_blended_probs = (best_weight * neural_val_metrics["probs"]) + (
            (1.0 - best_weight) * xgb_artifacts["val_probs"]
        )
        val_metrics = dict(val_metrics)
        val_metrics.update(
            summarize_predictions(neural_val_metrics["gts"], val_blended_probs.argmax(axis=1))
        )
        val_metrics["preds"] = val_blended_probs.argmax(axis=1)
        val_metrics["probs"] = val_blended_probs

        if test_metrics is not None and xgb_artifacts["test_probs"] is not None:
            test_blended_probs = (best_weight * neural_test_metrics["probs"]) + (
                (1.0 - best_weight) * xgb_artifacts["test_probs"]
            )
            test_metrics = dict(test_metrics)
            test_metrics.update(
                summarize_predictions(neural_test_metrics["gts"], test_blended_probs.argmax(axis=1))
            )
            test_metrics["preds"] = test_blended_probs.argmax(axis=1)
            test_metrics["probs"] = test_blended_probs

    return {
        "model": model,
        "history": history,
        "raw_prepared": raw_prepared,
        "prepared": prepared,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "neural_val_metrics": neural_val_metrics,
        "neural_test_metrics": neural_test_metrics,
        "xgb_artifacts": xgb_artifacts,
        "ensemble_summary": ensemble_summary,
    }


def get_random_search_space(args: argparse.Namespace, use_temporal: bool) -> Dict[str, List[object]]:
    search_space: Dict[str, List[object]] = {
        "learning_rate": [1e-4, 5e-4, args.learning_rate, 5e-3],
        "batch_size": [64, 128, args.batch_size, 256],
        "static_hid_dim": [64, 128, args.static_hid_dim, 256],
        "static_out_dim": [32, 64, args.static_out_dim, 128],
        "static_dropout": [0.0, 0.1, args.static_dropout, 0.3],
        "static_n_layers": [1, args.static_n_layers, 2],
        "static_use_batchnorm": [False, True],
        "classifier_hid_dim": [32, 64, args.classifier_hid_dim, 128],
        "classifier_n_layers": [1, args.classifier_n_layers, 2],
        "classifier_use_batchnorm": [False, True],
        "epochs": [max(10, args.epochs // 2), args.epochs, max(args.epochs, 30)],
        "optimizer": ["adamw", "adam", "sgd", "rmsprop"],
        "weight_decay": [0.0, args.weight_decay, 1e-3],
        "momentum": [0.9, args.momentum, 0.95],
        "scheduler": ["none", "plateau", "cosine"],
        "scheduler_factor": [0.1, args.scheduler_factor, 0.5],
        "scheduler_patience": [3, args.scheduler_patience, 10],
        "scheduler_eta_min": [1e-7, args.scheduler_eta_min, 1e-6],
        "early_stopping_patience": [3, args.early_stopping_patience, 10],
    }
    if args.model_family == "ft_transformer_gru_attention":
        search_space.update(
            {
                "transformer_layers": [1, args.transformer_layers, 3],
                "transformer_heads": [2, args.transformer_heads, 4],
                "transformer_dropout": [0.0, args.transformer_dropout, 0.2],
            }
        )
    if args.model_family == "deepfm_temporal_tcn":
        search_space.update({"fm_embed_dim": [16, args.fm_embed_dim, 32, 64]})
    if args.model_family == "xgboost_temporal_gru":
        search_space.update(
            {
                "xgb_num_boost_round": [50, args.xgb_num_boost_round, 300],
                "static_out_dim": [32, args.static_out_dim, 64],
                "static_dropout": [0.0, args.static_dropout, 0.2],
            }
        )
    if use_temporal:
        search_space.update(
            {
                "temporal_hid_dim": [32, 64, args.temporal_hid_dim, 128],
                "temporal_kernel_size": [2, 3, args.temporal_kernel_size, 4],
                "temporal_n_blocks": [1, args.temporal_n_blocks, 3],
                "temporal_dropout": [0.0, 0.1, args.temporal_dropout, 0.3],
                "temporal_recurrent_type": ["LSTM", "GRU"],
                "temporal_use_batchnorm": [False, True],
                "temporal_use_layernorm": [False, True],
            }
        )
    deduped_space: Dict[str, List[object]] = {}
    for key, values in search_space.items():
        unique_values: List[object] = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        deduped_space[key] = unique_values
    return deduped_space


def sample_random_configurations(
    search_space: Dict[str, List[object]], n_iter: int, seed: int
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    samples: List[Dict[str, object]] = []
    keys = list(search_space.keys())
    for _ in range(n_iter):
        sample = {key: rng.choice(search_space[key]) for key in keys}
        samples.append(sample)
    return samples


def save_history_plot(history: List[Dict[str, float]], output_dir: Path) -> None:
    history_df = pd.DataFrame(history)
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_accuracy"], label="train_accuracy")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=200)
    plt.close()


def save_confusion_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    names = [class_names[idx] for idx in labels]
    plt.figure(figsize=(12, 10))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=names,
            yticklabels=names,
        )
    else:
        plt.imshow(cm, cmap="Blues")
        plt.xticks(ticks=np.arange(len(names)), labels=names, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(names)), labels=names)
        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                plt.text(col_idx, row_idx, str(cm[row_idx, col_idx]), ha="center", va="center")
        plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def write_json(output_path: Path, payload: Dict[str, object]) -> None:
    serializable = json.loads(json.dumps(payload, default=_json_default))
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def build_xgboost_probability_features(prepared: PreparedData, args) -> PreparedData:
    if xgb is None:
        raise ImportError(
            "xgboost is required for model_family='xgboost_temporal_gru' but is not installed."
        )

    dtrain = xgb.DMatrix(prepared.x_static_train, label=prepared.y_train)
    dval = xgb.DMatrix(prepared.x_static_val, label=prepared.y_val)
    params = {
        "objective": "multi:softprob",
        "num_class": len(prepared.label_encoder.classes_),
        "eval_metric": "mlogloss",
        "eta": min(float(args.learning_rate), 0.3),
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": int(args.seed),
        "tree_method": "hist",
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=max(10, int(args.xgb_num_boost_round)),
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )

    x_static_train = booster.predict(dtrain).astype(np.float32)
    x_static_val = booster.predict(dval).astype(np.float32)
    x_static_test = None
    if prepared.x_static_test is not None:
        dtest = xgb.DMatrix(prepared.x_static_test)
        x_static_test = booster.predict(dtest).astype(np.float32)

    return PreparedData(
        static_cols=["xgb_probabilities"],
        num_cols=["xgb_probabilities"],
        cat_cols=[],
        static_preprocessor=prepared.static_preprocessor,
        label_encoder=prepared.label_encoder,
        temporal_feature_names=prepared.temporal_feature_names,
        x_static_train=x_static_train,
        x_static_val=x_static_val,
        x_static_test=x_static_test,
        y_train=prepared.y_train,
        y_val=prepared.y_val,
        y_test=prepared.y_test,
        x_seq_train=prepared.x_seq_train,
        x_seq_val=prepared.x_seq_val,
        x_seq_test=prepared.x_seq_test,
    )


def _table_cell(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.floating, float)):
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))
        return f"{float_value:.4f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows to display._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_table_cell(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def save_table_outputs(
    df: pd.DataFrame,
    csv_path: Path,
    markdown_path: Path,
    title: str,
    intro_lines: Optional[Sequence[str]] = None,
) -> None:
    df.to_csv(csv_path, index=False)
    markdown_lines = [f"# {title}", ""]
    if intro_lines:
        markdown_lines.extend(intro_lines)
        markdown_lines.append("")
    markdown_lines.append(dataframe_to_markdown_table(df))
    markdown_lines.append("")
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")


def write_experiment_summary_markdown(
    output_dir: Path,
    final_metrics: Dict[str, float],
    random_search_summary: Optional[Dict[str, object]],
    cross_validation_summary: Optional[Dict[str, object]],
) -> None:
    lines: List[str] = ["# Experiment Summary", ""]

    final_df = pd.DataFrame(
        [
            {
                "test_loss": final_metrics["test_loss"],
                "test_accuracy": final_metrics["test_accuracy"],
                "test_macro_precision": final_metrics["test_macro_precision"],
                "test_macro_recall": final_metrics["test_macro_recall"],
                "test_macro_f1": final_metrics["test_macro_f1"],
                "test_weighted_precision": final_metrics["test_weighted_precision"],
                "test_weighted_recall": final_metrics["test_weighted_recall"],
                "test_weighted_f1": final_metrics["test_weighted_f1"],
                "test_balanced_accuracy": final_metrics["test_balanced_accuracy"],
            }
        ]
    )
    lines.extend(["## Final Evaluation", "", dataframe_to_markdown_table(final_df), ""])

    if random_search_summary is not None:
        best_result = random_search_summary.get("best_result") or {}
        best_df = pd.DataFrame(
            [
                {
                    "selected_iteration": best_result.get("iteration"),
                    "val_accuracy": best_result.get("val_accuracy"),
                    "val_macro_f1": best_result.get("val_macro_f1"),
                    "epochs_ran": best_result.get("epochs_ran"),
                }
            ]
        )
        lines.extend(
            [
                "## Random Search",
                "",
                f"Full trial table: [random_search_trials.csv](./random_search_trials.csv) and [random_search_trials.md](./random_search_trials.md)",
                "",
                dataframe_to_markdown_table(best_df),
                "",
            ]
        )

    if cross_validation_summary is not None:
        cv_summary_df = pd.DataFrame(
            [
                {
                    "splitter": cross_validation_summary["splitter"],
                    "folds": cross_validation_summary["folds"],
                    "mean_val_accuracy": cross_validation_summary["mean_val_accuracy"],
                    "std_val_accuracy": cross_validation_summary["std_val_accuracy"],
                    "mean_val_macro_f1": cross_validation_summary["mean_val_macro_f1"],
                    "std_val_macro_f1": cross_validation_summary["std_val_macro_f1"],
                }
            ]
        )
        lines.extend(
            [
                "## Cross-Validation",
                "",
                f"Full fold table: [cross_validation_folds.csv](./cross_validation_folds.csv) and [cross_validation_folds.md](./cross_validation_folds.md)",
                "",
                dataframe_to_markdown_table(cv_summary_df),
                "",
            ]
        )

    (output_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_random_search(
    base_args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    schema: SchemaInfo,
    complete_temporal_groups: Dict[str, Dict[str, str]],
    engineered_cols: Sequence[str],
    device: torch.device,
    output_dir: Path,
) -> Tuple[argparse.Namespace, Dict[str, object]]:
    prepared = prepare_model_inputs(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        schema=schema,
        complete_temporal_groups=complete_temporal_groups,
        engineered_cols=engineered_cols,
    )
    use_temporal = prepared.x_seq_train is not None
    search_space = get_random_search_space(base_args, use_temporal)
    samples = sample_random_configurations(
        search_space=search_space,
        n_iter=base_args.random_search_iterations,
        seed=base_args.seed,
    )

    results: List[Dict[str, object]] = []
    best_result: Optional[Dict[str, object]] = None
    best_args = base_args

    for idx, params in enumerate(samples, start=1):
        trial_args = clone_args_with_overrides(base_args, params)
        print(
            f"Random search {idx}/{len(samples)} | "
            f"lr={trial_args.learning_rate} batch={trial_args.batch_size} "
            f"epochs={trial_args.epochs} optimizer={trial_args.optimizer}"
        )
        trial_run = execute_training_run(
            train_df=train_df,
            val_df=val_df,
            test_df=None,
            args=trial_args,
            schema=schema,
            complete_temporal_groups=complete_temporal_groups,
            engineered_cols=engineered_cols,
            device=device,
            verbose=False,
        )
        val_metrics = trial_run["val_metrics"]
        result = {
            "iteration": idx,
            "params": params,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_precision": float(val_metrics["macro_precision"]),
            "val_macro_recall": float(val_metrics["macro_recall"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "epochs_ran": len(trial_run["history"]),
        }
        results.append(result)

        if best_result is None:
            best_result = result
            best_args = trial_args
            continue

        current_key = (
            select_metric_value(
                {
                    "accuracy": result["val_accuracy"],
                    "macro_f1": result["val_macro_f1"],
                    "balanced_accuracy": result["val_balanced_accuracy"],
                },
                base_args.selection_metric,
            ),
            result["val_accuracy"],
            result["val_macro_f1"],
        )
        best_key = (
            select_metric_value(
                {
                    "accuracy": best_result["val_accuracy"],
                    "macro_f1": best_result["val_macro_f1"],
                    "balanced_accuracy": best_result["val_balanced_accuracy"],
                },
                base_args.selection_metric,
            ),
            best_result["val_accuracy"],
            best_result["val_macro_f1"],
        )
        if current_key > best_key:
            best_result = result
            best_args = trial_args

    summary = {
        "iterations": len(samples),
        "best_params": vars(best_args),
        "best_result": best_result,
        "results": results,
    }
    write_json(output_dir / "random_search_results.json", summary)
    random_search_rows: List[Dict[str, object]] = []
    for result in results:
        row = {
            "iteration": result["iteration"],
            "val_accuracy": result["val_accuracy"],
            "val_macro_precision": result["val_macro_precision"],
            "val_macro_recall": result["val_macro_recall"],
            "val_macro_f1": result["val_macro_f1"],
            "val_weighted_f1": result["val_weighted_f1"],
            "val_balanced_accuracy": result["val_balanced_accuracy"],
            "epochs_ran": result["epochs_ran"],
        }
        row.update(result["params"])
        random_search_rows.append(row)
    random_search_df = pd.DataFrame(random_search_rows)
    if not random_search_df.empty:
        random_search_df = random_search_df.sort_values(
            by=["val_accuracy", "val_macro_f1", "iteration"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        random_search_df.insert(0, "rank", np.arange(1, len(random_search_df) + 1))
    save_table_outputs(
        random_search_df,
        output_dir / "random_search_trials.csv",
        output_dir / "random_search_trials.md",
        title="Random Search Trials",
        intro_lines=[
            f"Trials run: {len(samples)}",
            (
                "Rows are sorted by validation accuracy, then validation macro-F1. "
                "The top row is the selected configuration."
            ),
        ],
    )
    return best_args, summary


def resolve_cv_splitter(
    train_val_df: pd.DataFrame,
    target_col: str,
    group_id: Optional[str],
    requested_folds: int,
    seed: int,
):
    group_count = train_val_df[group_id].nunique(dropna=False) if group_id is not None else 0
    if group_id is not None and 2 <= group_count < len(train_val_df):
        effective_folds = min(requested_folds, group_count)
        if effective_folds >= 2:
            return GroupKFold(n_splits=effective_folds), effective_folds, "group"

    min_class_count = int(train_val_df[target_col].value_counts().min())
    effective_folds = min(requested_folds, min_class_count)
    if effective_folds >= 2:
        return (
            StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed),
            effective_folds,
            "stratified",
        )

    effective_folds = min(requested_folds, len(train_val_df))
    if effective_folds >= 2:
        return KFold(n_splits=effective_folds, shuffle=True, random_state=seed), effective_folds, "kfold"
    return None, 0, "unavailable"


def run_cross_validation(
    train_val_df: pd.DataFrame,
    args: argparse.Namespace,
    schema: SchemaInfo,
    complete_temporal_groups: Dict[str, Dict[str, str]],
    engineered_cols: Sequence[str],
    device: torch.device,
    output_dir: Path,
) -> Optional[Dict[str, object]]:
    splitter, effective_folds, splitter_name = resolve_cv_splitter(
        train_val_df=train_val_df,
        target_col=schema.target_col,
        group_id=schema.group_id,
        requested_folds=args.cv_folds,
        seed=args.seed,
    )
    if splitter is None:
        print("Cross-validation skipped because there are not enough samples for two folds.")
        return None

    print(f"Running {effective_folds}-fold cross-validation using {splitter_name}.")
    fold_results: List[Dict[str, object]] = []
    groups = train_val_df[schema.group_id] if splitter_name == "group" and schema.group_id else None

    if splitter_name == "group":
        split_iter = splitter.split(train_val_df, groups=groups)
    elif splitter_name == "stratified":
        split_iter = splitter.split(train_val_df, train_val_df[schema.target_col])
    else:
        split_iter = splitter.split(train_val_df)

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        fold_train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        print(f"CV fold {fold_idx}/{effective_folds}")
        fold_run = execute_training_run(
            train_df=fold_train_df,
            val_df=fold_val_df,
            test_df=None,
            args=args,
            schema=schema,
            complete_temporal_groups=complete_temporal_groups,
            engineered_cols=engineered_cols,
            device=device,
            verbose=False,
        )
        val_metrics = fold_run["val_metrics"]
        fold_results.append(
            {
                "fold": fold_idx,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_precision": float(val_metrics["macro_precision"]),
                "val_macro_recall": float(val_metrics["macro_recall"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_weighted_f1": float(val_metrics["weighted_f1"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "epochs_ran": len(fold_run["history"]),
            }
        )

    summary = {
        "splitter": splitter_name,
        "folds": effective_folds,
        "fold_results": fold_results,
        "mean_val_accuracy": float(np.mean([r["val_accuracy"] for r in fold_results])),
        "std_val_accuracy": float(np.std([r["val_accuracy"] for r in fold_results])),
        "mean_val_macro_precision": float(
            np.mean([r["val_macro_precision"] for r in fold_results])
        ),
        "std_val_macro_precision": float(
            np.std([r["val_macro_precision"] for r in fold_results])
        ),
        "mean_val_macro_recall": float(np.mean([r["val_macro_recall"] for r in fold_results])),
        "std_val_macro_recall": float(np.std([r["val_macro_recall"] for r in fold_results])),
        "mean_val_macro_f1": float(np.mean([r["val_macro_f1"] for r in fold_results])),
        "std_val_macro_f1": float(np.std([r["val_macro_f1"] for r in fold_results])),
        "mean_val_weighted_f1": float(np.mean([r["val_weighted_f1"] for r in fold_results])),
        "std_val_weighted_f1": float(np.std([r["val_weighted_f1"] for r in fold_results])),
        "mean_val_balanced_accuracy": float(
            np.mean([r["val_balanced_accuracy"] for r in fold_results])
        ),
        "std_val_balanced_accuracy": float(
            np.std([r["val_balanced_accuracy"] for r in fold_results])
        ),
    }
    write_json(output_dir / "cross_validation_results.json", summary)
    cv_df = pd.DataFrame(fold_results)
    save_table_outputs(
        cv_df,
        output_dir / "cross_validation_folds.csv",
        output_dir / "cross_validation_folds.md",
        title="Cross-Validation Folds",
        intro_lines=[
            f"Splitter: {splitter_name}",
            f"Folds: {effective_folds}",
            f"Mean validation accuracy: {summary['mean_val_accuracy']:.4f}",
            f"Std validation accuracy: {summary['std_val_accuracy']:.4f}",
            f"Mean validation macro-precision: {summary['mean_val_macro_precision']:.4f}",
            f"Mean validation macro-recall: {summary['mean_val_macro_recall']:.4f}",
            f"Mean validation macro-F1: {summary['mean_val_macro_f1']:.4f}",
            f"Std validation macro-F1: {summary['std_val_macro_f1']:.4f}",
            f"Mean validation weighted F1: {summary['mean_val_weighted_f1']:.4f}",
            f"Mean validation balanced accuracy: {summary['mean_val_balanced_accuracy']:.4f}",
        ],
    )
    return summary


def run_xgboost_baseline(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    x_test,
    y_test: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
) -> Optional[Dict[str, object]]:
    if xgb is None:
        print("xgboost is not installed. Skipping XGBoost baseline.")
        return None

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    dtest = xgb.DMatrix(x_test)

    params = {
        "objective": "multi:softprob",
        "num_class": len(class_names),
        "eval_metric": "mlogloss",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "tree_method": "hist",
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=False,
    )
    probabilities = model.predict(dtest)
    preds = probabilities.argmax(axis=1)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_macro_precision": float(
            precision_score(y_test, preds, average="macro", zero_division=0)
        ),
        "test_macro_recall": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, preds, average="macro")),
        "test_weighted_f1": float(f1_score(y_test, preds, average="weighted")),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
    }
    save_confusion_plot(
        y_test,
        preds,
        class_names,
        output_dir / "xgboost_confusion_matrix.png",
        "XGBoost Confusion Matrix",
    )
    return metrics


def evaluate_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    sensitive_cols: Sequence[str],
    class_names: Sequence[str],
) -> Optional[Dict[str, object]]:
    if MetricFrame is None:
        print("fairlearn is not installed. Skipping fairness metrics.")
        return None
    if not sensitive_cols:
        print("No sensitive columns detected. Skipping fairness metrics.")
        return None

    fairness_results: Dict[str, object] = {}
    for sensitive_col in sensitive_cols:
        group_values = test_df[sensitive_col].astype(str)
        result: Dict[str, object] = {
            "group_accuracy": {},
            "group_selection_rate": {},
            "demographic_parity_difference": None,
            "equalized_odds_difference_per_class": {},
        }
        try:
            acc_frame = MetricFrame(
                metrics=accuracy_score,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=group_values,
            )
            result["group_accuracy"] = acc_frame.by_group.to_dict()
        except Exception as exc:
            result["group_accuracy_error"] = str(exc)

        try:
            sr_frame = MetricFrame(
                metrics=selection_rate,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=group_values,
            )
            result["group_selection_rate"] = sr_frame.by_group.to_dict()
        except Exception as exc:
            result["group_selection_rate_error"] = str(exc)

        try:
            result["demographic_parity_difference"] = float(
                demographic_parity_difference(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=group_values,
                )
            )
        except Exception as exc:
            result["demographic_parity_difference_error"] = str(exc)

        for class_idx, class_name in enumerate(class_names):
            try:
                eod = equalized_odds_difference(
                    y_true=(y_true == class_idx).astype(int),
                    y_pred=(y_pred == class_idx).astype(int),
                    sensitive_features=group_values,
                )
                result["equalized_odds_difference_per_class"][class_name] = float(eod)
            except Exception as exc:
                result["equalized_odds_difference_per_class"][class_name] = str(exc)

        fairness_results[sensitive_col] = result
    return fairness_results


@dataclass
class RunConfigSnapshot:
    csv_path: str
    output_dir: str
    model_family: str
    compare_model_families: bool
    seed: int
    test_size: float
    val_size: float
    min_class_samples: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    momentum: float
    scheduler: str
    scheduler_factor: float
    scheduler_patience: int
    scheduler_eta_min: float
    early_stopping_patience: int
    grad_clip: float
    static_hid_dim: int
    static_out_dim: int
    static_dropout: float
    static_n_layers: int
    static_use_batchnorm: bool
    temporal_hid_dim: int
    temporal_kernel_size: int
    temporal_n_blocks: int
    temporal_dropout: float
    temporal_recurrent_type: str
    temporal_use_batchnorm: bool
    temporal_use_layernorm: bool
    classifier_hid_dim: int
    classifier_n_layers: int
    classifier_use_batchnorm: bool
    transformer_layers: int
    transformer_heads: int
    transformer_dropout: float
    fm_embed_dim: int
    xgb_num_boost_round: int
    augment_with_xgb_probs: bool
    final_ensemble_with_xgb: bool
    ensemble_weight_step: float
    selection_metric: str
    use_class_weights: bool
    class_weight_power: float
    label_smoothing: float
    run_xgboost: bool
    run_fairness: bool
    run_random_search: bool
    random_search_iterations: int
    run_cross_validation: bool
    cv_folds: int
    final_retrain_val_size: float
    no_plots: bool


def argparse_namespace_to_dataclass(args, csv_path: str, output_dir: str) -> RunConfigSnapshot:
    return RunConfigSnapshot(
        csv_path=csv_path,
        output_dir=output_dir,
        model_family=args.model_family,
        compare_model_families=args.compare_model_families,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        min_class_samples=args.min_class_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        momentum=args.momentum,
        scheduler=args.scheduler,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_eta_min=args.scheduler_eta_min,
        early_stopping_patience=args.early_stopping_patience,
        grad_clip=args.grad_clip,
        static_hid_dim=args.static_hid_dim,
        static_out_dim=args.static_out_dim,
        static_dropout=args.static_dropout,
        static_n_layers=args.static_n_layers,
        static_use_batchnorm=args.static_use_batchnorm,
        temporal_hid_dim=args.temporal_hid_dim,
        temporal_kernel_size=args.temporal_kernel_size,
        temporal_n_blocks=args.temporal_n_blocks,
        temporal_dropout=args.temporal_dropout,
        temporal_recurrent_type=args.temporal_recurrent_type,
        temporal_use_batchnorm=args.temporal_use_batchnorm,
        temporal_use_layernorm=args.temporal_use_layernorm,
        classifier_hid_dim=args.classifier_hid_dim,
        classifier_n_layers=args.classifier_n_layers,
        classifier_use_batchnorm=args.classifier_use_batchnorm,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.transformer_dropout,
        fm_embed_dim=args.fm_embed_dim,
        xgb_num_boost_round=args.xgb_num_boost_round,
        augment_with_xgb_probs=args.augment_with_xgb_probs,
        final_ensemble_with_xgb=args.final_ensemble_with_xgb,
        ensemble_weight_step=args.ensemble_weight_step,
        selection_metric=args.selection_metric,
        use_class_weights=args.use_class_weights,
        class_weight_power=args.class_weight_power,
        label_smoothing=args.label_smoothing,
        run_xgboost=args.run_xgboost,
        run_fairness=args.run_fairness,
        run_random_search=args.run_random_search,
        random_search_iterations=args.random_search_iterations,
        run_cross_validation=args.run_cross_validation,
        cv_folds=args.cv_folds,
        final_retrain_val_size=args.final_retrain_val_size,
        no_plots=args.no_plots,
    )


def build_model_family_comparison_row(
    model_family: str,
    status: str,
    output_dir: Optional[Path] = None,
    note: str = "",
    summary: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "model_family": model_family,
        "status": status,
        "note": note,
        "output_dir": str(output_dir) if output_dir is not None else "",
        "test_accuracy": None,
        "test_macro_precision": None,
        "test_macro_recall": None,
        "test_macro_f1": None,
        "test_weighted_f1": None,
        "test_balanced_accuracy": None,
        "cv_mean_val_accuracy": None,
        "cv_mean_val_macro_precision": None,
        "cv_mean_val_macro_recall": None,
        "cv_mean_val_macro_f1": None,
        "cv_mean_val_weighted_f1": None,
        "cv_mean_val_balanced_accuracy": None,
        "random_search_best_val_accuracy": None,
        "random_search_best_val_macro_f1": None,
    }
    if summary is None:
        return row

    hybrid_metrics = summary.get("hybrid_metrics") or {}
    cross_validation = summary.get("cross_validation") or {}
    random_search = summary.get("random_search") or {}
    best_result = random_search.get("best_result") or {}

    row.update(
        {
            "test_accuracy": hybrid_metrics.get("test_accuracy"),
            "test_macro_precision": hybrid_metrics.get("test_macro_precision"),
            "test_macro_recall": hybrid_metrics.get("test_macro_recall"),
            "test_macro_f1": hybrid_metrics.get("test_macro_f1"),
            "test_weighted_f1": hybrid_metrics.get("test_weighted_f1"),
            "test_balanced_accuracy": hybrid_metrics.get("test_balanced_accuracy"),
            "cv_mean_val_accuracy": cross_validation.get("mean_val_accuracy"),
            "cv_mean_val_macro_precision": cross_validation.get("mean_val_macro_precision"),
            "cv_mean_val_macro_recall": cross_validation.get("mean_val_macro_recall"),
            "cv_mean_val_macro_f1": cross_validation.get("mean_val_macro_f1"),
            "cv_mean_val_weighted_f1": cross_validation.get("mean_val_weighted_f1"),
            "cv_mean_val_balanced_accuracy": cross_validation.get("mean_val_balanced_accuracy"),
            "random_search_best_val_accuracy": best_result.get("val_accuracy"),
            "random_search_best_val_macro_f1": best_result.get("val_macro_f1"),
        }
    )
    return row


def write_model_family_comparison(
    output_dir: Path,
    comparison_rows: List[Dict[str, object]],
) -> None:
    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(
            by=["status", "test_accuracy", "test_macro_f1", "model_family"],
            ascending=[True, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
    save_table_outputs(
        comparison_df,
        output_dir / "model_family_comparison.csv",
        output_dir / "model_family_comparison.md",
        title="Model Family Comparison",
        intro_lines=[
            "This table compares all requested hybrid model families run under the same pipeline.",
            "Rows with status `skipped` were not executed successfully, usually because an optional dependency was unavailable.",
        ],
    )
    write_json(
        output_dir / "model_family_comparison.json",
        {"results": comparison_rows},
    )


def run_model_family_pipeline(
    args: argparse.Namespace,
    csv_path: Path,
    output_dir: Path,
    device: torch.device,
    schema: SchemaInfo,
    complete_temporal_groups: Dict[str, Dict[str, str]],
    engineered_cols: Sequence[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    used_group_split: bool,
    removed_classes: Sequence[str],
) -> Dict[str, object]:
    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"=== Running model family: {args.model_family} ===")
    print("Device:", device)
    print("Model family:", args.model_family)
    print("Detected target column:", schema.target_col)
    print("Detected group column:", schema.group_id)
    print("Detected sensitive columns:", schema.sensitive_cols)
    print("Used group-aware split:", used_group_split)
    print("Removed minority classes:", list(removed_classes))
    print("Train/Val/Test shapes:", train_df.shape, val_df.shape, test_df.shape)
    print("Temporal pair bases:", len(complete_temporal_groups))
    print("Engineered temporal features:", len(engineered_cols))

    active_args = args
    random_search_summary = None
    if args.run_random_search:
        active_args, random_search_summary = run_random_search(
            base_args=args,
            train_df=train_df,
            val_df=val_df,
            schema=schema,
            complete_temporal_groups=complete_temporal_groups,
            engineered_cols=engineered_cols,
            device=device,
            output_dir=output_dir,
        )
        print("Random search selected parameters for final training.")

    train_val_pool = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    cross_validation_summary = None
    if args.run_cross_validation:
        cross_validation_summary = run_cross_validation(
            train_val_df=train_val_pool,
            args=active_args,
            schema=schema,
            complete_temporal_groups=complete_temporal_groups,
            engineered_cols=engineered_cols,
            device=device,
            output_dir=output_dir,
        )

    final_train_df, final_val_df, final_used_group_split = split_train_validation(
        train_val_pool,
        target_col=schema.target_col,
        group_id=schema.group_id,
        val_size=active_args.final_retrain_val_size,
        seed=active_args.seed,
    )
    print(
        "Final retrain split (train/internal-val/test):",
        final_train_df.shape,
        final_val_df.shape,
        test_df.shape,
    )
    print("Final retrain used group-aware split:", final_used_group_split)

    final_run = execute_training_run(
        train_df=final_train_df,
        val_df=final_val_df,
        test_df=test_df,
        args=active_args,
        schema=schema,
        complete_temporal_groups=complete_temporal_groups,
        engineered_cols=engineered_cols,
        device=device,
        verbose=True,
    )
    model = final_run["model"]
    history = final_run["history"]
    raw_prepared = final_run["raw_prepared"]
    prepared = final_run["prepared"]
    test_metrics = final_run["test_metrics"]
    neural_test_metrics = final_run["neural_test_metrics"]
    ensemble_summary = final_run["ensemble_summary"]
    xgb_artifacts = final_run["xgb_artifacts"]
    if test_metrics is None:
        raise RuntimeError("Expected test metrics for final run, but none were produced.")

    print(
        "Static features:",
        len(raw_prepared.static_cols),
        "| numeric:",
        len(raw_prepared.num_cols),
        "| categorical:",
        len(raw_prepared.cat_cols),
    )
    print("Temporal branch enabled:", prepared.x_seq_train is not None)

    class_report = classification_report(
        test_metrics["gts"],
        test_metrics["preds"],
        labels=np.arange(len(prepared.label_encoder.classes_)),
        target_names=[str(name) for name in prepared.label_encoder.classes_],
        zero_division=0,
    )

    (output_dir / "classification_report.txt").write_text(class_report, encoding="utf-8")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    torch.save(model.state_dict(), output_dir / "hybrid_best.pt")

    if not args.no_plots:
        save_history_plot(history, output_dir)
        save_confusion_plot(
            test_metrics["gts"],
            test_metrics["preds"],
            [str(name) for name in prepared.label_encoder.classes_],
            output_dir / "hybrid_confusion_matrix.png",
            "Hybrid Model Confusion Matrix",
        )

    xgboost_metrics = None
    if xgb_artifacts is not None and raw_prepared.y_test is not None and xgb_artifacts["test_probs"] is not None:
        xgb_preds = xgb_artifacts["test_probs"].argmax(axis=1)
        xgboost_metrics = summarize_predictions(raw_prepared.y_test, xgb_preds)
    elif args.run_xgboost:
        xgboost_metrics = run_xgboost_baseline(
            raw_prepared.x_static_train,
            raw_prepared.y_train,
            raw_prepared.x_static_val,
            raw_prepared.y_val,
            raw_prepared.x_static_test,
            raw_prepared.y_test,
            [str(name) for name in raw_prepared.label_encoder.classes_],
            output_dir,
        )

    fairness_metrics = None
    if args.run_fairness:
        fairness_metrics = evaluate_fairness(
            y_true=test_metrics["gts"],
            y_pred=test_metrics["preds"],
            test_df=test_df,
            sensitive_cols=schema.sensitive_cols,
            class_names=[str(name) for name in prepared.label_encoder.classes_],
        )

    summary = {
        "config": asdict(
            argparse_namespace_to_dataclass(
                active_args,
                csv_path=str(csv_path),
                output_dir=str(output_dir),
            )
        ),
        "schema": {
            "target_col": schema.target_col,
            "group_id": schema.group_id,
            "sensitive_cols": schema.sensitive_cols,
            "temporal_pair_bases": sorted(complete_temporal_groups.keys()),
            "engineered_temporal_cols": list(engineered_cols),
            "static_cols": raw_prepared.static_cols,
            "temporal_feature_names": prepared.temporal_feature_names,
        },
        "dataset": {
            "train_shape": list(train_df.shape),
            "val_shape": list(val_df.shape),
            "test_shape": list(test_df.shape),
            "used_group_split": used_group_split,
            "removed_minority_classes": list(removed_classes),
            "final_train_shape": list(final_train_df.shape),
            "final_internal_val_shape": list(final_val_df.shape),
            "final_retrain_used_group_split": final_used_group_split,
        },
        "hybrid_metrics": {
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_macro_precision": float(test_metrics["macro_precision"]),
            "test_macro_recall": float(test_metrics["macro_recall"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "test_weighted_precision": float(test_metrics["weighted_precision"]),
            "test_weighted_recall": float(test_metrics["weighted_recall"]),
            "test_weighted_f1": float(test_metrics["weighted_f1"]),
            "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        },
        "neural_only_metrics": None
        if neural_test_metrics is None
        else {
            "test_loss": float(neural_test_metrics["loss"]),
            "test_accuracy": float(neural_test_metrics["accuracy"]),
            "test_macro_precision": float(neural_test_metrics["macro_precision"]),
            "test_macro_recall": float(neural_test_metrics["macro_recall"]),
            "test_macro_f1": float(neural_test_metrics["macro_f1"]),
            "test_weighted_precision": float(neural_test_metrics["weighted_precision"]),
            "test_weighted_recall": float(neural_test_metrics["weighted_recall"]),
            "test_weighted_f1": float(neural_test_metrics["weighted_f1"]),
            "test_balanced_accuracy": float(neural_test_metrics["balanced_accuracy"]),
        },
        "ensemble_summary": ensemble_summary,
        "random_search": random_search_summary,
        "cross_validation": cross_validation_summary,
        "xgboost_metrics": xgboost_metrics,
        "fairness_metrics": fairness_metrics,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    final_metrics_df = pd.DataFrame(
        [
            {
                "test_loss": summary["hybrid_metrics"]["test_loss"],
                "test_accuracy": summary["hybrid_metrics"]["test_accuracy"],
                "test_macro_precision": summary["hybrid_metrics"]["test_macro_precision"],
                "test_macro_recall": summary["hybrid_metrics"]["test_macro_recall"],
                "test_macro_f1": summary["hybrid_metrics"]["test_macro_f1"],
                "test_weighted_precision": summary["hybrid_metrics"]["test_weighted_precision"],
                "test_weighted_recall": summary["hybrid_metrics"]["test_weighted_recall"],
                "test_weighted_f1": summary["hybrid_metrics"]["test_weighted_f1"],
                "test_balanced_accuracy": summary["hybrid_metrics"]["test_balanced_accuracy"],
            }
        ]
    )
    save_table_outputs(
        final_metrics_df,
        output_dir / "final_evaluation_summary.csv",
        output_dir / "final_evaluation_summary.md",
        title="Final Evaluation Summary",
    )
    write_experiment_summary_markdown(
        output_dir=output_dir,
        final_metrics=summary["hybrid_metrics"],
        random_search_summary=random_search_summary,
        cross_validation_summary=cross_validation_summary,
    )

    print()
    print(f"{active_args.model_family} test accuracy:", f"{test_metrics['accuracy']:.4f}")
    print(f"{active_args.model_family} test macro-F1:", f"{test_metrics['macro_f1']:.4f}")
    print("Artifacts saved to:", output_dir.resolve())
    return summary


def main() -> None:
    args = parse_args()
    if (
        not args.compare_model_families
        and args.model_family == "xgboost_temporal_gru"
        and xgb is None
    ):
        raise ImportError(
            "xgboost is required for --model-family xgboost_temporal_gru. "
            "Install xgboost to use this hybrid."
        )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = read_csv_with_fallback_encodings(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]

    schema = detect_schema(df)
    df = df.dropna(subset=[schema.target_col]).copy()
    df[schema.target_col] = df[schema.target_col].astype(str).str.strip()
    df, removed_classes = remove_minority_classes(
        df, schema.target_col, args.min_class_samples
    )
    df, engineered_cols, complete_temporal_groups = create_engineered_temporal_features(
        df, schema.temporal_groups
    )
    df, engineered_static_cols = create_engineered_static_features(df)
    if engineered_static_cols:
        print("Engineered static features added:", len(engineered_static_cols))

    split_bundle = split_dataframe(
        df=df,
        target_col=schema.target_col,
        group_id=schema.group_id,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    train_df, val_df, test_df = (
        split_bundle.train_df,
        split_bundle.val_df,
        split_bundle.test_df,
    )
    if args.compare_model_families:
        comparison_rows: List[Dict[str, object]] = []
        for family_name in MODEL_FAMILIES:
            family_args = clone_args_with_overrides(args, {"model_family": family_name})
            family_output_dir = output_dir / family_name
            if family_name == "xgboost_temporal_gru" and xgb is None:
                note = "Skipped because xgboost is not installed in this environment."
                print()
                print(f"=== Skipping model family: {family_name} ===")
                print(note)
                comparison_rows.append(
                    build_model_family_comparison_row(
                        model_family=family_name,
                        status="skipped",
                        note=note,
                    )
                )
                continue

            family_summary = run_model_family_pipeline(
                args=family_args,
                csv_path=csv_path,
                output_dir=family_output_dir,
                device=device,
                schema=schema,
                complete_temporal_groups=complete_temporal_groups,
                engineered_cols=engineered_cols,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                used_group_split=split_bundle.used_group_split,
                removed_classes=removed_classes,
            )
            comparison_rows.append(
                build_model_family_comparison_row(
                    model_family=family_name,
                    status="completed",
                    output_dir=family_output_dir,
                    summary=family_summary,
                )
            )

        write_model_family_comparison(output_dir, comparison_rows)
        print()
        print("Model family comparison saved to:", output_dir.resolve())
        return

    run_model_family_pipeline(
        args=args,
        csv_path=csv_path,
        output_dir=output_dir,
        device=device,
        schema=schema,
        complete_temporal_groups=complete_temporal_groups,
        engineered_cols=engineered_cols,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        used_group_split=split_bundle.used_group_split,
        removed_classes=removed_classes,
    )


if __name__ == "__main__":
    main()
