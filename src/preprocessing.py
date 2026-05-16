import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from src.config import DATA_RAW
from src.feature_engineering import add_engineered_features


def load_raw_data():
    """
    Load raw features and targets. Checks data/raw/ first (cached); falls back
    to ucimlrepo fetch and caches to disk for offline reruns.
    Returns (X, y) as DataFrames with original column names.
    """
    features_path = DATA_RAW / "features.csv"
    targets_path = DATA_RAW / "targets.csv"

    if features_path.exists() and targets_path.exists():
        X = pd.read_csv(features_path)
        y = pd.read_csv(targets_path)
        return X, y

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Try ucimlrepo first, fall back to bundled raw zip if API is unavailable
    try:
        from ucimlrepo import fetch_ucirepo
        steel = fetch_ucirepo(id=198)
        X = steel.data.features
        y = steel.data.targets
    except Exception:
        # Parse the locally downloaded UCI archive files
        from src.config import PROJECT_ROOT
        raw_zip_dir = PROJECT_ROOT / "data" / "raw_zip"
        target_cols = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
        col_file = raw_zip_dir / "Faults27x7_var"
        data_file = raw_zip_dir / "Faults.NNA"
        with open(col_file) as f:
            col_names = [line.strip() for line in f if line.strip()]
        df = pd.read_csv(data_file, sep="\t", header=None, names=col_names)
        X = df[[c for c in df.columns if c not in target_cols]]
        y = df[target_cols]

    X.to_csv(features_path, index=False)
    y.to_csv(targets_path, index=False)
    return X, y


def to_single_label(y: pd.DataFrame) -> pd.Series:
    """Convert 7-column binary target matrix to a single class-name Series."""
    return y.idxmax(axis=1).rename("defect_type")


def build_preprocessing_pipeline() -> Pipeline:
    """
    Return an UNFITTED preprocessing pipeline:
      1. FunctionTransformer: drop TypeOfSteel_A300 + add 6 engineered features
      2. StandardScaler: zero mean, unit variance
    Call pipeline.fit(X_train) then pipeline.transform(X_train) and pipeline.transform(X_test).
    """
    return Pipeline([
        ("feature_eng", FunctionTransformer(add_engineered_features, validate=False)),
        ("scaler", StandardScaler()),
    ])
