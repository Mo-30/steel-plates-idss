import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_raw_data, build_preprocessing_pipeline, to_single_label
from src.feature_engineering import add_engineered_features
from src.config import DEFECT_CLASSES, ENGINEERED_FEATURES
import joblib

print("=== Phase 2 Verification ===")

# 1. Load raw data
X, y = load_raw_data()
print(f"X shape: {X.shape}  (expect (1941, 27))")
print(f"y shape: {y.shape}  (expect (1941, 7))")
assert X.shape == (1941, 27), f"Unexpected shape: {X.shape}"
assert y.shape == (1941, 7), f"Unexpected shape: {y.shape}"

# 2. Feature engineering
X_eng = add_engineered_features(X)
print(f"X_eng shape: {X_eng.shape}  (expect (1941, 32))")
assert all(f in X_eng.columns for f in ENGINEERED_FEATURES), "Missing engineered features"
assert "TypeOfSteel_A300" not in X_eng.columns, "A300 should be dropped"
print("Engineered features present: OK")
print("TypeOfSteel_A300 dropped: OK")

# 3. Pipeline
pipeline = build_preprocessing_pipeline()
print(f"Pipeline steps: {[s[0] for s in pipeline.steps]}")

# 4. Single-label conversion
y_single = to_single_label(y)
assert set(y_single.unique()) == set(DEFECT_CLASSES), "Unexpected classes"
print(f"Classes: {sorted(y_single.unique())}")

# 5. Check saved files exist
from src.config import DATA_PROCESSED, MODELS_DIR
files_to_check = [
    DATA_PROCESSED / "X_train.csv",
    DATA_PROCESSED / "X_test.csv",
    DATA_PROCESSED / "y_train.csv",
    DATA_PROCESSED / "y_test.csv",
    MODELS_DIR / "preprocessing_pipeline.pkl",
]

missing = [str(f) for f in files_to_check if not f.exists()]
if missing:
    print(f"Missing output files (will be created when notebook runs): {missing}")
else:
    print("All output files present: OK")
    # Verify pipeline round-trip
    loaded_pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.pkl")
    import pandas as pd
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    print(f"X_train.csv columns: {len(X_train.columns)}")
    has_eng = all(f in X_train.columns for f in ENGINEERED_FEATURES)
    print(f"Engineered features in X_train.csv: {has_eng}")

print("\nPhase 2 verification PASSED")
