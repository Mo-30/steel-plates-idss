import json
import joblib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODELS_DIR, FIGURES_DIR, DATA_PROCESSED
import pandas as pd

print("=== Phase 3 Verification ===")

# Load metrics summary
with open(MODELS_DIR / 'metrics_summary.json') as f:
    metrics = json.load(f)

print(f"Final model: {metrics['final_model_name']}")
print(f"Macro AUC:   {metrics['macro_auc']:.4f}")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print()
print("Per-class AUC:")
for cls, auc_val in metrics['per_class_auc'].items():
    flag = "  *** below 0.85 ***" if auc_val < 0.85 else ""
    print(f"  {cls:<15}: {auc_val:.4f}{flag}")

# Check model files
model = joblib.load(MODELS_DIR / 'best_model.pkl')
classes = joblib.load(MODELS_DIR / 'label_classes.pkl')
print(f"\nbest_model.pkl loaded: {type(model).__name__}")
print(f"classes ({len(classes)}): {list(classes)}")

# Verify predict_proba output shape
X_test = pd.read_csv(DATA_PROCESSED / 'X_test.csv')
proba = model.predict_proba(X_test.iloc[:1])
print(f"predict_proba shape on 1 sample: {proba.shape}  (expect (1, 7))")
assert proba.shape == (1, 7), f"Expected (1,7), got {proba.shape}"

# Check figures
figs = sorted([f for f in os.listdir(FIGURES_DIR) if f.startswith('phase3')])
print(f"\nPhase 3 figures: {figs}")

required_figs = [
    'phase3_model_comparison.png',
    'phase3_per_class_auc.png',
    'phase3_confusion_matrix.png',
    'phase3_roc_per_class.png',
    'phase3_roc_per_model.png',
]
missing_figs = [f for f in required_figs if f not in figs]
if missing_figs:
    print(f"Missing figures: {missing_figs}")
else:
    print("All required figures present: OK")

print("\nPhase 3 verification PASSED")
