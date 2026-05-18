# CLAUDE.md — Steel Plates Defect IDSS

**Course:** AIS431 — Intelligent Decision Support Systems  
**Team:** Mohamed Sherif (221000142) · Mohamed Osama (221001647)  
**Repo:** https://github.com/Mo-30/steel-plates-idss  
**Python:** `C:\Users\moshe\anaconda3\python.exe` (only env with all packages)  
**Run app:** `cd prototype ; python -m streamlit run app.py`

---

## Project Status — ALL PHASES COMPLETE

| Phase | Deliverable | Status | Key Result |
|---|---|---|---|
| 1 | `notebooks/phase1_eda.ipynb` | ✅ Done | EDA, class imbalance confirmed |
| 2 | `notebooks/phase2_data_prep.ipynb` | ✅ Done | 32 features, pipeline saved |
| 3 | `notebooks/phase3_modelling.ipynb` | ✅ Done | Macro AUC **0.9545** |
| 4 | `notebooks/phase4_explainability.ipynb` | ✅ Done | SHAP + 6 recommendations |
| 5 | `prototype/app.py` + reports + PPTX | ✅ Done | Streamlit + PDF + PPTX |

---

## Dataset

- **Source:** UCI Steel Plates Faults (ID 198), 1,941 samples, 27 features
- **Task:** 7-class mutually exclusive classification (NOT multi-label — every row sums to 1)
- **Classes:** `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`
- **Imbalance:** 12:1 (Other_Faults=673 vs Dirtiness=55)
- **Primary metric:** `roc_auc_score(y, proba, multi_class='ovr', average='macro')`
- **Split:** 80/20 stratified, `random_state=42`
- **Raw data:** cached at `data/raw/features.csv` + `data/raw/targets.csv`; offline fallback from `data/raw_zip/Faults.NNA`

---

## Model

- **Final model:** `StackingClassifier` saved at `models/best_model.pkl`
  - Base: `RandomForestClassifier(n_estimators=300, class_weight='balanced')` → `named_estimators_['rf']`
  - Base: `SVC(kernel='rbf', class_weight='balanced', probability=True)`
  - Meta: `LogisticRegression(class_weight='balanced')`
- **Pipeline:** `models/preprocessing_pipeline.pkl` — FunctionTransformer (6 engineered features + drop TypeOfSteel_A300) → StandardScaler
- **Input contract:** pipeline receives all 27 original columns; drops `TypeOfSteel_A300` internally
- **Output:** `model.predict_proba(X_scaled)` → shape `(n, 7)` in `label_classes` order
- **SHAP:** TreeExplainer on `model.named_estimators_['rf']` (stacking meta not TreeExplainer-compatible)

### Performance
| Class | AUC |
|---|---|
| Z_Scratch | 0.9960 |
| Dirtiness | 0.9909 |
| K_Scatch | 0.9898 |
| Stains | 0.9611 |
| Pastry | 0.9436 |
| Bumps | 0.9101 |
| Other_Faults | 0.8900 |
| **Macro AUC** | **0.9545** |
| Accuracy | 76.4% |

### Risk Tiers
| Tier | Threshold | Coverage | Accuracy |
|---|---|---|---|
| High | prob ≥ 0.70 | 65.8% | 88.3% |
| Medium | 0.40–0.70 | 26.7% | ~57% |
| Low | < 0.40 | 7.5% | ~50% |

---

## src/ Modules

### `src/config.py`
All constants: `PROJECT_ROOT`, `DATA_RAW`, `DATA_PROCESSED`, `MODELS_DIR`, `FIGURES_DIR`, `RANDOM_STATE=42`, `TEST_SIZE=0.2`, `DEFECT_CLASSES`, `ENGINEERED_FEATURES`

### `src/feature_engineering.py`
`add_engineered_features(X)` — drops `TypeOfSteel_A300`, adds:
- `Defect_Area_Ratio` = `Pixels_Areas / (bounding_box + 1)`
- `Luminosity_Range` = `Max_Lum - Min_Lum`
- `Aspect_Ratio` = `X_Perimeter / (Y_Perimeter + 1e-6)`
- `Log_Pixels_Areas` = `log1p(Pixels_Areas)`
- `Edge_Strength` = `Edges_Index × Edges_X_Index × Edges_Y_Index`
- `Thickness_Area_Interaction` = `Steel_Plate_Thickness × Log_Pixels_Areas`

### `src/preprocessing.py`
- `load_raw_data()` — checks `data/raw/` → fallback to `data/raw_zip/Faults.NNA`
- `build_preprocessing_pipeline()` — returns unfitted Pipeline
- `to_single_label(y)` — `y.idxmax(axis=1)`

---

## Prototype (`prototype/app.py`)

Two tabs:
1. **🔍 Prediction** — manual entry (27 fields, median defaults) or CSV upload → probabilities + risk badge + SHAP waterfall
2. **🤖 QC Assistant** — Groq chatbot (Llama 3.3 70B) with prediction context; reads key from `prototype/.env`

**`prototype/.env`** (gitignored):
```
GROQ_API_KEY=your_key_here
```
**`prototype/.env.example`** — committed template for teammates.

**`prototype/chatbot.py`** — loads key via `python-dotenv`, calls Groq via `openai` client at `https://api.groq.com/openai/v1`

---

## Key Files Reference

| File | Purpose |
|---|---|
| `models/best_model.pkl` | Stacking Ensemble (Git LFS, 12.6 MB) |
| `models/preprocessing_pipeline.pkl` | Fitted pipeline |
| `models/metrics_summary.json` | `macro_auc`, `accuracy`, `per_class_auc` |
| `models/recommendation_evidence.json` | Evidence numbers for business recs |
| `models/label_classes.pkl` | `['Bumps','Dirtiness','K_Scatch','Other_Faults','Pastry','Stains','Z_Scratch']` |
| `data/processed/X_train.csv` | 1552 × 32 scaled features |
| `data/processed/X_test.csv` | 389 × 32 scaled features |
| `reports/business_insights_report.pdf` | 5-page business report |
| `reports/executive_summary.pdf` | 1-page non-technical summary |
| `presentation/final_presentation.pptx` | 13-slide deck |
| `prototype/test_sample_100.csv` | 100 test-set rows for batch demo |

---

## Environment Notes

- **sklearn 1.8.0** — `LogisticRegression` no longer accepts `multi_class` param (removed)
- **numpy 2.x** — in use; requires matplotlib ≥ 3.9 and shap ≥ 0.51
- **XGBoost** — trained on integer-encoded labels via `LabelEncoder` (saved as `models/label_encoder.pkl`); `use_label_encoder` param removed
- **`estimators_`** in sklearn 1.8 `StackingClassifier` — is a flat list of fitted estimators; use `named_estimators_['rf']` to access by name
- **Notebooks** — execute with: `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/phaseX.ipynb`

---

## What's Left / Open for Contribution

All core deliverables are done. Possible extensions:
- Improve `Other_Faults` AUC (0.89) — sub-type analysis by thickness quartile
- Add data drift monitoring dashboard
- Retrain pipeline with new labelled data
- Add more chatbot context (batch-mode SHAP summaries)
