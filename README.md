# Steel Plates Defect Prediction — Intelligent Decision Support System

> **AIS431 Final Project** | Mohamed Sherif (221000142) · Mohamed Osama (221001647)

A production-ready machine-learning system that classifies **seven types of surface defects** on steel plates from sensor measurements, explains every prediction with SHAP, routes each plate to the correct factory action based on a calibrated confidence threshold, and includes a **Groq-powered QC chatbot** for operator support.

---

## Results at a Glance

| Metric | Value |
|---|---|
| **Macro OVR AUC** (primary) | **0.9545** |
| Overall Accuracy | 79.5% |
| High-confidence automation coverage | 65.8% of plates |
| High-confidence accuracy | 88.3% |
| Model | Stacking Ensemble (RF + SVM + LR meta) |

**Per-class AUC:**

| Defect | AUC | Test Support |
|---|---|---|
| Z_Scratch | 0.9960 | 38 |
| Dirtiness | 0.9909 | 11 |
| K_Scatch | 0.9898 | 76 |
| Stains | 0.9611 | 22 |
| Pastry | 0.9436 | 49 |
| Bumps | 0.9101 | 76 |
| Other_Faults | 0.8900 | 117 |

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Mo-30/steel-plates-idss.git
cd steel-plates-idss
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up the chatbot API key
```bash
cp prototype/.env.example prototype/.env
# Edit prototype/.env and paste your Groq API key:
# GROQ_API_KEY=your_key_here
# Get a free key at: https://console.groq.com
```

### 4. Run the Streamlit prototype
```bash
cd prototype
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501).

> **Note:** The processed data (`data/processed/`) and model files (`models/`) are included in the repo via Git LFS. No need to re-run the notebooks unless you want to reproduce training.

---

## Project Structure

```
steel-plates-idss/
│
├── src/                              # Core Python modules (shared by notebooks + app)
│   ├── config.py                     # All constants: paths, random seed, class names
│   ├── feature_engineering.py        # 6 domain-engineered features + TypeOfSteel_A300 drop
│   └── preprocessing.py              # Pipeline factory (FunctionTransformer + StandardScaler)
│                                     # + load_raw_data() with offline fallback
│
├── notebooks/
│   ├── phase1_eda.ipynb              # Exploratory data analysis: distributions, correlations
│   ├── phase2_data_prep.ipynb        # Feature engineering, train/test split, pipeline fitting
│   ├── phase3_modelling.ipynb        # LR / RF / XGBoost / SVM / Stacking + evaluation
│   └── phase4_explainability.ipynb  # SHAP analysis, risk tiers, business recommendations
│
├── prototype/
│   ├── app.py                        # Streamlit app — prediction + SHAP + QC chatbot
│   ├── chatbot.py                    # Groq LLM chatbot (llama-3.3-70b) with prediction context
│   ├── .env                          # Your API keys — NOT committed (see .env.example)
│   ├── .env.example                  # Template: copy to .env and fill in your key
│   ├── requirements.txt              # App-specific dependencies
│   ├── sample_input.csv              # 5-row example for manual testing
│   └── test_sample_100.csv           # 100 test-set rows for batch prediction demo
│
├── models/                           # Trained artefacts (Git LFS for .pkl files)
│   ├── best_model.pkl                # Stacking Ensemble — main inference model
│   ├── preprocessing_pipeline.pkl    # Fitted FunctionTransformer + StandardScaler
│   ├── label_classes.pkl             # Ordered class names array
│   ├── label_encoder.pkl             # LabelEncoder for XGBoost integer targets
│   ├── feature_names.pkl             # List of 32 feature names post-engineering
│   ├── metrics_summary.json          # Final AUC, accuracy, per-class AUC
│   └── recommendation_evidence.json  # Numeric evidence cited in business recommendations
│
├── reports/
│   ├── business_insights_report.pdf  # 5-page business report (reportlab)
│   ├── executive_summary.pdf         # 1-page non-technical summary (reportlab)
│   ├── generate_business_report.py   # Script to regenerate business_insights_report.pdf
│   └── generate_executive_summary.py # Script to regenerate executive_summary.pdf
│
├── presentation/
│   ├── final_presentation.pptx       # 13-slide deck (python-pptx)
│   └── generate_presentation.py      # Script to regenerate the .pptx
│
├── figures/                          # All PNG plots saved during notebook execution
│   ├── phase2_class_distribution.png
│   ├── phase2_mutual_information.png
│   ├── phase3_model_comparison.png
│   ├── phase3_confusion_matrix.png
│   ├── phase3_per_class_auc.png
│   ├── phase3_roc_per_class.png
│   ├── phase3_roc_per_model.png
│   ├── phase4_shap_bar.png
│   ├── phase4_rf_importance.png
│   ├── phase4_risk_tiers.png
│   ├── phase4_waterfall_tp.png
│   ├── phase4_waterfall_edge.png
│   └── phase4_waterfall_wrong.png
│
├── data/
│   ├── raw_zip/                      # Original UCI archive files (offline fallback)
│   │   ├── Faults.NNA                # Raw tab-separated data (1941 × 34)
│   │   └── Faults27x7_var            # Column names file
│   └── data_dictionary.md            # Feature descriptions for all 33 columns
│
├── docs/                             # Project specification documents
│   ├── IDSS_Project.md
│   ├── CLAUDE.md
│   ├── Phase2_DataPrep.md
│   ├── Phase3_Modelling.md
│   ├── Phase4_Explainability.md
│   └── Phase5_Prototype.md
│
├── requirements.txt                  # Full project dependencies
├── verify_phase2.py                  # Smoke test: checks Phase 2 outputs
├── verify_phase3.py                  # Smoke test: checks Phase 3 model + metrics
├── .gitignore
└── .gitattributes                    # Git LFS tracking for .pkl / .pdf / .pptx
```

---

## Prototype Features

The Streamlit app (`prototype/app.py`) has two tabs:

### 🔍 Prediction Tab
- **Manual Entry** — 27 sensor fields with training-set median defaults; one-click predict
- **CSV Upload** — batch predictions on any CSV with the 27 original feature columns
- For each prediction:
  - 7-class probability bar chart
  - Risk tier badge (High / Medium / Low) with colour coding
  - Recommended factory action
  - SHAP waterfall plot explaining the top 10 feature contributions
  - Top-5 feature contribution table

### 🤖 QC Assistant Tab
- Groq-powered chatbot (Llama 3.3 70B) with full awareness of the current prediction context
- Context-aware suggested questions change based on the predicted defect class
- Answers factory-floor questions about defect types, QC procedures, model accuracy, and SHAP explanations
- Requires a free Groq API key (see setup above)

---

## Model Inference Pipeline

```
Raw input (27 sensor columns)
        │
        ▼
preprocessing_pipeline.pkl
  ├── FunctionTransformer  →  add 6 engineered features, drop TypeOfSteel_A300
  └── StandardScaler       →  zero mean, unit variance
        │
        ▼  32 scaled features
Stacking Ensemble (best_model.pkl)
  ├── Random Forest  (300 trees, class_weight='balanced')
  ├── SVM RBF        (C=10, class_weight='balanced', probability=True)
  └── LR meta-learner on stacked probabilities
        │
        ▼  7 class probabilities
Risk tier assignment  →  factory action recommendation
        │
        ▼
SHAP TreeExplainer (on RF base estimator)  →  feature attribution
```

---

## Engineered Features

| Feature | Formula | Manufacturing Meaning |
|---|---|---|
| `Defect_Area_Ratio` | `Pixels_Areas / (bounding_box + 1)` | Spread vs localised defect |
| `Luminosity_Range` | `Max_Lum − Min_Lum` | Sharp edge (scratch) vs diffuse (stain) |
| `Aspect_Ratio` | `X_Perimeter / (Y_Perimeter + ε)` | Elongated vs compact defect |
| `Log_Pixels_Areas` | `log1p(Pixels_Areas)` | Scale-invariant size signal |
| `Edge_Strength` | `Edges_Index × Edges_X_Index × Edges_Y_Index` | Combined sharpness |
| `Thickness_Area_Interaction` | `Steel_Plate_Thickness × Log_Pixels_Areas` | Process-level size interaction |

---

## Risk Tier Routing

| Tier | Threshold | Coverage | Accuracy | Factory Action |
|---|---|---|---|---|
| **High** | prob ≥ 0.70 | 65.8% | 88.3% | Auto pass/reject — no human needed |
| **Medium** | 0.40–0.70 | 26.7% | ~57% | Human review within 60 seconds |
| **Low** | prob < 0.40 | 7.5% | ~50% | Hold — escalate to senior QC |

---

## Reproduce from Scratch

```bash
# Phase 2 — data prep
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_prep.ipynb

# Phase 3 — modelling
jupyter nbconvert --to notebook --execute --inplace notebooks/phase3_modelling.ipynb

# Phase 4 — explainability
jupyter nbconvert --to notebook --execute --inplace notebooks/phase4_explainability.ipynb

# Regenerate PDF reports
python reports/generate_business_report.py
python reports/generate_executive_summary.py

# Regenerate PPTX
python presentation/generate_presentation.py
```

---

## Contributing

1. Fork the repo and create a feature branch: `git checkout -b feature/your-feature`
2. All shared logic goes in `src/` — import from there in notebooks and `prototype/app.py`
3. New figures should be saved to `figures/` as PNG
4. Test notebooks run clean: restart kernel → Run All, no errors
5. Open a pull request with a description of what you changed and why

---

## Tech Stack

| Library | Purpose |
|---|---|
| scikit-learn 1.8 | LR, RF, SVM, StackingClassifier, preprocessing |
| XGBoost 2.1 | Gradient boosting base model |
| SHAP 0.51 | Feature attribution (TreeExplainer) |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualisation |
| Streamlit 1.30 | Interactive prototype |
| openai + Groq | QC chatbot (Llama 3.3 70B via Groq API) |
| python-dotenv | API key management |
| reportlab | PDF report generation |
| python-pptx | Presentation generation |

---

## Authors

- **Mohamed Sherif** — 221000142
- **Mohamed Osama** — 221001647

*AIS431 — Intelligent Decision Support Systems, 2025/2026*
