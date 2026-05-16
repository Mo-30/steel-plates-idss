# CLAUDE.md — Master Project Guidance for Claude Code

## Project Identity

**Course:** AIS431 – Intelligent Decision Support Systems  
**Project:** Steel Plates Faults Prediction — End-to-End IDSS  
**Team:** Mohamed Sherif (221000142) · Mohamed Osama (221001647)  
**Total marks:** 30 (Phase 1 done = 5 marks secured; 25 marks remaining)  
**Goal:** A-grade quality. Every phase must connect technical work to business decisions.

---

## Current Status

Phase 1 is complete. The Phase 1 notebook (`notebooks/phase1_eda.ipynb`) contains all EDA work, plots, and findings. Before implementing any subsequent phase, read the Phase 1 notebook outputs to understand what was already discovered and decided.

**Key decisions already locked in from Phase 1:**

- **Task type:** 7-class mutually exclusive multi-class classification. Every plate has exactly one defect label. This was verified by `y.sum(axis=1).value_counts()` — all 1,941 rows sum to 1. Do NOT treat this as multi-label.
- **Target encoding:** `y_single = y.idxmax(axis=1)` gives a single string class label per row.
- **Primary metric:** AUC (one-vs-rest), macro-averaged across all 7 classes. This is explicitly required by the rubric.
- **Class imbalance:** 12:1 ratio. Other_Faults=673, Bumps=402, K_Scatch=391, Z_Scratch=190, Pastry=158, Stains=72, Dirtiness=55.
- **Strongest EDA finding:** Steel grade (A300 vs A400) near-perfectly separates K_Scatch, Z_Scratch, and Stains. This is the single most important feature.
- **Data quality:** Zero missing values, zero duplicates. TypeOfSteel_A300 and A400 are perfectly inverse — drop one.

---

## Rubric — What Gets Full Marks

The rubric's core principle (stated verbatim): *"A technically correct notebook or model should not receive the full score unless the team explains how the result supports a decision, what action should be taken, and why this matters in the chosen business domain."*

This means every notebook must contain markdown cells that explain the business "so what" after every technical step. Not just "accuracy is 0.92" but "this means the system would correctly route 92% of plates to the right inspection queue, reducing manual re-inspection by an estimated X%."

**Mark breakdown:**

| Phase | Marks | Key rubric criteria |
|-------|-------|---------------------|
| Phase 1 | 5 | ✅ DONE |
| Phase 2 | 3 | Data quality handling (1.5), Feature engineering quality (1.5), Business rationale & reproducibility (1.0) |
| Phase 3 | 5 | Model implementation (1.5), Evaluation quality (1.5), Model selection justification (1.0), Decision-support interpretation (1.0) |
| Phase 4 | 5 | Explainability & interpretation (2.0), Segmentation & insights (1.5), Business recommendation quality (2.5), Impact estimation (1.0) |
| Phase 5 | 5 | Prototype functionality, prediction output, explanation, usability |
| Presentation | 7 | Communication, justification, discussion, recommendation quality |

Phase 4's "business recommendation quality" at 2.5 marks is the single highest-weighted criterion in the entire rubric. Generic recommendations score zero even if the SHAP plots are technically correct.

---

## Project File Structure

```
steel-plates-idss/
│
├── CLAUDE.md                              ← You are here. Read this first.
├── Phase2_DataPrep.md                     ← Guidance for Phase 2 implementation
├── Phase3_Modelling.md                    ← Guidance for Phase 3 implementation
├── Phase4_Explainability.md               ← Guidance for Phase 4 implementation
├── Phase5_Prototype.md                    ← Guidance for Phase 5 implementation
│
├── notebooks/
│   ├── phase1_eda.ipynb                   ← Phase 1 (DONE — read first)
│   ├── phase2_data_prep.ipynb             ← Phase 2: data cleaning + feature eng
│   ├── phase3_modelling.ipynb             ← Phase 3: train + evaluate + select model
│   └── phase4_explainability.ipynb        ← Phase 4: SHAP + segmentation + recommendations
│
├── data/
│   ├── raw/                               ← Original UCI dataset (fetched by notebook)
│   ├── processed/
│   │   ├── X_train.csv                    ← Phase 2 output
│   │   ├── X_test.csv                     ← Phase 2 output
│   │   ├── y_train.csv                    ← Phase 2 output
│   │   └── y_test.csv                     ← Phase 2 output
│   └── data_dictionary.md                 ← Phase 2 output
│
├── models/
│   ├── best_model.pkl                     ← Phase 3 output: final ensemble
│   ├── preprocessing_pipeline.pkl         ← Phase 2/3 output: fitted scaler + feature eng
│   └── label_encoder.pkl                  ← Phase 2 output: class label mapping
│
├── prototype/
│   ├── app.py                             ← Phase 5: Streamlit dashboard
│   ├── requirements.txt                   ← Prototype dependencies
│   └── sample_input.csv                   ← Example input for demo
│
├── reports/
│   ├── business_insights_report.pdf       ← Phase 4 deliverable
│   └── executive_summary.pdf              ← Phase 5 deliverable
│
├── presentation/
│   └── final_presentation.pptx            ← Phase 5 deliverable (or PDF)
│
└── src/
    ├── __init__.py
    ├── config.py                          ← Constants: class names, feature lists, paths
    ├── preprocessing.py                   ← Reusable preprocessing pipeline
    └── feature_engineering.py             ← Feature creation functions
```

---

## Architecture Principles

### 1. Shared preprocessing pipeline
The same feature engineering and scaling logic must be used in Phase 2 (to create CSVs), Phase 3 (during cross-validation), and Phase 5 (in the prototype). Extract this into `src/preprocessing.py` and import it everywhere. Never duplicate preprocessing code.

### 2. One notebook per phase, self-contained but connected
Each notebook should be runnable independently (it loads its inputs from files, not from another notebook's memory). Phase 2 saves CSVs → Phase 3 loads those CSVs. Phase 3 saves the model pickle → Phase 4 loads it. Phase 5's prototype loads the same pickle.

### 3. Notebook structure convention
Every notebook follows this structure:
```
1. Setup & Imports
2. Load Data (from previous phase outputs)
3. Technical Work (the phase's core tasks)
4. Business Interpretation (markdown cells after every major result)
5. Save Outputs (CSVs, pickles, figures for the report)
6. Summary & Next Steps
```

### 4. Constants in one place
All class names, feature lists, color palettes, and file paths go in `src/config.py`. This prevents typos and inconsistencies across phases.

```python
# src/config.py
DEFECT_CLASSES = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = 'defect_type'
```

### 5. Business narrative is a first-class deliverable
After every code cell that produces a result, add a markdown cell with heading "**Business Interpretation**" or "**Decision-Support Implication**" that explains what this means for a quality control engineer or plant manager. This is not optional — it is what separates a 3/5 from a 5/5 on every phase.

---

## Implementation Order

Implement phases sequentially. Each depends on the previous:

1. **Phase 2** → Read `Phase2_DataPrep.md`. Outputs: 4 CSVs + data dictionary + preprocessing pipeline pickle.
2. **Phase 3** → Read `Phase3_Modelling.md`. Inputs: Phase 2 CSVs. Outputs: trained model pickle + evaluation report.
3. **Phase 4** → Read `Phase4_Explainability.md`. Inputs: Phase 3 model + Phase 2 data. Outputs: SHAP analysis + business report PDF.
4. **Phase 5** → Read `Phase5_Prototype.md`. Inputs: Phase 3 model + Phase 2 pipeline. Outputs: Streamlit app + presentation + exec summary.

---

## Critical Reminders

- **Do NOT use multi-label approaches** (no MultiOutputClassifier, no BinaryRelevance, no 7 independent classifiers). This is a single multi-class problem. Use standard classifiers with `predict_proba()` that output 7 class probabilities.
- **AUC must be computed via one-vs-rest** using `roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')`.
- **Stratified splits everywhere** — `StratifiedKFold`, `train_test_split(..., stratify=y)`.
- **Save everything reproducibly** — `random_state=42` on every randomized operation.
- **The prototype must use the actual trained model** — not hardcoded predictions. Load the pickle and run real inference.
- **Every recommendation in Phase 4 must cite specific model evidence** — a SHAP value, a feature importance rank, a confusion matrix pattern. "Improve quality control" is worth zero marks.
