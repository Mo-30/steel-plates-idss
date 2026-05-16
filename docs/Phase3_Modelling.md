# Phase 3: Predictive Modelling & Evaluation

**Rubric weight:** 5 marks  
**Criteria:** Model implementation (1.5) · Evaluation quality (1.5) · Model selection justification (1.0) · Decision-support interpretation (1.0)  
**Deliverables:** Extended notebook + comparison report + saved best model (pickle)

---

## Guiding Principle

The rubric does not prescribe specific algorithms. It grades whether models are "trained correctly and compared," whether the final model is "justified using performance, interpretability, and deployment practicality," and whether the team "explains how model quality affects decisions, actions, or business risk."

This means: choose models that are genuinely the best fit for this dataset, explain why, and connect results to factory-floor decisions. A well-justified selection of 3-4 appropriate models scores higher than a checklist of 5 generic ones.

---

## Dataset Characteristics That Drive Model Selection

Before choosing models, document these facts in a markdown cell. They are the evidence base for every modelling decision:

| Characteristic | Value | Implication |
|---|---|---|
| Sample size | 1,941 total (~1,553 train) | Small. Favors models that generalize well with limited data. Penalizes models that need large datasets (deep learning). |
| Classes | 7 mutually exclusive | Multi-class, not binary. Models must output calibrated probabilities across 7 classes. |
| Smallest class (train) | Dirtiness: ~44 samples | Extremely thin. High variance on rare-class metrics. Class weighting or resampling is essential. |
| Imbalance ratio | 12:1 | Aggressive. Without handling, models will ignore Dirtiness/Stains entirely. |
| Feature count | 27 original + ~6 engineered ≈ 33 | Moderate. Sample-to-feature ratio ~47:1. Fine for most models but watch for overfitting in deep trees. |
| Feature types | All numeric (int + float) | No categorical encoding complexity. All models can consume features directly. |
| Feature structure | Some near-perfect separators (steel grade), some highly correlated pairs, some right-skewed | Linear separability exists for some classes; non-linear interactions likely for others. Motivates trying both linear and non-linear models. |
| Evaluation metric | AUC one-vs-rest, macro-averaged | Ranking-based, threshold-independent. Models must produce well-calibrated probability outputs. |

---

## Model Selection Strategy

Based on the characteristics above, select models that cover three roles:

### Role 1: Interpretable baseline
A model whose decisions can be directly explained to quality engineers. Serves as the performance floor — anything more complex must beat it to justify the added complexity.

**Best fit: Logistic Regression (OVR, class_weight='balanced')**

Why for this data:
- Steel grade near-perfectly separates K_Scatch, Z_Scratch, and Stains. LR will capture this linear separation directly, and its coefficients will show exactly how much each feature contributes per class.
- On 1,941 samples, LR won't overfit.
- If LR achieves a strong AUC, it tells you the problem is partially linear — which is a useful insight about the manufacturing process (defect types are driven by a few dominant factors, not complex interactions).
- class_weight='balanced' directly addresses imbalance in the loss function.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    multi_class='ovr',
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
```

### Role 2: Non-linear models that handle small tabular data well
The core of the comparison. Select 2-3 models that approach the problem differently so you can assess which learning paradigm best fits the defect patterns.

**Option A: Random Forest**

Why for this data:
- Robust on small datasets — bagging reduces variance.
- Handles imbalance via class_weight='balanced' natively.
- Invariant to feature scaling (useful sanity check against scaled vs unscaled performance).
- Provides built-in feature importance for Phase 4.
- With 1,553 training samples, limit depth or increase min_samples_leaf to prevent memorizing rare classes.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    max_depth=20,           # Limit depth — small dataset
    min_samples_leaf=3,     # Prevent single-sample leaves
    random_state=42,
    n_jobs=-1
)
```

**Option B: XGBoost**

Why for this data:
- Gradient boosting is the standard winner on tabular data.
- Sequential error correction can capture subtle patterns that RF's parallel bagging misses.
- Regularization parameters (max_depth, min_child_weight, gamma) are critical on 1,941 samples — tune conservatively to prevent overfitting.
- For multi-class imbalance, use sample_weight (not scale_pos_weight, which is binary-only).

```python
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,              # Conservative — small data
    learning_rate=0.1,
    min_child_weight=5,       # Prevents splits on tiny rare-class subsets
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,                # Regularization
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
sample_weights = compute_sample_weight('balanced', y_train)
xgb.fit(X_train, y_train, sample_weight=sample_weights)
```

**Option C: SVM (RBF kernel)**

Why for this data:
- SVMs are historically among the best performers on small-to-medium tabular datasets (under ~5,000 samples). This is well-documented in the ML literature.
- The RBF kernel captures non-linear boundaries without the overfitting risk of deep trees.
- With StandardScaler already applied (Phase 2), SVM operates in its ideal input regime.
- Provides a fundamentally different learning paradigm from tree-based models — it maximizes margin rather than minimizing impurity/loss. This diversity is valuable for ensembling.
- class_weight='balanced' is natively supported.

```python
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',
    class_weight='balanced',
    probability=True,         # Required for predict_proba and AUC
    C=10,                     # Tune this
    gamma='scale',
    random_state=42
)
```

**Note:** probability=True adds computational cost (Platt scaling via internal CV). On 1,553 training samples this is fast. If SVM outperforms trees, its margin-based approach tells you something about the data geometry — defect classes form well-separated clusters in feature space.

**Option D: LightGBM (alternative to or alongside XGBoost)**

Consider if you want a second boosting model for comparison. LightGBM uses leaf-wise growth vs XGBoost's level-wise, which can be more efficient but risks overfitting on small data. If you include both XGBoost and LightGBM, be explicit about why: "We compare two boosting implementations to assess whether leaf-wise growth (LightGBM) provides an advantage over level-wise growth (XGBoost) on this small dataset."

```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    class_weight='balanced',
    min_child_samples=10,     # Higher than default — small data guard
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
```

### Recommended Selection: LR + RF + XGBoost + SVM (4 models)

This gives you:
- **Linear vs non-linear comparison** (LR vs everything else)
- **Bagging vs boosting comparison** (RF vs XGBoost)
- **Tree-based vs kernel-based comparison** (RF/XGB vs SVM)
- Maximum ensemble diversity for the final model

If you prefer 5 models, add LightGBM. But justify why — don't add it just to have more rows in the comparison table.

### Role 3: Final ensemble model

**Primary: Stacking Classifier**

Why stacking over soft voting for this data:
- Soft voting averages probabilities equally across all classes. If SVM handles Stains better than RF, and RF handles K_Scatch better than SVM, soft voting dilutes both strengths.
- Stacking with a Logistic Regression meta-learner learns per-class weights for each base model. On a 7-class problem with highly unequal representation, this class-specific routing outperforms blind averaging.
- The meta-learner is trained via 5-fold CV on the training set, so it doesn't see the test set.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold

stacking = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('svm', svm)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict_proba',
    n_jobs=-1
)
stacking.fit(X_train, y_train)
```

**Also compute soft voting for comparison:**

```python
# Manual soft voting (simpler than VotingClassifier for pre-fitted models)
y_proba_soft = (
    rf.predict_proba(X_test) +
    xgb.predict_proba(X_test) +
    svm.predict_proba(X_test)
) / 3
```

Compare stacking vs soft voting AUC. Report both. The winner is your final model. If they're close, prefer stacking — its ability to route classes to the best base model is more valuable for the decision-support use case (the factory cares about per-defect accuracy, not just the average).

---

## Class Imbalance Handling

### Primary approach: class_weight='balanced'

This is integrated directly into the loss function for LR, RF, SVM, and LightGBM. For XGBoost, use `compute_sample_weight('balanced', y_train)`.

**Business rationale:** "Undetected rare defects like Dirtiness (2.8%) and Stains (3.7%) have disproportionate business cost. A contamination defect that reaches a customer can trigger a product recall costing orders of magnitude more than the scrap cost of a single plate. Weighting the loss function to penalize rare-class misses ensures the model treats these costly failures seriously."

### Comparison experiment: run at least one model with vs without balancing

Train RF (or XGBoost) twice — once with class_weight='balanced' and once with default weights. Compare:
- Overall macro AUC
- Per-class AUC (especially Dirtiness and Stains)
- Confusion matrix

**Expected finding:** Without balancing, the model will achieve slightly higher overall accuracy (by predicting Other_Faults more often) but dramatically lower AUC on rare classes. With balancing, rare-class AUC improves at the cost of slightly more false positives on majority classes.

**Decision-support framing:** "We accept a small increase in false alarms for common defects (Other_Faults, Bumps) in exchange for reliably detecting rare but costly defects (Dirtiness, Stains). This tradeoff aligns with the quality control principle that missed defects cost more than unnecessary inspections."

### Optional: SMOTE comparison

If time allows, try SMOTE on the training set for one model:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```
Compare AUC with the class_weight approach. On small datasets, SMOTE can blur class boundaries by generating synthetic samples in overlapping regions. Document the finding either way.

---

## Hyperparameter Tuning

Tune the best-performing individual model (likely XGBoost or SVM) using RandomizedSearchCV before building the ensemble.

### XGBoost tuning:
```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7, 10],
    'gamma': [0, 0.1, 0.2, 0.3]
}

search = RandomizedSearchCV(
    xgb, param_distributions=param_dist,
    n_iter=50, scoring='roc_auc_ovr',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42, n_jobs=-1
)
search.fit(X_train, y_train, sample_weight=sample_weights)
```

### SVM tuning (if SVM is a top performer):
```python
param_dist_svm = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001]
}
```
SVM has fewer hyperparameters — GridSearchCV is fine here.

Report `best_params_` and the improvement in validation AUC. Even small improvements matter — document them.

---

## Evaluation Framework

### 1. Per-model metrics table

| Model | Macro AUC | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|-----------|----------|-----------------|--------------|----------|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |
| SVM (RBF) | | | | | |
| Soft Voting | | | | | |
| Stacking Ensemble | | | | | |

### 2. Per-class AUC for the final model

| Defect | AUC | Test Support |
|--------|-----|-------------|
| Pastry | | ~32 |
| Z_Scratch | | ~38 |
| K_Scatch | | ~78 |
| Stains | | ~14 |
| Dirtiness | | ~11 |
| Bumps | | ~80 |
| Other_Faults | | ~135 |
| **Macro Average** | | |

Flag any class with AUC below 0.85 and explain why. With only 11-14 test samples in Dirtiness/Stains, AUC estimates for these classes will have high variance — state this honestly.

### 3. Confusion matrix

Heatmap for the final model. Use class names as labels. Annotate with counts.

**Business interpretation (required):** Identify the top 2-3 misclassification pairs and explain their real-world consequences. For example: "The model confuses Pastry with Bumps in X cases. Both are surface irregularities but require different corrective actions — Pastry points to surface preparation issues while Bumps point to handling equipment damage. When this confusion occurs, the wrong corrective action is taken, wasting maintenance time and leaving the actual root cause unaddressed."

### 4. ROC curves

**Plot 1:** All 7 one-vs-rest ROC curves for the final model on a single figure. Include AUC in the legend.

**Plot 2:** Macro-average ROC curves overlaid for all models. This is the direct visual comparison.

### 5. Cross-validation scores

5-fold stratified CV for each model. Report mean ± standard deviation of AUC. High std indicates instability — note which models have it and why (likely driven by fold-to-fold variation in rare classes).

---

## Decision-Support Interpretation

**This section is worth a dedicated 1.0 mark.** Write it as a standalone markdown section titled "What This Means for the Factory Floor."

Address these questions with specific numbers from your results:

1. **Throughput:** "With a macro AUC of X, the system correctly ranks defective plates above non-defective ones X% of the time. At a production rate of ~200 plates/shift, approximately Y plates per shift would be correctly routed to the right corrective action queue."

2. **Error costs by class:** "The confusion matrix shows the model confuses [Class A] with [Class B] in N cases. This misrouting sends plates to [wrong action] instead of [right action], resulting in [specific consequence]."

3. **Rare defect reliability:** "The model achieves AUC of X on Dirtiness and Y on Stains — the two rarest defect types. While these estimates have wide confidence intervals due to small test samples (11 and 14 plates respectively), they indicate the model can reliably flag contamination-related defects for human review."

4. **Model vs no model:** "Without this system, a manual inspector must evaluate each plate against all 7 defect types. The model narrows the focus to the 1-2 most likely defect types with X% reliability, cutting average inspection time and reducing human error from fatigue."

5. **Threshold flexibility:** "Because AUC is threshold-independent, operators can adjust sensitivity per defect type. For safety-critical applications, lowering the threshold for K_Scatch catches more scratched plates at the cost of some false alarms — the ROC curve shows the exact tradeoff."

---

## Model Selection Justification

Write a markdown section titled "Why We Chose This Model" structured around the three rubric axes:

**1. Performance:**
"The [stacking ensemble / best model] achieves the highest macro AUC of X on the held-out test set. It outperforms the best individual model by Y points. Critically, it maintains strong AUC on the rarest classes (Dirtiness: X, Stains: Y), which is where the business cost of misclassification is highest."

**2. Interpretability:**
"While the ensemble itself is not directly interpretable, all base models support SHAP analysis (Phase 4). We can decompose any individual prediction into feature contributions, showing a quality engineer exactly why the system flagged a specific plate. The Logistic Regression baseline confirms that [X% / Y%] of the model's predictive power comes from interpretable linear relationships (primarily steel grade), with the ensemble adding value through non-linear interactions."

**3. Deployment practicality:**
"The ensemble consists of [N] model objects totaling approximately X MB on disk. Inference on a single plate takes under 100ms. Dependencies are standard Python ML libraries (scikit-learn, xgboost). The system can run on a factory-floor workstation without GPU or cloud infrastructure. The preprocessing pipeline is saved separately, ensuring consistent feature transformation between training and production."

---

## Save the Final Model

```python
import joblib

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(best_model.classes_, 'models/label_classes.pkl')
```

If using the stacking ensemble, the single StackingClassifier object contains all base models. One pickle file.

If using manual probability averaging, save each model individually and document the averaging logic.

---

## Deliverables Checklist

- [ ] Notebook: `notebooks/phase3_modelling.ipynb`
- [ ] All individual models trained and evaluated
- [ ] Class imbalance comparison (with/without balancing) documented
- [ ] Hyperparameter tuning on best individual model
- [ ] Ensemble model(s) trained and compared
- [ ] Per-model comparison table (AUC, accuracy, precision, recall, F1)
- [ ] Per-class AUC breakdown for final model
- [ ] Confusion matrix heatmap with business interpretation
- [ ] ROC curves (per-class + per-model overlay)
- [ ] Cross-validation scores (mean ± std)
- [ ] "What This Means for the Factory Floor" section
- [ ] "Why We Chose This Model" justification section (performance + interpretability + deployment)
- [ ] Model saved as `models/best_model.pkl`
- [ ] Every major result has a business rationale markdown cell
