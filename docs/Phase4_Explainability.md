# Phase 4: Explainability, Insights & Business Recommendations

**Rubric weight:** 5 marks  
**Criteria:** Explainability & interpretation (2.0) · Segmentation & insights (1.5) · Business recommendation quality (2.5) · Impact estimation (1.0)  
**Deliverables:** Business insights report (3-5 page PDF) + supporting notebook cells

**Warning:** This phase carries the rubric's single heaviest criterion — 2.5 marks for "specific, actionable, model-evidence-backed recommendations." The assessor note states: *"If the work does not end in clear, evidence-based business recommendations, the team should not receive the full mark even if the technical plots are correct."*

---

## Step 1: Load Model and Data

```python
import joblib
best_model = joblib.load('models/best_model.pkl')
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
```

---

## Step 2: Feature Importance — Model-Native vs SHAP

### 2a. Model-native feature importance

If using Random Forest or a single tree-based model: extract `.feature_importances_`. If using the ensemble: extract importance from each component model and average them, or pick the best individual model for this analysis.

Plot a horizontal bar chart of top 15 features by importance. Title it clearly.

### 2b. SHAP Analysis — Setup

**For multi-class with tree-based models, use TreeExplainer:**

```python
import shap

# Use one of the ensemble's component models (e.g., the best-performing one)
# TreeExplainer works directly with RF, XGBoost, LightGBM
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

**Critical: Multi-class SHAP output shape.** For 7-class problems, `shap_values` will be either:
- A list of 7 arrays, each with shape `(n_test, n_features)` — one per class
- A single array of shape `(n_test, n_features, 7)`

Handle both cases. Check the shape and convert if needed:
```python
if isinstance(shap_values, list):
    # shap_values[i] = SHAP values for class i, shape (n_test, n_features)
    pass
else:
    # shap_values[:, :, i] = SHAP values for class i
    shap_values_list = [shap_values[:, :, i] for i in range(7)]
```

### 2c. SHAP Summary Plot (Global)

Generate the SHAP summary (beeswarm) plot. For multi-class, either:
- Show one summary plot per class (pick the 3 most interesting classes — e.g., K_Scatch, Stains, Dirtiness based on Phase 1 findings)
- Or use `shap.summary_plot(shap_values, X_test, plot_type="bar")` for the overall mean absolute SHAP importance

**Business interpretation markdown cell:** Write plain English after the plot. Example: "The SHAP analysis confirms that `TypeOfSteel_A400` is the most influential feature overall. When a plate is made from A400 steel, the model strongly increases its probability of K_Scatch and Stains, while decreasing the probability of Z_Scratch. This aligns with our Phase 1 finding and suggests that steel grade alone could serve as a first-pass filter in the inspection workflow."

### 2d. Compare model importance vs SHAP

Create a side-by-side comparison of the top 10 features from:
- Model-native feature importance (impurity-based for RF, gain-based for XGB/LGBM)
- SHAP mean absolute values

Note any differences. Common pattern: impurity-based importance overweights high-cardinality or correlated features; SHAP provides more accurate attribution. If they largely agree, state that the model's internal importance is trustworthy.

### 2e. Top 5 drivers — link to manufacturing

For the top 5 features identified by SHAP, write a paragraph for each explaining what physical process it corresponds to:

| Feature | Manufacturing Meaning | Why it drives defects |
|---------|----------------------|----------------------|
| TypeOfSteel_A400 | Steel alloy grade | Different grades have different surface properties and rolling behavior |
| Pixels_Areas / Log_Pixels_Areas | Defect size in pixels | Larger defects indicate more severe process failures |
| Steel_Plate_Thickness | Plate gauge | Thinner plates are more susceptible to certain defect types |
| Luminosity_Index | Surface brightness measurement | Stains and contamination alter reflectivity |
| Edges_Index | Edge sharpness of defect | Mechanical scratches have sharp edges; chemical defects have diffuse edges |

Adjust based on actual SHAP results.

---

## Step 3: SHAP Individual Explanations

Generate SHAP force plots (or waterfall plots) for 3 specific test samples. The rubric and project doc require these specific cases.

### Case 1: Correct high-confidence prediction (True Positive)
- Find a test sample where the model correctly predicts the true class with probability > 0.9.
- Generate the SHAP force/waterfall plot for this sample.
- **Write 2-3 sentences in business English:** "This K_Scatch defect was correctly identified with 95% confidence. The model's decision was primarily driven by the A400 steel type and a large defect area (Pixels_Areas in the top 10th percentile). A quality engineer would see this as a textbook rolling-direction scratch — exactly the pattern the model was designed to catch."

### Case 2: Correct but low-confidence prediction (Edge case)
- Find a test sample where the model correctly predicts the true class but with probability between 0.3-0.6 (close to the decision boundary).
- Generate the SHAP force/waterfall plot.
- **Write 2-3 sentences:** "This Bumps defect was correctly identified but with only 42% confidence. The SHAP plot shows competing influences — the defect size pushed toward Bumps, but the luminosity pattern was more typical of Other_Faults. In a production setting, this plate would be flagged for human review because the model's confidence is below the recommended threshold."

### Case 3: Misclassification
- Find a test sample where the model predicted the wrong class.
- Generate the SHAP force/waterfall plot showing why the model was wrong.
- **Write 2-3 sentences:** "This plate was truly a Pastry defect but the model predicted Bumps. The SHAP analysis reveals that the plate had unusually high Edge_Strength for a Pastry defect, which the model associated with Bumps. This highlights a limitation — when defect morphology is atypical, the model can be misled. This case supports having a human inspector verify predictions below the 70% confidence threshold."

---

## Step 4: Defect Segmentation & Risk Profiling

### 4a. Create risk tiers

Using the ensemble's predicted probabilities on the test set, assign each plate to a risk tier based on the maximum predicted probability:

| Tier | Confidence Range | Meaning |
|------|-----------------|---------|
| **High Risk** | Max probability ≥ 0.7 | Strong defect signal — auto-route to corrective action |
| **Medium Risk** | 0.4 ≤ Max probability < 0.7 | Uncertain — flag for human review |
| **Low Risk** | Max probability < 0.4 | Very uncertain — manual inspection required |

Note: "Max probability" means the highest probability across all 7 classes for that plate. Even the "predicted" class might have low probability if the model is unsure.

### 4b. Profile each tier

For each risk tier, compute and display:
- Count of plates and percentage of total
- Distribution of actual defect types within the tier
- Mean values of the top 5 features (the SHAP-identified drivers)
- Accuracy within the tier (what % are correctly classified?)

**Expected pattern:** High-risk plates should have ~95%+ accuracy (model is confident and correct). Medium-risk plates should have lower accuracy (~70-80%). Low-risk plates may have poor accuracy — these are the cases where human judgment is needed.

### 4c. Business interpretation

"The segmentation reveals that X% of plates fall into the High Risk tier where the model's accuracy is Y%. These plates can be automatically routed to the appropriate corrective action without human review, saving an estimated Z minutes of inspector time per plate. The W% of plates in the Medium Risk tier need human verification, but the model's prediction narrows the inspector's focus to 2-3 likely defect types instead of 7."

---

## Step 5: Business Recommendations

**This is the highest-weighted criterion (2.5 marks).** Write 4-6 recommendations. Each MUST follow this structure:

### Recommendation template:
```
**Recommendation N: [Specific action title]**

**Evidence:** [Cite the specific SHAP finding, feature importance, or segmentation result]
**Action:** [Exactly what the factory should do]
**Expected impact:** [Quantified estimate — scrap reduction %, time saved, cost avoided]
**Priority:** [High / Medium / Low]
**Timeline:** [Immediate / 1-3 months / 3-6 months]
```

### Recommended recommendations (adjust based on actual model results):

**Recommendation 1: Implement grade-specific inspection protocols**
- Evidence: SHAP shows TypeOfSteel_A400 is the strongest predictor. K_Scatch and Stains occur almost exclusively on A400; Z_Scratch almost exclusively on A300.
- Action: When an A400 plate enters inspection, prioritize checking for rolling-direction scratches and surface stains. When A300, prioritize Z-direction scratches.
- Impact: Reduces average inspection time by ~40% by focusing on grade-relevant defects. For 200 plates/shift, this saves approximately 80 inspector-minutes per shift.
- Priority: High. Timeline: Immediate — requires no equipment changes, only updated inspection checklists.

**Recommendation 2: Trigger roller maintenance based on K_Scatch detection rate**
- Evidence: Phase 1 showed K_Scatch has the largest average defect area (Pixels_Areas). SHAP confirms defect size is a key driver for this class. Large K_Scatch defects indicate worn rollers.
- Action: Monitor the K_Scatch detection rate per rolling mill. When the rate exceeds 2x the historical baseline over any 4-hour window, automatically schedule roller inspection.
- Impact: Early roller replacement prevents cascading damage. Estimated 15-25% reduction in K_Scatch scrap by catching worn rollers before they damage multiple batches.
- Priority: High. Timeline: 1-3 months (requires integration with production monitoring system).

**Recommendation 3: Deploy the model as an automated first-pass filter**
- Evidence: The model achieves X% accuracy on High Risk tier plates (Y% of total volume). These plates can be auto-routed without human review.
- Action: Install the model as an inline quality gate. Plates scoring >0.7 confidence on any defect class are auto-routed to the corresponding corrective action queue. Plates scoring <0.7 are flagged for human review.
- Impact: Reduces manual inspection workload by X% (the proportion of High Risk plates). At $Z/inspector-hour, this saves approximately $W per month.
- Priority: High. Timeline: 3-6 months (requires IT integration, operator training, and a 2-week parallel-run validation period).

**Recommendation 4: Investigate environmental contamination controls for Dirtiness defects**
- Evidence: SHAP shows Dirtiness defects are associated with [specific features — e.g., high luminosity variance, specific thickness range]. Dirtiness is the rarest class (2.8%) but represents contamination that could indicate broader process environment issues.
- Action: Cross-reference Dirtiness detection timestamps with factory environmental logs (humidity, particulate count, cleaning schedules). Identify whether Dirtiness spikes correlate with specific shifts or conditions.
- Impact: If an environmental root cause is found, addressing it could eliminate Dirtiness defects entirely (55 plates in the dataset = ~2.8% of scrap).
- Priority: Medium. Timeline: 1-3 months (requires data correlation with existing environmental monitoring).

**Recommendation 5: Establish thickness-specific quality benchmarks**
- Evidence: Phase 1 median thickness analysis shows defect types cluster by thickness (K_Scatch at 40mm, Dirtiness at 100mm). SHAP confirms thickness as a top-5 feature.
- Action: Create thickness-band quality profiles. For each thickness range, define expected defect type probabilities. Plates whose defect probability significantly exceeds the baseline for their thickness band receive heightened scrutiny.
- Impact: Enables proactive quality adjustments when switching between thickness runs. Estimated 5-10% reduction in transition-related defects.
- Priority: Medium. Timeline: 3-6 months.

**Recommendation 6: Add model confidence score to quality reports**
- Evidence: The risk tier analysis shows model confidence is a reliable indicator of prediction accuracy.
- Action: Include the model's confidence score (max predicted probability) on every quality inspection report. Highlight plates below the 0.7 threshold in amber; plates below 0.4 in red.
- Impact: Gives inspectors a prioritization tool — they focus manual effort where the model is least certain, maximizing the value of human expertise.
- Priority: Low. Timeline: Immediate (software display change only).

---

## Step 6: Ethical & Operational Considerations

### Potential bias sources
- **Facility-specific bias:** The dataset may come from a single factory. Defect patterns at other facilities with different equipment, steel suppliers, or environmental conditions may differ. The model should not be deployed at a new facility without validation on local data.
- **Temporal drift:** Steel compositions, roller wear rates, and imaging system calibration change over time. The model's accuracy will degrade if not periodically retrained on recent data.
- **Class definition bias:** "Other_Faults" is a catch-all category (34.7% of data). Its heterogeneity may cause the model to overfit to specific subtypes of "Other" that were common in this dataset, while missing new subtypes.

### When human review must override
- Any plate where model confidence is below 0.4 (Low Risk tier)
- Any plate where the model predicts "Other_Faults" — since this is a catch-all, the specific defect needs human characterization
- Any plate destined for safety-critical applications (automotive, aerospace) regardless of model confidence
- When the K_Scatch detection rate spikes abnormally — this could indicate a new failure mode not in the training data

### Model monitoring strategy
- Track per-class AUC monthly on incoming labeled plates. If any class drops below X (the current test AUC minus 0.05), trigger retraining.
- Monitor the distribution of predicted probabilities. If the model starts producing more low-confidence predictions overall, the input data distribution may be shifting.
- Log all cases where human inspectors override the model. Analyze override patterns quarterly to identify systematic model weaknesses.

---

## Step 7: Business Insights Report (PDF)

Generate a 3-5 page PDF report. Structure:

**Page 1: Executive Context**
- Problem statement (2 paragraphs)
- Approach overview (1 paragraph)
- Key metric: final model AUC

**Page 2: What Drives Defects**
- SHAP summary plot (the bar chart version — cleaner for reports)
- Top 5 features explained in plain English
- One key insight per defect type (one sentence each)

**Page 3: Risk Segmentation**
- Risk tier table with counts and accuracy
- Feature profile of each tier
- Decision rule: auto-route vs human review

**Pages 4-5: Recommendations**
- All 4-6 recommendations in the template format above
- Implementation roadmap table (recommendation × priority × timeline)
- Total estimated impact summary

---

## Deliverables Checklist

- [ ] Notebook: `notebooks/phase4_explainability.ipynb`
- [ ] SHAP summary plot (global feature importance)
- [ ] SHAP bar plot (mean absolute values)
- [ ] Model-native importance vs SHAP comparison
- [ ] Top 5 features explained with manufacturing meaning
- [ ] 3 individual SHAP force/waterfall plots (TP, edge case, misclassification) with plain English explanations
- [ ] 3-tier risk segmentation with profiling
- [ ] 4-6 specific business recommendations with evidence, impact, priority, timeline
- [ ] Ethical and operational considerations section
- [ ] Business insights report PDF (3-5 pages): `reports/business_insights_report.pdf`
- [ ] Every plot and table has a business interpretation markdown cell
