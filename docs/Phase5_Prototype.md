# Phase 5: IDSS Prototype & Final Presentation

**Rubric weight:** 5 marks (prototype) + 7 marks (presentation & Q&A) = 12 marks total  
**Deliverables:** Working prototype + presentation slides + executive summary (1-page PDF)

---

## Part A: Prototype (5 marks)

### Architecture Decision: Streamlit

**Use Streamlit, not a static HTML file.** Reasons:

1. **Real model inference:** Streamlit loads the actual pickle file and runs `model.predict_proba()` on user input. A static HTML file cannot do this without embedding a JS ML runtime, which is complex, fragile, and limits model types.
2. **Real SHAP computation:** Streamlit can run SHAP on each prediction dynamically. HTML cannot.
3. **CSV upload:** Streamlit has built-in `st.file_uploader()` for CSV input. In HTML, you'd need to parse the file in JavaScript and still can't run inference.
4. **Explicitly listed as an accepted option** in the project doc.
5. **Fast to build, professional to demo:** A well-styled Streamlit app with custom CSS looks polished and takes far less development time than a custom HTML dashboard.

### Three Rubric-Required Functionalities

The project doc and rubric require the prototype to support all three:

**1. Input:** User can enter a steel plate's measurements manually OR upload a CSV row.
**2. Prediction:** Display confidence scores for all 7 defect classes + the top predicted class.
**3. Explanation:** Show which features drove the specific prediction (SHAP-based).

### Prototype Design — Streamlit App Structure

```
app.py
├── Header: "Steel Plates Fault Detection — IDSS Dashboard"
├── Sidebar: Input method selector (manual entry / CSV upload)
│   ├── Manual entry: 27 input fields with sensible defaults (use training set medians)
│   └── CSV upload: file uploader accepting .csv
├── Main panel (after prediction):
│   ├── Section 1: Prediction Result
│   │   ├── Predicted defect class (large, color-coded)
│   │   ├── Confidence score (progress bar or gauge)
│   │   └── Risk tier badge (High/Medium/Low)
│   ├── Section 2: All Class Probabilities
│   │   └── Horizontal bar chart showing probability for each of 7 classes
│   ├── Section 3: Recommended Action
│   │   └── Action text mapped from predicted class (see action mapping below)
│   ├── Section 4: Prediction Explanation (SHAP)
│   │   └── SHAP waterfall or bar plot for this specific prediction
│   └── Section 5: Feature Contributions Table
│       └── Top 5 features that influenced this prediction, with direction
```

### Action Mapping

Map each predicted defect class to a specific recommended action. These should be consistent with Phase 4 recommendations:

```python
ACTION_MAP = {
    'K_Scatch': {
        'action': 'Trigger roller maintenance inspection',
        'severity': 'high',
        'detail': 'K_Scatch defects indicate potential roller wear. Inspect rolling equipment and check for surface degradation.'
    },
    'Z_Scratch': {
        'action': 'Review A300 grade process parameters',
        'severity': 'high',
        'detail': 'Z_Scratch occurs almost exclusively on A300 steel. Check guide alignment and perpendicular tension settings.'
    },
    'Stains': {
        'action': 'Check A400 surface cleaning protocol',
        'severity': 'medium',
        'detail': 'Stains are associated with A400 steel. Verify chemical cleaning baths and rinse cycle effectiveness.'
    },
    'Bumps': {
        'action': 'Inspect material handling equipment',
        'severity': 'medium',
        'detail': 'Bumps indicate physical impact during handling. Check conveyor rollers and stacking mechanisms.'
    },
    'Dirtiness': {
        'action': 'Review environmental contamination controls',
        'severity': 'medium',
        'detail': 'Dirtiness suggests environmental contamination. Check air filtration, humidity, and cleaning schedules.'
    },
    'Pastry': {
        'action': 'Check surface preparation stage',
        'severity': 'medium',
        'detail': 'Pastry defects relate to surface texture. Review pre-rolling surface treatment and coating application.'
    },
    'Other_Faults': {
        'action': 'Manual inspection required',
        'severity': 'low',
        'detail': 'Defect does not fit standard categories. Route to senior inspector for manual classification.'
    }
}
```

### Styling

Apply custom CSS through `st.markdown()` to make the app look professional. Streamlit supports custom themes and CSS injection. Focus on:
- Clean, readable layout with clear visual hierarchy
- Color-coded severity (red for high, amber for medium, green for low/pass)
- Professional fonts and spacing
- Dark or light theme — either works, pick one and be consistent

Do NOT over-invest in styling. Functionality and usability matter more to the rubric than animations or glassmorphism.

### Implementation Notes

**Loading models:**
```python
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    pipeline = joblib.load('models/preprocessing_pipeline.pkl')
    return model, pipeline

model, pipeline = load_model()
```

**SHAP in Streamlit:**
```python
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)  # or component model
shap_values = explainer.shap_values(input_processed)

fig, ax = plt.subplots()
shap.waterfall_plot(shap.Explanation(
    values=shap_values[predicted_class_index][0],
    base_values=explainer.expected_value[predicted_class_index],
    data=input_processed.iloc[0],
    feature_names=feature_names
), show=False)
st.pyplot(fig)
```

**Handle the multi-class SHAP shape carefully.** Test the output shape and index correctly.

**Manual input defaults:** Pre-fill input fields with the training set median values so the user can immediately click "Predict" and see a result without entering all 27 values manually.

**Sample CSV:** Include a `prototype/sample_input.csv` file with 3-5 example rows that users can download and re-upload to test the system.

### Prototype requirements.txt

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
shap>=0.42.0
matplotlib>=3.6.0
joblib>=1.2.0
```

### Testing the Prototype

Before considering the prototype done, verify:
- [ ] Manual entry with default values produces a valid prediction
- [ ] CSV upload with sample_input.csv works and shows predictions
- [ ] SHAP plot renders without errors
- [ ] All 7 defect classes can be predicted (test with different inputs)
- [ ] Risk tier badge displays correctly
- [ ] Recommended action displays correctly
- [ ] App loads in under 5 seconds
- [ ] No error tracebacks visible to the user

---

## Part B: Final Presentation (7 marks)

### Assessor note (verbatim from rubric):
*"Strong presentations should show a clear line from problem → data → model → insight → recommendation → decision support value."*

Every slide must advance this narrative. No slide should exist that doesn't connect to the next.

### Slide Structure (15 minutes total)

**Slides 1-2: Problem & Business Context (2 min)**
- What: Steel plate defect detection in manufacturing
- Who cares: Quality control engineers, plant managers, operations directors
- Why it matters: Quantify the cost of undetected defects — scrap costs, rework costs, customer complaints, production delays
- What we built: An IDSS that automatically classifies defects and recommends corrective actions

**Slides 3-4: Dataset & Key EDA Findings (2 min)**
- Dataset overview: 1,941 plates, 27 measurements, 7 defect types
- Class imbalance visualization: show the bar chart
- **Headline finding:** Steel grade (A300 vs A400) is the single strongest predictor — show the steel grade × defect type cross-tabulation visual from Phase 1. This is your most striking visual.
- Thickness patterns: different defects cluster at different thickness ranges

**Slides 5-6: Model Results (3 min)**
- Model comparison: bar chart of macro AUC for all 5 models (LR, RF, XGB, LGBM, Ensemble)
- Show the ensemble wins — but not by a huge margin, which is typical and honest
- Confusion matrix for the ensemble — highlight the most common misclassification and explain its business impact in one sentence
- Per-class AUC: show which defect types the model handles best/worst

**Slides 7-8: SHAP Insights (2 min)**
- SHAP summary bar plot: top 10 features
- Pick ONE individual SHAP force plot — the most interesting one (the misclassification or the edge case)
- Translate: "These are the 5 things that most determine which defect type a plate has: [list them in plain language]"

**Slides 9-10: Business Recommendations (3 min)**
- Present the top 3-4 recommendations with their estimated impact
- Show the implementation roadmap as a simple timeline or priority matrix
- This is the payoff slide — the entire presentation builds to this. Spend time here.

**Slide 11: Live Prototype Demo (2-3 min)**
- Switch to the running Streamlit app
- Show manual input → prediction → SHAP explanation → recommended action
- Upload the sample CSV → show batch results
- Point out the risk tier and action text

**Slide 12: Ethical Considerations & Limitations (1 min)**
- Single-facility data bias
- Need for periodic retraining
- When human override is necessary
- "Other_Faults" catch-all limitation

**Slide 13: Conclusion & Next Steps (30 sec)**
- Summary: "We built an IDSS that classifies 7 defect types with X% macro AUC and provides actionable inspection recommendations"
- Next steps: deploy as inline quality gate, validate on second facility, add real-time monitoring dashboard

### Presentation Tips
- Use visuals, not text walls. Maximum 5 bullet points per slide. Most slides should be a chart + 1-2 sentences.
- Practice the demo beforehand — have a backup screenshot in case the live demo fails.
- Anticipate Q&A questions: "Why not a neural network?", "How often would you retrain?", "What happens when the model is wrong?", "Can this work at a different factory?"

---

## Part C: Executive Summary (1-page PDF)

### Audience
Non-technical business stakeholders — plant managers, operations directors, executives. Zero jargon.

### Structure

**Title:** Steel Plate Defect Detection: Intelligent Decision Support System

**Section 1: The Problem (2-3 sentences)**
Surface defects in steel plates lead to scrap, rework, and customer complaints. Manual inspection is slow and inconsistent. An automated system can classify defects faster and more reliably.

**Section 2: What We Built (2-3 sentences)**
We developed a machine learning system that analyzes 27 measurements from each steel plate and predicts which of 7 defect types (if any) is present. The system achieves [X]% accuracy in ranking defective plates and provides specific corrective action recommendations.

**Section 3: Key Findings (3-4 bullet points)**
- Steel grade is the strongest predictor: A400 steel is associated with K_Scatch and Stains; A300 with Z_Scratch
- The model correctly routes [X]% of plates to the right inspection queue without human review
- [Most impactful finding from SHAP]
- [Second most impactful finding]

**Section 4: Top 3 Recommendations**
1. Implement grade-specific inspection checklists → estimated 40% reduction in inspection time
2. Automate first-pass defect routing for high-confidence predictions → reduces manual inspection by X%
3. Set up roller maintenance alerts based on K_Scatch detection rate → estimated 15-25% reduction in scratch defects

**Section 5: Next Steps**
Pilot deployment on one production line for 2-week validation, then expand facility-wide.

**Include one visual:** Either the class distribution bar chart or the model's confusion matrix — whichever is more impactful in one small figure.

**Format:** 12pt font, 1-inch margins, fits on one page.

---

## Deliverables Checklist

- [ ] `prototype/app.py` — working Streamlit app with all 3 required functionalities
- [ ] `prototype/requirements.txt` — dependencies
- [ ] `prototype/sample_input.csv` — example input rows for demo
- [ ] Prototype tested: manual input, CSV upload, SHAP display all work
- [ ] `presentation/final_presentation.pptx` (or PDF) — 13 slides, 15-minute narrative
- [ ] `reports/executive_summary.pdf` — 1 page, plain language, 3 recommendations
- [ ] Demo rehearsed with backup screenshots
