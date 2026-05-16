# Phase 2: Data Preparation & Feature Engineering

**Rubric weight:** 3 marks  
**Criteria:** Data quality handling (1.5) · Feature engineering quality (1.5) · Business rationale & reproducibility (1.0)  
**Deliverables:** Updated notebook + X_train.csv, X_test.csv, y_train.csv, y_test.csv + data dictionary

---

## Step 1: Load Data and Confirm Phase 1 Findings

Load the dataset using the same method as Phase 1. Immediately verify the Phase 1 findings still hold:

```
- Shape: (1941, 27) features + 7 binary target columns
- Missing values: 0
- Duplicates: 0
- Target: each row sums to exactly 1 across the 7 target columns
```

Convert the 7 binary target columns to a single class label column using `y.idxmax(axis=1)`. This is your target variable for the rest of the project.

**Markdown cell after this step:** State that the data quality is confirmed clean from Phase 1. Explain that having no missing values is unusual and beneficial — it means the imaging system that captured these measurements is reliable, which supports confidence in model deployment.

---

## Step 2: Data Quality Handling

### 2a. Drop the redundant steel type column

TypeOfSteel_A300 and TypeOfSteel_A400 are perfectly inverse (correlation = -1.0). Keep `TypeOfSteel_A400` and drop `TypeOfSteel_A300`. Reason: A400 is the more common steel type in the dataset and retains the same information.

**Business rationale markdown cell:** "Since a plate is either A300 or A400 steel, one column encodes the same information as both. Dropping one avoids multicollinearity issues in Logistic Regression without losing any discriminative power. We keep A400 as it is the majority type."

### 2b. Outlier strategy

Phase 1 identified heavy right skew with extreme outliers in Pixels_Areas, X_Perimeter, Y_Perimeter, and Sum_of_Luminosity.

**Decision: Retain all outliers.** Justification:
- The primary models (Random Forest, XGBoost, LightGBM) are tree-based and split on rank order, making them inherently robust to outliers.
- These extreme values likely represent genuinely large defects (K_Scatch has visibly higher Pixels_Areas), so they carry real signal.
- For Logistic Regression (the baseline), outlier impact will be mitigated by StandardScaler, which centers data but doesn't clip. If LR performance is poor due to outlier sensitivity, note this in the evaluation — it reinforces why tree-based models are better for this domain.

**Business rationale markdown cell:** "In steel manufacturing, extreme defect sizes are not data errors — they represent severe faults that are most important to detect. Removing them would bias the model against catching the worst defects, which is the opposite of what quality control needs."

### 2c. Data type verification

Confirm all 27 features are numeric (14 int64 + 13 float64). No categorical encoding needed beyond the binary steel type already present.

---

## Step 3: Feature Engineering

Create new features that capture domain-meaningful relationships. The rubric requires "at least 3" but aim for 5-7 well-justified features. Each feature must have a formula, a rationale, and a validation check.

### Recommended Engineered Features

**Feature 1: `Defect_Area_Ratio`**
```python
X['Defect_Area_Ratio'] = X['Pixels_Areas'] / ((X['X_Maximum'] - X['X_Minimum']) * (X['Y_Maximum'] - X['Y_Minimum']) + 1)
```
- **What it captures:** How densely the defect fills its bounding box. A ratio near 1.0 means compact/round defect; near 0 means sparse/irregular.
- **Business rationale:** Compact defects (like Bumps) have different root causes than sprawling ones (like K_Scatch). This helps the model distinguish defect morphology, which maps to different corrective actions on the production line.

**Feature 2: `Luminosity_Range`**
```python
X['Luminosity_Range'] = X['Maximum_of_Luminosity'] - X['Minimum_of_Luminosity']
```
- **What it captures:** The contrast between the brightest and darkest pixels within the defect region.
- **Business rationale:** High contrast defects (Stains, Dirtiness) look different under inspection cameras than uniform-brightness defects (Bumps). This feature directly captures what a human inspector would see as "how visible is this defect."

**Feature 3: `Aspect_Ratio`**
```python
X['Aspect_Ratio'] = X['X_Perimeter'] / (X['Y_Perimeter'] + 1e-6)
```
- **What it captures:** Whether the defect is elongated horizontally vs vertically. Scratches (K_Scatch, Z_Scratch) will have extreme aspect ratios; Bumps will be closer to 1.0.
- **Business rationale:** Scratches in the rolling direction (K_Scatch) vs perpendicular (Z_Scratch) point to different equipment issues — worn rollers vs guide misalignment. The aspect ratio encodes this directionality.

**Feature 4: `Log_Pixels_Areas`**
```python
import numpy as np
X['Log_Pixels_Areas'] = np.log1p(X['Pixels_Areas'])
```
- **What it captures:** Log-transformed defect size, compressing the extreme right skew.
- **Business rationale:** Defect sizes span several orders of magnitude. The log transform makes the relationship between size and defect type more linear, which improves Logistic Regression performance and makes feature importance more interpretable (a unit increase in log-area represents a proportional, not absolute, size change).

**Feature 5: `Edge_Strength`**
```python
X['Edge_Strength'] = X['Edges_Index'] * X['Edges_X_Index'] * X['Edges_Y_Index']
```
- **What it captures:** Combined edge intensity across both axes. High values mean sharply defined defect boundaries.
- **Business rationale:** Sharp-edged defects (scratches, cuts) indicate mechanical damage, while diffuse-edged defects (stains, dirtiness) indicate chemical or environmental contamination. This distinction drives whether the corrective action is mechanical maintenance vs process environment cleanup.

**Feature 6: `Thickness_Area_Interaction`**
```python
X['Thickness_Area_Interaction'] = X['Steel_Plate_Thickness'] * X['Log_Pixels_Areas']
```
- **What it captures:** How defect size relates to plate thickness. Thin plates with large defects are more concerning.
- **Business rationale:** Phase 1 found that thickness varies by defect type (K_Scatch at 40mm median vs Dirtiness at 100mm). This interaction captures whether certain defects are thickness-dependent, which informs thickness-specific inspection protocols.

### Feature Validation Strategy

After creating all engineered features, validate them:

1. **Quick correlation check:** Compute correlation between each new feature and the target (encoded as integers). Features with near-zero correlation across all classes may not add value.
2. **Mutual information score:** Use `sklearn.feature_selection.mutual_info_classif` to rank all features (original + engineered) by information gain. Verify at least some engineered features appear in the top 15.
3. **Final validation comes in Phase 3:** Compare model AUC with and without engineered features. If they don't improve AUC, document this honestly and explain why (the rubric rewards honesty over inflated claims).

---

## Step 4: Scaling

### Strategy: Apply StandardScaler to all numeric features

**Why StandardScaler and not MinMaxScaler:**
- StandardScaler (zero mean, unit variance) is preferred when features have different ranges and some have outliers. MinMaxScaler would compress the majority of values into a tiny range for skewed features like Pixels_Areas.
- StandardScaler is the standard choice for Logistic Regression and doesn't hurt tree-based models (trees are scale-invariant, so scaling is harmless but ensures the pipeline is consistent).
- Since we're building a shared preprocessing pipeline used in the prototype, using one scaler for everything keeps the system simple and predictable.

**Implementation approach:**
- Fit the scaler on `X_train` only. Transform both `X_train` and `X_test`. This prevents data leakage.
- Save the fitted scaler as `models/preprocessing_pipeline.pkl` (or bundle it with feature engineering into a single sklearn Pipeline).

**Business rationale markdown cell:** "Scaling ensures that features with large absolute ranges (e.g., Pixels_Areas up to 200,000) don't dominate distance-based calculations in Logistic Regression. Tree-based models ignore scaling, but applying it uniformly keeps the preprocessing pipeline consistent across all models and simplifies the prototype deployment."

### Important: Scaling + Feature Engineering as a Pipeline

Build a reusable preprocessing function or sklearn Pipeline that does:
1. Drop TypeOfSteel_A300
2. Create engineered features
3. Apply StandardScaler

Save this pipeline object so the prototype can apply identical transformations to new data. The pipeline must be fitted on training data only.

```python
# Pseudocode for the pipeline concept
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

def add_engineered_features(X):
    X = X.copy()
    X['Defect_Area_Ratio'] = ...
    X['Luminosity_Range'] = ...
    # ... all features
    return X

preprocessing = Pipeline([
    ('feature_eng', FunctionTransformer(add_engineered_features)),
    ('scaler', StandardScaler())
])

preprocessing.fit(X_train_raw)
X_train = preprocessing.transform(X_train_raw)
X_test = preprocessing.transform(X_test_raw)

joblib.dump(preprocessing, 'models/preprocessing_pipeline.pkl')
```

---

## Step 5: Train/Test Split

### Configuration
- **Split ratio:** 80% train, 20% test
- **Stratification:** Stratify by the target class label to preserve class proportions
- **Random state:** 42 for reproducibility

### Implementation
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_single,
    test_size=0.2,
    random_state=42,
    stratify=y_single
)
```

### Verification
After splitting, print the class distribution in both train and test sets as percentages. Confirm they match the overall distribution within ±1%.

### Important: Split BEFORE scaling
The split must happen before fitting the scaler. Otherwise you leak test set statistics into the training data.

Correct order:
1. Create engineered features on the full dataset (no fitting involved, just formulas)
2. Split into train/test
3. Fit scaler on X_train only
4. Transform both X_train and X_test with the fitted scaler
5. Save CSVs

### Save outputs
```python
X_train_df.to_csv('data/processed/X_train.csv', index=False)
X_test_df.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
```

---

## Step 6: Data Dictionary

Create `data/data_dictionary.md` as a markdown table listing every feature (original + engineered) with:

| Column | Type | Range | Source | Business Meaning |
|--------|------|-------|--------|-----------------|
| X_Minimum | int | 0-1700 | Original | Min X coordinate of defect region on the plate |
| ... | ... | ... | ... | ... |
| Defect_Area_Ratio | float | 0-1 | Engineered | How densely the defect fills its bounding box |
| defect_type | str | 7 classes | Target | The fault type to predict |

Include a note at the top stating: "Features are StandardScaler-transformed in X_train.csv and X_test.csv. Raw ranges shown are pre-scaling."

---

## Deliverables Checklist

- [ ] Notebook: `notebooks/phase2_data_prep.ipynb` with all steps, code, outputs, and business rationale markdown cells
- [ ] `data/processed/X_train.csv` — scaled training features
- [ ] `data/processed/X_test.csv` — scaled test features
- [ ] `data/processed/y_train.csv` — training labels (single class strings)
- [ ] `data/processed/y_test.csv` — test labels (single class strings)
- [ ] `models/preprocessing_pipeline.pkl` — fitted pipeline (feature eng + scaler)
- [ ] `data/data_dictionary.md` — complete feature documentation
- [ ] Every major step has a markdown cell explaining the business rationale
- [ ] Class distribution verification printed for both train and test sets
