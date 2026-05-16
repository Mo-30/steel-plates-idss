# Data Dictionary — Steel Plates Faults Dataset

> **Note:** Features are StandardScaler-transformed in `X_train.csv` and `X_test.csv`. Ranges shown below are **pre-scaling** (original measurement units). The `defect_type` column appears only in `y_train.csv` and `y_test.csv`.

---

## Target Variable

| Column | Type | Values | Source | Business Meaning |
|--------|------|--------|--------|-----------------|
| `defect_type` | str | 7 classes (see below) | Derived | Single defect classification for each steel plate |

**Defect classes and frequencies (full dataset, n=1941):**

| Class | Count | % | Description |
|-------|-------|---|-------------|
| Other_Faults | 673 | 34.7% | Defects not fitting the 6 specific categories — heterogeneous catch-all |
| Bumps | 402 | 20.7% | Raised bumps and surface irregularities from handling/stacking |
| K_Scatch | 391 | 20.1% | Scratches along the rolling direction — indicate worn rollers |
| Z_Scratch | 190 | 9.8% | Scratches perpendicular to rolling — indicate guide misalignment |
| Pastry | 158 | 8.1% | Pastry-like surface texture from surface preparation issues |
| Stains | 72 | 3.7% | Surface stains/discoloration — chemical contamination on A400 steel |
| Dirtiness | 55 | 2.8% | Dirt/dust contamination — environmental control failure |

---

## Original Features (27 columns)

### Spatial & Geometric Features (8)

| Column | Type | Range (pre-scale) | Source | Business Meaning |
|--------|------|-------------------|--------|-----------------|
| `X_Minimum` | int | 0–1654 | Original | Min X coordinate of defect region bounding box |
| `X_Maximum` | int | 0–1654 | Original | Max X coordinate of defect region bounding box |
| `Y_Minimum` | int | 0–14980 | Original | Min Y coordinate of defect region bounding box |
| `Y_Maximum` | int | 0–14980 | Original | Max Y coordinate of defect region bounding box |
| `Pixels_Areas` | int | 0–152000 | Original | Total pixel count of defect area — proxy for defect severity |
| `X_Perimeter` | int | 0–1654 | Original | Perimeter length along X-axis |
| `Y_Perimeter` | int | 0–14980 | Original | Perimeter length along Y-axis |
| `Length_of_Conveyer` | int | 1227–2438 | Original | Position on conveyor belt — indicates production batch |

### Luminosity Features (4)

| Column | Type | Range (pre-scale) | Source | Business Meaning |
|--------|------|-------------------|--------|-----------------|
| `Sum_of_Luminosity` | float | 0–1.8M | Original | Total brightness of defect region |
| `Minimum_of_Luminosity` | int | 0–253 | Original | Darkest pixel value in defect — measures shadow depth |
| `Maximum_of_Luminosity` | int | 0–255 | Original | Brightest pixel value in defect — measures highlight intensity |
| `Luminosity_Index` | float | −0.38–0.50 | Original | Normalized luminosity metric — negative = darker than background |

### Steel & Material Properties (3)

| Column | Type | Range (pre-scale) | Source | Business Meaning |
|--------|------|-------------------|--------|-----------------|
| `TypeOfSteel_A400` | int | 0 or 1 | Original | Binary: 1 = A400 grade steel (kept); A400 associated with K_Scatch and Stains |
| ~~`TypeOfSteel_A300`~~ | ~~int~~ | ~~Dropped~~ | ~~Original~~ | ~~Perfectly inverse of A400 — dropped to avoid multicollinearity~~ |
| `Steel_Plate_Thickness` | int | 40–300 | Original | Plate gauge in mm — strong predictor; thinner plates show K_Scatch |

### Image Analysis Indices (12)

| Column | Type | Range (pre-scale) | Source | Business Meaning |
|--------|------|-------------------|--------|-----------------|
| `Edges_Index` | float | 0–1 | Original | Overall edge presence/intensity in defect region |
| `Empty_Index` | float | 0–1 | Original | Proportion of blank/empty area within bounding box |
| `Square_Index` | float | 0–1 | Original | Measure of rectangular shape — how square-like the defect is |
| `Outside_X_Index` | float | 0–1 | Original | Features outside normal X range |
| `Outside_Global_Index` | float | 0–1 | Original | Global measure of features outside norms |
| `Edges_X_Index` | float | 0–1 | Original | Edge detection along X-direction |
| `Edges_Y_Index` | float | 0–1 | Original | Edge detection along Y-direction |
| `LogOfAreas` | float | 0–12 | Original | Log-transformed pixel area (provided pre-computed) |
| `Log_X_Index` | float | 0–10 | Original | Logarithmic X-based index |
| `Log_Y_Index` | float | 0–10 | Original | Logarithmic Y-based index |
| `Orientation_Index` | float | −1–1 | Original | Orientation/angle of defect — captures directionality |
| `SigmoidOfAreas` | float | 0–1 | Original | Sigmoid-transformed area — compresses large values |

---

## Engineered Features (6 new columns)

| Column | Type | Range (pre-scale) | Formula | Business Meaning |
|--------|------|-------------------|---------|-----------------|
| `Defect_Area_Ratio` | float | 0–1 | `Pixels_Areas / ((X_Max - X_Min) * (Y_Max - Y_Min) + 1)` | Density of defect within bounding box. Near 1 = compact (Bumps); near 0 = sparse/elongated (scratches). Distinguishes defect morphology → different corrective actions. |
| `Luminosity_Range` | float | 0–255 | `Maximum_of_Luminosity - Minimum_of_Luminosity` | Brightness contrast of defect. High = stains/dirtiness; low = uniform surface defects. Maps to chemical vs mechanical root cause. |
| `Aspect_Ratio` | float | 0–∞ | `X_Perimeter / (Y_Perimeter + 1e-6)` | Elongation: horizontal vs vertical. Extreme values signal rolling-direction scratches (K_Scatch); near 1.0 signals rounded defects (Bumps). Encodes defect directionality. |
| `Log_Pixels_Areas` | float | 0–12 | `log(Pixels_Areas + 1)` | Log-scaled defect size. Compresses heavy right tail. Makes size comparisons proportional — improves LR performance and SHAP interpretability. |
| `Edge_Strength` | float | 0–1 | `Edges_Index * Edges_X_Index * Edges_Y_Index` | Combined edge sharpness in both directions. High = sharp mechanical damage (scratches); low = diffuse chemical contamination (stains). Maps directly to maintenance vs process-environment corrective action. |
| `Thickness_Area_Interaction` | float | 0–∞ | `Steel_Plate_Thickness * Log_Pixels_Areas` | Interaction between plate gauge and defect size. Captures whether defects are thickness-dependent (K_Scatch at thin 40mm; Dirtiness at thick 100mm). Enables thickness-specific inspection thresholds. |

---

## Notes

- All 32 feature columns in `X_train.csv` and `X_test.csv` are **StandardScaler-transformed** (mean=0, std=1 based on training set statistics).
- `TypeOfSteel_A300` has been **dropped** from all processed files. The pipeline receives all 27 original columns and drops it internally.
- The pipeline object (`models/preprocessing_pipeline.pkl`) applies both the feature engineering and scaling in a single `sklearn.Pipeline` call, ensuring identical transformations in training, evaluation, and the production prototype.
- Class imbalance ratio: **12:1** (Other_Faults vs Dirtiness). All models use `class_weight='balanced'` to compensate.
