# IDSS Final Course Project: Steel Plates Faults Prediction

## Course Information

| Field | Details |
|-------|---------|
| **Course Code** | AIS431 - Intelligent Decision Support Systems |
| **Instructor** | Dr. Nehal Ali |
| **Assignment Type** | Final Course Project — Team Assignment |
| **Team Size** | 2–3 students per team |
| **Duration** | Weeks 1–12 (parallel with the course) |
| **Weight** | 25% of final grade |
| **Deliverables** | Five phased submissions + final presentation |

---

## Project Overview

The Final Course Project is the capstone experience of the IDSS course. Working in teams of 2–3, you will design, build, and present a fully functional **Intelligent Decision Support System (IDSS)** applied to predicting defects in steel plates. 

The project mirrors a genuine data-science engagement: you start from raw data, move through analytics and modelling, and conclude with business-grade recommendations and a working prototype.

**Difficulty Level:** Moderate — tasks are tractable for final-year AI students yet substantial enough to demonstrate readiness for industry or graduate study.

---

## Dataset Information

### Dataset: Steel Plates Faults (UCI Machine Learning Repository)

**Source:** [Steel Plates Faults - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/198/steel%2Bplates%2Bfaults)  
**Donated:** October 25, 2010

### Dataset Overview

**Steel Plates Fault Detection:** Classify seven types of surface defects in steel plates from 27 physical measurements.

This is a **multivariate classification dataset** designed for automatic pattern recognition in manufacturing quality control. The goal is to train machine learning models to automatically detect and classify surface defects in steel plates, enabling real-time quality assurance in production environments.

### Dataset Characteristics

| Characteristic | Details |
|---|---|
| **Dataset Type** | Multivariate |
| **Subject Area** | Physics and Chemistry (Manufacturing) |
| **Associated Task** | Multi-label Binary Classification |
| **Feature Types** | Integer, Real-valued |
| **Number of Instances** | 1,941 steel plate samples |
| **Number of Features** | 27 independent variables |
| **Number of Targets** | 7 binary classification targets |

### Target Variables (7 Types of Steel Plates Faults)

Each sample can have one or more defects. The targets are:

1. **Pastry** - Pastry-like surface texture defects
2. **Z_Scratch** - Scratches in the Z-direction (perpendicular to rolling)
3. **K_Scatch** - Scratches in the K-direction (along rolling direction)
4. **Stains** - Surface stains and discoloration
5. **Dirtiness** - Dirt, dust, or contamination on surface
6. **Bumps** - Raised bumps and surface irregularities
7. **Other_Faults** - Other types of defects not fitting above categories

### Independent Variables (27 Features)

The dataset includes measurements extracted from image analysis and physical properties:

#### **Spatial & Geometric Features** (8 features)
- `X_Minimum` - Minimum X coordinate of defect region
- `X_Maximum` - Maximum X coordinate of defect region
- `Y_Minimum` - Minimum Y coordinate of defect region
- `Y_Maximum` - Maximum Y coordinate of defect region
- `Pixels_Areas` - Total number of pixels in defect area
- `X_Perimeter` - Perimeter length along X-axis
- `Y_Perimeter` - Perimeter length along Y-axis
- `Length_of_Conveyer` - Position on conveyor belt

#### **Luminosity Features** (4 features)
- `Sum_of_Luminosity` - Total luminosity (brightness) of defect region
- `Minimum_of_Luminosity` - Darkest pixel value in defect
- `Maximum_of_Luminosity` - Brightest pixel value in defect
- `Luminosity_Index` - Normalized luminosity metric

#### **Steel & Material Properties** (3 features)
- `TypeOfSteel_A300` - Binary indicator for A300 steel type
- `TypeOfSteel_A400` - Binary indicator for A400 steel type
- `Steel_Plate_Thickness` - Thickness of the steel plate

#### **Image Analysis Indices** (12 features)
- `Edges_Index` - Index measuring edge presence/intensity
- `Empty_Index` - Index measuring empty/blank areas
- `Square_Index` - Index for rectangular shape detection
- `Outside_X_Index` - Index for features outside normal X range
- `Outside_Global_Index` - Global measure of features outside norms
- `Edges_X_Index` - Edge detection along X-direction
- `Edges_Y_Index` - Edge detection along Y-direction
- `LogOfAreas` - Logarithmic transformation of pixel area
- `Log_X_Index` - Logarithmic X-based index
- `Log_Y_Index` - Logarithmic Y-based index
- `Orientation_Index` - Orientation/angle of defect
- `SigmoidOfAreas` - Sigmoid-transformed area measurement

**Note:** These 27 features are derived from digital image analysis and represent physical properties that can be automatically extracted from steel plate images or surface scanning data.

### Evaluation Metric

Submissions are evaluated based on the **AUC (Area Under the Receiver Operating Characteristic Curve)** for each of the 7 defect categories. 

**Final Score = Average AUC across all 7 defect categories**

This metric is ideal for imbalanced multi-label classification because:
- AUC is threshold-independent (good for tuning decision boundaries)
- Handles class imbalance better than accuracy
- Averages across all defect types for comprehensive assessment
- Reflects model's ability to rank predictions correctly

### Data Source & Competition Note

The dataset used in this competition was generated from a deep learning model trained on the original Steel Plates Faults dataset from UCI. While the feature distributions are close to the original dataset, they are not exactly the same. 

**Key Consideration:** Participants are encouraged to explore differences and assess whether incorporating the original dataset in training improves model performance.

### Business Context

**Manufacturing Scenario:** A steel production facility uses automated imaging systems to detect surface defects in real-time. Your IDSS will:
- Automatically classify detected defects into 7 categories
- Prioritize quality control actions based on defect type
- Enable faster decision-making on scrap vs. rework
- Reduce manual inspection costs and human error
- Improve overall production yield and efficiency

---

## Feature Engineering Opportunities

For Phase 2, consider creating new features that combine existing measurements. Here are some domain-informed ideas:

### Size & Area Features
- `Aspect_Ratio` = `X_Perimeter` / `Y_Perimeter` (detect elongated defects)
- `Defect_Compactness` = `Pixels_Areas` / (`X_Perimeter` × `Y_Perimeter`) (shape roundness)
- `Defect_Spread` = (`X_Maximum` - `X_Minimum`) × (`Y_Maximum` - `Y_Minimum`) (bounding box area)
- `Defect_Density` = `Pixels_Areas` / `Defect_Spread` (concentration of defect)

### Luminosity Features
- `Luminosity_Variance` = `Maximum_of_Luminosity` - `Minimum_of_Luminosity` (brightness range)
- `Luminosity_Ratio` = `Sum_of_Luminosity` / `Pixels_Areas` (average brightness per pixel)
- `Dark_Intensity` = `Minimum_of_Luminosity` / `Sum_of_Luminosity` (darkness concentration)

### Position Features
- `Conveyer_Position_Normalized` = `Length_of_Conveyer` / MAX_CONVEYER (normalized position)
- `X_Center` = (`X_Minimum` + `X_Maximum`) / 2 (center of defect in X)
- `Y_Center` = (`Y_Minimum` + `Y_Maximum`) / 2 (center of defect in Y)

### Edge & Shape Features
- `Edge_Strength` = `Edges_Index` × `Edges_X_Index` × `Edges_Y_Index` (combined edge measure)
- `Shape_Complexity` = `Orientation_Index` + `Square_Index` (shape irregularity)
- `Boundary_Quality` = `Edges_Index` / `Perimeter_Sum` (edge sharpness)

### Material Interaction Features
- `Steel_Type_Combined` = `TypeOfSteel_A300` + 2 × `TypeOfSteel_A400` (material encoding)
- `Thickness_to_Area` = `Steel_Plate_Thickness` × `Pixels_Areas` (thickness impact)

### Log-Scale Features (to handle skewed distributions)
- `Log_Pixels_Areas` = log(`Pixels_Areas` + 1) (handle outliers)
- `Log_Luminosity_Sum` = log(`Sum_of_Luminosity` + 1)
- Already provided: `LogOfAreas`, `Log_X_Index`, `Log_Y_Index`

### Interaction Features
- `Size_Luminosity_Interaction` = `Pixels_Areas` × `Luminosity_Index` 
- `Position_Defect_Interaction` = `Length_of_Conveyer` × `Defect_Spread`

**Recommendation:** Start with 3-5 engineered features that make business sense. Document the formula and motivation for each. Validate that they improve model performance in Phase 3.

---

## Learning Outcomes

By completing this project, each team member will be able to:

- Design an IDSS architecture suited to a specific business domain (manufacturing/quality control)
- Apply the full data science pipeline: ingestion, cleaning, EDA, feature engineering, modelling, and evaluation
- Train, compare, and select classification models using scikit-learn and XGBoost
- Interpret model behaviour using feature importance and SHAP analysis
- Translate analytical findings into actionable business recommendations (improve quality, reduce defects)
- Develop a simple interactive prototype and present it to a mixed technical/business audience

---


