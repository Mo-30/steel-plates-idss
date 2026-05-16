import numpy as np
import pandas as pd


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Drop TypeOfSteel_A300 and add 6 domain-informed engineered features.
    Accepts the original 27-column DataFrame; returns 32-column DataFrame.
    This is the single source of truth for raw -> engineered transformation.
    """
    X = X.copy()

    # Drop redundant column (perfectly inversely correlated with TypeOfSteel_A400)
    if "TypeOfSteel_A300" in X.columns:
        X = X.drop(columns=["TypeOfSteel_A300"])

    # 1. Defect_Area_Ratio: how densely the defect fills its bounding box
    #    Near 1.0 = compact/round; near 0 = sparse/irregular
    bounding_box = (X["X_Maximum"] - X["X_Minimum"]) * (X["Y_Maximum"] - X["Y_Minimum"]) + 1
    X["Defect_Area_Ratio"] = X["Pixels_Areas"] / bounding_box

    # 2. Luminosity_Range: contrast between brightest and darkest defect pixels
    X["Luminosity_Range"] = X["Maximum_of_Luminosity"] - X["Minimum_of_Luminosity"]

    # 3. Aspect_Ratio: elongation of defect (horizontal vs vertical)
    #    Scratches will have extreme ratios; Bumps near 1.0
    X["Aspect_Ratio"] = X["X_Perimeter"] / (X["Y_Perimeter"] + 1e-6)

    # 4. Log_Pixels_Areas: log-transform to compress right skew in defect size
    X["Log_Pixels_Areas"] = np.log1p(X["Pixels_Areas"])

    # 5. Edge_Strength: combined edge intensity across both axes
    #    High = sharp mechanical damage; low = diffuse chemical contamination
    X["Edge_Strength"] = X["Edges_Index"] * X["Edges_X_Index"] * X["Edges_Y_Index"]

    # 6. Thickness_Area_Interaction: captures thickness-dependent defect severity
    X["Thickness_Area_Interaction"] = X["Steel_Plate_Thickness"] * X["Log_Pixels_Areas"]

    return X
