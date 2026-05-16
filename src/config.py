from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "defect_type"

DEFECT_CLASSES = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults"
]

ORIGINAL_DROP_COLS = ["TypeOfSteel_A300"]
ENGINEERED_FEATURES = [
    "Defect_Area_Ratio", "Luminosity_Range", "Aspect_Ratio",
    "Log_Pixels_Areas", "Edge_Strength", "Thickness_Area_Interaction"
]
