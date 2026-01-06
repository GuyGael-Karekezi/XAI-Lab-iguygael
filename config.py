import os
import pandas as pd

# Paths
DATA_PATH = "data/credit_dataset2.csv"
MODEL_DIR = "model"
VIZ_DIR = "visualizations"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_credit_model2.pkl")
RANDOM_STATE = 42

# Data schema
TARGET = "Risk"
PROTECTED_ATTR = "Sex"

# Categorical and numerical columns
CATEGORICAL_COLS = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Purpose",
]

NUMERICAL_COLS = [
    "Age",
    "Job",
    "Checking account",
    "Credit amount",
    "Duration",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    validate_schema(df)
    return df


def encode_target(y: pd.Series) -> pd.Series:
    return y.map({"Good": 1, "Bad": 0})


def validate_schema(df: pd.DataFrame) -> None:
    required = set([TARGET] + CATEGORICAL_COLS + NUMERICAL_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")