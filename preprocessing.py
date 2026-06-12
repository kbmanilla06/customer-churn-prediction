"""
preprocessing.py
----------------
All feature engineering and preprocessing logic.
Keeps the pipeline file clean — import build_preprocessor() and
engineer_features() from here.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ── Column groups ─────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "num_services",           # engineered
    "charge_per_month_ratio", # engineered
    "tenure_bucket",          # engineered (ordinal numeric)
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
    "payment_method",
]

BINARY_FEATURES = [
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
]

TARGET = "churn"
DROP_COLS = ["customer_id"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model signal."""
    df = df.copy()

    # Count how many add-on services the customer has (0–8)
    service_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
        "multiple_lines", "phone_service",
    ]
    df["num_services"] = df[service_cols].sum(axis=1)

    # Average monthly charge per unit of tenure (cost intensity)
    df["charge_per_month_ratio"] = (
        df["monthly_charges"] / (df["tenure_months"] + 1)
    ).round(4)

    # Ordinal tenure bucket (0=new, 1=growing, 2=established, 3=loyal)
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
    ).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a fitted-ready ColumnTransformer:
      - Numeric: StandardScaler
      - Categorical: OneHotEncoder (drop first to avoid multicollinearity)
      - Binary: pass-through (already 0/1)
    """
    numeric_pipe = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipe,    NUMERIC_FEATURES),
            ("cat",  categorical_pipe, CATEGORICAL_FEATURES),
            ("bin",  "passthrough",   BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, engineer features, and return (X, y)."""
    df = pd.read_csv(path)
    df = engineer_features(df)
    X = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
    y = df[TARGET]
    return X, y
