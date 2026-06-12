import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "num_services",
    "charge_per_month_ratio",
    "tenure_bucket",
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
    df = df.copy()
    service_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
        "multiple_lines", "phone_service",
    ]
    df["num_services"] = df[service_cols].sum(axis=1)
    df["charge_per_month_ratio"] = (
        df["monthly_charges"] / (df["tenure_months"] + 1)
    ).round(4)
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
    ).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("scaler", StandardScaler())])
    categorical_pipe = Pipeline([
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe,     NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("bin", "passthrough",    BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = engineer_features(df)
    X = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
    y = df[TARGET]
    return X, y