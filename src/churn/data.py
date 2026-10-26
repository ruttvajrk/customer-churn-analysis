"""Load, clean and validate the IBM Telco churn data.

Every step here was done by hand in the notebook. Moving it into one function
means training, the API and the tests all use exactly the same cleaning.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_INT_COLS = ["SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_COLS + BINARY_INT_COLS + CATEGORICAL_COLS

# Allowed values, taken from the dataset. The API uses the same lists to reject bad input.
CATEGORY_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check",
    ],
}


def file_sha256(path: str | Path) -> str:
    """Hash of the raw data file, stored with the model so every model can be traced to its data."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning as the original notebook.

    - TotalCharges is text with 11 blanks; all 11 are customers with tenure 0, so blank = 0.
    - Churn Yes/No -> 1/0 (only if the column is present, so the API can reuse this).
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    """Fail loudly if the data does not look like the data the model was built for."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if require_target:
        if TARGET not in df.columns:
            raise ValueError("Missing target column 'Churn'")
        bad = set(df[TARGET].dropna().unique()) - {0, 1}
        if bad or df[TARGET].isna().any():
            raise ValueError(f"Target must be 0/1, found {bad or 'NaN'}")
    for col, allowed in CATEGORY_VALUES.items():
        unknown = set(df[col].unique()) - set(allowed)
        if unknown:
            raise ValueError(f"Unexpected values in {col}: {sorted(unknown)}")
    if (df[NUMERIC_COLS] < 0).any().any():
        raise ValueError("Negative values in numeric columns")
    if df[NUMERIC_COLS].isna().any().any():
        raise ValueError("Missing values in numeric columns after cleaning")


def load(path: str | Path) -> pd.DataFrame:
    df = clean(pd.read_csv(path))
    validate(df)
    return df
