"""Preprocessing and model candidates, all wrapped in scikit-learn Pipelines.

Why a Pipeline: in the old notebook the scaler, the one-hot columns and the model were saved
as three separate files, and the Streamlit app had to repeat the encoding by hand. A Pipeline
keeps them together, so (1) the scaler is re-fitted inside every cross-validation fold
(no information from the validation fold leaks into training), and (2) one saved file is all
that is needed to make a prediction.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import BINARY_INT_COLS, CATEGORICAL_COLS, NUMERIC_COLS

SEED = 42


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_COLS),
            ("bin", "passthrough", BINARY_INT_COLS),
            # drop="if_binary": Yes/No columns become one 0/1 column; handle_unknown keeps
            # serving safe if a category is ever missing from training data
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,
    )


def candidates() -> dict:
    """Each candidate = (pipeline, hyper-parameter grid searched with 5-fold CV).

    No class weights: class imbalance (26.5% churners) is handled by tuning the decision
    threshold instead (see train.py). Re-weighting classes pushes predicted probabilities up
    (the v1 model predicted 68% risk for a group whose real churn rate was 52.5%), and we need
    honest probabilities to estimate revenue at risk.
    """
    return {
        "baseline_prior": (
            Pipeline([("prep", make_preprocessor()),
                      ("model", DummyClassifier(strategy="prior"))]),
            {},
        ),
        "logistic_regression": (
            Pipeline([("prep", make_preprocessor()),
                      ("model", LogisticRegression(max_iter=2000))]),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            Pipeline([("prep", make_preprocessor(scale_numeric=False)),
                      ("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
            {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [1, 5, 20]},
        ),
        "xgboost": (
            Pipeline([("prep", make_preprocessor(scale_numeric=False)),
                      ("model", XGBClassifier(n_estimators=300, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8,
                                              eval_metric="logloss",
                                              random_state=SEED, n_jobs=2, tree_method="hist"))]),
            {"model__max_depth": [2, 3, 4], "model__min_child_weight": [1, 5],
             "model__reg_lambda": [1, 5]},
        ),
    }
