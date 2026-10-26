import pandas as pd
import pytest

from churn.data import CATEGORY_VALUES, FEATURES, clean, load, validate

DATA = "Telco-Customer-Churn.csv"


def test_load_shape_and_target():
    df = load(DATA)
    assert df.shape[0] == 7043
    assert set(df["Churn"].unique()) == {0, 1}
    assert round(df["Churn"].mean(), 3) == 0.265


def test_blank_total_charges_become_zero_for_new_customers():
    raw = pd.read_csv(DATA)
    blanks = pd.to_numeric(raw["TotalCharges"], errors="coerce").isna()
    assert blanks.sum() == 11
    assert (raw.loc[blanks, "tenure"] == 0).all()
    assert (clean(raw).loc[blanks, "TotalCharges"] == 0).all()


def test_category_lists_match_data():
    df = load(DATA)
    for col, allowed in CATEGORY_VALUES.items():
        assert set(df[col].unique()) == set(allowed), col


def test_validate_rejects_unknown_category():
    df = load(DATA).head(5).copy()
    df.loc[df.index[0], "Contract"] = "Three year"
    with pytest.raises(ValueError, match="Contract"):
        validate(df)


def test_validate_rejects_missing_column():
    df = load(DATA).head(5).drop(columns=["tenure"])
    with pytest.raises(ValueError, match="tenure"):
        validate(df)


def test_features_exclude_id_and_target():
    assert "customerID" not in FEATURES and "Churn" not in FEATURES
