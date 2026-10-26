"""Tests for the saved pipeline and the API.

Run `python -m churn.train` first (CI runs a quick training into a temp folder, see ci.yml).
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "churn_pipeline.joblib").exists(),
                                reason="train the model first")

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0, "TotalCharges": 190.0,
}
LOW_RISK = {**HIGH_RISK, "tenure": 70, "Contract": "Two year", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "TechSupport": "Yes", "PaymentMethod": "Credit card (automatic)",
            "PaperlessBilling": "No", "MonthlyCharges": 60.0, "TotalCharges": 4200.0}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["PREDICTION_LOG"] = str(tmp_path_factory.mktemp("logs") / "pred.jsonl")
    from churn import api
    api.LOG_PATH = Path(os.environ["PREDICTION_LOG"])
    with TestClient(api.app) as c:
        yield c


def test_model_card_is_complete():
    card = json.loads((MODEL_DIR / "model_card.json").read_text())
    for key in ("data_sha256", "chosen_model", "decision_threshold", "test", "reference_stats"):
        assert key in card
    assert 0 < card["decision_threshold"] < 1
    assert card["test"]["roc_auc"] > 0.80          # guard against a broken retrain


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_risk_ordering_makes_sense(client):
    high = client.post("/predict", json=HIGH_RISK).json()
    low = client.post("/predict", json=LOW_RISK).json()
    assert high["churn_probability"] > low["churn_probability"]
    assert high["at_risk"] is True and low["at_risk"] is False


def test_bad_input_returns_422(client):
    assert client.post("/predict", json={**HIGH_RISK, "Contract": "Three year"}).status_code == 422
    assert client.post("/predict", json={**HIGH_RISK, "tenure": -1}).status_code == 422
    missing = {k: v for k, v in HIGH_RISK.items() if k != "tenure"}
    assert client.post("/predict", json=missing).status_code == 422


def test_batch_and_stats(client):
    r = client.post("/predict/batch", json={"customers": [HIGH_RISK, LOW_RISK] * 20})
    assert r.status_code == 200 and len(r.json()) == 40
    s = client.get("/stats").json()
    assert s["predictions"] >= 40
    assert "tenure" in s["numeric_drift"]


def test_drift_alert_fires_on_shifted_inputs(client):
    shifted = {**LOW_RISK, "tenure": 72, "TotalCharges": 8000.0}
    client.post("/predict/batch", json={"customers": [shifted] * 500})
    drift = client.get("/stats").json()["numeric_drift"]
    assert drift["tenure"]["alert"] is True


def test_batch_limit(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
