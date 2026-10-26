"""FastAPI service that serves the saved churn pipeline.

    uvicorn churn.api:app --port 8000      (docs at http://localhost:8000/docs)

Endpoints
  GET  /health          model version, model type, decision threshold
  POST /predict         one customer  -> churn probability, at_risk flag, top 3 reasons
  POST /predict/batch   up to 1,000 customers
  GET  /stats           simple monitoring: request count, latency, average predicted risk,
                        and input drift compared with the training data
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from .data import FEATURES, NUMERIC_COLS

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
LOG_PATH = Path(os.getenv("PREDICTION_LOG", "logs/predictions.jsonl"))
MAX_BATCH = 1000
DRIFT_MIN_ROWS = 30          # do not report drift on a handful of requests
DRIFT_ALERT_STD = 0.5        # mean moved by more than half a training std -> flag it

log = logging.getLogger("churn.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

YesNo = Literal["Yes", "No"]
InternetDep = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer. Allowed values match the training data, so bad input gets a clear 422."""
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetDep
    OnlineBackup: InternetDep
    DeviceProtection: InternetDep
    TechSupport: InternetDep
    StreamingTV: InternetDep
    StreamingMovies: InternetDep
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 256.5}}}


class Batch(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    churn_probability: float
    at_risk: bool
    threshold: float
    top_reasons: list[str]


# --------------------------------------------------------------------------- model loading
class ModelBundle:
    def __init__(self, model_dir: Path):
        self.pipeline = joblib.load(model_dir / "churn_pipeline.joblib")
        self.card = json.loads((model_dir / "model_card.json").read_text())
        self.threshold = float(self.card["decision_threshold"])
        self.ref = self.card["reference_stats"]["numeric"]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[FEATURES])[:, 1]

    def reasons(self, df: pd.DataFrame, k: int = 3) -> list[list[str]]:
        """Top factors pushing each customer's risk up.

        Only for the logistic regression model: its log-odds are a sum of (coefficient x
        feature value), so each input column's share can be read off exactly.
        Other model types return an empty list rather than an approximate explanation."""
        model = self.pipeline.named_steps["model"]
        if not isinstance(model, LogisticRegression):
            return [[] for _ in range(len(df))]
        prep = self.pipeline.named_steps["prep"]
        Z = prep.transform(df[FEATURES])
        names = prep.get_feature_names_out()
        contrib = Z * model.coef_[0]
        out = []
        for row in contrib:
            by_col: dict[str, float] = {}
            for name, value in zip(names, row):
                col = next(c for c in FEATURES if name == c or name.startswith(c + "_"))
                by_col[col] = by_col.get(col, 0.0) + float(value)
            top = sorted(((v, c) for c, v in by_col.items() if v > 0), reverse=True)[:k]
            out.append([f"{c} = {df.iloc[len(out)][c]}" for _, c in top])
        return out


bundle: ModelBundle | None = None
latencies_ms: deque[float] = deque(maxlen=1000)
recent_inputs: deque[dict] = deque(maxlen=5000)
recent_probs: deque[float] = deque(maxlen=5000)
started = datetime.now(timezone.utc)

def load_model() -> None:
    global bundle
    try:
        bundle = ModelBundle(MODEL_DIR)
        log.info("Loaded %s, threshold %.2f", bundle.card["chosen_model"], bundle.threshold)
    except FileNotFoundError:
        bundle = None
        log.error("No model in %s - run `python -m churn.train` first", MODEL_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Telco churn API", version="2.0.0", lifespan=lifespan)


def _require_model() -> ModelBundle:
    if bundle is None:
        raise HTTPException(503, "Model not loaded. Train it with `python -m churn.train`.")
    return bundle


def _log_predictions(rows: list[dict], probs: np.ndarray, ms: float) -> None:
    latencies_ms.append(ms)
    recent_inputs.extend(rows)
    recent_probs.extend(float(p) for p in probs)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as f:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for row, p in zip(rows, probs):
                f.write(json.dumps({"ts": ts, "prob": round(float(p), 4), **row}) + "\n")
    except OSError as exc:          # logging must never break a prediction
        log.warning("Could not write prediction log: %s", exc)


def _score(customers: list[Customer]) -> list[Prediction]:
    b = _require_model()
    rows = [c.model_dump() for c in customers]
    t0 = time.perf_counter()
    df = pd.DataFrame(rows)
    try:
        probs = b.predict(df)
        reasons = b.reasons(df)
    except Exception as exc:  # noqa: BLE001 - return a clean error, keep details in the log
        log.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed") from exc
    _log_predictions(rows, probs, (time.perf_counter() - t0) * 1000)
    return [Prediction(churn_probability=round(float(p), 4), at_risk=bool(p >= b.threshold),
                       threshold=b.threshold, top_reasons=r) for p, r in zip(probs, reasons)]


@app.get("/health")
def health() -> dict:
    b = _require_model()
    return {"status": "ok", "model": b.card["chosen_model"],
            "model_version": b.card["model_version"], "threshold": b.threshold,
            "trained_utc": b.card["created_utc"], "data_sha256": b.card["data_sha256"][:12]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return _score([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: Batch) -> list[Prediction]:
    return _score(batch.customers)


@app.get("/stats")
def stats() -> dict:
    b = _require_model()
    n = len(recent_probs)
    out = {"since_utc": started.isoformat(timespec="seconds"), "predictions": n,
           "requests": len(latencies_ms)}
    if latencies_ms:
        lat = np.array(latencies_ms)
        out["latency_ms"] = {"p50": round(float(np.percentile(lat, 50)), 2),
                             "p95": round(float(np.percentile(lat, 95)), 2)}
    if n:
        out["mean_predicted_risk"] = round(float(np.mean(recent_probs)), 4)
        out["training_churn_rate"] = b.card["reference_stats"]["churn_rate"]
    if n >= DRIFT_MIN_ROWS:
        live = pd.DataFrame(list(recent_inputs))
        drift = {}
        for col in NUMERIC_COLS:
            shift = (live[col].mean() - b.ref[col]["mean"]) / b.ref[col]["std"]
            drift[col] = {"live_mean": round(float(live[col].mean()), 2),
                          "train_mean": b.ref[col]["mean"],
                          "shift_in_std": round(float(shift), 3),
                          "alert": bool(abs(shift) > DRIFT_ALERT_STD)}
        out["numeric_drift"] = drift
    else:
        out["numeric_drift"] = f"needs at least {DRIFT_MIN_ROWS} predictions"
    return out
