# Customer Churn Prediction and Retention Targeting

Predicts which telecom customers are likely to leave, ranks them for a retention team, and estimates the monthly revenue at risk. Version 2 turns the original notebook into a reproducible pipeline: one command trains and compares models with cross-validation, tracks every run in MLflow, and saves a single model file that a FastAPI service and a Streamlit app both use.

**Tools:** Python · Pandas · Scikit-learn · XGBoost · MLflow · FastAPI · Pydantic · Docker · pytest · GitHub Actions · Streamlit

**Results (held-out test set, 1,409 customers):** ROC-AUC 0.842 · the riskiest 20% of customers contain 49.7% of all churners (2.5× random) and 54.3% of churners' monthly revenue · predicted 377 churners vs 374 actual (calibrated probabilities)

## The business problem

A retention team cannot call every customer. It needs a ranked list: who is most likely to leave, why, and how much monthly revenue is at stake. So the model is judged on **ranking quality** (PR-AUC, lift), **honest probabilities** (calibration), and the **recall/precision trade-off** at the chosen threshold, not on accuracy (predicting "no churn" for everyone already gives 73%).

## Dataset

IBM Telco Customer Churn: 7,043 customers, 21 columns, 26.5% churn. Source: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) (`Telco-Customer-Churn.csv`, included). The notebook `customer_churn_analysis.ipynb` keeps the original exploratory analysis.

## Pipeline (`src/churn/`)

| Step | File | What it does |
|---|---|---|
| Load + validate | `data.py` | Cleans `TotalCharges` (11 blanks = tenure-0 customers → 0), checks columns, allowed category values and value ranges; fails loudly on bad data |
| Leakage / data checks | `train.py` → `reports/data_checks.json` | ID unique and not used as a feature; no customer in both train and test; no single column with AUC > 0.90 (highest: tenure 0.74); duplicates counted |
| Preprocessing | `features.py` | One `ColumnTransformer` (scaling + one-hot) inside each model's `Pipeline`, so it is re-fitted in every CV fold |
| Model comparison | `train.py` | Baseline, Logistic Regression, Random Forest, XGBoost, each tuned with `GridSearchCV`, 5-fold stratified CV, on the training set only |
| Selection rule | `train.py` | Highest mean CV PR-AUC; if a simpler model is within 1 standard deviation, take the simpler one. Fixed before looking at the test set |
| Threshold | `train.py` | Chosen on out-of-fold training predictions (max F1; max F2 is available with `--threshold-rule f2`) |
| Evaluation | `reports/` | Test metrics, lift table, segment risk, calibration, permutation importance |
| Versioning | `models/model_card.json` | SHA-256 of the data file, split, parameters, threshold, metrics, library versions; every run also logged to MLflow |

```bash
pip install -r requirements-dev.txt && pip install -e .
python -m churn.train                 # ~75 s on a 2-core CPU; writes models/ and reports/
mlflow ui --backend-store-uri sqlite:///mlflow.db    # compare runs
pytest -q                             # 13 tests
```

## Model comparison (5-fold CV on the 5,634 training customers)

| Model | CV PR-AUC | CV ROC-AUC |
|---|---|---|
| Baseline (always predicts the churn rate) | 0.265 | 0.500 |
| Logistic Regression (C=1) | 0.662 ± 0.020 | 0.846 ± 0.013 |
| Random Forest | 0.664 ± 0.021 | 0.848 ± 0.010 |
| XGBoost (depth 2) | 0.673 ± 0.020 | 0.851 ± 0.012 |

XGBoost was best by 0.011 PR-AUC, but that is inside one standard deviation of Logistic Regression, so the rule kept **Logistic Regression**: easier to explain, and its per-customer reasons are exact. On the test set XGBoost also scored higher (PR-AUC 0.666 vs 0.634, `reports/test_comparison_at_0.5.csv`). The model was not switched after seeing that, because choosing on the test set would make the reported test score optimistic.

## Results on the test set (1,409 customers, 374 churners)

At the F1 threshold (0.32): recall 0.727 (272 of 374 churners caught, 76.9% of their monthly revenue), precision 0.531, ROC-AUC 0.842, PR-AUC 0.634, Brier score 0.138.

**Calibration:** the model predicts 377 churners in total vs 374 actual, and each risk decile's average prediction is within about 7 points of the real churn rate (`reports/calibration.csv`). Version 1 used class weights, which pushed predictions up (68% predicted vs 52.5% actual in the highest-risk segment); v2 handles the class imbalance with the threshold instead.

**Lift: who should the retention team call?** (`reports/lift.csv`)

| Customers contacted (riskiest first) | Churners reached | Churners' revenue reached | Lift vs random |
|---|---|---|---|
| Top 10% (141) | 28.3% | 31.8% | 2.8× |
| Top 20% (282) | 49.7% | 54.3% | 2.5× |
| Top 30% (423) | 65.0% | 69.4% | 2.2× |

**Segments at risk** (`reports/segments.csv`): month-to-month customers hold 87% of the expected monthly revenue at risk. The single largest group is month-to-month customers in their first 12 months: 28% of test customers, 52.5% churn, 47% of revenue at risk.

**What drives the predictions** (permutation importance, drop in test PR-AUC when a column is shuffled): tenure 0.243, internet service 0.080, monthly charges 0.062, contract 0.049.

## Serving

```bash
uvicorn churn.api:app --port 8000      # docs at http://localhost:8000/docs
docker build -t churn-api . && docker run -p 8000:8000 churn-api
streamlit run app.py
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model type, version, threshold, data hash |
| `POST /predict` | One customer → probability, `at_risk` flag, top 3 factors raising the risk |
| `POST /predict/batch` | Up to 1,000 customers |
| `GET /stats` | Monitoring: request count, latency p50/p95, average predicted risk vs training churn rate, input drift per numeric column (alert when the mean moves more than 0.5 training std) |

Inputs are validated with Pydantic (unknown categories or negative values return 422). Every prediction is appended to a JSONL log. Measured locally on a 2-core CPU: about 21 ms per single prediction at the server (p50), 441 ms for a 1,000-row batch.

## Limitations

- One snapshot with no dates: a time-based split is impossible, and some fields may have changed near the churn date.
- 13 test rows have an identical feature row in training (different customers with the same plan); a small optimism in test scores.
- Revenue-at-risk uses current monthly charges and assumes a retention call saves the customer; real offer cost and success rate are unknown.
- The Docker image and the CI workflow are written and the same steps were run locally, but the image has not been pushed or deployed to a cloud host yet.

## Recommendations (unchanged from v1, now with numbers per segment)

1. Move month-to-month customers to longer contracts, starting with the first-year group (largest revenue at risk).
2. First-year onboarding: tenure is the strongest single driver.
3. Review fiber optic pricing and service quality.
4. Promote automatic payment over electronic check.
