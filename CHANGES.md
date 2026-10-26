# What changed in version 2 (and why)

Read this before interviews: you should be able to explain every change and every number.

| Change | File | Why |
|---|---|---|
| Cleaning + validation moved into one function | `src/churn/data.py` | Training, API and tests now use identical cleaning. Bad input (new category, negative value, missing column) raises a clear error. |
| Preprocessing inside a scikit-learn `Pipeline` | `src/churn/features.py` | v1 saved scaler, columns and model separately and the app repeated the encoding by hand. Now one file holds everything, and the scaler is re-fitted inside each CV fold. |
| XGBoost and a baseline added to the comparison | `features.py`, `train.py` | A baseline shows what "no skill" scores (PR-AUC 0.265). XGBoost tests whether a stronger model is worth it. Result: +0.011 CV PR-AUC, within noise, so Logistic Regression kept. |
| Model chosen by CV PR-AUC with a rule fixed in advance | `train.py` (`choose_model`) | PR-AUC suits a 26.5% minority class. The rule (simpler model if within 1 std) stops picking a model because of random fold noise. The test set is never used for choosing. |
| Class weights removed; threshold tuned instead | `features.py`, `train.py` | Class weights inflated probabilities (68% predicted vs 52.5% actual for the riskiest segment). Unweighted model + tuned threshold gives honest probabilities: 377 predicted vs 374 actual churners. |
| Threshold chosen on out-of-fold training predictions | `train.py` (`best_threshold`) | Choosing it on the test set would leak. F1 (0.32) is the default; F2 (0.14) is available when missing a churner is much costlier than an unneeded call. |
| Leakage and data checks saved to a file | `reports/data_checks.json` | Single-column AUCs (none above 0.90), ID not a feature, no customer in train and test, duplicate counts. |
| Lift table and segment table | `reports/lift.csv`, `reports/segments.csv` | Answers the business question: "if we can call the top 20%, how many churners and how much revenue do we reach?" |
| Permutation importance | `reports/permutation_importance.csv` | Measured on the test set; works for any model, unlike raw coefficients. |
| MLflow tracking (SQLite backend) | `train.py` | Every candidate's CV scores and the final model, card and reports are logged, so runs can be compared in `mlflow ui`. |
| Model card with data hash and library versions | `models/model_card.json` | Any saved model can be traced back to the exact data and settings that produced it. |
| FastAPI service with validation, batch endpoint, prediction log and drift check | `src/churn/api.py` | Turns the model into something another system can call; `/stats` gives basic monitoring. |
| Per-customer reasons | `api.py` (`ModelBundle.reasons`) | For Logistic Regression, log-odds = sum of coefficient × value, so each column's contribution is exact. |
| Streamlit app uses the same pipeline | `app.py` | No duplicated encoding logic; also shows expected monthly revenue at risk. |
| 13 pytest tests | `tests/` | Data cleaning, validation errors, model card, API responses, 422 on bad input, drift alert. |
| Dockerfile (non-root, health check, slim serving requirements) | `Dockerfile`, `requirements-serve.txt` | Small image with only what serving needs. The serving requirements were tested in a clean virtual environment. |
| GitHub Actions CI | `.github/workflows/ci.yml` | On every push: unit tests, a quick retrain, API tests on the new model, Docker build. |
| Old `churn_model.pkl`, `scaler.pkl`, `model_columns.pkl` removed | root | Replaced by `models/churn_pipeline.joblib`. |

## Not verified here

- `docker build` could not run in the environment used to prepare this version (Docker Hub was blocked), and the CI workflow has not run on GitHub yet. Push the repo and check the Actions tab; fix anything it reports before mentioning CI or Docker in an interview.
- Exact numbers can shift slightly on another machine or library version. Re-run `python -m churn.train` and use your own numbers.
