"""Train, compare and save the churn model in one reproducible run.

    python -m churn.train --data Telco-Customer-Churn.csv

What it does, in order:
 1. load + clean + validate the data (churn.data)
 2. run leakage / data checks and save them to reports/data_checks.json
 3. stratified 80/20 split with the same seed as the notebook (same 1,409 test customers)
 4. tune every candidate with 5-fold stratified CV on the TRAINING set only (MLflow logs each run)
 5. pick the model with a rule fixed in advance (see choose_model) - the test set is not used
 6. pick the decision threshold from out-of-fold training predictions (max F1 by default; F2 available)
 7. score the untouched test set once; save metrics, lift table, segments and feature importance
 8. save ONE pipeline file + a model card (data hash, params, threshold, metrics, library versions)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)

from . import __version__
from .data import (CATEGORICAL_COLS, FEATURES, ID_COL, NUMERIC_COLS, TARGET, file_sha256,
                   load)
from .features import SEED, candidates

# Simpler / easier-to-explain models first. Used only to break near-ties (see choose_model).
SIMPLICITY_ORDER = ["logistic_regression", "random_forest", "xgboost"]
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 m", "13-24 m", "25-48 m", "49-72 m"]


# --------------------------------------------------------------------------- checks
def data_checks(df: pd.DataFrame) -> dict:
    """Checks for leakage and data problems. Results are saved, not just printed."""
    X = df[FEATURES]
    y = df[TARGET]
    checks = {
        "rows": int(len(df)),
        "churn_rate": round(float(y.mean()), 4),
        "customer_id_unique": bool(df[ID_COL].is_unique),
        "customer_id_used_as_feature": ID_COL in FEATURES,
        "exact_duplicate_feature_rows": int(X.duplicated().sum()),
        "duplicate_rows_with_conflicting_label": int(
            df.groupby(FEATURES, observed=True)[TARGET].nunique().gt(1).sum()),
    }
    # A single column that predicts churn almost perfectly is the usual sign of leakage
    # (e.g. a field filled in after the customer left). Flag anything above 0.90 AUC.
    single_auc = {}
    for col in NUMERIC_COLS + ["SeniorCitizen"]:
        auc = roc_auc_score(y, X[col])
        single_auc[col] = round(max(auc, 1 - auc), 3)
    for col in CATEGORICAL_COLS:
        rate = y.groupby(X[col]).mean()
        auc = roc_auc_score(y, X[col].map(rate))
        single_auc[col] = round(max(auc, 1 - auc), 3)
    checks["single_feature_auc"] = dict(sorted(single_auc.items(), key=lambda kv: -kv[1]))
    checks["single_feature_auc_above_0.90"] = [c for c, a in single_auc.items() if a > 0.90]
    # TotalCharges is roughly tenure x MonthlyCharges: redundant, not leakage. Measured, not assumed.
    approx = X["tenure"] * X["MonthlyCharges"]
    checks["corr_TotalCharges_vs_tenure_x_MonthlyCharges"] = round(
        float(np.corrcoef(approx, X["TotalCharges"])[0, 1]), 4)
    checks["note"] = ("The dataset is one snapshot with no dates, so a time-based split is not "
                      "possible; contract and service fields could have changed close to the "
                      "churn date. This cannot be checked with this data.")
    return checks


# --------------------------------------------------------------------------- metrics
def classification_metrics(y_true, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "f2": round(fbeta_score(y_true, pred, beta=2), 4),
        "roc_auc": round(roc_auc_score(y_true, prob), 4),
        "pr_auc": round(average_precision_score(y_true, prob), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def best_threshold(y_true, prob, beta: float) -> float:
    """Threshold that maximises F-beta. beta=2 weights recall twice as much as precision
    (a missed churner costs more than an unneeded retention call); beta=1 weights them equally."""
    grid = np.round(np.arange(0.05, 0.91, 0.01), 2)
    scores = [fbeta_score(y_true, (prob >= t).astype(int), beta=beta) for t in grid]
    return float(grid[int(np.argmax(scores))])


def calibration_table(y_true, prob, bins: int = 10) -> pd.DataFrame:
    """Mean predicted risk vs actual churn rate in 10 equal-size groups (lowest to highest risk)."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": prob})
    d["risk_decile"] = pd.qcut(d["p"].rank(method="first"), bins, labels=range(1, bins + 1))
    return (d.groupby("risk_decile", observed=True)
             .agg(customers=("y", "size"), mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
             .reset_index().round(4))


def choose_model(cv_table: pd.DataFrame) -> tuple[str, str]:
    """Rule fixed before looking at the test set:
    take the best mean CV PR-AUC; if a simpler model is within one standard deviation
    of the best, take the simpler one (the 'one-standard-error' idea)."""
    real = cv_table[cv_table.model != "baseline_prior"].set_index("model")
    best = real["cv_pr_auc_mean"].idxmax()
    floor = real.loc[best, "cv_pr_auc_mean"] - real.loc[best, "cv_pr_auc_std"]
    for name in SIMPLICITY_ORDER:
        if real.loc[name, "cv_pr_auc_mean"] >= floor:
            reason = ("highest CV PR-AUC" if name == best else
                      f"within 1 std of best ({best}) and simpler")
            return name, reason
    return best, "highest CV PR-AUC"


def lift_table(y_true: pd.Series, prob: np.ndarray, revenue: pd.Series) -> pd.DataFrame:
    """If the retention team can only contact the top X% riskiest customers,
    what share of churners and of churners' monthly revenue do they reach?"""
    order = np.argsort(-prob)
    y_sorted = y_true.to_numpy()[order]
    rev_sorted = (revenue.to_numpy() * y_true.to_numpy())[order]
    rows = []
    for pct in (10, 20, 30, 40, 50):
        k = int(round(len(prob) * pct / 100))
        rows.append({
            "top_pct_contacted": pct,
            "customers_contacted": k,
            "churners_reached": int(y_sorted[:k].sum()),
            "share_of_churners": round(y_sorted[:k].sum() / y_sorted.sum(), 4),
            "share_of_churned_revenue": round(rev_sorted[:k].sum() / rev_sorted.sum(), 4),
            "churn_rate_in_group": round(y_sorted[:k].mean(), 4),
            "lift_vs_random": round((y_sorted[:k].sum() / y_sorted.sum()) / (pct / 100), 2),
        })
    return pd.DataFrame(rows)


def segment_table(test: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    seg = test.assign(
        tenure_band=pd.cut(test["tenure"], TENURE_BINS, labels=TENURE_LABELS),
        predicted_prob=prob,
        expected_revenue_at_risk=prob * test["MonthlyCharges"],
    )
    out = (seg.groupby(["Contract", "tenure_band"], observed=True)
              .agg(customers=(TARGET, "size"),
                   actual_churn_rate=(TARGET, "mean"),
                   mean_predicted_risk=("predicted_prob", "mean"),
                   monthly_revenue=("MonthlyCharges", "sum"),
                   expected_monthly_revenue_at_risk=("expected_revenue_at_risk", "sum"))
              .reset_index()
              .sort_values("expected_monthly_revenue_at_risk", ascending=False))
    total = out["expected_monthly_revenue_at_risk"].sum()
    out["share_of_revenue_at_risk"] = out["expected_monthly_revenue_at_risk"] / total
    return out.round(4)


def reference_stats(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Training-data profile kept in the model card; the API compares live inputs to it."""
    return {
        "churn_rate": round(float(y_train.mean()), 4),
        "numeric": {c: {"mean": round(float(X_train[c].mean()), 4),
                        "std": round(float(X_train[c].std()), 4)} for c in NUMERIC_COLS},
        "categorical": {c: X_train[c].value_counts(normalize=True).round(4).to_dict()
                        for c in CATEGORICAL_COLS},
    }


# --------------------------------------------------------------------------- main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="Telco-Customer-Churn.csv")
    ap.add_argument("--out", default=".", help="folder for models/ and reports/")
    ap.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--threshold-rule", choices=["f1", "f2"], default="f1")
    ap.add_argument("--quick", action="store_true", help="tiny grids + 3 folds (CI smoke test)")
    args = ap.parse_args(argv)

    t0 = time.time()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    df = load(args.data)
    checks = data_checks(df)
    (out / "reports" / "data_checks.json").write_text(json.dumps(checks, indent=2))

    train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET])
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    assert set(train[ID_COL]).isdisjoint(test[ID_COL]), "customer appears in train and test"
    # Different customers can have identical plans; count test rows whose exact feature row
    # also appears in training (a small source of optimism in the test score).
    train_keys = set(map(tuple, X_train.astype(str).to_numpy()))
    checks["test_rows_with_identical_feature_row_in_train"] = int(
        sum(tuple(r) in train_keys for r in X_test.astype(str).to_numpy()))
    (out / "reports" / "data_checks.json").write_text(json.dumps(checks, indent=2))

    n_folds = 3 if args.quick else 5
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc",
               "recall": "recall", "precision": "precision", "f1": "f1"}

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("telco-churn")

    cv_rows, fitted = [], {}
    for name, (pipe, grid) in candidates().items():
        if args.quick:
            grid = {k: v[:1] for k, v in grid.items()}
        search = GridSearchCV(pipe, grid or {}, scoring=scoring, refit="pr_auc", cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        i = search.best_index_
        res = search.cv_results_
        row = {"model": name, "best_params": json.dumps(search.best_params_)}
        for m in scoring:
            row[f"cv_{m}_mean"] = round(float(res[f"mean_test_{m}"][i]), 4)
            row[f"cv_{m}_std"] = round(float(res[f"std_test_{m}"][i]), 4)
        cv_rows.append(row)
        fitted[name] = search.best_estimator_
        with mlflow.start_run(run_name=f"cv_{name}"):
            mlflow.log_params({"candidate": name, "folds": n_folds, **search.best_params_})
            mlflow.log_metrics({k: v for k, v in row.items() if k.startswith("cv_")})
        print(f"{name:20s} CV PR-AUC {row['cv_pr_auc_mean']:.4f} ± {row['cv_pr_auc_std']:.4f}")

    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(out / "reports" / "cv_results.csv", index=False)

    chosen, reason = choose_model(cv_table)
    model = fitted[chosen]

    # Threshold from out-of-fold predictions on the training set only.
    oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thresholds = {"f1": best_threshold(y_train, oof, beta=1),
                  "f2": best_threshold(y_train, oof, beta=2)}
    threshold = thresholds[args.threshold_rule]

    # ---- one look at the test set
    prob = model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, prob, threshold)
    test_metrics_at_05 = classification_metrics(y_test, prob, 0.5)
    caught = (prob >= threshold) & (y_test.to_numpy() == 1)
    churner_rev = test.loc[y_test == 1, "MonthlyCharges"].sum()
    test_metrics["brier_score"] = round(float(brier_score_loss(y_test, prob)), 4)
    test_metrics["flagged_customers"] = int((prob >= threshold).sum())
    test_metrics["predicted_vs_actual_churners"] = [round(float(prob.sum()), 1), int(y_test.sum())]
    calibration_table(y_test, prob).to_csv(out / "reports" / "calibration.csv", index=False)
    test_metrics["churners_in_test"] = int(y_test.sum())
    test_metrics["churners_caught"] = int(caught.sum())
    test_metrics["share_of_churner_revenue_caught"] = round(
        float(test.loc[caught, "MonthlyCharges"].sum() / churner_rev), 4)

    # All candidates on test, for the README table only (selection above did not use these).
    comparison = []
    for name, est in fitted.items():
        p = est.predict_proba(X_test)[:, 1]
        comparison.append({"model": name, **classification_metrics(y_test, p, 0.5)})
    pd.DataFrame(comparison).to_csv(out / "reports" / "test_comparison_at_0.5.csv", index=False)

    lift = lift_table(y_test, prob, test["MonthlyCharges"])
    lift.to_csv(out / "reports" / "lift.csv", index=False)
    segments = segment_table(test, prob)
    segments.to_csv(out / "reports" / "segments.csv", index=False)

    # Permutation importance on raw input columns: how much test PR-AUC drops when a column
    # is shuffled. Works for any model and is measured on data the model never saw.
    perm = permutation_importance(model, X_test, y_test, scoring="average_precision",
                                  n_repeats=5 if args.quick else 20, random_state=SEED, n_jobs=1)
    importance = (pd.DataFrame({"feature": FEATURES,
                                "pr_auc_drop_mean": perm.importances_mean,
                                "pr_auc_drop_std": perm.importances_std})
                  .sort_values("pr_auc_drop_mean", ascending=False).round(4))
    importance.to_csv(out / "reports" / "permutation_importance.csv", index=False)

    # ---- save model + model card
    model_path = out / "models" / "churn_pipeline.joblib"
    joblib.dump(model, model_path)
    card = {
        "model_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_file": Path(args.data).name,
        "data_sha256": file_sha256(args.data),
        "split": {"test_size": 0.2, "random_state": SEED, "stratified": True,
                  "train_rows": int(len(train)), "test_rows": int(len(test))},
        "chosen_model": chosen,
        "selection_reason": reason,
        "best_params": json.loads(cv_table.set_index("model").loc[chosen, "best_params"]),
        "decision_threshold": threshold,
        "threshold_rule": f"max {args.threshold_rule.upper()} on out-of-fold training predictions",
        "thresholds_considered": thresholds,
        "cv": cv_table.set_index("model").loc[chosen].drop("best_params").to_dict(),
        "test": test_metrics,
        "test_at_threshold_0.5": test_metrics_at_05,
        "reference_stats": reference_stats(X_train, y_train),
        "features": FEATURES,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__},
        "quick_mode": bool(args.quick),
    }
    (out / "models" / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    metrics_out = {"chosen_model": chosen, "selection_reason": reason,
                   "threshold": threshold, "thresholds_considered": thresholds,
                   "test": test_metrics,
                   "test_at_threshold_0.5": test_metrics_at_05,
                   "runtime_seconds": round(time.time() - t0, 1)}
    (out / "reports" / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    with mlflow.start_run(run_name=f"final_{chosen}"):
        mlflow.log_params({"chosen_model": chosen, "threshold": threshold,
                           "data_sha256": card["data_sha256"][:12], **card["best_params"]})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(out / "models" / "model_card.json"))
        for f in (out / "reports").glob("*"):
            mlflow.log_artifact(str(f), artifact_path="reports")

    print(f"\nChosen: {chosen} ({reason}); threshold {threshold}")
    print(json.dumps(test_metrics, indent=2))
    print(f"Done in {time.time() - t0:.0f} s")
    return metrics_out


if __name__ == "__main__":
    main()
