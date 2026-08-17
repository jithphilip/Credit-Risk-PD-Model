"""
run_pipeline.py
End-to-end execution: data prep -> fit both models -> benchmark -> stability checks.
Saves all figures to reports/figures/ and prints a full results summary (also
dumped to reports/results.json) that feeds directly into the validation report.
"""
import sys
sys.path.insert(0, "src")

import json
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from modeling import get_splits, fit_logistic_regression, fit_xgboost
from validation import (compute_core_metrics, plot_roc_overlay, psi_table,
                         plot_calibration)

RESULTS_PATH = "reports/results.json"


def main():
    results = {}

    X_train, X_test, y_train, y_test = get_splits()
    results["n_train"] = len(X_train)
    results["n_test"] = len(X_test)
    results["train_default_rate"] = float(y_train.mean())
    results["test_default_rate"] = float(y_test.mean())

    # --- Fit models ---
    lr_model, scaler = fit_logistic_regression(X_train, y_train)
    xgb_model = fit_xgboost(X_train, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_score_lr_test = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_score_xgb_test = xgb_model.predict_proba(X_test)[:, 1]

    X_train_scaled = scaler.transform(X_train)
    y_score_lr_train = lr_model.predict_proba(X_train_scaled)[:, 1]
    y_score_xgb_train = xgb_model.predict_proba(X_train)[:, 1]

    # --- LR coefficients ---
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": lr_model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    results["lr_coefficients"] = coef_df.to_dict(orient="records")
    results["lr_intercept"] = float(lr_model.intercept_[0])

    # --- VIF ---
    X_vif = X_train.astype(float).copy()
    X_vif.insert(0, "const", 1.0)
    vif_data = pd.DataFrame({
        "feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    })
    vif_data = vif_data[vif_data["feature"] != "const"].sort_values("VIF", ascending=False)
    results["vif"] = vif_data.to_dict(orient="records")
    results["max_vif"] = float(vif_data["VIF"].max())

    # --- Core benchmarking metrics at threshold=0.5 ---
    # We also report at the threshold that maximizes KS, which is a common
    # PD-model convention for picking an operating point when the business
    # decision threshold isn't already fixed by policy.
    metrics_lr_050 = compute_core_metrics(y_test, y_score_lr_test, threshold=0.5)
    metrics_xgb_050 = compute_core_metrics(y_test, y_score_xgb_test, threshold=0.5)

    # threshold that maximizes KS on the TEST set score distribution (reported
    # for transparency; in practice this should be chosen on a separate
    # validation fold, not test -- flagged as a limitation in the report)
    from sklearn.metrics import roc_curve
    def best_ks_threshold(y_true, y_score):
        fpr, tpr, thresh = roc_curve(y_true, y_score)
        idx = np.argmax(tpr - fpr)
        return float(thresh[idx])

    thr_lr = best_ks_threshold(y_test, y_score_lr_test)
    thr_xgb = best_ks_threshold(y_test, y_score_xgb_test)
    metrics_lr_ks = compute_core_metrics(y_test, y_score_lr_test, threshold=thr_lr)
    metrics_xgb_ks = compute_core_metrics(y_test, y_score_xgb_test, threshold=thr_xgb)

    def clean_metrics(m):
        m = dict(m)
        m["confusion_matrix"] = m["confusion_matrix"].tolist()
        return m

    results["metrics_lr_thr050"] = clean_metrics(metrics_lr_050)
    results["metrics_xgb_thr050"] = clean_metrics(metrics_xgb_050)
    results["metrics_lr_thr_ks_optimal"] = clean_metrics(metrics_lr_ks)
    results["metrics_xgb_thr_ks_optimal"] = clean_metrics(metrics_xgb_ks)

    # --- ROC overlay plot ---
    plot_roc_overlay(y_test, {
        "Logistic Regression": y_score_lr_test,
        "XGBoost": y_score_xgb_test,
    })

    # --- PSI: train vs test, on raw features + both model scores ---
    feature_cols = ["Age", "Credit amount", "Duration", "Saving accounts_ord", "Checking account_ord"]
    from validation import psi as psi_fn
    psi_feat = psi_table(X_train, X_test, cols=feature_cols)
    psi_scores = pd.DataFrame([
        {"feature": "LR_score", "PSI": float(psi_fn(y_score_lr_train, y_score_lr_test))},
        {"feature": "XGB_score", "PSI": float(psi_fn(y_score_xgb_train, y_score_xgb_test))},
    ])
    psi_all = pd.concat([psi_feat, psi_scores], ignore_index=True).sort_values("PSI", ascending=False)
    results["psi"] = psi_all.to_dict(orient="records")

    # --- Calibration plot (LR) ---
    calib_grp = plot_calibration(y_test, y_score_lr_test)
    results["calibration_lr"] = calib_grp.to_dict(orient="records")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
