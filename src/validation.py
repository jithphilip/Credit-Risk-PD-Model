"""
validation.py
Independent validation / benchmarking utilities:
  - AUC-ROC, KS statistic, Gini coefficient
  - Confusion matrix, precision, recall, F1 at a chosen threshold
  - ROC curve plotting (both models on one chart)
  - Population Stability Index (PSI) between train and test
  - Calibration plot (predicted probability vs observed default rate)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                              precision_score, recall_score, f1_score)

FIG_DIR = "reports/figures"


def ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov statistic: max separation between cumulative
    good/bad distributions across score thresholds."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = np.max(np.abs(tpr - fpr))
    return ks


def gini_from_auc(auc):
    return 2 * auc - 1


def compute_core_metrics(y_true, y_score, threshold=0.5):
    auc = roc_auc_score(y_true, y_score)
    ks = ks_statistic(y_true, y_score)
    gini = gini_from_auc(auc)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "AUC": auc,
        "KS": ks,
        "Gini": gini,
        "threshold": threshold,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_roc_overlay(y_true, score_dict, save_path=f"{FIG_DIR}/05_roc_overlay.png"):
    """score_dict: {'Logistic Regression': y_score_lr, 'XGBoost': y_score_xgb}"""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"Logistic Regression": "#4C72B0", "XGBoost": "#C44E52"}
    for name, y_score in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=colors.get(name), linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Logistic Regression vs. XGBoost")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def psi(expected, actual, buckets=10):
    """
    Population Stability Index between an 'expected' (e.g. train) and
    'actual' (e.g. test) distribution of a continuous variable or score.
    PSI < 0.1: no significant shift. 0.1-0.25: moderate shift, monitor.
    > 0.25: significant shift, investigate.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # avoid divide-by-zero / log(0)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)

    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return psi_value


def psi_table(X_train, X_test, score_train=None, score_test=None, cols=None):
    """Compute PSI for a list of feature columns, and optionally for the model score."""
    rows = []
    cols = cols if cols is not None else X_train.columns.tolist()
    for col in cols:
        val = psi(X_train[col].values, X_test[col].values)
        rows.append({"feature": col, "PSI": val})
    if score_train is not None and score_test is not None:
        rows.append({"feature": "model_score", "PSI": psi(score_train, score_test)})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False)


def plot_calibration(y_true, y_score, n_bins=10, save_path=f"{FIG_DIR}/06_calibration_lr.png"):
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df["bin"] = pd.qcut(df["score"], q=n_bins, duplicates="drop")
    grp = df.groupby("bin", observed=True).agg(
        predicted=("score", "mean"), observed=("y", "mean"), n=("y", "size")
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(grp["predicted"], grp["observed"], "o-", color="#4C72B0", label="Logistic Regression")
    lims = [0, max(grp["predicted"].max(), grp["observed"].max()) * 1.1]
    ax.plot(lims, lims, linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted probability of default (bin mean)")
    ax.set_ylabel("Observed default rate (bin mean)")
    ax.set_title("Calibration Plot — Logistic Regression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return grp
