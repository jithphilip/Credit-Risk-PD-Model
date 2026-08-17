"""
modeling.py
Train/test split, baseline Logistic Regression PD model, and XGBoost challenger.

Class imbalance approach (documented for the validation report):
The target is 70% good / 30% bad. This is a MILD imbalance, not severe, and it
reflects a real-world default rate that a PD model should preserve. We deliberately
do NOT apply SMOTE/oversampling/undersampling, because:
  - Resampling techniques distort the class prior, which biases the model's
    predicted probabilities away from the true population default rate.
  - For a PD model, calibrated probabilities (not just rank-ordering) matter --
    they typically feed into expected-loss and capital calculations. A model
    trained on an artificially rebalanced 50/50 sample will systematically
    over-predict PD unless probabilities are recalibrated back to the true prior.
  - 70/30 is not severe enough to cause the class-collapse problems (e.g. a
    classifier that only ever predicts the majority class) that resampling is
    meant to fix.
Instead we: (1) stratify the train/test split so both sets preserve the 70/30
prior, and (2) select an operating threshold deliberately in the benchmarking
step rather than defaulting to 0.5, since 0.5 is not necessarily the right
decision threshold under class imbalance.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from data_prep import load_model_ready

RANDOM_STATE = 42


def get_splits(test_size=0.25, random_state=RANDOM_STATE):
    X, y, df = load_model_ready()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def fit_logistic_regression(X_train, y_train):
    """
    Standardize features (helps optimizer convergence and makes coefficients
    comparable in magnitude), then fit a plain (unregularized-ish, high C) logistic
    regression for maximum coefficient interpretability -- this is the point of
    using LR for a regulated PD model: coefficients must be explainable to a
    validator/regulator, not just accurate.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000, C=1e6, solver="lbfgs")
    model.fit(X_train_scaled, y_train)
    return model, scaler


def fit_xgboost(X_train, y_train):
    """
    Modest, documented hyperparameters -- deliberately not heavily tuned.
    The goal is a fair, realistic challenger benchmark, not a leaderboard score.
    """
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_splits()
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train target rate:", y_train.mean().round(3), "Test target rate:", y_test.mean().round(3))

    lr_model, scaler = fit_logistic_regression(X_train, y_train)
    xgb_model = fit_xgboost(X_train, y_train)
    print("Both models fit successfully.")
