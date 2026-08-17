"""
data_prep.py
Data loading, cleaning, and feature preparation for the German Credit PD model.

Key data decisions (documented here so they're auditable, as a validator would expect):

1. Missing values in 'Saving accounts' and 'Checking account' are NOT random missingness.
   In this dataset's source (UCI German Credit / Kaggle kabure version), a blank in these
   fields means the customer does not hold that type of account. We therefore encode NaN
   as an explicit category 'none' rather than imputing a mode/median, which would silently
   invent an account status that doesn't exist for that customer.

2. 'Job' is provided as an ordinal integer (0=unskilled non-resident, 1=unskilled resident,
   2=skilled, 3=highly skilled/self-employed) and is kept as-is (ordinal), not one-hot encoded.

3. 'Saving accounts' and 'Checking account' have a natural order (none < little < moderate
   < quite rich / rich) and are ordinal-encoded rather than one-hot encoded, to preserve
   that ordering for the logistic regression coefficients' interpretability.

4. 'Sex', 'Housing', 'Purpose' have no natural order and are one-hot encoded.

5. Target 'Risk' is mapped to a binary PD target: bad=1 (default/event of interest),
   good=0. This follows standard PD modelling convention where 1 = the event being
   predicted (default).
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/german_credit_data.csv"

SAVINGS_ORDER = {"none": 0, "little": 1, "moderate": 2, "quite rich": 3, "rich": 4}
CHECKING_ORDER = {"none": 0, "little": 1, "moderate": 2, "rich": 3}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Saving accounts"] = df["Saving accounts"].fillna("none")
    df["Checking account"] = df["Checking account"].fillna("none")
    # Defensive: strip whitespace on string/categorical columns
    for col in ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose", "Risk"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["default"] = (df["Risk"].str.lower() == "bad").astype(int)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a fully numeric feature matrix (no target column) ready for modelling.
    Ordinal encodes Saving accounts / Checking account / Job (Job already ordinal int).
    One-hot encodes Sex, Housing, Purpose.
    """
    df = df.copy()

    df["Saving accounts_ord"] = df["Saving accounts"].map(SAVINGS_ORDER)
    df["Checking account_ord"] = df["Checking account"].map(CHECKING_ORDER)

    if df["Saving accounts_ord"].isna().any() or df["Checking account_ord"].isna().any():
        raise ValueError("Unmapped category found in Saving/Checking account encoding.")

    cat_onehot_cols = ["Sex", "Housing", "Purpose"]
    df_onehot = pd.get_dummies(df[cat_onehot_cols], drop_first=True)

    numeric_cols = ["Age", "Job", "Credit amount", "Duration",
                     "Saving accounts_ord", "Checking account_ord"]

    X = pd.concat([df[numeric_cols].reset_index(drop=True),
                   df_onehot.reset_index(drop=True)], axis=1)

    # Ensure all-numeric, boolean -> int
    X = X.apply(lambda c: c.astype(int) if c.dtype == bool else c)

    return X


def load_model_ready(path: str = RAW_PATH):
    """
    Full pipeline: load -> clean -> add target -> encode.
    Returns (X, y, df_clean) where df_clean retains original readable categories
    (useful for EDA) and X/y are the modelling-ready numeric matrix and target.
    """
    df = load_raw(path)
    df = clean(df)
    df = add_target(df)
    X = encode_features(df)
    y = df["default"]
    return X, y, df


if __name__ == "__main__":
    X, y, df = load_model_ready()
    print("Feature matrix shape:", X.shape)
    print("Target balance:\n", y.value_counts(normalize=True))
    print("\nFeature columns:", list(X.columns))
