"""
eda.py
Exploratory data analysis for the German Credit PD dataset.
Produces summary stats and 4 key visualizations saved to reports/figures/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import load_raw, clean, add_target

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def default_rate_by(df, col, ax, order=None):
    rates = df.groupby(col)["default"].mean().sort_values(ascending=False)
    if order is not None:
        rates = rates.reindex(order)
    rates.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.axhline(df["default"].mean(), color="red", linestyle="--", linewidth=1,
               label=f"Overall rate ({df['default'].mean():.0%})")
    ax.set_ylabel("Default rate")
    ax.set_title(f"Default rate by {col}")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=35)


def run_eda():
    df = load_raw()
    df = clean(df)
    df = add_target(df)

    # --- Fig 1: Target balance ---
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Risk"].value_counts().plot(kind="bar", color=["#4C72B0", "#C44E52"], ax=ax)
    ax.set_title("Target class balance (Risk)")
    ax.set_ylabel("Count")
    for i, v in enumerate(df["Risk"].value_counts()):
        ax.text(i, v + 5, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_target_balance.png")
    plt.close(fig)

    # --- Fig 2: Default rate by checking/savings account status ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    default_rate_by(df, "Checking account", axes[0],
                     order=["none", "little", "moderate", "rich"])
    default_rate_by(df, "Saving accounts", axes[1],
                     order=["none", "little", "moderate", "quite rich", "rich"])
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_default_by_account_status.png")
    plt.close(fig)

    # --- Fig 3: Credit amount & duration distributions by risk ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=df, x="Risk", y="Credit amount", ax=axes[0],
                order=["good", "bad"], palette=["#4C72B0", "#C44E52"])
    axes[0].set_title("Credit amount by outcome")
    sns.boxplot(data=df, x="Risk", y="Duration", ax=axes[1],
                order=["good", "bad"], palette=["#4C72B0", "#C44E52"])
    axes[1].set_title("Loan duration (months) by outcome")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_amount_duration_by_risk.png")
    plt.close(fig)

    # --- Fig 4: Default rate by purpose and housing ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    default_rate_by(df, "Purpose", axes[0])
    default_rate_by(df, "Housing", axes[1])
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_default_by_purpose_housing.png")
    plt.close(fig)

    # --- Summary stats to print/log ---
    summary = {
        "n_rows": len(df),
        "n_duplicates": int(df.duplicated().sum()),
        "missing_savings_pct": float(df["Saving accounts"].eq("none").mean()) if "none" in df["Saving accounts"].values else None,
        "default_rate_overall": float(df["default"].mean()),
        "default_rate_no_checking": float(df.loc[df["Checking account"] == "none", "default"].mean()),
        "default_rate_rich_checking": float(df.loc[df["Checking account"] == "rich", "default"].mean()),
        "age_mean_good": float(df.loc[df["Risk"] == "good", "Age"].mean()),
        "age_mean_bad": float(df.loc[df["Risk"] == "bad", "Age"].mean()),
        "credit_amount_mean_good": float(df.loc[df["Risk"] == "good", "Credit amount"].mean()),
        "credit_amount_mean_bad": float(df.loc[df["Risk"] == "bad", "Credit amount"].mean()),
        "duration_mean_good": float(df.loc[df["Risk"] == "good", "Duration"].mean()),
        "duration_mean_bad": float(df.loc[df["Risk"] == "bad", "Duration"].mean()),
    }
    return summary


if __name__ == "__main__":
    import json
    s = run_eda()
    print(json.dumps(s, indent=2))
