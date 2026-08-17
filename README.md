# Credit Risk PD Model — Development & Independent Validation

A demonstration project covering end-to-end PD (Probability of Default)
model development and independent model validation, built to mirror how a
bank's Model Risk Management / Validation function would review a credit
risk model.

**Dataset:** [German Credit Data with Risk](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) (kabure, Kaggle / UCI Statlog German Credit), 1,000 rows.

## What's in here

- **Champion model**: Logistic Regression PD model, chosen for coefficient
  interpretability (a regulatory requirement for PD models).
- **Challenger model**: XGBoost, benchmarked on the identical train/test
  split as a fair performance comparison.
- **Full independent validation**: benchmarking metrics (AUC, KS, Gini,
  precision/recall/F1), multicollinearity (VIF), linearity-of-log-odds
  checks, population stability index (PSI), and a calibration plot —
  followed by a written validation report with an honest recommendation.

**Headline finding**: the Logistic Regression model has a real
interpretability flaw (the `Checking account` ordinal encoding assumes a
monotonic risk relationship that the data doesn't show), and XGBoost beats
it by a wide margin on every discrimination metric (Gini 0.559 vs. 0.367).
Full detail and recommendation in `reports/validation_report.md`.

## Project structure

```
credit-risk-pd-model/
├── data/
│   └── german_credit_data.csv       # raw dataset (not modified in place)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb    # exploration + modelling walkthrough, pre-executed
├── src/
│   ├── data_prep.py                 # loading, cleaning, encoding
│   ├── eda.py                       # EDA visualizations
│   ├── modeling.py                  # train/test split, LR + XGBoost fitting
│   ├── validation.py                # AUC/KS/Gini/PSI/calibration/ROC utilities
│   └── run_pipeline.py              # end-to-end script; regenerates all results
├── reports/
│   ├── figures/                     # all generated charts (6 PNGs)
│   ├── results.json                 # full numeric results, machine-readable
│   └── validation_report.md         # the written validation report (start here)
├── requirements.txt
└── README.md
```

## Reproducing the results

```bash
pip install -r requirements.txt

# Run the full pipeline (fits both models, computes all metrics, regenerates figures)
python3 src/run_pipeline.py

# Or open the notebook for a walkthrough with narrative
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

All scripts assume they're run from the project root (`credit-risk-pd-model/`),
since data and report paths are relative. `run_pipeline.py` writes
`reports/results.json` and all six figures in `reports/figures/`.

## Key results at a glance

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| AUC-ROC | 0.683 | 0.780 |
| KS statistic | 0.299 | 0.463 |
| Gini coefficient | 0.367 | 0.559 |
| F1 @ 0.5 threshold | 0.327 | 0.511 |

Full metrics, VIF table, PSI table, and calibration results are in
`reports/results.json` and discussed in `reports/validation_report.md`.

## Notes on methodology decisions

- **Missing values** in `Saving accounts` / `Checking account` are encoded
  as an explicit `"none"` category, not imputed — a blank in this dataset
  means "holds no account of that type," not missing data.
- **Class imbalance** (70% good / 30% bad) was handled via stratified
  train/test splitting, not resampling — resampling would distort the PD
  model's calibrated probabilities.
- **XGBoost hyperparameters were kept modest and undertuned by design**, to
  serve as a fair benchmark rather than a leaderboard entry.
