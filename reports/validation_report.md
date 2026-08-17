# Independent Validation Report: Retail Credit PD Model

**Model reviewed:** Logistic Regression Probability-of-Default (PD) model
**Challenger benchmark:** XGBoost classifier
**Dataset:** German Credit Data with Risk (kabure, Kaggle / UCI Statlog German Credit), n=1,000
**Reviewer role:** Independent Model Validation

---

## 1. Objective and Dataset Description

This report documents an independent validation of a Logistic Regression PD
model built to estimate the probability that a retail credit applicant
defaults (`Risk = bad`), and benchmarks it against an XGBoost challenger
model trained on the same data split. The purpose of this review is to
assess the champion model's soundness, performance relative to a more
flexible benchmark, and fitness for a regulated credit decisioning context —
not simply to declare a "winner."

The dataset contains 1,000 applicants with 9 candidate predictors (age, sex,
job skill level, housing status, savings account status, checking account
status, credit amount, loan duration, and loan purpose) and a binary outcome,
`Risk` (good/bad), which we treat as `default = 1` for `bad`. The sample is
complete (no true missingness or duplicates once account-status blanks are
correctly interpreted — see Section 4) and reflects a 70% good / 30% bad
class split.

## 2. Methodology

**Data treatment.** Blank values in `Saving accounts` (18.3% of rows) and
`Checking account` (39.4% of rows) were encoded as an explicit `none`
category rather than imputed, since a blank in the source data denotes "does
not hold this type of account," not a missing observation. `Job` was kept as
its native ordinal integer. `Saving accounts` and `Checking account` were
ordinal-encoded (none < little < moderate < rich); `Sex`, `Housing`, and
`Purpose` were one-hot encoded, having no natural order. Data was split
75/25 train/test, stratified on the target, preserving the 70/30 class rate
in both splits.

**Class imbalance.** The 70/30 split is a mild imbalance that reflects a real
default rate. We did not apply SMOTE, oversampling, or undersampling: these
techniques alter the class prior seen by the model, which biases predicted
probabilities away from the true population rate — a material problem for a
PD model whose output typically feeds expected-loss or provisioning
calculations downstream. Preserving the natural prior via stratified
sampling was judged the more defensible choice for a probability-calibrated
model.

**Champion model (Logistic Regression).** Fit on standardized features with
negligible regularization (C=1e6), prioritizing coefficient interpretability
over squeezing out marginal performance — appropriate for a model whose
coefficients must be explainable to a credit committee or regulator.

**Challenger model (XGBoost).** Fit on the identical train/test split with
deliberately modest, documented hyperparameters (150 trees, max depth 3,
learning rate 0.05, subsampling 0.8) to serve as a fair, realistic benchmark
rather than a tuned leaderboard entry.

## 3. Benchmarking Results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| AUC-ROC | 0.683 | **0.780** |
| KS statistic | 0.299 | **0.463** |
| Gini coefficient | 0.367 | **0.559** |
| Precision @ 0.5 threshold | 0.586 | 0.586 |
| Recall @ 0.5 threshold | 0.227 | **0.453** |
| F1 @ 0.5 threshold | 0.327 | **0.511** |
| Precision @ KS-optimal threshold | 0.455 | 0.573 |
| Recall @ KS-optimal threshold | 0.613 | **0.680** |
| F1 @ KS-optimal threshold | 0.523 | **0.622** |

*(KS-optimal thresholds: LR = 0.312, XGBoost = 0.383. These were selected on
the test set itself for illustration; in a production setting the operating
threshold should be chosen on a separate validation fold to avoid
optimistic bias — see Limitations.)*

XGBoost outperforms the Logistic Regression baseline on every discrimination
metric, by a wide margin: **+0.097 AUC, +0.164 KS, +0.192 Gini**. At the
default 0.5 threshold, both models achieve identical precision (0.586), but
XGBoost roughly doubles recall (0.453 vs. 0.227) — the linear model is
missing a large share of actual defaulters at that cut-off. See
`reports/figures/05_roc_overlay.png` for the ROC comparison.

This performance gap is consistent with what the assumption checks below
explain: several predictors have non-linear or non-monotonic relationships
with default risk that a linear model structurally cannot capture, while a
tree ensemble can.

## 4. Stability and Assumption Checks

**Multicollinearity (VIF).** Maximum VIF across all Logistic Regression
features is 3.05 (`Purpose_car`), well under the common concern threshold of
5–10. Multicollinearity is not a material issue for this feature set.

**Linearity of log-odds.** Binned empirical log-odds were checked for the
three continuous predictors. `Age` and `Duration` show reasonably monotonic,
close-to-linear trends. **`Credit amount` shows a non-monotonic, U-shaped
pattern** (log-odds of roughly -0.83, -1.15, -0.99, -1.05, -0.30 across
quintiles) — risk dips in the middle of the distribution and rises sharply
only at the highest amounts. This likely explains why `Credit amount`
receives a near-zero coefficient (-0.007) in the fitted model: a single
linear term cannot represent this shape, so the model effectively treats the
variable as uninformative when it is not.

**Non-monotonic categorical encoding — flagged issue.** The ordinal encoding
of `Checking account` (none=0 < little=1 < moderate=2 < rich=3) assumes a
monotonic risk relationship. The raw default rates by category are **11.7%
(none), 49.3% (little), 39.0% (moderate), 22.2% (rich)** — clearly
non-monotonic (a hump, not a slope). As a direct consequence, this feature
receives the **largest coefficient in the model (+0.471) with a sign that
has no clean business interpretation**: the model reads as "more money in
checking → higher risk," which is not a defensible statement to a credit
committee and is really an artifact of forcing a straight line through a
hump-shaped relationship. **This is the single most important finding of
this validation** and should be remediated (see Recommendations) before this
model is considered for production use in its current form.

**Population Stability Index (PSI).** PSI between train and test was below
0.07 for every feature and both model scores (`LR_score` = 0.025,
`XGB_score` = 0.041), well inside the "no significant shift" band (<0.1).
**Caveat:** train/test here is a random split of one cross-sectional sample,
not a genuine out-of-time holdout — low PSI under these conditions is close
to guaranteed by construction and should not be read as evidence of
real-world temporal stability. This dataset has no date field, so a true
stability assessment against a future vintage was not possible and would be
required before relying on PSI as a monitoring signal in production.

**Calibration.** The Logistic Regression model's predicted probabilities
track observed default rates reasonably closely across deciles (see
`reports/figures/06_calibration_lr.png`), with the scatter expected from a
250-row test set (~25 observations per bin). No systematic over- or
under-prediction pattern is evident.

## 5. Limitations

- **Small sample size** (1,000 rows, 250 in the test set) limits the
  precision of all point estimates in this report, particularly the
  deciled calibration curve and per-category default rates for sparser
  `Purpose` categories.
- **No temporal dimension.** All findings — especially PSI — are based on a
  random split of a single cross-sectional sample, not separate vintages.
  Real-world population and concept drift cannot be assessed here.
- **Checking account encoding flaw** (Section 4) means the current champion
  model's largest coefficient is not interpretable in a business sense and
  should not be presented to a credit committee as-is.
- **Credit amount's true relationship is non-linear** and is not well
  captured by the linear model, likely understating that variable's real
  predictive contribution.
- **KS-optimal thresholds were chosen on the test set**, which will be
  mildly optimistic; a proper implementation would select the operating
  threshold on an independent validation fold.
- **XGBoost was not tuned aggressively** by design (to keep the benchmark
  fair, not to find its ceiling); its true performance advantage over a
  well-specified linear model could differ from what's reported here in
  either direction with further tuning of either model.

## 6. Recommendation

XGBoost is the stronger model on every discrimination metric measured, by a
margin large enough (Gini +0.19) to be practically, not just statistically,
meaningful. However, discrimination performance alone does not make it the
right choice for this use case. For a regulated PD model where coefficient
sign, magnitude, and business rationale must be defended to a credit
committee or regulator, an uninterpretable ranking model is a real cost, not
a footnote.

**As an independent validator, I would not approve the current Logistic
Regression model for production as-is** — the `Checking account`
encoding issue in Section 4 produces a headline coefficient with an
indefensible sign, which is precisely the kind of finding this review
process exists to catch. My recommendation is:

1. **Remediate the champion model first**: re-encode `Checking account` and
   `Saving accounts` using a method that doesn't assume monotonicity (e.g.
   Weight-of-Evidence encoding, or treating `none` as its own dummy rather
   than the base of an ordinal scale), and consider a spline or binned term
   for `Credit amount`. Re-run this validation on the remediated model.
2. **Only after remediation**, compare fairly: if the performance gap to
   XGBoost narrows substantially, the interpretable model remains preferable
   for this use case. If a large gap persists, the business should
   explicitly weigh the performance gain against the explainability cost —
   potentially via a hybrid approach (e.g. XGBoost for a challenger
   early-warning score, with the interpretable model as the model of
   record for adverse-action decisions).
3. **Do not rely on the PSI results here as a stability sign-off** for
   either model; require an out-of-time sample before deployment monitoring
   thresholds are set.

This is not a "which model wins" conclusion — it is a finding that the
champion model, though interpretable in form, is not yet interpretable in
substance, and that gap needs to close before the performance trade-off
against XGBoost can even be fairly evaluated.
