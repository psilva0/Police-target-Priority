# Criminal Priority Classification — Municipalities of São Paulo State

Machine Learning project that classifies municipalities in the state of São Paulo into
**Low**, **Medium** and **High** criminal priority levels from lagged historical crime
indicators.

The project combines a composite priority index, temporal validation, a comparison of
eight algorithms against explicit baselines, and Explainable AI (SHAP).

> **Disclaimer.** Academic project. The model learns statistical patterns in *recorded*
> crime, not in crime that occurred, and establishes no causality. It must not be used
> on its own for police deployment or public security decisions. See **Limitations**.

---

## Headline result

**No model beat the persistence baseline.**

On the temporal test set (2016–2020), the trivial rule "a municipality stays in the
class it was in last year" outperforms all eight tuned algorithms on every metric:

| Model | Accuracy | Macro F1 | Quadratic Kappa |
|---|---|---|---|
| **Persistence baseline (class at *t−1*)** | **0.818** | **0.776** | **0.833** |
| Logistic Regression | 0.805 | 0.760 | 0.824 |
| SVM | 0.792 | 0.745 | 0.814 |
| XGBoost | 0.790 | 0.739 | 0.809 |
| KNN | 0.782 | 0.732 | 0.807 |
| Random Forest | 0.772 | 0.713 | 0.791 |
| Gradient Boosting | 0.770 | 0.704 | 0.785 |
| Naive Bayes | 0.739 | 0.683 | 0.746 |
| Decision Tree | 0.732 | 0.670 | 0.733 |
| Majority-class baseline | 0.127 | 0.075 | 0.000 |

This is the finding of the project, not a failure of it. The target at year *t* is a
deterministic function of the crime indicators at *t*, and the features are those same
indicators at *t−1*. Because municipal crime rates are strongly inertial, a
cross-validated ROC-AUC of 0.92 measures **temporal persistence**, not additional
predictive power contributed by machine learning.

The honest conclusion: **with lagged crime indicators alone, municipal criminal
priority is essentially inertial.** Beating the persistence floor would require
information persistence does not carry — socioeconomic, demographic or policing
covariates, or modelling *class transitions* rather than class levels.

---

## Objective

Investigate whether lagged crime indicators can estimate the probability that a
municipality belongs to the High-priority group in the following year — **and whether
that estimate improves on a trivial persistence rule.**

---

## Dataset

Official crime-rate data from the São Paulo State Public Security Secretariat (SSP-SP),
via Kaggle.

**Scope.** The source covers only the larger municipalities: **79 municipalities**, not
the 645 in the state. All conclusions are limited to this subset.

| | |
|---|---|
| Raw rows | 2,304 |
| Empty separator rows removed | 566 |
| Clean municipality-year observations | **1,738** |
| Municipalities | **79** (22 years each — balanced panel) |
| Period | 1999–2020 |
| Regions | 12 |
| Observations after lagging | 1,659 |

Features used by the model (all lagged one year):

- Homicide rate per 100,000 inhabitants
- Theft rate per 100,000 inhabitants
- Robbery rate per 100,000 inhabitants
- Vehicle theft and robbery rate per 100,000 inhabitants

Rates per 100,000 **vehicles** were excluded: unavailable in 1999–2000, and containing
logically impossible zeros (see Data quality).

---

## Methodology

### 1. Data preparation

- Brazilian numeric format conversion (`1.085,19` → `1085.19`).
- Removal of 566 fully empty rows (CSV separator artefacts — `Ano` is also null in all
  of them).
- Outlier investigation: extremes were **kept**, with the denominator caveats below.

**Data quality — impossible zeros.** Boxplots by region exposed municipality-years in
the Santos region reporting `0.00` simultaneously in all three vehicle-denominated
columns, while the *same row* records 140–200 vehicle crimes per 100,000 inhabitants.
A positive per-inhabitant rate with a zero per-vehicle rate is internally
contradictory: the fault is in the denominator (missing or zeroed municipal fleet), not
an absence of crime. These zeros are now treated as missing. On a log-scale plot they
were also being dropped silently, which is what made the Santos boxes look truncated.

**Denominator bias.** Rates use *resident* population, but crimes are recorded where
the police district sits. Regional hubs (Barretos: 4,053 thefts/100k) absorb an entire
microregion's records, and coastal resort towns (Praia Grande: 4,919 vehicle thefts per
100k vehicles) have summer populations far above their resident count. These outliers
are coherent as records and misleading as rates.

### 2. Priority index

A composite index from four Z-score standardized indicators, equally weighted:

```
Priority Index = Z(Homicide) + Z(Theft) + Z(Robbery) + Z(Vehicle theft & robbery)
```

Split into three classes by the 33rd and 66th percentiles (thresholds: −1.148 and
0.523).

| Class | Definition |
|---|---|
| 🟢 Low | ≤ 33rd percentile |
| 🟡 Medium | 33rd–66th percentile |
| 🔴 High | > 66th percentile |

**Equal weights are a value judgement, not the absence of one.** Weighting z-scores
equally asserts that one standard deviation of theft is equivalent to one standard
deviation of homicide. A sensitivity analysis was run:

| Alternative specification | Agreement with equal weights |
|---|---|
| Homicide weighted ×2 | 87.8% |
| PCA first component | **61.7%** |

PC1 explains only **48.1%** of the variance. The correlation matrix explains why: theft
correlates *negatively* with homicide (−0.17) and is essentially uncorrelated with
vehicle crime (−0.01), while robbery and vehicle crime correlate strongly (**0.73**).
There is no single latent "crime factor" in this data, and the tercile classification
is **not robust** to the weighting choice.

### 3. Temporal design

| | |
|---|---|
| Training | 2000–2015 — 1,264 observations |
| Test | 2016–2020 — 395 observations |

Leakage controls:

- Z-score means and standard deviations estimated **only on the training period**.
- Tercile thresholds computed **only on the training period**.
- Features lagged one year, with an assertion that `Ano − lag_Ano == 1` so that a
  missing year can never be silently treated as the previous year.

**Prior shift — read every metric with this in mind.** Because crime fell in 2016–2020
while the thresholds stay anchored to the 1999–2015 scale, "High priority" in the test
set means "as high as the top tercile of the 1999–2015 era", not "high relative to
contemporaries":

| Class | Train | Test |
|---|---|---|
| Low | 33.5% | 63.5% |
| Medium | 33.0% | 23.8% |
| High | 33.5% | **12.7%** |

### 4. Baselines

Two reference points, both evaluated on the same test set:

1. **Majority class** (`DummyClassifier`) — the absolute floor.
2. **Persistence** — the class at *t* is the class at *t−1*. Uses exactly the
   information available to the models, while learning nothing.

A useful artefact: the majority-class baseline scores **recall = 1.00 on the High
class** with 12.7% accuracy, because it predicts "High" for everything. Recall in
isolation measures nothing.

### 5. Model comparison

Eight algorithms, each tuned by grid search inside the cross-validation:

- **Grouping by municipality.** `StratifiedGroupKFold(groups=Cidade)` replaces a
  shuffled split. A shuffled split would put Diadema 2003 in training and Diadema 2004
  in validation — effectively the same observation — inflating AUC and distorting the
  ranking.
- **Metric:** multiclass ROC-AUC (one-vs-rest).

| Model | CV ROC-AUC | Std | Best params |
|---|---|---|---|
| 🥇 Logistic Regression | 0.920 | 0.021 | `C=10` |
| SVM | 0.916 | 0.023 | `C=1, gamma=0.1` |
| KNN | 0.903 | 0.026 | `k=31, distance` |
| XGBoost | 0.897 | 0.027 | `lr=0.1, depth=2, n=300` |
| Gradient Boosting | 0.893 | 0.028 | `lr=0.1, depth=2, n=100` |
| Random Forest | 0.888 | 0.040 | `depth=None, leaf=5, n=300` |
| Naive Bayes | 0.864 | 0.042 | `var_smoothing=1e-6` |
| Decision Tree | 0.830 | 0.051 | `depth=8, leaf=10` |

With folds varying by ~0.02, Logistic Regression and SVM are a **statistical tie**.
Logistic Regression was selected for interpretability, not for superiority.

*Note:* Logistic Regression, SVM, Decision Tree and Random Forest use
`class_weight='balanced'`; Gradient Boosting, XGBoost, KNN and Naive Bayes have no
equivalent parameter. This asymmetry is acknowledged rather than hidden.

---

## Final model results

Logistic Regression on the temporal test set (2016–2020):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.98 | 0.83 | 0.90 | 251 |
| Medium | 0.58 | 0.69 | 0.63 | 94 |
| High | 0.65 | 0.90 | 0.76 | 50 |
| **Accuracy** | | | **0.81** | 395 |
| **Macro avg** | 0.73 | 0.81 | 0.76 | 395 |

The 0.90 recall on the High class comes with 0.65 precision: the model predicts 69
"High" against 50 actual. This over-prediction is **not** caused by `class_weight` —
`balanced` and `None` produce identical predictions here, because with `C=10` the
classes are nearly linearly separable. It is caused by the prior shift: the model
learned a world where one third of municipalities are High, and applies it to a test
period where 12.7% are.

---

## Explainability

**Logistic regression coefficients** (standardized features, therefore comparable):

| Lagged feature | Low | Medium | High |
|---|---|---|---|
| Vehicle theft & robbery | −2.449 | 0.198 | **2.251** |
| Robbery | −1.966 | 0.058 | 1.907 |
| Theft | −1.974 | 0.111 | 1.864 |
| Homicide | −1.737 | 0.136 | 1.600 |

Vehicle crime carries the most weight toward the High class, homicide the least — a
direct consequence of the index being dominated by property crime.

**SHAP** adds per-observation attribution. For a linear model the coefficients are
already the exact explanation; SHAP redistributes them per observation, which is useful
for local questions ("why did *this* municipality-year get *this* prediction?"). Global
and local (waterfall) plots are included.

**Ranking caveat.** 21 of 395 test observations receive a High probability above 0.99
and 200 below 0.01. The ordering at the top of the ranking is not informative:
separating 0.999999 from 0.999995 does not meaningfully distinguish municipalities.

---

## Technologies

Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Matplotlib · Plotly · Jupyter

---

## Limitations

- **Target circularity (central limitation).** The target is a deterministic function of
  the same indicators used as features, lagged one year. Performance reflects temporal
  persistence — confirmed by the persistence baseline outperforming every model.
- **Index sensitivity.** Tercile classification agrees with a PCA-derived index in only
  61.7% of observations. The classification is not robust to the weighting choice.
- **Scope.** 79 larger municipalities, not the 645 in the state.
- **Denominator bias.** Resident-population rates overstate regional hubs and coastal
  resort towns; three columns contained logically impossible zeros.
- **Recorded crime ≠ crime occurred.** The model learns about police notification,
  which depends on policing coverage and reporting propensity. Modelling police
  allocation from recorded crime carries a known **feedback loop** risk: more policing
  produces more records, which produce higher predicted priority (Lum & Isaac, 2016,
  *To predict and serve?*).
- **No covariates.** No socioeconomic, demographic, urban or police-strength variables.
  Classes are relative to the constructed index, not to an external severity criterion.
- **No causality.** The model identifies statistical association only.

---

## Possible extensions

- Model **class transitions** (which municipalities change level) instead of class
  levels — the persistence baseline cannot do this, so it is where machine learning
  could actually add signal.
- Add socioeconomic and demographic covariates (GDP per capita, Gini, urbanization,
  age structure, police strength).
- Use **ordinal** models (ordinal logistic regression) since Low < Medium < High.
- Recalibrate the index against an external severity criterion instead of equal weights.

---

## Author

**Pedro Silva**
Bachelor's student in Science and Technology — Federal University of Bahia (UFBA)
