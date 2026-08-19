#  Criminal Priority Prediction

Machine Learning project for classifying municipalities in São Paulo into **Low, Medium, and High criminal priority levels** using historical crime data.

The project combines **Machine Learning, temporal modeling, model comparison, and Explainable AI (SHAP)** to identify patterns associated with higher criminal priority.

>  **Disclaimer:** This is an academic project. Predictions are based on historical data and should not be interpreted as operational recommendations for police deployment or public security decisions.

---

## Objective

Investigate whether historical crime indicators can be used to estimate the probability that a municipality will belong to the **High-priority** group in a subsequent year.

---

##  Dataset

Historical crime-rate data from municipalities in the state of São Paulo.

The final model uses:

- Homicide Rate per 100,000 inhabitants
- Theft Rate per 100,000 inhabitants
- Robbery Rate per 100,000 inhabitants
- Vehicle Theft and Robbery Rate per 100,000 inhabitants

The original dataset contains **2,304 municipality-year observations** covering **1999–2020**.

---

## 🔎 Methodology

### 1. Data Preparation

- Missing value treatment
- Numerical conversion
- Municipality name standardization
- Outlier investigation
- Feature standardization

### 2. Priority Index

A composite index was created using Z-score standardized crime indicators:

```text
Priority Index =
Z(Homicide)
+ Z(Theft)
+ Z(Robbery)
+ Z(Vehicle Theft and Robbery)
```

Equal weights were used for all indicators.

The index was divided into three classes:

| Class | Definition |
|---|---|
| 🟢 Low | ≤ 33rd percentile |
| 🟡 Medium | 33rd–66th percentile |
| 🔴 High | > 66th percentile |

### 3. Temporal Modeling

Instead of a random split, the project uses a temporal approach:

```text
1999–2015 → Training
2016–2020 → Testing
```

Lagged features were created so that previous-year crime indicators are used to classify the following year.

This helps reduce **data leakage** and provides a more realistic forecasting scenario.

---

##  Models

Eight Machine Learning algorithms were compared:

- Logistic Regression
- SVM
- Random Forest
- Gradient Boosting
- XGBoost
- KNN
- Naive Bayes
- Decision Tree

### Cross-Validation Results

| Model | ROC-AUC |
|---|---:|
| 🥇 Logistic Regression | **0.922** |
| SVM | 0.918 |
| Gradient Boosting | 0.901 |
| Random Forest | 0.899 |
| XGBoost | 0.893 |
| KNN | 0.881 |
| Naive Bayes | 0.870 |
| Decision Tree | 0.766 |

**Logistic Regression was selected as the final model.**

---

## 📈 Final Results

On the temporal test set (2016–2020):

| Metric | Score |
|---|---:|
| Accuracy | **0.81** |
| Macro F1 | **0.76** |
| Recall — Low | 0.83 |
| Recall — Medium | 0.69 |
| Recall — High | **0.90** |
| F1 — High | **0.76** |

The **90% recall for the High-priority class** indicates that the model identified most observations belonging to the highest-priority group according to the project's target definition.

---

## Explainable AI

**SHAP (SHapley Additive exPlanations)** was used to understand the model's predictions.

The analysis includes:

- **Global explanations:** which features have the greatest influence on predictions.
- **Local explanations:** why a specific municipality-year observation received its prediction.

---

##  Priority Ranking

The final model generates predicted probabilities for the **High-priority class**, allowing municipality-year observations to be ranked according to their predicted priority.

---

##  Technologies

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `SHAP` · `Matplotlib` · `Seaborn` · `Jupyter`

---

##  Limitations

- The model relies on historical crime indicators.
- Socioeconomic, demographic, and urban variables are not included.
- The priority classes are relative to the constructed index.
- The model identifies statistical patterns and does not establish causality.
- Results should not be used independently for real-world public security decisions.

---

##  Author

**Pedro Silva**

Bachelor's student in Science and Technology — **Federal University of Bahia (UFBA)**



---

⭐ If you find this project interesting, consider giving the repository a star!
