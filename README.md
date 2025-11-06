# Credit Risk Scoring Model

This project demonstrates the process of building a **Credit Risk Scoring System** using various **Machine Learning algorithms** — Decision Tree, Random Forest, and XGBoost — to predict whether a client will **default** or **repay** a loan.

---

## Overview

The workflow includes:
1. **Data Cleaning & Preprocessing**  
2. **Model Building** (Decision Tree, Random Forest, XGBoost)  
3. **Hyperparameter Tuning**  
4. **Model Evaluation**  
5. **Final Model Selection**

---

## Data Cleaning and Preparation

- **Dataset:** `CreditScoring.csv`  
- Transformed categorical columns: `home`, `marital`, `records`, `job`  
- Replaced invalid values (`99999999`) in `income`, `assets`, and `debt` with `NaN`  
- Removed rows with unknown `status`  
- Split the dataset into:
  - **Train:** 60%
  - **Validation:** 20%
  - **Test:** 20%

---

## Decision Tree Model

### Manual Rule-Based Risk Assessment

A simple hand-crafted function was initially implemented to assess risk based on:
- Credit record (`records`)
- Job type (`job`)
- Asset value (`assets`)

### Automated Learning Model

Trained using `DecisionTreeClassifier` from Scikit-learn.

**Observation:**
- Overfitting at full depth (`AUC = 1.0` train, `0.67` validation)
- Controlled overfitting with:

`DecisionTreeClassifier(max_depth=10, min_samples_leaf=15)`

**Validation AUC:** ≈ **0.73**

---

##  Random Forest Model

The **Random Forest** algorithm was used to build an ensemble of decision trees that improves generalization and reduces overfitting. It averages predictions from multiple trees, each trained on random subsets of data and features.

###  Parameters Tuned

- `n_estimators`: Number of trees in the forest  
- `max_depth`: Maximum depth of each tree  
- `min_samples_leaf`: Minimum samples required at a leaf node  

###  Best Configuration

`RandomForestClassifier(n_estimators=160, max_depth=10, min_samples_leaf=3, random_state=1, n_jobs=-1)`

###  Model Performance

| Dataset | ROC-AUC Score |
|----------|---------------|
| **Training** | ~0.88 |
| **Validation** | **~0.80** |

**Interpretation:**  
- Random Forest successfully reduced overfitting compared to Decision Tree.  
- It provided a strong baseline before moving to Gradient Boosting.

---

##  XGBoost Model

**XGBoost (Extreme Gradient Boosting)** was used for further improvement by sequentially building trees that correct errors of previous ones.

### Parameters Tuned

| Parameter | Description | Best Value | Effect |
|------------|--------------|-------------|--------|
| `eta` | Learning rate | **0.05** | Smooth convergence, prevents overfitting |
| `max_depth` | Tree depth | **3** | Best generalization |
| `min_child_weight` | Minimum weight of child nodes | **30** | Strong regularization |
| `num_boost_round` | Boosting rounds | **190** | Optimal convergence |

###  Final Parameters

`eta=0.05, max_depth=3, min_child_weight=30, objective='binary:logistic', eval_metric='auc', nthread=8, seed=1`

**Validation AUC:** **0.835**  
**Test AUC:** **0.8212 (82.12%)**

---

## Hyperparameter Insights

### Learning Rate (`eta`)

| η | Behavior | Notes |
|---|-----------|-------|
| 0.01 | Slow learning, needs more rounds | Underfits |
| **0.05** | ✅ Best AUC (~0.835), smooth convergence | Balanced |
| 0.1 | Slight overfitting | Acceptable |
| 0.3 | Fast but unstable | Overfits |
| 1.0 | Diverges | Poor |

### Tree Depth (`max_depth`)

| Depth | AUC | Behavior |
|--------|------|-----------|
| **3** | 0.83 | ✅ Best generalization |
| 4 | 0.829 | Good |
| 6 | 0.828 | Slight overfit |
| 10 | 0.80 | Poor generalization |

### Minimum Child Weight (`min_child_weight`)

| min_child_weight | AUC | Stability | Verdict |
|------------------|------|------------|----------|
| **30** | 0.835 | ✅ Very High | Best |
| 10 | 0.834 | High | Good |
| 1 | 0.833 | Medium | Acceptable |

---

##  Model Comparison

| Model | Key Parameters | Validation AUC | Test AUC |
|--------|----------------|----------------|-----------|
| Decision Tree | max_depth=10, min_samples_leaf=15 | 0.73 | 0.70 |
| Random Forest | n_estimators=160, max_depth=10, min_samples_leaf=3 | 0.80 | 0.79 |
| **XGBoost (Final)** | eta=0.05, max_depth=3, min_child_weight=30 | **0.835** | **0.821** |

---

##  Final Model Implementation

`xgb.train(xgb_params, dtrain, num_boost_round=190)`

**Final Test AUC:** **82.12%**

---

## 🚀 How to Run

### Install Dependencies
`pip install pandas numpy seaborn matplotlib scikit-learn xgboost`

### Run the Notebook
`jupyter notebook Credit_Risk_Scoring.ipynb`

### Folder Structure
├── credit-risk-scoring.ipynb
├── CreditScoring.csv
├── LICENSE
└── README.md

1 directory, 4 files

---

##  Summary

- **Best Model:** XGBoost  
- **Best Parameters:** `eta=0.05`, `max_depth=3`, `min_child_weight=30`  
- **Final Test AUC:** **82.12%**  
- The model generalizes well and is suitable for real-world credit scoring applications.

---

## Author

**Purushothama D (Codevalhalla)**  
Machine Learning Engineer | Data Enthusiast  
🔗 [GitHub: @Codevalhalla](https://github.com/Codevalhalla)

