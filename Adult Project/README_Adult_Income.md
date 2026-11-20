# 💼 Adult Census Income Prediction

A machine learning project that predicts whether a person earns **more than $50K per year** from census data (age, education, occupation, marital status, working hours and more). The notebook covers cleaning, encoding, class-imbalance handling with SMOTE, and a comparison of multiple classifiers.

## 📌 Overview

| | |
|---|---|
| **Task** | Binary classification: income `<=50K` vs `>50K` |
| **Dataset** | `Adult Census Income.csv`, 32,561 records × 15 columns |
| **Class balance** | Imbalanced: about 24% of records earn `>50K` |
| **Notebook** | `Adult.ipynb` |

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age in years |
| `workclass` | Categorical | Employment type (Private, Self-emp, Gov, ...) |
| `fnlwgt` | Numeric | Census sampling weight (dropped) |
| `education` | Categorical | Education level (dropped, redundant with `education.num`) |
| `education.num` | Numeric | Education level as a number (1–16) |
| `marital.status` | Categorical | Marital status |
| `occupation` | Categorical | Job type |
| `relationship` | Categorical | Household relationship |
| `race` | Categorical | Race |
| `sex` | Categorical | Sex |
| `capital.gain` / `capital.loss` | Numeric | Capital gains and losses |
| `hours.per.week` | Numeric | Weekly working hours |
| `native.country` | Categorical | Country of origin |
| `income` | Categorical | **Target**: `<=50K` or `>50K` |

## 🔧 Workflow

1. **Data understanding**: `info()`, `describe()`, null checks.
2. **Cleaning**
   - Removed 24 duplicate rows (32,537 remain)
   - Missing values were stored as `"?"`, so converted them to `NaN` (found in `occupation`, `workclass`, `native.country`)
   - Filled them with the most frequent value
   - Dropped `fnlwgt` and `education`
3. **Exploratory analysis**: income vs. marital status, sex and age.
4. **Encoding**
   - One-hot encoding (`drop='first'`) for `workclass`, `marital.status`, `occupation`, `relationship`, `race`, `sex`
   - Label encoding for `native.country`
5. **Train/test split**: 80/20, stratified, `random_state=42` (26,029 train / 6,508 test).
6. **Scaling**: `StandardScaler` on `age`, `capital.gain`, `capital.loss`, `hours.per.week`, fit on train only.
7. **Imbalance handling**: `class_weight='balanced'` for several models, and **SMOTE** on the training set only.
8. **Modeling & evaluation**: Logistic Regression, KNN, Naive Bayes, SVC, then a broader comparison including tree ensembles, boosting, Voting and Stacking, evaluated with confusion matrices, classification reports and ROC-AUC.

## 🤖 Models

Logistic Regression · KNN · Gaussian Naive Bayes · SVC (RBF) · Decision Tree · Random Forest · Extra Trees · AdaBoost · Gradient Boosting · HistGradientBoosting · XGBoost · LightGBM · Voting Classifier · Stacking Classifier

## 📏 Evaluation

Because only ~24% of the records are `>50K`, accuracy alone is not enough. Models are compared on:

- Precision, recall and F1 for the `>50K` class
- Balanced accuracy
- ROC-AUC and PR-AUC
- Confusion matrices

| Model | Accuracy | F1 (`>50K`) | ROC-AUC |
|---|---|---|---|
| _To be added_ | | | |

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm tqdm jupyter
```

**3. Run the notebook**

Make sure `Adult Census Income.csv` is in the same folder as the notebook, then:
```bash
jupyter notebook Adult.ipynb
```

## 🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · imbalanced-learn · XGBoost · LightGBM

## 🔮 Future Improvements

- Hyperparameter tuning with stratified cross-validation
- Compare SMOTE against class weights and threshold tuning
- One-hot or target encoding for `native.country` instead of label encoding
- Feature importance / SHAP analysis
- Fairness analysis across `sex` and `race`
- Save the best model with `joblib` and wrap it in a simple app (Streamlit / Flask)

## 👤 Author

**Farag**
AI Undergraduate, Faculty of Computers & Artificial Intelligence, Benha University

---

⭐ If you found this project useful, consider giving it a star!
