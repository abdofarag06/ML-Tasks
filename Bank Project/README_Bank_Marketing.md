# 🏦 Bank Marketing: Term Deposit Prediction

A machine learning project that predicts whether a bank customer will subscribe to a term deposit after a marketing campaign. The notebook covers data understanding, cleaning, feature engineering, encoding, scaling, and a comparison of five classification algorithms.

## 📌 Overview

| | |
|---|---|
| **Task** | Binary classification: predict `deposit` (yes / no) |
| **Dataset** | `bank.csv`, 8,912 customers × 17 columns |
| **Best model** | SVM, linear kernel (Test accuracy ≈ **82.5%**) |
| **Notebook** | `Bannk_preprocessing.ipynb` |

## 📊 Dataset

| Group | Features |
|---|---|
| **Customer info** | `age`, `job`, `marital`, `education` |
| **Financial info** | `default` (credit in default), `balance` (average yearly balance), `housing` (housing loan), `loan` (personal loan) |
| **Current campaign** | `contact` (communication type), `day`, `month` (last contact), `duration` (last call length in seconds), `campaign` (contacts during this campaign) |
| **Previous campaign** | `pdays` (days since last contact, `-1` = never contacted), `previous` (earlier contacts), `poutcome` (earlier outcome) |
| **Target** | `deposit`: did the customer subscribe to a term deposit? |

The data has missing values in almost every column (up to 169 per column) and no duplicate rows.

## 🔧 Workflow

1. **Data understanding**: shape, `info()`, `describe()`, null and duplicate checks.
2. **Exploratory analysis**: deposit rate by education, marital status, job, contact type and number of campaign contacts.
3. **Skewness check**: histograms and skewness for `pdays`, `day`, `balance`, `campaign` and `previous` to choose an imputation strategy.
4. **Feature engineering**: the value `-1` in `pdays` means "never contacted". Added a binary `previously_contacted` feature and replaced `-1` with `NaN`.
5. **Missing values**
   - Median imputation for numeric columns
   - Most frequent imputation for categorical columns
   - Dropped rows with a missing target (`deposit`)
6. **Encoding**: one-hot encoding (`drop='first'`) for all 9 categorical columns, giving 43 features.
7. **Train/test split**: 80/20, stratified, `random_state=42` (7,122 train / 1,781 test).
8. **Scaling**: `StandardScaler` on numeric columns, fit on train only.
9. **Modeling**: five classifiers with hyperparameter tuning for the Decision Tree.
10. **Feature importance**: top predictors from a Decision Tree.

## 🤖 Models

- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors (`k=5`, Minkowski `p=3`)
- Support Vector Machine (linear kernel, `C=0.1`)
- Decision Tree (manual settings and `GridSearchCV`-tuned)

## 🏆 Results

Evaluated on the held-out test set:

| Model | Accuracy | Macro F1 |
|---|---|---|
| **SVM (linear)** | **0.825** | 0.82 |
| Logistic Regression | 0.820 | 0.819 |
| KNN | 0.810 | 0.809 |
| Decision Tree (tuned) | 0.806 | 0.81 |
| Decision Tree (manual) | 0.775 | 0.78 |
| Gaussian Naive Bayes | 0.705 | 0.692 |

**Takeaways**

- Linear models (SVM, Logistic Regression) lead, and the class balance is healthy, so accuracy and F1 tell the same story.
- Naive Bayes struggles: it catches only 53% of actual subscribers (recall for `yes`), because its independence assumption doesn't fit one-hot encoded features.
- The most important features were `duration`, `poutcome_success`, `day`, `balance` and `age`. Call duration dominates, with importance ≈ 0.33.

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**3. Run the notebook**

Make sure `bank.csv` is in the same folder as the notebook, then:
```bash
jupyter notebook Bannk_preprocessing.ipynb
```

## 🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn

## 🔮 Future Improvements

- Tune SVM hyperparameters (the grid search is prepared in the notebook but commented out)
- Try ensemble models (Random Forest, XGBoost, LightGBM)
- Handle class-specific metrics such as recall for `yes` if the business goal is finding subscribers
- Add cross-validation for a more reliable comparison
- Save the best model with `joblib` and wrap it in a simple app (Streamlit / Flask)

## 👤 Author

**Farag**
AI Undergraduate, Faculty of Computers & Artificial Intelligence, Benha University

---

⭐ If you found this project useful, consider giving it a star!
