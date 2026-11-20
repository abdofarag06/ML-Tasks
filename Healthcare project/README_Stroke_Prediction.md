# 🧠 Stroke Prediction: Healthcare Data

A machine learning project that predicts whether a patient is likely to have a stroke from demographic, medical and lifestyle features. The notebook walks through preprocessing, exploratory analysis, and a comparison of 12 classifiers on a **highly imbalanced** dataset, evaluated with metrics that go beyond accuracy.

## 📌 Overview

| | |
|---|---|
| **Task** | Binary classification: predict `stroke` (0 / 1) |
| **Dataset** | `healthcare-dataset-stroke-data.csv`, 5,110 patients × 12 columns |
| **Class balance** | Only ~4.9% of patients had a stroke |
| **Notebook** | `Preprocessing_Healthcare.ipynb` |

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `id` | Identifier | Patient ID (dropped) |
| `gender` | Categorical | Male / Female / Other |
| `age` | Numeric | Age in years |
| `hypertension` | Binary | 1 = has hypertension |
| `heart_disease` | Binary | 1 = has heart disease |
| `ever_married` | Categorical | Yes / No |
| `work_type` | Categorical | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Categorical | Urban / Rural |
| `avg_glucose_level` | Numeric | Average blood glucose (mg/dL) |
| `bmi` | Numeric | Body mass index (201 missing values) |
| `smoking_status` | Categorical | formerly smoked, never smoked, smokes, Unknown |
| `stroke` | Binary | **Target**: 1 = had a stroke |

## 🔧 Workflow

1. **Data understanding**: `describe()`, `info()`, null and duplicate checks (no duplicates).
2. **Cleaning**: dropped `id`; filled missing `bmi` with the **median**, since its distribution is right-skewed (skew ≈ 1.06).
3. **Exploratory analysis**: stroke vs. BMI, glucose, age, gender and marital status.
4. **Encoding**: one-hot encoding (`drop='first'`) for all 5 categorical columns.
5. **Train/test split**: 80/20, stratified so both sets keep the same stroke ratio (4,088 train / 1,022 test).
6. **Scaling**: `StandardScaler` on `age`, `avg_glucose_level` and `bmi`, fit on train only.
7. **Modeling**: 10 base classifiers plus a Voting and a Stacking ensemble.
8. **Evaluation**: confusion matrix, classification report, ROC curve with ROC-AUC, and precision-recall curve with PR-AUC for every model.
9. **Dimensionality reduction**: PCA (7 components) to inspect which features drive the main variance.

## 🤖 Models Compared

Logistic Regression · KNN · SVC · Decision Tree · Gaussian Naive Bayes · Random Forest · Extra Trees · AdaBoost · Gradient Boosting · HistGradientBoosting · Voting Classifier (soft) · Stacking Classifier

## 🏆 Results

Test-set results, sorted by F1 for the stroke class:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Voting Classifier** | 0.933 | 0.263 | 0.20 | **0.227** | 0.829 | 0.217 |
| Decision Tree | 0.909 | 0.123 | 0.14 | 0.131 | 0.544 | 0.059 |
| Naive Bayes | 0.230 | 0.059 | **0.98** | 0.111 | 0.799 | 0.173 |
| HistGradientBoosting | 0.945 | 0.250 | 0.06 | 0.097 | 0.825 | 0.174 |
| Extra Trees | 0.943 | 0.214 | 0.06 | 0.094 | 0.752 | 0.122 |
| Logistic Regression | 0.952 | 1.000 | 0.02 | 0.039 | **0.842** | **0.270** |
| Stacking | 0.950 | 0.333 | 0.02 | 0.038 | 0.831 | 0.235 |
| Random Forest | 0.950 | 0.333 | 0.02 | 0.038 | 0.791 | 0.159 |

**Takeaways**

- **Accuracy is misleading here.** Always predicting "no stroke" already gives ≈ 95% accuracy, so most models at 95% learned almost nothing about the minority class. SVC, KNN, AdaBoost and Gradient Boosting caught **zero** strokes.
- **Logistic Regression ranks best by ROC-AUC (0.84) and PR-AUC (0.27)**, which means it ranks risky patients well even though its default 0.5 threshold is too strict.
- **Naive Bayes** catches 98% of strokes but with very low precision (many false alarms), a trade-off that could suit a screening scenario.
- Random Forest, Extra Trees and the Decision Tree fit the training data perfectly (train accuracy 1.0) and generalize poorly.
- **PCA:** age and BMI contribute most to the first principal component, followed by glucose level.

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm tqdm jupyter
```

**3. Run the notebook**

Make sure `healthcare-dataset-stroke-data.csv` is in the same folder as the notebook, then:
```bash
jupyter notebook Preprocessing_Healthcare.ipynb
```

## 🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · LightGBM

## 🔮 Future Improvements

- **Handle class imbalance:** `class_weight='balanced'`, SMOTE or undersampling
- **Tune the decision threshold** (using the precision-recall curve) instead of the default 0.5
- Optimize for **recall / PR-AUC** rather than accuracy
- Hyperparameter tuning with stratified cross-validation
- Enable XGBoost, LightGBM and CatBoost, which are already prepared in the notebook
- Add feature importance / SHAP analysis for explainability

## ⚠️ Disclaimer

This project is for educational purposes only and is **not** a medical diagnostic tool.

## 👤 Author

**Farag**
AI Undergraduate, Faculty of Computers & Artificial Intelligence, Benha University

---

⭐ If you found this project useful, consider giving it a star!
