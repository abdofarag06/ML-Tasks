# 🚗 Used Car Price Prediction

A machine learning project that predicts used car prices from vehicle attributes. The notebook covers the full workflow: data cleaning, visualization, feature engineering, and a comparison of **18 regression models**, including ensembles.

## 📌 Overview

| | |
|---|---|
| **Task** | Regression: predict car `Price` |
| **Dataset** | `cars_data.csv`, 1,000 rows × 8 columns |
| **Best model** | HistGradientBoosting (Test R² ≈ **0.78**) |
| **Notebook** | `Cars_modeling.ipynb` |

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `Brand` | Categorical | Car manufacturer (e.g. Kia, BMW, Mercedes, Nissan) |
| `Year` | Numeric | Manufacturing year (2010–2023) |
| `Mileage` | Numeric | Distance driven |
| `Engine_Size` | Numeric | Engine size in liters (1.0–4.0) |
| `Fuel_Type` | Categorical | Petrol, Diesel, Electric, Hybrid |
| `Transmission` | Categorical | Manual / Automatic |
| `Previous_Owners` | Numeric | Number of previous owners (1–5) |
| `Price` | Numeric | **Target**: car price (3,000–43,994) |

The raw data contains missing values in every column (up to 50 per column) and blank-space entries in `Previous_Owners`. There are no duplicate rows.

## 🔧 Workflow

1. **Data loading & inspection**: shape, `describe()`, `info()`, null and duplicate checks.
2. **Missing value handling** (`SimpleImputer`)
   - Mean for `Year`, `Mileage`, `Engine_Size` (near-zero skew)
   - Median for `Price` (right-skewed, skew ≈ 2.1)
   - Most frequent for categorical columns and `Previous_Owners`
3. **Exploratory analysis**: transmission vs. fuel type counts, price vs. mileage / engine size / previous owners, correlation heatmap.
4. **Encoding**
   - One-hot encoding for `Fuel_Type` and `Transmission` (`drop='first'`)
   - Label encoding for `Brand`
5. **Train/test split**: 80/20, `random_state=42`
6. **Scaling**: `StandardScaler` on `Year` and `Mileage`, fit on train only to avoid leakage.
7. **Modeling & evaluation**: 18 models compared on MAE, MSE, RMSE and R² for both train and test sets, with actual-vs-predicted plots for each model.

## 🤖 Models Compared

- **Linear:** Linear Regression, Lasso, Ridge
- **Instance / kernel-based:** KNN, SVM (SVR)
- **Trees:** Decision Tree
- **Bagging:** Random Forest, Extra Trees, Bagging Regressor
- **Boosting:** AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM
- **Ensembles:** Voting (simple and weighted), Stacking (RF + GB + LightGBM → XGBoost meta-model)

## 🏆 Results

Top models ranked by test R²:

| Model | MAE (Test) | RMSE (Test) | R² (Train) | R² (Test) |
|---|---|---|---|---|
| **HistGradientBoosting** | 1,666.6 | 2,969.8 | 0.954 | **0.778** |
| Stacking | 1,717.9 | 3,038.6 | 0.895 | 0.772 |
| LightGBM | 1,717.9 | 3,038.6 | 0.954 | 0.768 |
| Gradient Boosting | 1,794.7 | 3,040.8 | 0.916 | 0.767 |
| XGBoost | 1,844.0 | 3,216.1 | 0.991 | 0.740 |
| Random Forest | 2,173.8 | 3,633.7 | 0.950 | 0.668 |
| KNN | 2,225.1 | 3,800.6 | 0.756 | 0.637 |

**Takeaways**

- Gradient boosting methods perform best on this dataset, and linear models (R² ≈ 0.45) fall well behind, which suggests non-linear relationships between features and price.
- Tree-based models show a large gap between train and test R² (e.g. Decision Tree: 0.97 → 0.18), a clear sign of overfitting without tuning.
- Extra Trees reaches R² = 1.0 on train data, so it memorizes the training set.

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

Make sure `cars_data.csv` is in the same folder as the notebook, then:
```bash
jupyter notebook Cars_modeling.ipynb
```

## 🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · LightGBM

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation for more reliable model comparison
- Outlier handling for `Price`
- Feature importance analysis
- Use one-hot encoding for `Brand` instead of label encoding
- Save the best model with `joblib` and wrap it in a simple app (Streamlit / Flask)

## 👤 Author

**Farag**
AI Undergraduate, Faculty of Computers & Artificial Intelligence, Benha University

---

⭐ If you found this project useful, consider giving it a star!
