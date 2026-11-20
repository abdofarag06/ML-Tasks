# 🏠 House Rent Prediction

A machine learning project that predicts house rent from property attributes such as size, BHK, location, furnishing and tenant preference. The notebook covers data cleaning, exploratory analysis, encoding, scaling, and a comparison of **18 regression models**, including ensembles.

## 📌 Overview

| | |
|---|---|
| **Task** | Regression: predict `Rent` |
| **Dataset** | `House_Rent_Dataset.csv`, 4,746 listings × 12 columns |
| **Best model** | HistGradientBoosting (Test R² ≈ **0.77**) |
| **Notebook** | `House_Rent_Regression.ipynb` |

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `Posted On` | Date | Listing date (dropped) |
| `BHK` | Numeric | Number of bedrooms, hall, kitchen (1–6) |
| `Rent` | Numeric | **Target**: monthly rent |
| `Size` | Numeric | Property size in sq. ft. |
| `Floor` | Categorical | Floor number out of total floors |
| `Area Type` | Categorical | Super Area / Carpet Area / Built Area |
| `Area Locality` | Categorical | Locality name (2,233 unique values) |
| `City` | Categorical | 6 cities |
| `Furnishing Status` | Categorical | Furnished / Semi-Furnished / Unfurnished |
| `Tenant Preferred` | Categorical | Bachelors / Family / Bachelors-Family |
| `Bathroom` | Numeric | Number of bathrooms (1–10) |
| `Point of Contact` | Categorical | Contact Owner / Agent / Builder |

## 🔧 Workflow

1. **Data loading & inspection**: shape, `describe()`, `info()`, null and duplicate checks.
2. **Cleaning**
   - Dropped `Posted On`
   - Imputed `Bathroom` (median), `Size` (mean) and `Area Type` (most frequent)
   - Dropped the few rows with a missing `Rent`
   - Removed 8 duplicate rows
3. **Exploratory analysis**: category distributions, area type by city, size and rent by BHK and furnishing status, rent vs. size scatter plots, box plots.
4. **Outlier removal**: kept listings with `Size < 5000` and `Rent < 1,500,000` (4,727 rows remain).
5. **Encoding**
   - Label encoding for `Floor`, `Area Locality`, `City`
   - One-hot encoding for `Area Type`, `Furnishing Status`, `Tenant Preferred`, `Point of Contact`
6. **Scaling**: `RobustScaler`, chosen for its resistance to outliers.
7. **Train/test split**: 80/20, `random_state=42`.
8. **Modeling & evaluation**: Linear Regression, SGDRegressor, then a full comparison of 18 models on MAE, MSE, RMSE and R² (train and test), with actual-vs-predicted plots for each.

## 🤖 Models Compared

- **Linear:** Linear Regression, Lasso, Ridge, SGDRegressor
- **Instance / kernel-based:** KNN, SVM (SVR)
- **Trees:** Decision Tree
- **Bagging:** Random Forest, Extra Trees, Bagging Regressor
- **Boosting:** AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM
- **Ensembles:** Voting (simple and weighted), Stacking (RF + GB + LightGBM → XGBoost meta-model)

## 🏆 Results

Top models ranked by test R²:

| Model | MAE (Test) | RMSE (Test) | R² (Train) | R² (Test) |
|---|---|---|---|---|
| **HistGradientBoosting** | 11,612.6 | 25,854.0 | 0.877 | **0.769** |
| LightGBM | 11,750.9 | 26,625.4 | 0.884 | 0.755 |
| Extra Trees | 11,962.5 | 29,800.3 | 1.000 | 0.693 |
| Gradient Boosting | 11,829.8 | 30,186.2 | 0.873 | 0.685 |
| Random Forest | 11,735.2 | 30,305.5 | 0.944 | 0.682 |
| XGBoost | 11,645.8 | 31,433.5 | 0.970 | 0.658 |
| KNN | 13,693.0 | 32,007.4 | 0.739 | 0.646 |
| Linear Regression | 23,042.1 | 38,005.6 | 0.451 | 0.500 |

**Takeaways**

- Boosting methods give the best results, while linear models (R² ≈ 0.50) fall far behind, which points to strongly non-linear relationships in rent.
- Tree-based models overfit noticeably (Extra Trees reaches R² ≈ 1.0 on train data, Decision Tree 0.97 → 0.43), so regularization and tuning could help.

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

Make sure `House_Rent_Dataset.csv` is in the same folder as the notebook, then:
```bash
jupyter notebook House_Rent_Regression.ipynb
```

## 🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · LightGBM

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) and cross-validation
- Log-transform the skewed `Rent` target
- Parse `Floor` into numeric floor and total-floor features
- Better handling of the high-cardinality `Area Locality` column (target or frequency encoding)
- Feature importance analysis
- Save the best model with `joblib` and wrap it in a simple app (Streamlit / Flask)

## 👤 Author

**Farag**
AI Undergraduate, Faculty of Computers & Artificial Intelligence, Benha University

---

⭐ If you found this project useful, consider giving it a star!
