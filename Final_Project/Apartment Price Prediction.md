# Apartment Price Prediction

This project explores an apartment-prices dataset and builds regression models to predict apartment prices in Egyptian pounds (EGP). The notebook covers exploratory data analysis, feature engineering, missing-value handling, preprocessing with scikit-learn pipelines, model comparison, ensemble regression, and a Gradio prediction interface.

## Project files

```text
.
├── Apartments_prices.ipynb       # Main analysis and modeling notebook
└── README_Apartments_Prices.md   # Project documentation
```

The dataset used by the notebook is named:

```text
Apartments Prices Dataset(Target column price).csv
```

Place the CSV file in the project directory and update the notebook’s data-loading path before running it.

## Objective

The goal is to predict an apartment’s total `price` using property characteristics such as area, number of rooms, bathrooms, floor level, year built, seller type, view, payment method, and finishing type.

The target variable is:

```text
price
```

Prices are represented in EGP.

## Dataset columns

| Column | Description | Role / treatment |
|---|---|---|
| `price` | Total apartment price in EGP | Regression target; converted from text to numeric |
| `area_sqm` | Apartment area | Converted from values such as `120 m²` to numeric |
| `rooms` | Number of rooms | Converted from values such as `3 rooms` to numeric |
| `bathrooms` | Number of bathrooms | Converted from values such as `2 bathroom` to numeric |
| `floor_level` | Apartment floor | Converted to numeric; invalid values become missing |
| `year_built` | Construction year | Numeric feature |
| `seller_type` | Seller category | Missing values imputed; one-hot encoded |
| `view` | Apartment view | Missing values imputed; one-hot encoded |
| `price_per_sqm` | Price per square meter | Removed during the notebook’s exploratory feature-engineering stage; see the implementation note below |
| `payment_method` | Payment option | Missing values imputed; one-hot encoded |
| `finishing_type` | Finishing level | One-hot encoded |

## Notebook workflow

### 1. Load and inspect the data

The notebook loads the CSV file with pandas and examines:

- The first rows of the dataset
- Data types and non-null counts
- Descriptive statistics
- Missing-value counts
- Duplicate rows
- Dataset shape

The original dataset contains 5,578 rows and 11 columns before duplicate removal. The notebook identifies 1,298 duplicate rows and removes them.

### 2. Feature engineering

The notebook performs the following transformations:

- Drops `price_per_sqm` during the initial feature-engineering section.
- Converts `area_sqm` from strings such as `120 m²` to floating-point values.
- Converts `price` from strings such as `2,163,000 EGP` to numeric values.
- Converts `bathrooms` from strings such as `3 bathroom` to numeric values.
- Converts `rooms` from strings such as `2 rooms` to numeric values.
- Converts `floor_level` to numeric values using `errors='coerce'`.
- Replaces rare `seller_type` values (`Cash`, `Cash or Installments`, and `Installments`) with missing values so they can be imputed consistently.

### 3. Handle missing values

Numerical columns are imputed with their median:

```python
['floor_level', 'bathrooms', 'area_sqm', 'rooms', 'price']
```

Categorical columns are imputed with their most frequent value:

```python
['seller_type', 'view', 'payment_method']
```

The notebook also plots distributions and calculates skewness for several numerical features.

### 4. Visualize relationships

The notebook generates visualizations including:

- Average price by number of rooms
- Price versus apartment area
- Price versus construction year
- Average price by view
- A numerical correlation heatmap
- Finishing-type counts
- Boxplots for rooms, bathrooms, and year built

### 5. Build the preprocessing pipeline

The final pipeline is designed to prevent preprocessing leakage by fitting transformations only on the training split.

For numerical features, it applies:

1. Median imputation
2. IQR-based outlier clipping
3. Standard scaling

For categorical features, it applies:

1. Most-frequent-value imputation
2. One-hot encoding with `handle_unknown='ignore'`

The data is split into training and test sets using:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

The processed outputs are stored as:

```python
X_train_df
X_test_df
y_train
y_test
```

### 6. Train and compare regression models

The notebook evaluates several regression algorithms, including:

- Linear Regression
- Lasso
- Ridge
- K-Nearest Neighbors
- Linear Support Vector Regression
- Decision Tree
- Random Forest
- Extra Trees
- Bagging
- AdaBoost
- Gradient Boosting
- Histogram Gradient Boosting
- XGBoost
- LightGBM
- Voting regressors
- A stacking regressor

The following metrics are calculated on both the training and test sets:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score

The results are collected in `errors_df` and sorted by test-set R² score. Bar charts compare train and test performance for each metric.

### 7. Launch the prediction interface

The final section creates a Gradio interface intended to predict apartment prices from user-provided details. The interface uses the fitted preprocessing pipeline and an XGBoost regressor, returning a formatted price such as:

```text
3,250,000.00 EGP
```

## Installation

Install the required packages with:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn tqdm xgboost lightgbm gradio
```

Depending on the installed versions of scikit-learn and XGBoost, minor API adjustments may be required.

## How to run

1. Put the notebook and CSV dataset in the same project directory.
2. Open the notebook in Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

3. Update the `pd.read_csv(...)` path to point to the local CSV file.
4. Run the notebook cells in order.
5. Review `errors_df` to compare model performance.
6. Run the final Gradio cell only after the preprocessing pipeline and selected model have been trained successfully.

## Important implementation notes

The notebook currently contains a few inconsistencies that should be corrected before running it from a clean environment:

1. **Dataset path is machine-specific.** The notebook uses a Windows path such as `D:\Courses\...`. Replace it with a portable path, for example:

   ```python
   df = pd.read_csv("Apartments Prices Dataset(Target column price).csv")
   ```

2. **`price_per_sqm` is dropped but later included in the pipeline.** The notebook removes `price_per_sqm` early, but the final `num_features` list still includes it. Either keep the column and convert it to numeric, or remove it from `num_features`:

   ```python
   num_features = [
       "area_sqm",
       "rooms",
       "bathrooms",
       "floor_level",
       "year_built",
   ]
   ```

3. **The Gradio function and inputs do not match.** `predict_apartment_price` includes a `price_per_sqm` argument, but the Gradio input list does not provide an input for it. The function signature and input list must contain the same fields. If `price_per_sqm` is excluded from the pipeline, remove it from the function signature as well.

4. **Use the fitted pipeline for inference.** New apartment records must contain the same raw feature names expected by the fitted `preprocessing_pipeline`.

5. **The notebook evaluates models on the same preprocessed split.** For more reliable model selection, consider cross-validation and a separate final holdout set.

6. **Outlier processing clips values rather than deleting rows.** The `IQRClipper` transformer calculates bounds from the training data and clips extreme values during transformation. This is different from dropping outlier rows.

7. **The final app uses XGBoost specifically.** Make sure the `XGBoost` model has been trained before launching Gradio:

   ```python
   xgb_model = models["XGBoost"]
   xgb_model.fit(X_train_df, y_train)
   ```

## Possible improvements

- Keep `price_per_sqm` as a derived feature calculated from `price / area_sqm`, rather than using it as an independent input if it is mathematically derived from the target.
- Add cross-validation and hyperparameter tuning.
- Log-transform the highly skewed `price` target and transform predictions back to EGP.
- Compare models using a consistent validation protocol.
- Save the fitted preprocessing pipeline and best model with `joblib`.
- Add prediction intervals or uncertainty estimates.
- Add input validation to the Gradio interface.
- Separate exploratory analysis, preprocessing, training, evaluation, and deployment code into reusable Python modules.

## License and data attribution

The notebook does not specify a license or dataset source. Add the appropriate attribution and licensing information before publishing or redistributing the project.
