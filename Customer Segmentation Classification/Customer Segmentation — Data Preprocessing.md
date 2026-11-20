# Customer Segmentation — Data Preprocessing

This project prepares customer data for a machine-learning customer-segmentation task. The accompanying Jupyter notebook performs exploratory data analysis, data cleaning, missing-value imputation, duplicate removal, categorical encoding, target encoding, and numerical feature scaling.

The notebook produces model-ready feature matrices and encoded segmentation labels, but it does not train or evaluate a predictive model.

## Project files

```text
.
├── Preprocessinf_Customer_Segmentation.ipynb  # Main preprocessing notebook
├── Train.csv                                 # Training data, including Segmentation
├── Test.csv                                  # Test data used by the notebook
└── README.md
```

> The uploaded notebook filename contains the spelling `Preprocessinf`. You may rename it to `Preprocessing_Customer_Segmentation.ipynb` for clarity, but update any links or references accordingly.

## Dataset

The dataset contains demographic, professional, and household information about customers.

| Column | Description | Treatment in notebook |
|---|---|---|
| `ID` | Customer identifier | Removed before preprocessing |
| `Gender` | Customer gender | One-hot encoded |
| `Ever_Married` | Whether the customer has ever been married | Missing values filled with the most frequent value, then one-hot encoded |
| `Age` | Customer age | Standardized |
| `Graduated` | Whether the customer graduated | Missing values filled with the most frequent value, then one-hot encoded |
| `Profession` | Customer profession | Missing values filled with the most frequent value, then one-hot encoded |
| `Work_Experience` | Years of work experience | Median imputation and standardization |
| `Spending_Score` | Spending category: `Low`, `Average`, or `High` | Ordinal mapping: `Low → 0`, `Average → 1`, `High → 2` |
| `Family_Size` | Number of people in the customer’s family | Median imputation |
| `Var_1` | Categorical variable such as `Cat_4` or `Cat_6` | Prefix removed and converted to a numeric category |
| `Segmentation` | Customer segment label, with classes such as `A`, `B`, `C`, and `D` | Label-encoded; used as the target |

The training data contains 8,068 rows before duplicate removal, while the test data contains 2,627 rows before duplicate removal. The notebook checks for missing values and duplicate rows in both datasets before continuing.

## Processing workflow

The notebook follows these steps:

1. Load `Train.csv` and `Test.csv` with pandas.
2. Inspect the datasets with `head()`, `describe()`, and `info()`.
3. Remove the `ID` column and reset the dataframe indexes.
4. Check for missing values and duplicate rows.
5. Remove duplicate rows from both datasets.
6. Impute missing numerical values in `Family_Size` and `Work_Experience` with the training-data median.
7. Impute missing categorical values in `Ever_Married`, `Graduated`, `Profession`, and `Var_1` with the most frequent training-data value.
8. Compare train/test categorical counts and numerical distributions visually.
9. One-hot encode `Gender`, `Ever_Married`, `Graduated`, and `Profession` using `drop='first'` and `handle_unknown='ignore'`.
10. Convert `Spending_Score` to an ordinal numeric feature.
11. Convert `Var_1` values such as `Cat_4` into numeric values such as `4`.
12. Label-encode the `Segmentation` target.
13. Split the data into features and target:
    - `X_train`, `y_train`
    - `X_test`, `y_test`
14. Standardize `Age` and `Work_Experience` using `StandardScaler` fitted on the training data and applied to the test data.

## Requirements

Install the required Python packages with:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn category-encoders
```

The notebook imports `category_encoders`, although no encoder from that package is ultimately used. It can be removed from the imports if it is not needed elsewhere.

## How to run

1. Place the notebook and both CSV files in the same directory.
2. Start Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

3. Open the customer-segmentation notebook.
4. Run the cells from top to bottom.
5. Use the final variables in downstream modeling code:

   ```python
   X_train  # processed training features
   y_train  # encoded training labels
   X_test   # processed test features
   y_test   # encoded test labels, if available
   ```

## Important implementation notes

- The preprocessing statistics are learned from the training data and reused for the test data, which helps prevent preprocessing leakage.
- `OneHotEncoder` is configured with `handle_unknown='ignore'`, so unseen categorical values in the test data do not cause encoding errors.
- The notebook creates `cleaned_df` after applying an IQR-based outlier filter to `Family_Size` and `Work_Experience`, but it does not assign that filtered dataframe back to `train` or `test`. Therefore, outlier rows remain in the final `X_train` and `X_test` unless this behavior is changed.
- The outlier helper functions use the global `train` or `test` dataframe rather than their `data` argument. If outlier removal is required, the functions should be rewritten to use `data` consistently.
- `X_test` is created by dropping `Segmentation` from `test`. This assumes that `Test.csv` contains that column. If the real test file does not contain labels, use `X_test = test.copy()` instead.
- The notebook fits `LabelEncoder` on the training target. Save or reuse that encoder when converting future predictions back to segment names.
- `plt.show` appears in a few cells without parentheses. Use `plt.show()` to explicitly render those plots when needed.

## Suggested next steps

The processed data can be used to train and compare classification models such as logistic regression, decision trees, random forests, gradient-boosted trees, or other suitable classifiers. Recommended next steps include:

- Create a validation split or use cross-validation.
- Evaluate class balance across the segmentation labels.
- Compare models using accuracy, macro F1-score, and a confusion matrix.
- Tune hyperparameters using validation data.
- Persist the fitted imputers, encoder, scaler, and final model for consistent inference on new customers.

## License and data attribution

No license or dataset attribution is specified in the notebook. Add the relevant license and source information before publishing or redistributing the project.
