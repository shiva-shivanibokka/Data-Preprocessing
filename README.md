# Data Preprocessing for Machine Learning — Complete Reference Notebook

---

## Overview

This project is a **complete, structured reference notebook** for data preprocessing in machine learning.
It covers every major preprocessing technique you will encounter in real-world ML workflows —
from detecting missing values all the way to dimensionality reduction and handling class imbalance.

The notebook is designed to be **a one-stop resource**: readable, well-explained, and practical.
Every technique is demonstrated with clean code, detailed plain-language explanations, and
visualizations that make the results immediately understandable.

---

## Dataset

**Titanic — Survival Prediction**
Loaded directly from seaborn — no download, no file, no account needed.

```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

The Titanic dataset was chosen because it naturally covers everything we need:

| Property | Detail |
|---|---|
| Rows | 891 |
| Numeric features | `age`, `fare`, `sibsp`, `parch`, `pclass` |
| Categorical features | `sex`, `embarked`, `who`, `deck` |
| Real missing values | `age` (177 missing), `deck` (688 missing), `embarked` (2 missing) |
| Apparent duplicates | 107 rows become duplicates after subsetting to selected columns |
| Class imbalance | 62% did not survive vs 38% survived |
| Regression target | `fare` (continuous ticket price) |
| Classification target | `survived` (binary: 0 or 1) |

---

## ML Tasks Demonstrated

- **Linear Regression** — predicting `fare` (a continuous value)
- **Logistic Regression** — predicting `survived` (binary classification)

These two tasks together allow every preprocessing technique to be demonstrated in a
meaningful, real context.

---

## Requirements

```bash
pip install numpy pandas seaborn scikit-learn imbalanced-learn joblib matplotlib
```

> **Note:** `imbalanced-learn` is a separate package from scikit-learn and must be installed explicitly.

---

## What This Notebook Covers

### 0) Setup and Imports

All required libraries imported and explained. Seaborn theme applied for consistent visuals.

---

### 1–4) Load, Subset, Targets, Feature/Target Split

Initial data loading, column selection, target variable definition, and splitting features from targets.

---

### 5) Exploratory Data Analysis (EDA)

Getting to know your data before touching it. Includes a 7-panel visual dashboard showing:
- Age distribution (histogram)
- Survival class balance (bar chart)
- Passenger class distribution (bar chart)
- Age distribution by survival outcome (overlapping histograms)
- Correlation heatmap (numeric features)
- Survival rate by sex (bar chart)
- Survival rate by family size / alone status

| Step | What it does |
|---|---|
| `unique()` | See all distinct values in a column |
| `nunique()` | Count how many unique values exist (cardinality) |
| `value_counts()` | Frequency of each category |
| `value_counts(normalize=True)` | Percentage distribution |
| `describe()` | Statistical summary for numeric columns |
| `describe(include="all")` | Summary including categorical columns |
| `shape` | Number of rows and columns |
| `columns` | List all column names |
| `info()` | Data types, non-null counts, memory usage |

---

### 6) Missing Values: Detection and Handling

| Step | What it does |
|---|---|
| `isna()` | Returns True/False for every cell |
| `isna().sum()` | Count missing values per column — also includes a bar chart of missing counts |
| `isna().sum(axis=1)` | Count missing values per row |
| `dropna()` | Drop rows with any missing value |
| `dropna(axis=1)` | Drop columns with any missing value |
| `dropna(subset=[...])` | Drop rows only if a specific column is missing |
| `fillna(value)` | Fill all missing values with a constant |
| `fillna(mean)` | Mean imputation (single column) |
| `fillna(median)` | Median imputation (single column) |
| `fillna(mode)` | Mode imputation — best for categorical columns |
| `SimpleImputer(strategy="mean")` | sklearn mean imputation |
| `SimpleImputer(strategy="median")` | sklearn median imputation |
| `SimpleImputer(strategy="most_frequent")` | sklearn mode imputation |
| `SimpleImputer(strategy="constant")` | Fill with a fixed value |
| `KNNImputer` | Fill using k nearest neighbor rows |

---

### 7) Handling Duplicates

| Step | What it does |
|---|---|
| `duplicated().sum()` | Count duplicate rows |
| `drop_duplicates()` | Remove duplicate rows, keep first occurrence |

---

### 8) Data Type Fixes and Text Cleaning

| Step | What it does |
|---|---|
| `dtypes` | Inspect the data type of every column |
| `pd.to_numeric(errors="coerce")` | Convert messy strings to numbers safely |
| `pd.to_datetime()` | Parse date strings into proper datetime objects |
| `astype("category")` | Mark a column as a fixed categorical type |
| `.str.strip()` | Remove leading and trailing whitespace |
| `.str.lower()` | Standardize text to lowercase |
| `try / except` | Handle errors gracefully during data cleaning |

---

### 9) Index and Structure Management

| Step | What it does |
|---|---|
| `reset_index(drop=True)` | Renumber rows from 0 after dropping rows |
| `rename(columns={...})` | Rename specific columns |
| `df.columns = [...]` | Set all column names at once |
| `reindex(columns=[...])` | Enforce a specific column order |
| Alignment checks | Verify columns, dtypes, and indices match between DataFrames |

---

### 10) Z-Score Standardization (Manual)

- Computing `(x - mean) / std` manually to understand what scalers do internally
- Includes before/after distribution plot

---

### 11) Feature Scaling

Before demonstrating scalers, a train/test split is performed (Section 11.0) so all scalers
are fit on training data only and applied to both train and test — the correct practice for
preventing data leakage.

Each scaler is demonstrated individually, followed by a side-by-side boxplot comparison of
all scalers applied to the same column (`age`).

| Scaler | Formula | Best For |
|---|---|---|
| `StandardScaler` | (x - mean) / std | Most ML models, PCA |
| `MinMaxScaler` | (x - min) / (max - min) | Neural networks, bounded inputs |
| `RobustScaler` | (x - median) / IQR | Data with outliers |
| `MaxAbsScaler` | x / max(\|x\|) | Sparse data |
| `Normalizer` | x / row magnitude | Text / vector data |

---

### 12) Categorical Encoding

| Method | Output | Best For |
|---|---|---|
| `pd.get_dummies(drop_first=True)` | N-1 binary columns | Quick pandas encoding |
| `OneHotEncoder` | N binary columns | sklearn pipelines, unseen category handling |
| `OrdinalEncoder` | 1 integer column per feature | Ordered categories |
| `LabelEncoder` | 1 integer column | Target variable `y`, single column encoding |

---

### 13) Train/Test Split and Preventing Data Leakage

| Step | What it does |
|---|---|
| `train_test_split(random_state=...)` | Reproducible split for regression |
| `train_test_split(stratify=y)` | Preserves class balance in both sets |
| Fit imputer on train only | Learn fill values from training data, apply to test |
| Fit scaler on train only | Learn scale from training data, apply to test |

Key principle: **never fit any transformer on the full dataset or on test data**.
Fitting on test data causes data leakage — your model appears better than it really is.

---

### 14) Data Reshaping

| Step | What it does |
|---|---|
| `df.T` | Transpose — flip rows and columns |
| `pd.melt()` | Wide format → Long format |
| `df.pivot()` | Long format → Wide format |
| `df.unstack()` | Multi-index → Wide format |
| `pd.concat(axis=0)` | Stack DataFrames vertically (add rows) |
| `pd.concat(axis=1)` | Stack DataFrames horizontally (add columns) |
| `pd.merge(how="inner/left/right/outer")` | Join two DataFrames on a shared key |

---

### 15) Feature Engineering

| Step | What it does |
|---|---|
| `family_size = sibsp + parch + 1` | Total family members aboard (including the passenger) |
| `is_alone` | Binary flag: 1 if travelling solo, 0 otherwise — derived from `family_size` |
| `age_x_class` | Interaction feature: `age × pclass` — captures joint effect of age and passenger class |
| `pd.cut()` | Bin a continuous column into fixed-width ranges (e.g., age groups by domain knowledge) |
| `pd.qcut()` | Bin a continuous column into equal-frequency ranges (e.g., age quartiles) |

Engineered features are visualized with a 3-panel chart: family size distribution,
survival rate by family size, and survival rate for passengers travelling alone vs with family.

---

### 16) Feature Validation and Selection

| Step | What it does |
|---|---|
| `VarianceThreshold` | Remove constant or near-constant features |
| `RFE` (Recursive Feature Elimination) | Iteratively remove least important features using a base model |

---

### 17) Dimensionality Reduction

| Step | What it does |
|---|---|
| `PCA` | Reduce numeric features to fewer principal components while maximizing variance explained |
| `TruncatedSVD` | Similar to PCA but works on sparse matrices (e.g., after one-hot encoding) |

Includes scree plot and 2D scatter coloured by survival outcome.

---

### 18) Handling Class Imbalance — SMOTE

| Step | What it does |
|---|---|
| Check class distribution | `value_counts()` on the target column |
| `SMOTE` | Generate synthetic minority class samples to balance the dataset |

SMOTE is applied **only on training data** after the train/test split.
The test set must remain imbalanced to reflect the real-world distribution.
Includes a before/after bar chart showing class balance pre- and post-SMOTE.

---

### 19) Converting Between NumPy and Pandas

| Step | What it does |
|---|---|
| `df.to_numpy()` | DataFrame → NumPy array (modern approach) |
| `df.values` | DataFrame → NumPy array (older approach) |
| `pd.DataFrame(arr, columns=[...])` | NumPy array → DataFrame with column names |

---

### 20) Prebuilt sklearn Workflow — Pipelines

| Step | What it does |
|---|---|
| `ColumnTransformer` | Apply different pipelines to numeric vs categorical columns |
| `Pipeline` | Chain preprocessing steps and a model into one object |
| Full regression pipeline | Preprocessing + LinearRegression — includes actual vs predicted plot |
| Full classification pipeline | Preprocessing + LogisticRegression — includes confusion matrix |

---

### 21) Saving and Loading Models

| Step | What it does |
|---|---|
| `joblib.dump()` | Save a trained pipeline (preprocessing + model) to disk |
| `joblib.load()` | Load and reuse a saved pipeline without retraining |

> **Note for GitHub users:** Add `*.joblib` to your `.gitignore` to avoid committing binary model files.

---

## Does This Apply Only to Linear and Logistic Regression?

No. The preprocessing techniques shown here apply broadly to:

- Ridge / Lasso Regression
- Support Vector Machines
- K-Nearest Neighbors
- Neural Networks
- Random Forest, Gradient Boosting, XGBoost

Only minor adjustments are needed depending on the algorithm.
For example, tree-based models are less sensitive to feature scaling,
but imputation, encoding, and leakage prevention apply universally.

---

## Key Principles Reinforced Throughout

- Always split data before fitting any transformer
- Never fit preprocessing on the full dataset or the test set
- Handle missing values before scaling
- Encode categorical features before modeling
- Apply SMOTE only on training data
- Use Pipelines to keep preprocessing consistent and leakage-free
- Save full pipelines — not just models — for deployment

---

## Final Takeaway

Data preprocessing is not a minor step — it is foundational.

A model is only as good as the data it is trained on.
Well-structured preprocessing improves model performance, prevents leakage, ensures reproducibility,
and makes the entire workflow production-ready.

Mastering preprocessing is one of the highest-leverage skills in machine learning.
