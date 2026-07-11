# Data Preprocessing for Machine Learning — A Visual, Hands-On Reference

A complete, beginner-friendly reference for **data preprocessing** — the highest-leverage
and most-skipped skill in machine learning. Three runnable notebooks take you from raw,
messy data to a leakage-free, production-ready pipeline, and an accompanying **interactive
web app** lets you *see* what every step does to the data.

> **A model is only as good as the data it is trained on.** Everything here is about earning that.

---

## 🔗 Live demo

**[→ Open the interactive explainer](#)** &nbsp;·&nbsp; a static, no-server web app (deployed on Vercel)
that visualizes every preprocessing step — before and after — right in your browser.

> Replace the link above with your Vercel URL after the first deploy (see [Web app](#-web-app) below).

---

## What's inside

| # | Notebook | What it covers |
|---|---|---|
| 01 | **`01_pandas_fundamentals.ipynb`** | The pure **pandas / NumPy** toolkit: EDA, missing values by hand (`fillna` / `dropna`), duplicates, dtype & text cleaning, index management, manual z-score, reshaping, feature engineering, NumPy ↔ pandas. |
| 02 | **`02_sklearn_preprocessing.ipynb`** | The **scikit-learn** way: `SimpleImputer` / `KNNImputer`, all five scalers, encoders, train/test splitting & **leakage prevention**, feature selection, PCA / TruncatedSVD, **SMOTE**, full `Pipeline` / `ColumnTransformer` workflows, and model persistence. |
| 03 | **`03_advanced_concepts.ipynb`** | The techniques a "complete" reference usually skips: **outlier detection** (IQR, z-score, `IsolationForest`), **datetime feature extraction** with cyclical encoding, and leakage-safe **target & frequency encoding**. |

Notebooks 01 and 02 share the **Titanic** dataset (loaded from seaborn — no download).
Notebook 03 adds seaborn's **taxis** dataset for the datetime section.

---

## The dataset

**Titanic — Survival Prediction**, loaded directly from seaborn:

```python
import seaborn as sns
df = sns.load_dataset("titanic")
```

It was chosen because it naturally exercises everything: real missing values
(`age`, `deck`, `embarked`), apparent duplicates, mixed numeric/categorical types,
class imbalance (62% died / 38% survived), a continuous target (`fare`) and a binary
target (`survived`).

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter and open the notebooks in order
jupyter lab        # or: jupyter notebook
```

Run the notebooks top to bottom — each is self-contained and reloads its own data.

---

## 🌐 Web app

The `web/` folder is a **Next.js static site** that turns the notebooks into an interactive,
before/after tour. It has **no backend**: a Python script precomputes small JSON artifacts,
and the browser just draws them — so it deploys to Vercel as pure static content.

**Tabs:** Overview · Missing Values · Scaling · Encoding · Outliers · Datetime · PCA · Class Imbalance.

### Run it locally

```bash
# 1. Regenerate the JSON artifacts from the notebooks' logic (writes to web/public/)
python scripts/export_web_artifacts.py

# 2. Start the dev server
cd web
npm install
npm run dev          # http://localhost:3000
```

### Deploy to Vercel

1. Push this repo to GitHub and import it at [vercel.com/new](https://vercel.com/new).
2. Set the project's **Root Directory** to `web`. Vercel auto-detects Next.js — no other config needed.
3. Deploy. Re-run `export_web_artifacts.py` and commit whenever the notebooks change.

---

## Techniques at a glance

**Missing values** — `isna`, `dropna`, `fillna` (constant/mean/median/mode), `SimpleImputer`, `KNNImputer`
**Scaling** — `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`
**Encoding** — `get_dummies`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `TargetEncoder`, frequency encoding
**Outliers** — IQR rule, z-score rule, `IsolationForest`
**Datetime** — `.dt` accessor parts, `is_weekend`, cyclical sine/cosine encoding
**Feature work** — engineering (`family_size`, `is_alone`, interactions), binning (`cut` / `qcut`), `VarianceThreshold`, `RFE`
**Dimensionality** — `PCA`, `TruncatedSVD`
**Imbalance** — `SMOTE` (training set only)
**Pipelines** — `ColumnTransformer`, `Pipeline`, `joblib` persistence

---

## The principles this repo drills

- **Split before you fit.** Never fit a transformer on the full dataset or the test set.
- Handle missing values **before** scaling; encode categoricals **before** modeling.
- Apply **SMOTE and target encoding on training data only** — both leak otherwise.
- Detect outliers before deciding to cap, drop, or keep them — never delete blindly.
- Ship **full pipelines**, not just models, so preprocessing stays consistent in production.

---

## Repository layout

```
├── 01_pandas_fundamentals.ipynb     # pandas / NumPy
├── 02_sklearn_preprocessing.ipynb   # scikit-learn transformers & pipelines
├── 03_advanced_concepts.ipynb       # outliers, datetime, target encoding
├── scripts/
│   └── export_web_artifacts.py      # notebooks → web/public/*.json
├── web/                             # Next.js static explainer (deploys to Vercel)
├── requirements.txt
└── LICENSE                          # MIT
```

---

## License

[MIT](LICENSE) © 2026 Shivani Bokka
