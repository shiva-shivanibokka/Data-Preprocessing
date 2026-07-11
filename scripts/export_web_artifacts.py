# -*- coding: utf-8 -*-
"""Compute compact JSON artifacts for the client-side preprocessing explainer.

Every visualization on the web app is a *precomputed* before/after result - no server,
no live sklearn in the browser. Run this whenever the notebooks change:

    python scripts/export_web_artifacts.py

Outputs go to web/public/*.json (one file per tab).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, TargetEncoder,
)
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

OUT = os.path.join("web", "public")
os.makedirs(OUT, exist_ok=True)


def dump(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"  wrote {name} ({os.path.getsize(path) // 1024 + 1} KB)")


def r(x, n=4):
    """Round scalars/arrays for compact JSON."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [round(float(v), n) for v in x]
    return round(float(x), n)


def hist(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    return [int(c) for c in counts]


titanic = sns.load_dataset("titanic")
print("Titanic:", titanic.shape)

# ----------------------------------------------------------------------------
# 1) MISSING VALUES  -> missing.json
# ----------------------------------------------------------------------------
print("missing...")
miss = titanic.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
counts = [{"col": c, "missing": int(n), "pct": r(100 * n / len(titanic), 1)}
          for c, n in miss.items()]

age = titanic["age"]
edges = np.linspace(0, 80, 17).tolist()          # 5-year bins
observed = age.dropna().values

strategies = {}
for label, filled in {
    "none": age.dropna().values,
    "mean": age.fillna(age.mean()).values,
    "median": age.fillna(age.median()).values,
}.items():
    strategies[label] = {
        "hist": hist(filled, edges),
        "fill_value": None if label == "none" else r(age.mean() if label == "mean" else age.median(), 1),
    }
# KNN imputation uses correlated numeric columns
knn_src = titanic[["age", "pclass", "fare", "sibsp", "parch"]].copy()
knn_filled = KNNImputer(n_neighbors=5).fit_transform(knn_src)[:, 0]
strategies["knn"] = {"hist": hist(knn_filled, edges), "fill_value": None}

dump("missing.json", {
    "counts": counts,
    "total_rows": int(len(titanic)),
    "age": {"edges": r(edges, 1), "n_missing": int(age.isna().sum()), "strategies": strategies},
})

# ----------------------------------------------------------------------------
# 2) SCALING  -> scaling.json  (fit on train only, report distribution shape)
# ----------------------------------------------------------------------------
print("scaling...")
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler(),
}
scaling = {}
for col in ["age", "fare"]:
    vals = titanic[col].dropna().values.reshape(-1, 1)
    tr, te = train_test_split(vals, test_size=0.25, random_state=42)
    per_scaler = {"Original": tr.ravel()}
    for name, sc in scalers.items():
        per_scaler[name] = sc.fit(tr).transform(tr).ravel()
    scaling[col] = {
        name: {
            "min": r(v.min()), "q1": r(np.quantile(v, 0.25)), "median": r(np.median(v)),
            "q3": r(np.quantile(v, 0.75)), "max": r(v.max()),
            "sample": r(np.random.default_rng(0).choice(v, size=min(150, len(v)), replace=False)),
        }
        for name, v in per_scaler.items()
    }
dump("scaling.json", scaling)

# ----------------------------------------------------------------------------
# 3) ENCODING  -> encoding.json
# ----------------------------------------------------------------------------
print("encoding...")
enc_df = titanic[["embarked", "class", "who"]].dropna().copy()
sample = enc_df.head(6).reset_index(drop=True)
onehot = pd.get_dummies(sample, dtype=int)
widths = {
    "One-Hot": int(pd.get_dummies(enc_df, dtype=int).shape[1]),
    "Ordinal": int(enc_df.shape[1]),
    "Target": int(enc_df.shape[1]),
}
dump("encoding.json", {
    "original": {"cols": list(sample.columns), "rows": sample.astype(str).values.tolist()},
    "onehot": {"cols": list(onehot.columns), "rows": onehot.values.tolist()},
    "widths": widths,
})

# ----------------------------------------------------------------------------
# 4) OUTLIERS  -> outliers.json  (IQR vs z-score vs IsolationForest)
# ----------------------------------------------------------------------------
print("outliers...")
fare = titanic["fare"].dropna()
Q1, Q3 = fare.quantile(0.25), fare.quantile(0.75)
IQR = Q3 - Q1
low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
z = (fare - fare.mean()) / fare.std()
z_cut = fare.mean() + 3 * fare.std()

feat = titanic[["age", "fare"]].dropna().copy()
iso_pred = IsolationForest(contamination=0.05, random_state=42).fit_predict(feat)
feat["outlier"] = iso_pred == -1
pts = feat.sample(min(300, len(feat)), random_state=1)

dump("outliers.json", {
    "fare_sample": r(fare.sample(min(250, len(fare)), random_state=2).values, 2),
    "iqr": {"low": r(low, 2), "high": r(high, 2), "count": int(((fare < low) | (fare > high)).sum())},
    "z": {"cut": r(z_cut, 2), "count": int((z.abs() > 3).sum())},
    "iso": {
        "count": int(feat["outlier"].sum()),
        "points": [{"age": r(a, 1), "fare": r(f, 2), "out": bool(o)}
                   for a, f, o in zip(pts["age"], pts["fare"], pts["outlier"])],
    },
    "n_total": int(len(fare)),
})

# ----------------------------------------------------------------------------
# 5) DATETIME  -> datetime.json  (seaborn taxis)
# ----------------------------------------------------------------------------
print("datetime...")
taxis = sns.load_dataset("taxis")
taxis["pickup"] = pd.to_datetime(taxis["pickup"])
by_hour = taxis["pickup"].dt.hour.value_counts().sort_index()
clock = [{"h": int(h), "sin": r(np.sin(2 * np.pi * h / 24)), "cos": r(np.cos(2 * np.pi * h / 24))}
         for h in range(24)]
dump("datetime.json", {
    "by_hour": [int(by_hour.get(h, 0)) for h in range(24)],
    "clock": clock,
})

# ----------------------------------------------------------------------------
# 6) PCA  -> pca.json  (numeric Titanic features, colored by survival)
# ----------------------------------------------------------------------------
print("pca...")
num = titanic[["pclass", "age", "sibsp", "parch", "fare"]].copy()
num = num.fillna(num.median())
Xs = StandardScaler().fit_transform(num)
pca = PCA(n_components=5).fit(Xs)
proj = pca.transform(Xs)[:, :2]
idx = np.random.default_rng(3).choice(len(proj), size=min(400, len(proj)), replace=False)
dump("pca.json", {
    "explained": r(pca.explained_variance_ratio_),
    "cumulative": r(np.cumsum(pca.explained_variance_ratio_)),
    "scatter": [{"x": r(proj[i, 0], 3), "y": r(proj[i, 1], 3), "s": int(titanic["survived"].iloc[i])}
                for i in idx],
})

# ----------------------------------------------------------------------------
# 7) SMOTE  -> smote.json  (class balance before/after, train only)
# ----------------------------------------------------------------------------
print("smote...")
d = titanic[["pclass", "age", "sibsp", "parch", "fare", "survived"]].dropna()
Xd, yd = d.drop(columns="survived"), d["survived"]
Xtr, _, ytr, _ = train_test_split(Xd, yd, test_size=0.25, random_state=42, stratify=yd)
before = ytr.value_counts().sort_index()
_, ysm = SMOTE(random_state=42).fit_resample(Xtr, ytr)
after = pd.Series(ysm).value_counts().sort_index()
dump("smote.json", {
    "before": {"died": int(before.get(0, 0)), "survived": int(before.get(1, 0))},
    "after": {"died": int(after.get(0, 0)), "survived": int(after.get(1, 0))},
})

print("done ->", OUT)
