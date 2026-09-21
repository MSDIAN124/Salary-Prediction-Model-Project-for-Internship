"""
model_comparison.py – Does the model actually beat a simple baseline?
=====================================================================
Compares several models with 5-fold cross-validation on the same features
that MukeshSamrit_Salarypredictor.py uses. Run from the project root:

    python model_comparison.py

The mean-value baseline (DummyRegressor) is the most important row: any
useful model must score clearly better than it.
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = pathlib.Path(__file__).resolve().parent / "Cleaned_DS_Jobs.csv"

NUMERIC = ["Rating", "company_age", "python", "excel", "hadoop", "spark",
           "aws", "tableau", "big_data", "same_state"]
CATEGORICAL = ["Size", "Type of ownership", "Industry", "Sector",
               "job_simp", "seniority", "job_state"]
TARGET = "avg_salary"


def load():
    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df.loc[df["Rating"] < 0, "Rating"] = np.nan
    df.loc[df["company_age"] < 0, "company_age"] = np.nan
    for c in CATEGORICAL:
        df[c] = (df[c].astype(str).str.strip()
                 .replace({"-1": np.nan, "unknown": np.nan,
                           "Unknown": np.nan, "nan": np.nan}))
    return df


def make_pipeline(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False,
                                                min_frequency=5))]), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("model", model)])


MODELS = {
    "Mean baseline (predict average)": DummyRegressor(),
    "Ridge regression": Ridge(alpha=10),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=42),
    "Gradient Boosting (depth 5, 300 trees)": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42),
    "Gradient Boosting (depth 3, 150 trees)": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=5, random_state=42),
}


if __name__ == "__main__":
    df = load()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Rows: {len(df)}  |  5-fold cross-validation\n")
    print(f"{'Model':42s} {'R2':>8s} {'MAE ($K)':>10s}")
    print("-" * 62)
    for name, model in MODELS.items():
        r2 = cross_val_score(make_pipeline(model), X, y, cv=kf, scoring="r2")
        mae = -cross_val_score(make_pipeline(model), X, y, cv=kf,
                               scoring="neg_mean_absolute_error")
        print(f"{name:42s} {r2.mean():8.3f} {mae.mean():10.1f}")
