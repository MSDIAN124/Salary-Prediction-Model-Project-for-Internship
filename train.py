"""
train.py – Load the Cleaned_DS_Jobs dataset, train a GradientBoostingRegressor,
and persist the pipeline + metadata.
Run once before starting the API:  python -m salary_predictor.train
"""

import json
import pickle
import pathlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT / "Cleaned_DS_Jobs.csv"
MODEL_DIR = ROOT / "salary_predictor" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
META_PATH     = MODEL_DIR / "meta.json"

# ── Feature definitions ───────────────────────────────────────────────────────
# Numeric: already computed in the dataset
NUMERIC_FEATURES = [
    "Rating", "company_age",
    "python", "excel", "hadoop", "spark", "aws", "tableau", "big_data",
    "same_state",
]
# Categorical: label-encoded as strings in the dataset
CATEGORICAL_FEATURES = [
    "Size", "Type of ownership", "Industry", "Sector",
    "job_simp", "seniority", "job_state",
]
# Target: average salary in $K
TARGET = "avg_salary"


# ── 1. Load & clean ───────────────────────────────────────────────────────────
def load_data(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # Coerce Rating -1 sentinel → NaN
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df.loc[df["Rating"] < 0, "Rating"] = np.nan

    # Coerce company_age negatives → NaN
    df["company_age"] = pd.to_numeric(df["company_age"], errors="coerce")
    df.loc[df["company_age"] < 0, "company_age"] = np.nan

    # Clean categorical sentinels
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"-1": np.nan, "unknown": np.nan, "Unknown": np.nan})

    print(f"Clean rows after dropping missing target: {len(df):,}")
    return df


# ── 2. Build sklearn pipeline ─────────────────────────────────────────────────
def build_pipeline(num_features: list[str], cat_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )),
    ])


# ── 3. Train & evaluate ───────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    num_avail = [f for f in NUMERIC_FEATURES    if f in df.columns]
    cat_avail = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    all_feats = num_avail + cat_avail

    X = df[all_feats]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(num_avail, cat_avail)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"\n✓ Training complete  |  MAE: {mae:.2f}K  |  R²: {r2:.4f}\n")

    # Persist pipeline
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    # Collect dropdown options for the frontend
    cat_options: dict[str, list] = {}
    for col in cat_avail:
        cat_options[col] = sorted(df[col].dropna().unique().tolist())

    meta = {
        "numeric_features":     num_avail,
        "categorical_features": cat_avail,
        "cat_options":          cat_options,
        "mae":     round(mae, 2),
        "r2":      round(r2, 4),
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Pipeline saved → {PIPELINE_PATH}")
    print(f"Metadata saved → {META_PATH}")
    return pipeline, meta


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    train(df)
