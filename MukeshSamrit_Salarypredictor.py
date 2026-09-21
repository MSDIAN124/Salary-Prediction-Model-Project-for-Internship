"""
train.py  –  DS Jobs Salary Predictor
======================================
Loads the Cleaned_DS_Jobs dataset, engineers features, trains a
GradientBoostingRegressor pipeline, evaluates it, and saves the
trained model to disk.

How to run
----------
Option 1 (recommended – from project root):
    python -m salary_predictor.train

Option 2 (run directly – place Cleaned_DS_Jobs.csv in same folder):
    python train.py

Output
------
    salary_predictor/model/pipeline.pkl   – serialised sklearn pipeline
    salary_predictor/model/meta.json      – feature lists + eval metrics
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
# Works whether run as `python train.py` OR `python -m salary_predictor.train`
THIS_FILE  = pathlib.Path(__file__).resolve()
ROOT       = THIS_FILE.parent.parent          # project root
MODEL_DIR  = THIS_FILE.parent / "model"       # salary_predictor/model/
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Dataset can be in project root OR same folder as this file
_candidates = [ROOT / "Cleaned_DS_Jobs.csv", THIS_FILE.parent / "Cleaned_DS_Jobs.csv"]
DATA_PATH   = next((p for p in _candidates if p.exists()), _candidates[0])

PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
META_PATH     = MODEL_DIR / "meta.json"

# ── Feature definitions ───────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Rating",
    "company_age",
    "python",
    "excel",
    "hadoop",
    "spark",
    "aws",
    "tableau",
    "big_data",
    "same_state",
]

CATEGORICAL_FEATURES = [
    "Size",
    "Type of ownership",
    "Industry",
    "Sector",
    "job_simp",
    "seniority",
    "job_state",
]

TARGET = "avg_salary"   # average salary in $K (pre-computed in dataset)


# ── 1. Load & clean ───────────────────────────────────────────────────────────
def load_data(path):
    """Load CSV and apply basic cleaning."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {path}"
            "\nPlease place Cleaned_DS_Jobs.csv in the project root folder.\n"
        )

    df = pd.read_csv(path)
    print(f"[1/3] Loaded {len(df):,} rows, {df.shape[1]} columns from {path.name}")

    # Drop rows where the target salary is missing
    before = len(df)
    df = df.dropna(subset=[TARGET])
    dropped = before - len(df)
    if dropped:
        print(f"      Dropped {dropped} rows with missing '{TARGET}'")

    # Rating: Glassdoor uses -1 for unknown → coerce to NaN
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df.loc[df["Rating"] < 0, "Rating"] = np.nan

    # company_age: negative values are data errors → coerce to NaN
    if "company_age" in df.columns:
        df["company_age"] = pd.to_numeric(df["company_age"], errors="coerce")
        df.loc[df["company_age"] < 0, "company_age"] = np.nan

    # Categorical columns: replace sentinel strings with NaN
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(
                {"-1": np.nan, "unknown": np.nan, "Unknown": np.nan, "nan": np.nan}
            )

    print(f"      Clean dataset: {len(df):,} rows ready for training")
    return df


# ── 2. Build sklearn pipeline ─────────────────────────────────────────────────
def build_pipeline(num_features, cat_features):
    """
    Build a full sklearn Pipeline:
      Numeric  branch : SimpleImputer(median)        → StandardScaler
      Categorical branch: SimpleImputer(most_frequent) → OneHotEncoder
      Estimator : GradientBoostingRegressor
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,     num_features),
        ("cat", categorical_transformer, cat_features),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )),
    ])

    return pipeline


# ── 3. Train & evaluate ───────────────────────────────────────────────────────
def train(df):
    """Train the pipeline and evaluate on a held-out test set."""

    # Only use features that actually exist in this dataset
    num_avail = [f for f in NUMERIC_FEATURES    if f in df.columns]
    cat_avail = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    all_feats = num_avail + cat_avail

    print(f"\n[2/3] Training on {len(num_avail)} numeric + {len(cat_avail)} categorical features")
    print(f"      Numeric    : {num_avail}")
    print(f"      Categorical: {cat_avail}")

    X = df[all_feats]
    y = df[TARGET]

    # 80 / 20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    # Build and fit
    pipeline = build_pipeline(num_avail, cat_avail)
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"\n      Results:")
    print(f"      R² Score           : {r2:.4f}  (1.0 = perfect)")
    print(f"      Mean Absolute Error: ${mae:.2f}K")
    print(f"\n✓  Training complete!")

    # ── Save pipeline ──────────────────────────────────────────────────────────
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    # ── Save metadata (feature lists + eval metrics + dropdown options) ────────
    cat_options = {}
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

    print(f"\n[3/3] Model saved:")
    print(f"      Pipeline → {PIPELINE_PATH}")
    print(f"      Metadata → {META_PATH}\n")

    return pipeline, meta


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   DS Jobs Salary Predictor — Model Training")
    print("=" * 55)
    dataframe = load_data(DATA_PATH)
    train(dataframe)
    print("=" * 55)
    print("   Done! Run the backend next:")
    print("   uvicorn salary_predictor.backend.app:app --reload --port 8000")
    print("=" * 55)
