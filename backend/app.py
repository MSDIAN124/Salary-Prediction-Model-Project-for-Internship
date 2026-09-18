"""
backend/app.py – FastAPI service that loads the trained pipeline
and exposes prediction + metadata endpoints.

Start with:  uvicorn salary_predictor.backend.app:app --reload --port 8000
"""

import re
import json
import pickle
import pathlib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = pathlib.Path(__file__).parent.parent
MODEL_DIR     = ROOT / "model"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
META_PATH     = MODEL_DIR / "meta.json"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DS Jobs Salary Predictor API",
    description="Predict salary (in $K) from Cleaned_DS_Jobs dataset features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artefacts at startup ─────────────────────────────────────────────────
_pipeline = None
_meta: dict = {}


def _load_artefacts():
    global _pipeline, _meta
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {PIPELINE_PATH}. "
            "Run `python -m salary_predictor.train` first."
        )
    with open(PIPELINE_PATH, "rb") as f:
        _pipeline = pickle.load(f)
    with open(META_PATH) as f:
        _meta = json.load(f)


@app.on_event("startup")
def startup_event():
    _load_artefacts()


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    job_title:         str            = Field(...,  example="Data Scientist")
    rating:            Optional[float]= Field(None, ge=0, le=5, example=3.8)
    company_age:       Optional[float]= Field(None, ge=0, example=20)
    size:              Optional[str]  = Field(None, example="1001 to 5000 employees")
    type_of_ownership: Optional[str]  = Field(None, example="Company - Public")
    industry:          Optional[str]  = Field(None, example="Internet")
    sector:            Optional[str]  = Field(None, example="Information Technology")
    job_state:         Optional[str]  = Field(None, example="CA")
    seniority:         Optional[str]  = Field(None, example="senior")
    same_state:        Optional[int]  = Field(None, example=1)
    # Skill flags (0 or 1)
    python:    Optional[int] = Field(None, example=1)
    excel:     Optional[int] = Field(None, example=0)
    hadoop:    Optional[int] = Field(None, example=0)
    spark:     Optional[int] = Field(None, example=0)
    aws:       Optional[int] = Field(None, example=0)
    tableau:   Optional[int] = Field(None, example=0)
    big_data:  Optional[int] = Field(None, example=0)

    class Config:
        json_schema_extra = {
            "example": {
                "job_title": "Senior Data Scientist",
                "rating": 4.0,
                "company_age": 15,
                "size": "1001 to 5000 employees",
                "type_of_ownership": "Company - Public",
                "industry": "Internet",
                "sector": "Information Technology",
                "job_state": "CA",
                "seniority": "senior",
                "same_state": 1,
                "python": 1,
                "excel": 0,
                "spark": 1,
                "aws": 1,
            }
        }


class PredictResponse(BaseModel):
    predicted_salary_k:      float
    predicted_salary_annual: float
    confidence_note:         str
    model_r2:                float
    model_mae_k:             float


# ── Helper ────────────────────────────────────────────────────────────────────
def _build_row(req: PredictRequest) -> dict:
    """Map request fields to the exact feature names the pipeline expects."""
    title_lower = req.job_title.lower()

    # Derive seniority from title if not provided
    seniority = req.seniority
    if not seniority:
        if re.search(r"senior|sr\.?|lead|principal|staff", title_lower):
            seniority = "senior"
        elif re.search(r"junior|jr\.?|entry", title_lower):
            seniority = "junior"
        else:
            seniority = "na"

    # Derive job_simp from title
    if re.search(r"data scientist", title_lower):
        job_simp = "data scientist"
    elif re.search(r"data engineer", title_lower):
        job_simp = "data engineer"
    elif re.search(r"analyst", title_lower):
        job_simp = "analyst"
    elif re.search(r"machine learning|ml engineer", title_lower):
        job_simp = "mle"
    elif re.search(r"manager|director|head|vp", title_lower):
        job_simp = "manager"
    else:
        job_simp = "other"

    return {
        "Rating":            req.rating,
        "company_age":       req.company_age,
        "python":            req.python    if req.python    is not None else 0,
        "excel":             req.excel     if req.excel     is not None else 0,
        "hadoop":            req.hadoop    if req.hadoop    is not None else 0,
        "spark":             req.spark     if req.spark     is not None else 0,
        "aws":               req.aws       if req.aws       is not None else 0,
        "tableau":           req.tableau   if req.tableau   is not None else 0,
        "big_data":          req.big_data  if req.big_data  is not None else 0,
        "same_state":        req.same_state if req.same_state is not None else 0,
        "Size":              req.size,
        "Type of ownership": req.type_of_ownership,
        "Industry":          req.industry,
        "Sector":            req.sector,
        "job_simp":          job_simp,
        "seniority":         seniority,
        "job_state":         req.job_state,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.get("/meta", summary="Model metadata and dropdown options")
def get_meta():
    if not _meta:
        raise HTTPException(503, "Model metadata not loaded.")
    return _meta


@app.post("/predict", response_model=PredictResponse, summary="Predict salary")
def predict(req: PredictRequest):
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded.")

    row       = _build_row(req)
    num_feats = _meta.get("numeric_features", [])
    cat_feats = _meta.get("categorical_features", [])
    all_feats = num_feats + cat_feats

    aligned = {f: row.get(f, np.nan) for f in all_feats}
    X = pd.DataFrame([aligned])

    pred_k      = float(_pipeline.predict(X)[0])
    pred_annual = round(pred_k * 1_000, 2)

    return PredictResponse(
        predicted_salary_k=round(pred_k, 2),
        predicted_salary_annual=pred_annual,
        confidence_note=(
            f"Estimate based on {_meta.get('n_train', '?')} training samples. "
            f"Typical error ≈ ${_meta.get('mae', '?')}K."
        ),
        model_r2=_meta.get("r2", 0.0),
        model_mae_k=_meta.get("mae", 0.0),
    )


@app.get("/feature-importance", summary="Top 20 feature importances")
def feature_importance():
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded.")

    model        = _pipeline.named_steps["model"]
    preprocessor = _pipeline.named_steps["preprocessor"]

    feature_names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = transformer.named_steps["ohe"]
            feature_names.extend(ohe.get_feature_names_out(cols).tolist())

    importances = model.feature_importances_
    paired = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
    return {"features": [{"name": n, "importance": round(float(v), 5)} for n, v in paired]}
