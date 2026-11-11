
import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from config import MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict")

EXAMPLE = {
    "baseline_value": 120,
    "accelerations": 0.0,
    "fetal_movement": 0.0,
    "uterine_contractions": 0.0,
    "light_decelerations": 0.0,
    "severe_decelerations": 0.0,
    "prolongued_decelerations": 0.0,
    "abnormal_short_term_variability": 73,
    "mean_value_of_short_term_variability": 0.5,
    "percentage_of_time_with_abnormal_long_term_variability": 43,
    "mean_value_of_long_term_variability": 2.4,
    "histogram_width": 64,
    "histogram_min": 62,
    "histogram_max": 126,
    "histogram_number_of_peaks": 2,
    "histogram_number_of_zeroes": 0,
    "histogram_mode": 120,
    "histogram_mean": 137,
    "histogram_median": 121,
    "histogram_variance": 73,
    "histogram_tendency": 1,
}


class CTGFeatures(BaseModel):
    """The 21 CTG-derived input features. Names match the training data."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    baseline_value: float
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    light_decelerations: float
    severe_decelerations: float
    prolongued_decelerations: float  # spelling matches the original dataset column
    abnormal_short_term_variability: float
    mean_value_of_short_term_variability: float
    percentage_of_time_with_abnormal_long_term_variability: float
    mean_value_of_long_term_variability: float
    histogram_width: float
    histogram_min: float
    histogram_max: float
    histogram_number_of_peaks: float
    histogram_number_of_zeroes: float
    histogram_mode: float
    histogram_mean: float
    histogram_median: float
    histogram_variance: float
    histogram_tendency: float


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    path = Path(os.environ.get("MODEL_PATH", MODEL_PATH))
    if not path.exists():
        raise RuntimeError(f"Model file not found at {path}. Run `python src/train.py` first.")

    with open(path, "rb") as f:
        artifact = pickle.load(f)

    # Fail fast if the API schema and the trained model disagree on features
    if set(artifact["features"]) != set(CTGFeatures.model_fields):
        raise RuntimeError("Model features do not match the API input schema.")

    state["artifact"] = artifact
    log.info("Loaded %s model from %s", artifact["model_name"], path)
    yield
    state.clear()


app = FastAPI(
    title="Fetal Health Classifier",
    description="Predicts fetal health status (normal, suspect, pathological) from CTG features. "
    "Learning project, not a medical device.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model": state["artifact"]["model_name"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CTGFeatures):
    artifact = state["artifact"]
    # Enforce the exact column order used in training
    row = pd.DataFrame([features.model_dump()])[artifact["features"]]
    proba = artifact["model"].predict_proba(row)[0]

    labels = artifact["labels"]
    return PredictionResponse(
        prediction=labels[int(proba.argmax())],
        probabilities={labels[i]: round(float(p), 4) for i, p in enumerate(proba)},
    )