"""Smoke tests for the API. Train the model first: python src/train.py"""
import copy
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config import MODEL_PATH
from predict import EXAMPLE, app

pytestmark = pytest.mark.skipif(
    not Path(os.environ.get("MODEL_PATH", MODEL_PATH)).exists(),
    reason="models/model.bin not found. Run `python src/train.py` first.",
)


@pytest.fixture(scope="module")
def client():
    # The context manager runs the lifespan handler, which loads the model
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_response(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"normal", "suspect", "pathological"}
    assert set(body["probabilities"]) == {"normal", "suspect", "pathological"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-2)


def test_prediction_matches_highest_probability(client):
    body = client.post("/predict", json=EXAMPLE).json()
    top = max(body["probabilities"], key=body["probabilities"].get)
    assert body["prediction"] == top


def test_missing_field_is_rejected(client):
    payload = copy.deepcopy(EXAMPLE)
    payload.pop("baseline_value")
    assert client.post("/predict", json=payload).status_code == 422


def test_non_numeric_value_is_rejected(client):
    payload = copy.deepcopy(EXAMPLE)
    payload["baseline_value"] = "high"
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_field_is_rejected(client):
    payload = copy.deepcopy(EXAMPLE)
    payload["not_a_feature"] = 1
    assert client.post("/predict", json=payload).status_code == 422