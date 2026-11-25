# ML Zoomcamp 2026

Coursework and projects for the [DataTalks.Club Machine Learning Zoomcamp 2026](https://courses.datatalks.club/ml-zoomcamp-2026/). This folder contains course notebooks, datasets, deep-learning exercises, a model deployment exercise, and a fetal-health classification project.

## Repository contents

| Path | Description |
| --- | --- |
| `notebooks/` | Notebooks for analysis, regression, classification, model evaluation, and tree-based methods. |
| `data/` | CSV datasets for car fuel-efficiency regression and course lead-conversion scoring. |
| `deep_learning/` | Deep-learning exercise answers and reference training code and histories. |
| `deployment/` | Lead-conversion model training, a FastAPI prediction service, and Docker setup. |
| `projects/fetal_health_classifier/` | End-to-end fetal-health classification project, including its own data, notebooks, API, tests, and Dockerfile. |
| `requirements.txt` | CPU PyTorch and torchvision dependencies used for deep-learning exercises. |

## Getting started

Use Python 3.11 for the deployment exercise and fetal-health project. Their dependencies are managed separately with [uv](https://docs.astral.sh/uv/); the top-level `requirements.txt` is for the deep-learning exercises.

### Deep-learning dependencies

From this directory:

```bash
python -m pip install -r requirements.txt
```

### Train and run the lead-scoring service

From `deployment/`, install the locked dependencies and train the model:

```bash
uv sync --locked
uv run python train.py
```

Training reads `../data/course_lead_scoring_2026.csv` by default and writes the model, feature defaults, and metadata into `deployment/`.

Start the API locally:

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

The service provides `GET /health` and `POST /predict`. For example:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304,
    "interaction_count": 7,
    "lead_score": 0.74
  }'
```

To build and run the same service with Docker, from `deployment/`:

```bash
docker build -t ml-zoomcamp-lead-scoring .
docker run --rm -p 9696:9696 ml-zoomcamp-lead-scoring
```

The deployed model artifact is already included. To run the local prediction smoke check, from `deployment/`:

```bash
uv run python smoke_test.py
```

### Fetal-health classifier

See [`projects/fetal_health_classifier/readme.md`](projects/fetal_health_classifier/readme.md) for the project overview, training steps, API usage, and Docker instructions. It has its own `pyproject.toml` and lockfile.
