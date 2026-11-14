# Fetal Health Classification from Cardiotocography (CTG)

A machine learning service that classifies fetal health status (normal, suspect, pathological) from cardiotocography exam measurements. Built as the midterm project for [DataTalks.Club ML Zoomcamp 2026](https://courses.datatalks.club/ml-zoomcamp-2026/).

> **Disclaimer:** This is a learning project, not a medical device. It must not be used for real clinical decisions.

## Problem description

Cardiotocography (CTG) records the fetal heart rate and uterine contractions during pregnancy. Interpreting these traces requires trained specialists, and in low-resource settings such expertise is often scarce. Late identification of a fetus at risk is a major contributor to preventable maternal and neonatal complications.

This project trains a classifier that takes the numeric features extracted from a CTG exam and predicts one of three outcomes:

| Label | Class | Meaning |
|-------|-------|---------|
| 1 | Normal | No concerning pattern |
| 2 | Suspect | Needs closer monitoring |
| 3 | Pathological | Needs urgent attention |

**How it would be used:** as a triage aid that flags suspect and pathological cases for prioritized review by a clinician. Because missing a pathological case is far more costly than a false alarm, the project optimizes and reports **per-class recall** and **macro F1**, not accuracy alone.

## Dataset

- **Source:** Fetal Health Classification dataset (Kaggle), derived from the UCI Cardiotocography data. https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
- **Size:** 2,126 records, 21 input features, 1 target (`fetal_health`).
- **Features:** baseline fetal heart rate, accelerations, fetal movement, uterine contractions, several types of decelerations, short and long term variability measures, and histogram summary statistics.
- **Labels:** assigned by expert obstetricians.
- **Class balance:** strongly imbalanced, with roughly 78% normal, 14% suspect and 8% pathological.
- **Location in this repo:** `data/fetal_health.csv`

## Approach

1. **EDA** (`notebooks/notebook.ipynb`): target distribution, missing values and duplicates, feature distributions, correlations, and feature importance.
2. **Models compared:** logistic regression, random forest, and XGBoost, each tuned with cross-validation.
3. **Imbalance handling:** stratified splits and class weighting, evaluated with macro F1 and per-class recall.
4. **Interpretability:** SHAP values for the final model.
5. **Final model:** retrained on the full training set by `src/train.py` and saved to `models/model.bin`.
6. **Serving:** a FastAPI app (`src/predict.py`) with input validation, containerized with Docker.

## Results

| Model              | Stage   | CV macro F1        | Recall: normal | Recall: suspect | Recall: pathological |
|--------------------|---------|--------------------|----------------:|----------------:|---------------------:|
| Logistic regression| baseline| 0.781 (±0.030)     | 0.872           | 0.829           | 0.850                |
| Logistic regression| tuned   | 0.782 (±0.031)     | 0.859           | 0.876           | 0.829                |
| Random forest      | baseline| 0.877 (±0.018)     | 0.949           | 0.808           | 0.907                |
| Random forest      | tuned   | 0.883 (±0.019)     | 0.951           | 0.820           | 0.914                |
| XGBoost            | baseline| 0.897 (±0.022)     | 0.970           | 0.816           | 0.907                |
| XGBoost            | tuned   | 0.898 (±0.022)     | 0.969           | 0.816           | 0.914                |


## Findings

- **Tree models clearly outperform logistic regression.** The gap is approximately **0.10 macro F1**, far larger than the fold-to-fold variation (standard deviation of about **0.02–0.03**). This suggests that the relationships between CTG features and fetal status are non-linear.

- **XGBoost was selected, but it performs close to random forest.** The **0.015** difference in macro F1 is within one standard deviation. XGBoost was chosen because it achieved the highest macro F1, the best recall on the normal class (**0.969**), and equal recall on the pathological class (**0.914**).

- **Hyperparameter tuning provided only marginal gains.** XGBoost improved from **0.897** to **0.898**, while random forest improved from **0.877** to **0.883**. This indicates that the default settings were already close to optimal and that performance is likely limited by the available data rather than hyperparameter choices.

- **The suspect class is the hardest to identify.** Recall is approximately **0.82**

Final test-set performance of the selected model:

| Class | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Normal | 0.967|0.991 | 0.979|
| Suspect|0.939 | 0.793|0.860 |
| Pathological | 0.972| 1.000| 0.986|

## Repository structure

```
fetal-health-classifier/
├── data/                 # dataset
├── notebooks/            # EDA, feature analysis, model selection
├── src/
│   ├── train.py          # trains the final model and saves it
│   ├── predict.py        # FastAPI prediction service
│   └── config.py         # paths and shared settings
├── models/               # saved model artifact
├── tests/                # API smoke tests
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Getting started

### 1. Clone and install dependencies

```bash
git clone https://github.com/onchonga-programmer/fetal-health-classifier.git
cd fetal-health-classifier
uv sync
```

### 2. Train the model

```bash
uv run python src/train.py
```

This reads `data/fetal_health.csv` and writes the trained model to `models/model.bin`.

### 3. Run the API locally

```bash
uv run uvicorn src.predict:app --host 0.0.0.0 --port 8000
```

### 4. Run with Docker

```bash
docker build -t fetal-health-classifier .
docker run -p 8000:8000 fetal-health-classifier
```

### 5. Send a test request

The service expects all 21 features. The request below is abbreviated; see `tests/test_predict.py` for a complete payload.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "baseline_value": 120,
        "accelerations": 0.0,
        "fetal_movement": 0.0,
        "uterine_contractions": 0.0
      }'
```

Example response:

```json
{
  "prediction": "normal",
  "probabilities": {
    "normal": 0.92,
    "suspect": 0.06,
    "pathological": 0.02
  }
}
```

Other endpoints: `GET /health` returns service status.

## Deployment


Live service: _not yet deployed_

## Reproducibility

- Dependencies are pinned in `uv.lock`.
- The dataset is included in the repo, so every step runs without external downloads.
- Random seeds are fixed in `src/config.py`.

## Acknowledgements

Dataset: Fetal Health Classification (Kaggle), based on the UCI Cardiotocography data. Project completed as part of ML Zoomcamp by DataTalks.Club.