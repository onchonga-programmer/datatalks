from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fetal_health.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.bin"
METADATA_PATH = MODEL_DIR / "metadata.json"

SEED = 42
TEST_SIZE = 0.2

TARGET = "fetal_health"
LABELS = {0: "normal", 1: "suspect", 2: "pathological"}

# TODO: replace with the best XGBoost params printed in section 6 of the notebook.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1,
}