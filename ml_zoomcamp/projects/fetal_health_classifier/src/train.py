
import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    DATA_PATH,
    LABELS,
    MODEL_DIR,
    SEED,
    TARGET,
    TEST_SIZE,
    XGB_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and apply the same cleaning as the notebook."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("Loaded %d rows, dropped %d duplicates", n_before, n_before - len(df))

    # Raw target is 1.0 / 2.0 / 3.0; XGBoost needs 0 / 1 / 2
    df[TARGET] = df[TARGET].astype(int) - 1
    if set(df[TARGET].unique()) != set(LABELS):
        raise ValueError(f"Unexpected target values: {sorted(df[TARGET].unique())}")
    return df


def split(df: pd.DataFrame):
    """Stratified split. Same seed and size as the notebook, so the test set is identical."""
    features = [c for c in df.columns if c != TARGET]
    X, y = df[features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        features,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Fit XGBoost with class-balanced sample weights to handle the class imbalance."""
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        **XGB_PARAMS,
    )
    model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    return model


def evaluate(model, X_test, y_test) -> float:
    """Report held-out performance and return macro F1."""
    pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, pred, average="macro")
    log.info("Test macro F1: %.3f", macro_f1)
    print(classification_report(y_test, pred, target_names=list(LABELS.values()), digits=3))
    return macro_f1


def save(model, features, macro_f1, output_dir: Path) -> None:
    """Bundle the model with its feature order and label map so serving cannot drift."""
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "model_name": "xgb",
        "params": XGB_PARAMS,
        "features": features,
        "labels": LABELS,
    }
    with open(output_dir / "model.bin", "wb") as f:
        pickle.dump(artifact, f)

    metadata = {
        "model_name": "xgb",
        "params": XGB_PARAMS,
        "features": features,
        "labels": LABELS,
        "test_macro_f1": round(float(macro_f1), 4),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Saved model and metadata to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fetal health classifier")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to fetal_health.csv")
    parser.add_argument("--output", type=Path, default=MODEL_DIR, help="Directory for the model files")
    args = parser.parse_args()

    df = load_data(args.data)
    X_train, X_test, y_train, y_test, features = split(df)
    log.info("Train: %s  Test: %s", X_train.shape, X_test.shape)

    model = train_model(X_train, y_train)
    macro_f1 = evaluate(model, X_test, y_test)
    save(model, features, macro_f1, args.output)


if __name__ == "__main__":
    main()