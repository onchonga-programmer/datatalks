from pathlib import Path
import json
import numpy as np

reference_dir = Path(__file__).resolve().parent / "reference"

hb = json.loads((reference_dir / "history_baseline.json").read_text())
ha = json.loads((reference_dir / "history_augmented.json").read_text())

print("Q3", round(float(np.median(hb["train_accuracy"])), 2))
print("Q4", round(float(np.std(hb["train_loss"], ddof=0)), 3))
print("Q5", round(float(np.mean(ha["evaluation_loss"])), 3))
print("Q6", round(float(np.mean(ha["evaluation_accuracy"][-5:])), 2))