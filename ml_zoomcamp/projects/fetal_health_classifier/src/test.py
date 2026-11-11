import pandas as pd
import httpx
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "fetal_health.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")
target = "fetal_health"                     

for cls in sorted(df[target].unique()):
    row = df[df[target] == cls].iloc[0].drop(target).to_dict()
    r = httpx.post("http://127.0.0.1:8001/predict", json=row)
    print("true class:", cls, "| status:", r.status_code)
    print(r.json(), "\n")