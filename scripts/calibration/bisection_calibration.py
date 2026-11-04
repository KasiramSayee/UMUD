import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = PROJECT / "submissions"

sub = pd.read_csv(SUBMISSIONS_DIR / "submission_v11b.csv")
tif = sub["image_id"].str.endswith(".tif")
s = sub.copy()
s.loc[tif, "fl_mm"] = (s.loc[tif, "fl_mm"] * 0.90).clip(30, 200)
s.loc[tif, "mt_mm"] = (s.loc[tif, "mt_mm"] * 0.90).clip(10, 50)
s.to_csv(SUBMISSIONS_DIR / "submission_v12_m10.csv", index=False)
print(s.head())