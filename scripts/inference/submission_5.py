import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = PROJECT / "submissions"

p = SUBMISSIONS_DIR / "submission_v4.csv"
sub = pd.read_csv(p)
sub["fl_mm"] = (sub["fl_mm"] * 0.85).clip(30, 200)
sub.to_csv(SUBMISSIONS_DIR / "submission_v5_ab.csv", index=False)
print(sub.head())