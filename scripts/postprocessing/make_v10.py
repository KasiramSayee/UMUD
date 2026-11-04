import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = PROJECT / "submissions"

v9 = pd.read_csv(SUBMISSIONS_DIR / "submission_v9.csv")
v9b = pd.read_csv(SUBMISSIONS_DIR / "submission_v9b.csv")

sub = v9b.copy()
ratio = np.sin(np.radians(v9["pa_deg"])) / np.sin(np.radians(v9b["pa_deg"]))
sub["fl_mm"] = (v9["fl_mm"] * ratio).clip(30, 200)

assert len(sub) == 309 and sub["image_id"].nunique() == 309
assert sub[["pa_deg", "fl_mm", "mt_mm"]].isna().sum().sum() == 0
sub.to_csv(SUBMISSIONS_DIR / "submission_v10.csv", index=False)
print(sub.head(8).to_string())
print("FL mean: v9b {:.1f} -> v10 {:.1f}".format(v9b["fl_mm"].mean(), sub["fl_mm"].mean()))