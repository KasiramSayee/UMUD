import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = PROJECT / "submissions"

sub = pd.read_csv(SUBMISSIONS_DIR / "submission_v11b.csv")
tif = sub["image_id"].str.endswith(".tif")
for k, name in [(1.05, "submission_v12_p5.csv"), (0.95, "submission_v12_m5.csv")]:
    s = sub.copy()
    s.loc[tif, "fl_mm"] = (s.loc[tif, "fl_mm"] * k).clip(30, 200)
    s.loc[tif, "mt_mm"] = (s.loc[tif, "mt_mm"] * k).clip(10, 50)
    s.to_csv(SUBMISSIONS_DIR / name, index=False)
    print(name, "written")