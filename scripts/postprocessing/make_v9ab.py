import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = PROJECT / "submissions"

p = SUBMISSIONS_DIR / "submission_v9.csv"
sub = pd.read_csv(p)

# v9a: mean-centered un-shrink (keeps the mean at 15.5, widens the spread)
a = sub.copy()
MU, B = 15.5, 0.59
a["pa_deg"] = (MU + (a["pa_deg"] - MU) / B).clip(5, 45)
a.to_csv(SUBMISSIONS_DIR / "submission_v9a.csv", index=False)

# v9b: raw twin fit (applies the measured twin relationship directly)
b = sub.copy()
b["pa_deg"] = (7.54 + 0.59 * b["pa_deg"]).clip(5, 45)
b.to_csv(SUBMISSIONS_DIR / "submission_v9b.csv", index=False)

for name, df in [("v9a", a), ("v9b", b)]:
    assert len(df) == 309 and df["image_id"].nunique() == 309
    assert df["pa_deg"].isna().sum() == 0
    print(name, "OK:", len(df), "rows")

print("\nv9  (original):\n", sub.head(3).to_string())
print("\nv9a (un-shrink):\n", a.head(3).to_string())
print("\nv9b (twin fit):\n", b.head(3).to_string())