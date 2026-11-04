# classical_pa.py -- structure-tensor PA vs pipeline, on the twins
# CPU-only, ~5 minutes, runs alongside GPU training
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage

# New universal path structure
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data"
MODELS_DIR = PROJECT / "models"
ARTIFACTS_DIR = PROJECT / "artifacts"
SUBMISSIONS_DIR = PROJECT / "submissions"
OUTPUTS_DIR = PROJECT / "outputs"

ROOT = DATA_DIR
twins = pd.read_csv(ARTIFACTS_DIR / "twins.csv")
ev = pd.read_csv(ARTIFACTS_DIR / "twin_eval.csv")

def classical_pa(gray, apo_mask):
    # find the two aponeurosis bands (from the real mask)
    lbl, n = ndimage.label(apo_mask, structure=np.ones((3, 3)))
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(ys) >= 500: comps.append((ys, xs))
    if len(comps) < 2: return None
    comps.sort(key=lambda c: c[0].mean())
    sup, deep = comps[0], comps[1]
    mh = deep[0].mean() - sup[0].mean()
    if mh < 60: return None
    # region between aponeuroses (upper quarter excluded to skip superficial connective tissue)
    y0 = int(sup[0].mean() + 0.25 * mh)
    y1 = int(deep[0].mean() - 0.15 * mh)
    if y1 - y0 < 30: return None
    # deep aponeurosis angle (reference for PA)
    apo_slope = np.polyfit(deep[1], deep[0], 1)[0]
    apo_ang = abs(np.degrees(np.arctan(apo_slope)))
    # structure tensor: gradients weighted by magnitude, circular mean of doubled angles
    reg = gray[y0:y1, :].astype(np.float32)
    gx = ndimage.gaussian_filter(reg, 2.0, order=(0, 1))
    gy = ndimage.gaussian_filter(reg, 2.0, order=(1, 0))
    w = np.sqrt(gx * gx + gy * gy)
    phi = 2.0 * np.arctan2(gy, gx)          # doubled line angles
    R, I = (w * np.cos(phi)).sum(), (w * np.sin(phi)).sum()
    if R == 0 and I == 0: return None
    line_ang = abs((np.degrees(0.5 * np.arctan2(I, R)) + 90.0) % 180.0)
    if line_ang > 90: line_ang = 180.0 - line_ang
    return abs(line_ang - apo_ang)

rows = []
for _, r in twins.iterrows():
    img = Image.open(r.fasc_img).convert("L")
    m = Image.open(r.apo_mask).convert("L")
    if m.size != img.size:
        m = m.resize(img.size, Image.BILINEAR)
    pa = classical_pa(np.asarray(img), np.asarray(m) > 127)
    rows.append(dict(name=Path(r.fasc_img).name, pa_class=pa))

df = pd.DataFrame(rows).merge(ev[["name", "gt_pa", "pr_pa"]], on="name", how="left")
ok = df.dropna()
ok["pa_class"] = ok["pa_class"] + 12.08   # bias correction
print("solved: {}/{}".format(len(ok), len(df)))
err_c, err_p = ok.pa_class - ok.gt_pa, ok.pr_pa - ok.gt_pa
print("classical: MAE={:.2f} bias={:+.2f}".format(err_c.abs().mean(), err_c.mean()))
print("pipeline : MAE={:.2f} bias={:+.2f}".format(err_p.abs().mean(), err_p.mean()))
print("ERROR CORRELATION: {:.2f}   (<0.5 means blending helps)".format(
    np.corrcoef(err_c, err_p)[0, 1]))
for wgt in [0.0, 0.2, 0.3, 0.4, 0.5]:
    b = wgt * ok.pa_class + (1 - wgt) * ok.pr_pa
    print("  blend w={:.1f}: PA MAE={:.2f}".format(wgt, (b - ok.gt_pa).abs().mean()))