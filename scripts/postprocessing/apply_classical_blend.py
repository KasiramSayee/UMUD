# apply_classical_blend.py -- compute classical PA on test images, blend with submission
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage
import segmentation_models_pytorch as smp

# New universal path structure
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data"
MODELS_DIR = PROJECT / "models"
ARTIFACTS_DIR = PROJECT / "artifacts"
SUBMISSIONS_DIR = PROJECT / "submissions"
OUTPUTS_DIR = PROJECT / "outputs"

ROOT = DATA_DIR
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 512
IMG_EXTS = {".tif", ".tiff", ".png"}
BIAS = 12.08       # twin-measured bias correction
W_CLASS = 0.35     # blend weight (from twin table: 2.60° optimal)
IN_CSV = "submission_v15.csv"   # your current best submission

def classify(d):
    s = str(d.relative_to(ROOT)).lower()
    m = "mask" in s
    if "apo" in s:  return "apo_masks" if m else "apo_imgs"
    if "fasc" in s: return "fasc_masks" if m else "fasc_imgs"
    if "test" in s: return "test_imgs"
    return None

pools = {}
for d in sorted([dd for dd in ROOT.rglob("*") if dd.is_dir()],
                key=lambda x: len(x.parts)):
    c = classify(d)
    if c and c not in pools:
        pools[c] = sorted(f for f in d.rglob("*")
                          if f.is_file() and f.suffix.lower() in IMG_EXTS)

CONN = np.ones((3, 3))

print("loading ApoNet...")
models_a = []
for s in range(2):
    ma = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                  in_channels=1, classes=1).to(DEV)
    ma.load_state_dict(torch.load(MODELS_DIR / ("aponet_ens_{}.pt".format(s)),
                                   map_location=DEV))
    ma.eval()
    models_a.append(ma)

def predict_apo(img):
    W0, H0 = img.size
    x = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), np.float32)
    x = (x / 255.0 - 0.5) / 0.5
    with torch.no_grad():
        out = 0.0
        for m in models_a:
            out = out + torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(x[None, None])).to(DEV)))
        p = (out / len(models_a))[0, 0].cpu().numpy()
    pm = Image.fromarray((p * 255).astype(np.uint8)).resize((W0, H0), Image.BILINEAR)
    return np.asarray(pm) > 127

def classical_pa(gray, apo_mask):
    lbl, n = ndimage.label(apo_mask, structure=CONN)
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(ys) >= 500: comps.append((ys, xs))
    if len(comps) < 2: return None
    comps.sort(key=lambda c: c[0].mean())
    sup, deep = comps[0], comps[1]
    mh = deep[0].mean() - sup[0].mean()
    if mh < 60: return None
    y0 = int(sup[0].mean() + 0.25 * mh)
    y1 = int(deep[0].mean() - 0.15 * mh)
    if y1 - y0 < 30: return None
    apo_slope = np.polyfit(deep[1], deep[0], 1)[0]
    apo_ang = abs(np.degrees(np.arctan(apo_slope)))
    reg = gray[y0:y1, :].astype(np.float32)
    gx = ndimage.gaussian_filter(reg, 2.0, order=(0, 1))
    gy = ndimage.gaussian_filter(reg, 2.0, order=(1, 0))
    w = np.sqrt(gx * gx + gy * gy)
    phi = 2.0 * np.arctan2(gy, gx)
    R, I = (w * np.cos(phi)).sum(), (w * np.sin(phi)).sum()
    if R == 0 and I == 0: return None
    line_ang = abs((np.degrees(0.5 * np.arctan2(I, R)) + 90.0) % 180.0)
    if line_ang > 90: line_ang = 180.0 - line_ang
    return abs(line_ang - apo_ang) + BIAS

sub = pd.read_csv(SUBMISSIONS_DIR / IN_CSV)
results = []
for k, f in enumerate(pools["test_imgs"]):
    img = Image.open(f).convert("L")
    gray = np.asarray(img)
    apo = predict_apo(img)
    cp = classical_pa(gray, apo)
    results.append(dict(image_id=f.name, pa_class=cp))
    if (k + 1) % 50 == 0:
        print("  ...{}/{}".format(k + 1, len(pools["test_imgs"])))

cls = pd.DataFrame(results)
sub = sub.merge(cls, on="image_id", how="left")
n_blend = sub.pa_class.notna().sum()
sub["pa_deg"] = np.where(sub.pa_class.notna(),
                          W_CLASS * sub.pa_class + (1 - W_CLASS) * sub.pa_deg,
                          sub.pa_deg)
sub["pa_deg"] = sub.pa_deg.clip(5, 45)
sub = sub[["image_id", "pa_deg", "fl_mm", "mt_mm"]]
sub.to_csv(SUBMISSIONS_DIR / "submission_v15_blended.csv", index=False)
assert len(sub) == 309 and sub["image_id"].nunique() == 309
print("blended {}/309 images ({} used pipeline PA only)".format(n_blend, 309 - n_blend))
print(sub.head())