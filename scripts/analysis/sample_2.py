import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from PIL import Image
try:
    from scipy import ndimage
except ImportError:
    raise SystemExit("Need scipy: pip install scipy")

# New universal path structure
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data"
MODELS_DIR = PROJECT / "models"
ARTIFACTS_DIR = PROJECT / "artifacts"
SUBMISSIONS_DIR = PROJECT / "submissions"
OUTPUTS_DIR = PROJECT / "outputs"

ROOT = DATA_DIR
OUT = OUTPUTS_DIR; OUT.mkdir(exist_ok=True)
IMG_EXTS = {".tif", ".tiff", ".png"}

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

def mask_frac(p):
    m = np.asarray(Image.open(p).convert("L"))
    return float((m > 127).mean()), m.shape

print("=== Y. MASK CONTENT AUDIT (full pools, few minutes) ===")
for pname, mkey in [("FASC", "fasc_masks"), ("APO", "apo_masks")]:
    rows = []
    for f in pools[mkey]:
        idx = int("".join(ch for ch in f.stem if ch.isdigit()) or 0)
        frac, shape = mask_frac(f)
        rows.append((idx, frac, shape))
    rows.sort()
    fr = np.array([r[1] for r in rows])
    print("  [{}] n={} | nonempty(frac>0.001): {} | >0.005: {} | >0.02: {}".format(
        pname, len(fr), int((fr > 0.001).sum()),
        int((fr > 0.005).sum()), int((fr > 0.02).sum())))
    for lo in range(0, 2800, 200):
        sub = [r for r in rows if lo <= r[0] < lo + 200]
        if not sub: continue
        f_ = np.array([r[1] for r in sub])
        shapes = Counter(r[2] for r in sub).most_common(2)
        print("    {}-{}: n={} med_frac={:.4f} nonempty={} shapes={}".format(
            lo, lo + 200, len(sub), float(np.median(f_)),
            int((f_ > 0.001).sum()), shapes))

print("\n=== Z. TWIN MASK FRACTIONS ===")
tdf = pd.read_csv(ARTIFACTS_DIR / "twins.csv")
tfr, afr = [], []
for _, r in tdf.iterrows():
    tf, _ = mask_frac(Path(r.fasc_mask))
    af, _ = mask_frac(Path(r.apo_mask))
    tfr.append(tf); afr.append(af)
tfr, afr = np.array(tfr), np.array(afr)
print("  twin fasc masks: median={:.5f} | nonempty: {}/{}".format(
    float(np.median(tfr)), int((tfr > 0.001).sum()), len(tfr)))
print("  twin apo masks:  median={:.5f} | nonempty: {}/{}".format(
    float(np.median(afr)), int((afr > 0.001).sum()), len(afr)))

print("\n=== W2. TEST SCALE DETECTION (tick columns) ===")
def detect_scales(gray):
    H, W = gray.shape
    found = []
    for side, c0, c1 in [("L", 0, min(130, W)), ("R", max(0, W - 130), W)]:
        bw = gray[:, c0:c1] > 90
        lbl, n = ndimage.label(bw)
        dots = []
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            if 1 <= h <= 12 and 1 <= w <= 12 and len(ys) <= 150:
                dots.append((c0 + xs.mean(), ys.mean()))
        if len(dots) < 8: continue
        xs_ = np.array([d[0] for d in dots]); ys_ = np.array([d[1] for d in dots])
        best = None
        for x0 in range(c0, c1 - 5, 6):
            sel = (xs_ >= x0) & (xs_ < x0 + 6)
            if sel.sum() < 8: continue
            yv = np.sort(ys_[sel])
            key = (int(sel.sum()), int(yv[-1] - yv[0]))
            if best is None or key > best[0]: best = (key, x0, yv)
        if best:
            (cnt, span), x0, yv = best
            found.append((side, x0, cnt, int(yv[0]), int(yv[-1])))
    return found

n_with = 0
for k, f in enumerate(pools["test_imgs"]):
    gray = np.asarray(Image.open(f).convert("L"))
    sc = detect_scales(gray)
    if sc: n_with += 1
    if k < 12 or (k >= 12 and k % 30 == 0):
        print("  {}: {}".format(f.name, sc))
    if k < 10 and sc:
        im = Image.open(f).convert("RGB")
        side, x0, cnt, y0, y1 = sc[0]
        box = (max(0, x0 - 70), max(0, y0 - 50),
               min(im.width, x0 + 70), min(im.height, y1 + 50))
        crop = im.crop(box)
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.NEAREST)
        crop.save(OUT / ("calib_" + f.stem + ".png"))
print("  test images with a detected tick column: {}/{}".format(
    n_with, len(pools["test_imgs"])))
print("  saved calib_*.png crops for the first 10 -- READ THESE")