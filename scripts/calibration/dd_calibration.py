import re
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps
from scipy import ndimage

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
OCR_OK, _ocr_warned = True, False
try:
    import pytesseract
    for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p; break
except ImportError:
    OCR_OK = False
    print("pytesseract missing -> pip install pytesseract (mechanical analysis only)")

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

def detect_scales(gray):
    H, W = gray.shape
    found = []
    for side, c0, c1 in [("L", 0, min(130, W)), ("R", max(0, W - 130), W)]:
        bw = gray[:, c0:c1] > 90
        lbl, n = ndimage.label(bw)
        dots = []
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
            if 1 <= h <= 12 and 1 <= w <= 12 and len(ys) <= 150:
                dots.append((c0 + xs.mean(), ys.mean()))
        if len(dots) < 8: continue
        xs_ = np.array([x for x, _ in dots]); ys_ = np.array([y for _, y in dots])
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

def ruler_table(gray, x0, maxh=14):
    c0 = max(0, x0 - 6); c1 = min(gray.shape[1], x0 + 18)
    prof = (gray[:, c0:c1] > 90).sum(axis=1)
    on = prof > 0
    ticks = []; y = 0; H = len(on)
    while y < H:
        if on[y]:
            y2 = y
            while y2 < H and on[y2]: y2 += 1
            if (y2 - y) <= maxh:
                ticks.append(dict(y0=y, y1=y2, yc=(y + y2) / 2.0,
                                  ln=int(prof[y:y2].max())))
            y = y2
        else: y += 1
    return ticks

def ocr(im, box):
    global _ocr_warned
    if not OCR_OK: return ""
    crop = im.crop(box)
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    crop = ImageOps.autocontrast(crop.convert("L"))
    try:
        return pytesseract.image_to_string(crop, config="--psm 6").strip()
    except Exception:
        if not _ocr_warned:
            _ocr_warned = True
            print("  OCR engine not reachable -- check Tesseract install")
        return ""

DEPTH_RE = re.compile(r"\b(?:De|Depth|Tief\w*)\b\D{0,15}?(\d{1,2}(?:[.,]\d)?)\s*(mm|cm)?", re.I)

rows, sigs = [], {}
for k, f in enumerate(pools["test_imgs"]):
    gray = np.asarray(Image.open(f).convert("L"))
    im = Image.open(f).convert("RGB")
    H, W = gray.shape
    sc = detect_scales(gray)
    key = tuple((s, x, y0 // 10 * 10, y1 // 10 * 10) for s, x, c, y0, y1 in sc)
    fam = sigs.setdefault(key, len(sigs))
    spans, nlongs, lspaces = [], [], []
    for side, x0, cnt, y0, y1 in sc:
        t = ruler_table(gray, x0)
        if not t: continue
        ys = [q["yc"] for q in t]; lns = [q["ln"] for q in t]
        spans.append(max(ys) - min(ys))
        big = [q["yc"] for q in t if q["ln"] >= max(6, int(0.6 * max(lns)))]
        nlongs.append(len(big))
        if len(big) > 1: lspaces.append(float(np.median(np.diff(big))))
    texts = [ocr(im, (0, 0, min(150, W), H)),
             ocr(im, (max(0, W - 150), 0, W, H)),
             ocr(im, (0, max(0, H - 100), W, H)),
             ocr(im, (0, 0, W, min(100, H)))]
    blob = " | ".join(t for t in texts if t)
    dep = DEPTH_RE.search(blob)
    mm_px, method = "", ""
    if dep:
        val = float(dep.group(1).replace(",", "."))
        unit = (dep.group(2) or "mm").lower()
        dmm = val * 10 if unit == "cm" else val
        if spans:
            mm_px = round(dmm / max(spans), 5); method = "ocr_depth/span"
    rows.append(dict(name=f.name, family=fam, n_rulers=len(sc),
                     span=round(max(spans), 1) if spans else -1,
                     n_long=max(nlongs) if nlongs else 0,
                     long_space=round(max(lspaces), 1) if lspaces else -1,
                     depth_txt=(dep.group(0) if dep else "")[:24],
                     mm_px=mm_px, method=method,
                     ocr_snip=" ".join(blob.split())[:70]))
    if (k + 1) % 50 == 0:
        print("  ...{}/{}".format(k + 1, len(pools["test_imgs"])))

df = pd.DataFrame(rows)
df.to_csv(ARTIFACTS_DIR / "calibration_v1.csv", index=False)

print("\n=== DD. CALIBRATION SUMMARY ({} images, {} families) ===".format(len(df), len(sigs)))
for fam, g in df.groupby("family"):
    wd = g[g.mm_px != ""]
    sp = g[g.span > 0].span.median() if (g.span > 0).any() else -1
    ls = g[g.long_space > 0].long_space.median() if (g.long_space > 0).any() else -1
    print("  fam {:2d}: n={:3d} | span med={:6.1f} | long-space med={:6.1f} | depth OCR: {}/{}".format(
        fam, len(g), sp, ls, len(wd), len(g)))
    for _, r in wd.head(2).iterrows():
        print("       {} -> '{}' -> {} mm/px".format(r["name"], r.depth_txt, r.mm_px))

print("\n  PNG family: mechanical check (2cm markers -> mm/px = 20/long_space):")
for _, r in df[df["name"].str.endswith(".png")].head(8).iterrows():
    if r.long_space > 0:
        print("    {}: long_space={} -> {:.4f} mm/px | OCR: {}".format(
            r["name"], r.long_space, 20.0 / r.long_space, r.depth_txt or "-"))

big_fams = df.family.value_counts().head(8).index
need = df[(df.mm_px == "") & (df.family.isin(big_fams))]
print("\n  big-family images with NO calibration yet:", len(need))
print("  ", need["name"].head(12).tolist())