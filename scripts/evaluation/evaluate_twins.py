# evaluate_twins.py v2
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import re
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
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
OUT = OUTPUTS_DIR; OUT.mkdir(exist_ok=True)
SIZE = 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"
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

CONN = np.ones((3, 3))

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
                ticks.append((y, y2, (y + y2) / 2.0, int(prof[y:y2].max())))
            y = y2
        else: y += 1
    return ticks

def components(mask, min_area):
    lbl, n = ndimage.label(mask, structure=CONN)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < min_area: continue
        pts = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float64)
        evals, evecs = np.linalg.eigh(np.cov(pts))
        v = evecs[:, -1]; v = v / np.linalg.norm(v)
        try: fit = robust_polyfit(xs, ys, 1)
        except Exception: continue
        out.append(dict(area=int(len(xs)), cx=float(xs.mean()), cy=float(ys.mean()),
                        vx=float(v[0]), vy=float(v[1]), fit=fit,
                        elong=float(np.sqrt(evals[-1] / max(evals[0], 1e-9))),
                        vh=int(ys.max() - ys.min() + 1)))
    return out

def angle_between(v, d):
    c = min(1.0, abs(float(np.dot(v, d))))
    return np.degrees(np.arccos(c))

def robust_polyfit(xs, ys, degree=1, n_iter=5):
    fit = np.polyfit(xs, ys, degree)
    for _ in range(n_iter):
        residuals = ys - np.polyval(fit, xs)
        med = np.median(np.abs(residuals))
        sigma = max(1.4826 * med, 1e-6)
        w = np.minimum(1.0, 1.345 * sigma / (np.abs(residuals) + 1e-6))
        fit = np.polyfit(xs, ys, degree, w=np.sqrt(w))
    return fit

def geometry(fasc_m, apo_m, strict=True):
    ac = sorted(components(apo_m, 500), key=lambda c: -c["area"])
    if len(ac) < 2 or abs(ac[0]["cy"] - ac[1]["cy"]) < 60: return None
    sup, deep = sorted(ac[:2], key=lambda c: c["cy"])
    bd, ad = deep["fit"]; bs, as_ = sup["fit"]
    d = np.array([1.0, bd]);  d /= np.linalg.norm(d)
    ds = np.array([1.0, bs]); ds /= np.linalg.norm(ds)
    fc = []
    for c in components(fasc_m, 80):
        a = angle_between(np.array([c["vx"], c["vy"]]), d)
        if strict:
            if not (6 <= a <= 60 and c["elong"] >= 4
                    and sup["cy"] + 20 < c["cy"] < deep["cy"] - 20):
                continue
            fc.append((c, a))
        else:
            if 2 <= a <= 70:
                fc.append((c, a))
    if not fc: return None
    if strict:
        angs = np.array([a for _, a in fc])
        wts = np.array([c["area"] * np.sin(np.radians(a)) for c, a in fc],
                       dtype=float)
        best_mode, best_score = float(angs[0]), -1.0
        for a0 in angs:
            score = float(wts[np.abs(angs - a0) <= 3.5].sum())
            if score > best_score:
                best_score, best_mode = score, float(a0)
        fc = [(c, a) for c, a in fc if abs(a - best_mode) <= 3.5]
    fc = sorted(fc, key=lambda ca: -ca[0]["area"])[:3]
    W = fasc_m.shape[1]
    gaps = [abs((as_ + bs * x) - (ad + bd * x)) for x in (0.25 * W, 0.5 * W, 0.75 * W)]
    mt_px = float(np.mean(gaps) * np.cos(np.arctan((bs + bd) / 2)))
    pas, fls = [], []
    for c, a in fc:
        v = np.array([c["vx"], c["vy"]]); cpt = np.array([c["cx"], c["cy"]])
        pts = []
        for P, dd in [(np.array([0.0, ad]), d), (np.array([0.0, as_]), ds)]:
            den = v[0] * dd[1] - v[1] * dd[0]
            if abs(den) < 1e-6: pts.append(None); continue
            t = ((P[0] - cpt[0]) * dd[1] - (P[1] - cpt[1]) * dd[0]) / den
            pts.append(cpt + t * v)
        if pts[0] is None or pts[1] is None: continue
        fl = float(np.linalg.norm(pts[1] - pts[0]))
        if 0.4 <= fl * np.sin(np.radians(a)) / max(mt_px, 1e-6) <= 2.5:
            pas.append(a); fls.append(fl)
    if not pas: return None
    return dict(pa=float(np.mean(pas)), fl_px=float(np.mean(fls)), mt_px=mt_px)

def load_real_fasc(p, size):
    m = Image.open(p).convert("L")
    if m.size != size: m = m.resize(size, Image.BILINEAR)
    return ndimage.binary_dilation(np.asarray(m) > 32, CONN, iterations=2)

def load_real_apo(p, size):
    m = Image.open(p).convert("L")
    if m.size != size: m = m.resize(size, Image.BILINEAR)
    return np.asarray(m) > 127

try:
    import pytesseract
    for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p; break
    OCR_OK = True
except ImportError:
    OCR_OK = False

def ocr3(im, box):
    if not OCR_OK: return ""
    crop = im.crop(box)
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
    crop = ImageOps.autocontrast(crop.convert("L"))
    try:
        return pytesseract.image_to_string(crop, config="--psm 6").strip()
    except Exception:
        return ""

DEPTH_RE = re.compile(r"\b(?:De|Depth|Tief\w*)\b\D{0,15}?(\d{1,2}(?:[.,]\d)?)\s*(mm|cm)?", re.I)

def twin_scale(im_p, verbose=False):
    img = Image.open(im_p); W, H = img.size
    rgb = img.convert("RGB")
    regions = [("L", (0, 0, min(160, W), H)),
               ("R", (max(0, W - 160), 0, W, H)),
               ("B", (0, max(0, H - 160), W, H)),
               ("T", (0, 0, W, min(120, H)))]
    texts = {}
    for k, box in regions:
        t = ocr3(rgb, box)
        if t: texts[k] = " ".join(t.split())[:90]
    if verbose:
        print("    {} -> {}".format(im_p.name, texts if texts else "NOTHING READ"))
    blob = " | ".join(texts.values())
    m = DEPTH_RE.search(blob)
    if m:
        v = float(m.group(1).replace(",", "."))
        dmm = v * 10 if (m.group(2) or "mm").lower() == "cm" else v
        gray = np.asarray(img.convert("L"))
        t = ruler_table(gray, 6)
        span = (max(q[2] for q in t) - min(q[2] for q in t)) if t else 0
        if span > 200 and 20 <= dmm <= 120:
            s = dmm / span
            if 0.03 <= s <= 0.15:
                return s
    return None

print("loading models...")
model_a = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                   in_channels=1, classes=1).to(DEV)
model_a.load_state_dict(torch.load(MODELS_DIR / "aponet_v1.pt", map_location=DEV))
model_a.eval()
model_f1 = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=1, classes=1).to(DEV)
model_f1.load_state_dict(torch.load(MODELS_DIR / "fascnet_v1.pt", map_location=DEV))
model_f1.eval()
models_f = []
for k in [0,1]:
    mm = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                  in_channels=1, classes=1).to(DEV)
    mm.load_state_dict(torch.load(MODELS_DIR / ("fascnet_v3_fold{}.pt".format(k)),
                                  map_location=DEV))
    mm.eval()
    models_f.append(mm)

def predict(models, img, tta=True, size=512):
    if not isinstance(models, list): models = [models]
    W0, H0 = img.size
    def forward(im):
        x = np.asarray(im.resize((size, size), Image.BILINEAR), np.float32)
        x = (x / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            out = 0.0
            for m in models:
                out = out + torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(x[None, None])).to(DEV)))
            return (out / len(models))[0, 0].cpu().numpy()
    p = forward(img)
    if tta:
        p = 0.5 * p + 0.5 * forward(img.transpose(Image.FLIP_LEFT_RIGHT))[:, ::-1]
    pm = Image.fromarray((p * 255).astype(np.uint8)).resize((W0, H0), Image.BILINEAR)
    return ndimage.binary_closing(np.asarray(pm) > 127, CONN)

twins = pd.read_csv(ARTIFACTS_DIR / "twins.csv")
tw_by_name = {Path(r.fasc_img).name: r for _, r in twins.iterrows()}

print("=== twin scale probe (first 6 twins, all OCR text found) ===")
scales = []
for i, (_, r) in enumerate(twins.iterrows()):
    s = twin_scale(Path(r.fasc_img), verbose=(i < 6))
    scales.append(s)
valid = [s for s in scales if s]
print("  valid twin scales: {}/122".format(len(valid)))
med = float(np.median(valid)) if valid else None

rows = []
for i, r in twins.iterrows():
    im_p = Path(r.fasc_img)
    img = Image.open(im_p).convert("L")
    size = img.size
    mm_px = scales[i] if scales[i] else med
    gt = geometry(load_real_fasc(Path(r.fasc_mask), size),
                  load_real_apo(Path(r.apo_mask), size), strict=False)
    g = geometry(predict(models_f, img, size=640), predict(model_a, img))
    src = "ens"
    if g is None:
        g = geometry(predict(model_f1, img, tta=True), predict(model_a, img)); src = "v1"
    if g is None:
        g = geometry(predict(model_f1, img, tta=False), predict(model_a, img, tta=False)); src = "v1s"
    rows.append(dict(name=im_p.name, mm_px=mm_px,
                     gt_pa=gt["pa"] if gt else None,
                     gt_fl_px=gt["fl_px"] if gt else None,
                     gt_mt_px=gt["mt_px"] if gt else None,
                     pr_pa=g["pa"] if g else None,
                     pr_fl_px=g["fl_px"] if g else None,
                     pr_mt_px=g["mt_px"] if g else None,
                     src=src if g else "FAIL"))

df = pd.DataFrame(rows)
df.to_csv(ARTIFACTS_DIR / "twin_eval.csv", index=False)
ok = df.dropna(subset=["gt_pa", "pr_pa"])
print("\n=== ERRORS (pixel space; {} of 122 pairs) ===".format(len(ok)))
for t in ["pa", "fl_px", "mt_px"]:
    err = ok["pr_" + t] - ok["gt_" + t]
    print("  {}: MAE={:.2f} bias={:+.2f}".format(t, err.abs().mean(), err.mean()))

b, a = np.polyfit(ok.pr_pa, ok.gt_pa, 1)
corr = float(np.corrcoef(ok.pr_pa, ok.gt_pa)[0, 1])
pa_corr = a + b * ok.pr_pa
print("\n=== PA SHRINKAGE FIT ===")
print("  corr={:.2f} | gt_pa = {:.2f} + {:.2f} * pr_pa".format(corr, a, b))
print("  PA MAE: raw={:.2f} corrected={:.2f}".format(
    (ok.pr_pa - ok.gt_pa).abs().mean(), (pa_corr - ok.gt_pa).abs().mean()))

if med:
    sc = ok[ok.mm_px.notna()]
    print("\n=== mm-space (scale={:.4f}, n={}) ===".format(med, len(sc)))
    for t in ["fl", "mt"]:
        e = ((sc["pr_" + t + "_px"] - sc["gt_" + t + "_px"]).abs() * sc.mm_px)
        print("  {}: MAE={:.2f} mm".format(t, e.mean()))

df["comb"] = ((df.pr_pa - df.gt_pa).abs() / 6 +
              (df.pr_fl_px - df.gt_fl_px).abs() / (df.gt_fl_px.abs() + 1e-9) +
              (df.pr_mt_px - df.gt_mt_px).abs() / (df.gt_mt_px + 1e-9))
worst = df.sort_values("comb", ascending=False).head(6)
print("\n  worst 6 (PA truth vs pred):")
print(worst[["name", "gt_pa", "pr_pa", "src"]].round(1).to_string())
for _, w in worst.iterrows():
    nm = w["name"]
    tr = tw_by_name[nm]
    img = Image.open(tr.fasc_img).convert("L")
    size = img.size
    rf = load_real_fasc(Path(tr.fasc_mask), size)
    ra = load_real_apo(Path(tr.apo_mask), size)
    pf = predict(models_f, img, size=640)
    pa_ = predict(model_a, img)
    ac = sorted(components(ra, 500), key=lambda c: -c["area"])[:2]
    if len(ac) == 2:
        sup, deep = sorted(ac, key=lambda c: c["cy"])
        bd, ad = deep["fit"]
        d = np.array([1.0, bd]); d /= np.linalg.norm(d)
        real_angs = sorted((angle_between(np.array([c["vx"], c["vy"]]), d)
                            for c in components(rf, 80)), reverse=True)[:8]
        pred_angs = sorted((angle_between(np.array([c["vx"], c["vy"]]), d)
                            for c in components(pf, 80)), reverse=True)[:8]
        print("  {}: real ang {} | pred ang {}".format(
            nm, [round(x, 1) for x in real_angs], [round(x, 1) for x in pred_angs]))
    gray = np.asarray(img)
    rgb = np.stack([gray] * 3, -1).astype(np.uint8)
    rgb[ra, 1] = 255; rgb[ra, 2] = 255          # real aponeurosis: cyan
    rgb[pa_, 0] = 255; rgb[pa_, 2] = 255        # predicted apo: magenta
    rgb[rf, 1] = 255                             # real fascicles: green
    rgb[pf, 0] = 255; rgb[pf, 1] = 60           # predicted fascicles: red
    Image.fromarray(rgb).save(OUT / ("twinQA_" + Path(nm).stem + ".png"))
    print("  saved twinQA_{}.png".format(Path(nm).stem))