import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
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

tdf = pd.read_csv(ARTIFACTS_DIR / "twins.csv")
CONN = np.ones((3, 3))                       # 8-connectivity -- THE FIX

def load_mask(path, size):
    m = Image.open(path).convert("L")
    if m.size != size: m = m.resize(size, Image.BILINEAR)
    b = np.asarray(m) > 32
    return ndimage.binary_dilation(b, structure=CONN)

def components(mask, min_area):
    lbl, n = ndimage.label(mask, structure=CONN)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < min_area: continue
        pts = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float64)
        evals, evecs = np.linalg.eigh(np.cov(pts))
        v = evecs[:, -1]; v = v / np.linalg.norm(v)
        try: fit = np.polyfit(xs, ys, 1)
        except Exception: continue
        out.append(dict(area=int(len(xs)), cx=float(xs.mean()), cy=float(ys.mean()),
                        vx=float(v[0]), vy=float(v[1]),
                        ang=float(np.degrees(np.arctan2(abs(v[1]), abs(v[0])))),
                        w=int(xs.max() - xs.min() + 1),
                        hgt=int(ys.max() - ys.min() + 1), fit=fit))
    return out

print("=== AA1. FASC CENSUS v2 (8-conn + dilation, 15 twins) ===")
for i, r in list(tdf.iterrows())[:15]:
    im_p = Path(r.fasc_img)
    size = Image.open(im_p).size
    comps = sorted(components(load_mask(Path(r.fasc_mask), size), 80),
                   key=lambda c: -c["area"])
    desc = " | ".join("a={} ang={:.0f} {}x{}".format(c["area"], c["ang"], c["w"], c["hgt"])
                      for c in comps[:6])
    print("  {}: {} comps | {}".format(im_p.name, len(comps), desc))

print("\n  angle histogram, 60 sampled non-twin fasc masks:")
hist = Counter()
fasc_img = {f.stem: f for f in pools["fasc_imgs"]}
inv = []
for f in pools["fasc_masks"]:
    size = Image.open(fasc_img[f.stem]).size
    m = np.asarray(Image.open(f).convert("L"))
    frac = float((m > 127).mean())
    inv.append(dict(stem=f.stem, frac=frac, native=m.shape))
    if frac > 0.001:
        for c in components(load_mask(f, size), 80):
            hist[int(c["ang"] // 5) * 5] += 1
pd.DataFrame(inv).to_csv(ARTIFACTS_DIR / "fasc_mask_inventory.csv", index=False)
for k in sorted(hist):
    print("    {}-{} deg: {}".format(k, k + 5, hist[k]))
print("  saved fasc_mask_inventory.csv ({} masks, {} nonempty)".format(
    len(inv), sum(1 for x in inv if x["frac"] > 0.001)))

print("\n=== AA2. GEOMETRY v0.2 ON TWINS ===")
def angle_between(v, d):
    c = min(1.0, abs(float(np.dot(v, d))))
    return np.degrees(np.arccos(c))

results, fail_apo, fail_fasc, qa = [], 0, 0, 0
for _, r in tdf.iterrows():
    im_p = Path(r.fasc_img)
    size = Image.open(im_p).size
    gray = np.asarray(Image.open(im_p).convert("L"))
    ac = sorted(components(load_mask(Path(r.apo_mask), size), 500),
                key=lambda c: -c["area"])
    if len(ac) < 2 or abs(ac[0]["cy"] - ac[1]["cy"]) < 60:
        fail_apo += 1; continue
    sup, deep = sorted(ac[:2], key=lambda c: c["cy"])
    bd, ad = deep["fit"]; bs, as_ = sup["fit"]
    d = np.array([1.0, bd]);  d /= np.linalg.norm(d)
    ds = np.array([1.0, bs]); ds /= np.linalg.norm(ds)
    fcomps = sorted(components(load_mask(Path(r.fasc_mask), size), 80),
                    key=lambda c: -c["area"])
    fc = []
    for c in fcomps:
        a = angle_between(np.array([c["vx"], c["vy"]]), d)
        if 3 <= a <= 60: fc.append((c, a))
    if not fc:
        fail_fasc += 1; continue
    fc = fc[:3]
    pas, fls = [], []
    for c, a in fc:
        pas.append(a)
        v = np.array([c["vx"], c["vy"]])
        cpt = np.array([c["cx"], c["cy"]])
        pts = []
        for P, dd in [(np.array([0.0, ad]), d), (np.array([0.0, as_]), ds)]:
            den = v[0] * dd[1] - v[1] * dd[0]
            if abs(den) < 1e-6: pts.append(None); continue
            t = ((P[0] - cpt[0]) * dd[1] - (P[1] - cpt[1]) * dd[0]) / den
            pts.append(cpt + t * v)
        if pts[0] is not None and pts[1] is not None:
            fls.append(float(np.linalg.norm(pts[1] - pts[0])))
    W = size[0]
    gaps = [abs((as_ + bs * x) - (ad + bd * x)) for x in (0.25 * W, 0.5 * W, 0.75 * W)]
    mt_px = float(np.mean(gaps) * np.cos(np.arctan((bs + bd) / 2)))
    results.append(dict(fasc=im_p.name, pa=float(np.mean(pas)),
                        fl_px=float(np.mean(fls)) if fls else np.nan,
                        mt_px=mt_px, n_fasc=len(pas)))
    if len(results) % 10 == 1 and qa < 12:
        img = Image.fromarray(np.stack([gray] * 3, -1).astype(np.uint8))
        dr = ImageDraw.Draw(img)
        dr.line([(0, ad), (W, ad + bd * W)], fill=(0, 120, 255), width=3)
        dr.line([(0, as_), (W, as_ + bs * W)], fill=(0, 255, 255), width=3)
        for c, a in fc:
            if abs(c["vx"]) < 1e-6: continue
            sl = c["vy"] / c["vx"]
            dr.line([(0, c["cy"] - sl * c["cx"]), (W, c["cy"] + sl * (W - c["cx"]))],
                    fill=(255, 60, 60), width=3)
        img.save(OUT / ("geomQA_" + im_p.stem + ".png"))
        qa += 1

res = pd.DataFrame(results)
res.to_csv(ARTIFACTS_DIR / "geometry_v2_twins.csv", index=False)
print("  solved: {} | apo failures: {} | no-fascicle failures: {}".format(
    len(res), fail_apo, fail_fasc))
if len(res):
    print(res[["pa", "fl_px", "mt_px", "n_fasc"]].describe().to_string())
    pa_ok = ((res.pa >= 5) & (res.pa <= 45)).sum()
    print("  PA within physiological 5-45 deg: {}/{}".format(pa_ok, len(res)))
    ok = res.dropna(subset=["fl_px"])
    if len(ok):
        ratio = ok.fl_px * np.sin(np.radians(ok.pa)) / ok.mt_px
        print("  consistency FL*sin(PA)/MT: median={:.2f} p25={:.2f} p75={:.2f}".format(
            ratio.median(), ratio.quantile(0.25), ratio.quantile(0.75)))
print("  saved {} QA overlays".format(qa))

print("\n=== BB. TEST UI-LAYOUT GROUPS (calibration reading plan) ===")
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

sigs, det = {}, {}
for f in pools["test_imgs"]:
    gray = np.asarray(Image.open(f).convert("L"))
    sc = detect_scales(gray)
    det[f.name] = sc
    key = tuple((s, x, y0 // 10 * 10, y1 // 10 * 10) for s, x, c, y0, y1 in sc)
    sigs.setdefault(key, []).append(f.name)
print("  {} layout groups for {} images".format(len(sigs), len(pools["test_imgs"])))
for gi, k in enumerate(sigs):
    print("  group {:02d}: n={:3d}  {}   e.g. {}".format(
        gi, len(sigs[k]), k, sigs[k][0]))

for gi, k in enumerate(sigs):
    fname = sigs[k][0]
    sc = det[fname]
    if not sc: continue
    im = Image.open(next(x for x in pools["test_imgs"] if x.name == fname)).convert("RGB")
    W, H = im.size
    panels = []
    for side, x0, cnt, y0, y1 in sc:
        box = (max(0, x0 - 120), max(0, y0 - 70), min(W, x0 + 120), min(H, y1 + 70))
        crop = im.crop(box)
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.NEAREST)
        panels.append(np.asarray(crop))
    hmax = max(p.shape[0] for p in panels)
    fixed = []
    for p in panels:
        if p.shape[0] < hmax:
            pad = np.full((hmax - p.shape[0], p.shape[1], 3), 255, np.uint8)
            p = np.concatenate([p, pad], axis=0)
        fixed.append(p)
    Image.fromarray(np.concatenate(fixed, axis=1)).save(
        OUT / ("calibG_{:02d}_{}.png".format(gi, Path(fname).stem)))
print("  saved calibG_*.png -- READ EVERY ONE OF THESE")