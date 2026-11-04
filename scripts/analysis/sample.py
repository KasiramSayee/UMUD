import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from collections import Counter
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
tdf = pd.read_csv(ARTIFACTS_DIR / "twins.csv")

def load_mask(path, size, thresh):
    m = Image.open(path).convert("L")
    if m.size != size: m = m.resize(size, Image.BILINEAR)
    return np.asarray(m) > thresh

def components(mask, min_area):
    lbl, n = ndimage.label(mask)
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
                        w=int(xs.max()-xs.min()+1), hgt=int(ys.max()-ys.min()+1),
                        fit=fit))
    return out

print("=== X1. FASC MASK CENSUS (15 twins: what is in these masks?) ===")
for i, r in list(tdf.iterrows())[:15]:
    im_p = Path(r.fasc_img)
    size = Image.open(im_p).size
    comps = sorted(components(load_mask(Path(r.fasc_mask), size, 32), 50),
                   key=lambda c: -c["area"])
    desc = " | ".join("a={} ang={:.0f} {}x{}".format(c["area"], c["ang"], c["w"], c["hgt"])
                      for c in comps[:6])
    print("  {}: {} comps | {}".format(im_p.name, len(comps), desc))

print("\n  angle histogram, ALL twin fasc components (5-deg bins):")
hist = Counter()
for _, r in tdf.iterrows():
    size = Image.open(r.fasc_img).size
    for c in components(load_mask(Path(r.fasc_mask), size, 32), 50):
        hist[int(c["ang"] // 5) * 5] += 1
for k in sorted(hist):
    print("    {}-{} deg: {}".format(k, k + 5, hist[k]))

print("\n=== X2. APO MASK CENSUS (8 twins) ===")
for i, r in list(tdf.iterrows())[:8]:
    im_p = Path(r.fasc_img)
    size = Image.open(im_p).size
    comps = sorted(components(load_mask(Path(r.apo_mask), size, 127), 500),
                   key=lambda c: -c["area"])
    desc = " | ".join("a={} ang={:.0f} {}x{}".format(c["area"], c["ang"], c["w"], c["hgt"])
                      for c in comps[:4])
    print("  {}: {} comps | {}".format(im_p.name, len(comps), desc))

print("\n=== X3. GEOMETRY v0.1 (bilinear resize, relaxed filters) ===")
def angle_between(v, d):
    c = min(1.0, abs(float(np.dot(v, d))))
    return np.degrees(np.arccos(c))

results, fail_apo, fail_fasc, qa = [], 0, 0, 0
reject_angs = Counter()
for _, r in tdf.iterrows():
    im_p = Path(r.fasc_img)
    size = Image.open(im_p).size
    gray = np.asarray(Image.open(im_p).convert("L"))
    ac = sorted(components(load_mask(Path(r.apo_mask), size, 127), 500),
                key=lambda c: -c["area"])
    if len(ac) < 2 or abs(ac[0]["cy"] - ac[1]["cy"]) < 60:
        fail_apo += 1; continue
    sup, deep = sorted(ac[:2], key=lambda c: c["cy"])
    bd, ad = deep["fit"]; bs, as_ = sup["fit"]
    d = np.array([1.0, bd]);  d /= np.linalg.norm(d)
    ds = np.array([1.0, bs]); ds /= np.linalg.norm(ds)
    fcomps = sorted(components(load_mask(Path(r.fasc_mask), size, 32), 80),
                    key=lambda c: -c["area"])
    fc, rejected = [], []
    for c in fcomps:
        a = angle_between(np.array([c["vx"], c["vy"]]), d)
        if 3 <= a <= 60: fc.append((c, a))
        else: rejected.append((c, a))
    for c, a in rejected[:5]: reject_angs[int(a // 5) * 5] += 1
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
        for c, a in rejected[:6]:
            if abs(c["vx"]) < 1e-6: continue
            sl = c["vy"] / c["vx"]
            dr.line([(0, c["cy"] - sl * c["cx"]), (W, c["cy"] + sl * (W - c["cx"]))],
                    fill=(255, 165, 0), width=2)
        img.save(OUT / ("geomQA_" + im_p.stem + ".png"))
        qa += 1

res = pd.DataFrame(results)
res.to_csv(ARTIFACTS_DIR / "geometry_v1_twins.csv", index=False)
print("  solved: {} | apo failures: {} | no-fascicle failures: {}".format(
    len(res), fail_apo, fail_fasc))
if len(res):
    print(res[["pa", "fl_px", "mt_px", "n_fasc"]].describe().to_string())
    ok = res.dropna(subset=["fl_px"])
    if len(ok):
        ratio = ok.fl_px * np.sin(np.radians(ok.pa)) / ok.mt_px
        print("  consistency FL*sin(PA)/MT: median={:.2f} p25={:.2f} p75={:.2f}".format(
            ratio.median(), ratio.quantile(0.25), ratio.quantile(0.75)))
print("  rejected fascicle angle bins:", dict(sorted(reject_angs.items())))
print("  saved {} QA overlays (red=accepted, orange=rejected)".format(qa))