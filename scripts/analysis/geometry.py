import numpy as np, pandas as pd
from pathlib import Path
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

df = pd.read_csv(ARTIFACTS_DIR / "content_hashes.csv")
apo_msk = {f.stem: f for f in pools["apo_masks"]}
fasc_msk = {f.stem: f for f in pools["fasc_masks"]}
img_size = {f.stem: Image.open(f).size for f in pools["apo_imgs"]}

def load_mask(path, size):
    m = Image.open(path).convert("L")
    if m.size != size: m = m.resize(size, Image.NEAREST)
    return np.asarray(m) > 127

def components(mask, min_area=300):
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
                        fit=fit))
    return out

print("=== U. 19 PAIRS: STRUCTURE ORIENTATION (0=horizontal, 90=vertical) ===")
fi = df[df.pool == "apo_imgs"]
am = df[df.pool == "apo_masks"].set_index("stem")["hash"]
for h, g in fi[fi.duplicated("hash", keep=False)].groupby("hash"):
    a, b = g.stem.iloc[0], g.stem.iloc[1]
    if am.get(a) == am.get(b): continue
    size = img_size[a]
    for tag, stem in [("A", a), ("B", b)]:
        comps = sorted(components(load_mask(apo_msk[stem], size)),
                       key=lambda c: -c["area"])[:4]
        desc = ", ".join("ang={:.0f} area={}".format(c["ang"], c["area"]) for c in comps)
        print("  {} [{}]: {} comps | {}".format(stem, tag, len(comps), desc))
am_h = set(df[df.pool == "apo_masks"].hash)
fm_h = set(df[df.pool == "fasc_masks"].hash)
print("  byte-identical masks across apo/fasc folders:", len(am_h & fm_h))

print("\n=== V. TWIN CO-LABELED DATASET ===")
z = np.load(ARTIFACTS_DIR / "thumbs32.npz")
F, names = z["F"].astype(np.float32), z["names"].astype(str)
C = (F @ F.T) / F.shape[1]
np.fill_diagonal(C, -2.0)
path_by_name = {}
for key, tag in [("fasc_imgs","fasc"), ("apo_imgs","apo"), ("test_imgs","test")]:
    for f in pools[key]: path_by_name[tag + "|" + f.name] = f
ii, jj = np.where(np.triu(np.ones_like(C, dtype=bool), 1) & (C > 0.9995))
pairs = []
for a, b in zip(ii, jj):
    na, nb = names[a], names[b]
    pa, pb = na.split("|")[0], nb.split("|")[0]
    if pa == pb or "test" in (pa, pb): continue
    fa, fb = path_by_name[na], path_by_name[nb]
    if Image.open(fa).size != Image.open(fb).size: continue
    if pa == "apo": fa, fb = fb, fa
    pairs.append((fa, fb))
tdf = pd.DataFrame([dict(fasc_img=str(fa), fasc_mask=str(fasc_msk[fa.stem]),
                          apo_img=str(fb), apo_mask=str(apo_msk[fb.stem]))
                    for fa, fb in pairs]).drop_duplicates(subset=["fasc_img"])
tdf.to_csv(ARTIFACTS_DIR / "twins.csv", index=False)
print("  co-labeled twins saved:", len(tdf))
diffs = []
for _, r in tdf.head(10).iterrows():
    a = np.asarray(Image.open(r.fasc_img).convert("L"), np.float32)
    b = np.asarray(Image.open(r.apo_img).convert("L"), np.float32)
    if a.shape == b.shape:
        diffs.append(float(np.abs(a - b).mean()))
print("  twin pixel diff (10 samples): mean={:.2f} max={:.2f} (0 = identical render)".format(
    np.mean(diffs), np.max(diffs)))

print("\n=== W. GEOMETRY v0 ON TWINS (PA in degrees, FL/MT in pixels) ===")
def angle_between(v, d):
    c = min(1.0, abs(float(np.dot(v, d))))
    return np.degrees(np.arccos(c))

results, fail_apo, fail_fasc, qa = [], 0, 0, 0
for _, r in tdf.iterrows():
    im_p = Path(r.fasc_img)
    size = Image.open(im_p).size
    gray = np.asarray(Image.open(im_p).convert("L"))
    ac = sorted(components(load_mask(Path(r.apo_mask), size), min_area=500),
                key=lambda c: -c["area"])
    if len(ac) < 2 or abs(ac[0]["cy"] - ac[1]["cy"]) < 60:
        fail_apo += 1; continue
    sup, deep = sorted(ac[:2], key=lambda c: c["cy"])
    bd, ad = deep["fit"]; bs, as_ = sup["fit"]
    d = np.array([1.0, bd]);  d /= np.linalg.norm(d)
    ds = np.array([1.0, bs]); ds /= np.linalg.norm(ds)
    fc = [c for c in components(load_mask(Path(r.fasc_mask), size), min_area=300)
          if 4 <= angle_between(np.array([c["vx"], c["vy"]]), d) <= 50]
    if not fc:
        fail_fasc += 1; continue
    fc = sorted(fc, key=lambda c: -c["area"])[:3]
    pas, fls = [], []
    for c in fc:
        v = np.array([c["vx"], c["vy"]])
        pas.append(angle_between(v, d))
        cpt = np.array([c["cx"], c["cy"]])
        pts = []
        for P, dd in [(np.array([0.0, ad]), d), (np.array([0.0, as_]), ds)]:
            den = v[0]*dd[1] - v[1]*dd[0]
            if abs(den) < 1e-6: pts.append(None); continue
            t = ((P[0]-cpt[0])*dd[1] - (P[1]-cpt[1])*dd[0]) / den
            pts.append(cpt + t*v)
        if pts[0] is not None and pts[1] is not None:
            fls.append(float(np.linalg.norm(pts[1] - pts[0])))
    W = size[0]
    gaps = [abs((as_ + bs*x) - (ad + bd*x)) for x in (0.25*W, 0.5*W, 0.75*W)]
    mt_px = float(np.mean(gaps) * np.cos(np.arctan((bs + bd)/2)))
    results.append(dict(fasc=im_p.name, pa=float(np.mean(pas)),
                        fl_px=float(np.mean(fls)) if fls else np.nan,
                        mt_px=mt_px, n_fasc=len(pas)))
    if len(results) % 10 == 1 and qa < 12:
        img = Image.fromarray(np.stack([gray]*3, -1).astype(np.uint8))
        dr = ImageDraw.Draw(img)
        dr.line([(0, ad), (W, ad + bd*W)], fill=(0, 120, 255), width=3)   # deep
        dr.line([(0, as_), (W, as_ + bs*W)], fill=(0, 255, 255), width=3) # superficial
        for c in fc:
            if abs(c["vx"]) < 1e-6: continue
            sl = c["vy"] / c["vx"]
            dr.line([(0, c["cy"] - sl*c["cx"]), (W, c["cy"] + sl*(W - c["cx"]))],
                    fill=(255, 60, 60), width=3)
        img.save(OUT / ("geomQA_" + im_p.stem + ".png"))
        qa += 1

res = pd.DataFrame(results)
res.to_csv(ARTIFACTS_DIR / "geometry_v0_twins.csv", index=False)
print("  solved: {} | apo failures: {} | no-fascicle failures: {}".format(
    len(res), fail_apo, fail_fasc))
if len(res):
    print(res[["pa", "fl_px", "mt_px", "n_fasc"]].describe().to_string())
    print(res.head(10).to_string())
    ok = res.dropna(subset=["fl_px"])
    ratio = ok.fl_px * np.sin(np.radians(ok.pa)) / ok.mt_px
    print("  consistency FL*sin(PA)/MT: median={:.2f} p25={:.2f} p75={:.2f} (expect ~1)".format(
        ratio.median(), ratio.quantile(0.25), ratio.quantile(0.75)))
print("  saved {} QA overlays (geomQA_*.png)".format(qa))