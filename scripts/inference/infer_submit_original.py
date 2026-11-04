# infer_submit.py -- predict -> geometry -> calibrate -> bursts -> submission_v2.csv
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import re
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
OUT = OUTPUTS_DIR; OUT.mkdir(exist_ok=True)
SIZE = 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".tif", ".tiff", ".png"}
C_PA, C_FL, C_MT = 18.0, 80.0, 23.0          # LB-validated constants fallback
FL_SCALE = 0.925     # global FL scale (A/B-testable)
PRIOR = {(1200, 800): 0.0656, (1088, 644): 0.0830}
PA_A, PA_B = 7.23, 0.59   # LB-validated (v9b), NOT the twin-refit 

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
                        elong=float(np.sqrt(evals[-1] / max(evals[0], 1e-9)))))
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

cal = pd.read_csv(ARTIFACTS_DIR / "calibration_v1.csv")
cal_by_name = {r["name"]: r for _, r in cal.iterrows()}

def parse_depth_mm(txt):
    if not isinstance(txt, str): return None
    m = re.search(r"(\d{1,2}(?:[.,]\d)?)\s*(mm|cm)", txt)
    if not m: return None
    v = float(m.group(1).replace(",", "."))
    return v * 10 if m.group(2).lower() == "cm" else v

def get_mm_px(name, gray, size):
    row = cal_by_name.get(name)
    dep = parse_depth_mm(row.depth_txt) if row is not None else None
    if dep:
        t = ruler_table(gray, 6)
        if t:
            lns = [q[3] for q in t]
            big = [q[2] for q in t if q[3] >= max(6, int(0.6 * max(lns)))]
            if len(big) >= 2:
                sp = float(np.median(np.diff(big)))
                if 120 <= sp <= 420:
                    return 20.0 / sp, "marker"
        return dep / 604.0, "depth/604"
    return PRIOR.get(size, 0.070), "prior"

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

def geometry(fasc_m, apo_m):
    ac = sorted(components(apo_m, 500), key=lambda c: -c["area"])
    if len(ac) < 2 or abs(ac[0]["cy"] - ac[1]["cy"]) < 60:
        return None
    sup, deep = sorted(ac[:2], key=lambda c: c["cy"])
    bd, ad = deep["fit"]; bs, as_ = sup["fit"]
    d = np.array([1.0, bd]);  d /= np.linalg.norm(d)
    ds = np.array([1.0, bs]); ds /= np.linalg.norm(ds)
    fc = []
    for c in components(fasc_m, 80):
        a = angle_between(np.array([c["vx"], c["vy"]]), d)
        if 6 <= a <= 60 and c["elong"] >= 4 and sup["cy"] + 20 < c["cy"] < deep["cy"] - 20:
            fc.append((c, a))
    if not fc:
        return None
    angs = np.array([a for _, a in fc])
    wts = np.array([c["area"] * np.sin(np.radians(a)) for c, a in fc], dtype=float)
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
    pas, fls, oks = [], [], []
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
        r = fl * np.sin(np.radians(a)) / max(mt_px, 1e-6)
        if 0.4 <= r <= 2.5:
            pas.append(a); fls.append(fl); oks.append(r)
    if not pas: return None
    return dict(pa=float(np.mean(pas)), fl_px=float(np.mean(fls)),
                mt_px=mt_px, n=len(pas), ratio=float(np.median(oks)))

print("loading models...")
model_a = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                   in_channels=1, classes=1).to(DEV)
model_a.load_state_dict(torch.load(MODELS_DIR / "aponet_v1.pt", map_location=DEV))
model_a.eval()

models_f = []
for k in [0, 1]:
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=1, classes=1).to(DEV)
    m.load_state_dict(torch.load(MODELS_DIR / ("fascnet_v3_fold{}.pt".format(k)),
                                 map_location=DEV))
    m.eval()
    models_f.append(m)

model_f1 = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=1, classes=1).to(DEV)
model_f1.load_state_dict(torch.load(MODELS_DIR / "fascnet_v1.pt", map_location=DEV))
model_f1.eval()

recs = []
for k, f in enumerate(pools["test_imgs"]):
    img = Image.open(f).convert("L")
    gray = np.asarray(img)
    mm_px, meth = get_mm_px(f.name, gray, img.size)
    if meth == "prior":
        mm_px = mm_px * 0.95
    g = geometry(predict(models_f, img, size=640), predict(model_a, img))
    src = "ens"
    if g is None:
        g = geometry(predict(model_f1, img, tta=True), predict(model_a, img))
        src = "v1"
    if g is None:
        g = geometry(predict(model_f1, img, tta=False), predict(model_a, img, tta=False))
        src = "v1s"
    if g is None:
        recs.append(dict(name=f.name, pa=C_PA, fl=C_FL * FL_SCALE, mt=C_MT,
                         mm_px=round(mm_px, 5), method=meth, geom="FALLBACK"))
    else:
        recs.append(dict(name=f.name, pa=PA_A + PA_B * g["pa"], fl=g["fl_px"] * mm_px * FL_SCALE,
                         mt=g["mt_px"] * mm_px, mm_px=round(mm_px, 5),
                         method=meth,
                         geom="ok n={} r={:.2f} {}".format(g["n"], g["ratio"], src)))
    if (k + 1) % 50 == 0:
        print("  ...{}/{}".format(k + 1, len(pools["test_imgs"])))

assert len(recs) == len(pools["test_imgs"]), "record count mismatch!"
print("records:", len(recs))

df = pd.DataFrame(recs)

# burst averaging
z = np.load(ARTIFACTS_DIR / "thumbs32.npz")
F, names = z["F"].astype(np.float32), z["names"].astype(str)
C = (F @ F.T) / F.shape[1]
np.fill_diagonal(C, -2.0)
tidx = [i for i, n in enumerate(names) if n.startswith("test|")]
r_adj = [C[tidx[i], tidx[i + 1]] for i in range(len(tidx) - 1)]
runs, cur = [], [0]
for i in range(len(r_adj)):
    if r_adj[i] > 0.99: cur.append(i + 1)
    else: runs.append(cur); cur = [i + 1]
runs.append(cur)
name_run = {}
for ri, run in enumerate(runs):
    if len(run) >= 2:
        for i in run: name_run[names[tidx[i]].split("|")[1]] = ri
for ri, grp in df.groupby(df["name"].map(name_run)):
    if len(grp) >= 2:
        df.loc[grp.index, ["pa", "fl", "mt"]] = grp[["pa", "fl", "mt"]].mean().values

df["pa"] = df.pa.clip(5, 45)
df["fl"] = df.fl.clip(30, 200)
df["mt"] = df.mt.clip(10, 50)
df["family"] = df["name"].map({r["name"]: r.family for _, r in cal.iterrows()})

sub = df[["name", "pa", "fl", "mt"]].rename(columns={
    "name": "image_id", "pa": "pa_deg", "fl": "fl_mm", "mt": "mt_mm"})
sub.to_csv(SUBMISSIONS_DIR / "submission_v16.csv", index=False)
chk = pd.read_csv(SUBMISSIONS_DIR / "submission_v16.csv")
assert len(chk) == 309 and chk["image_id"].nunique() == 309, "submission row check failed"
print("submission file OK:", len(chk), "rows,", chk["pa_deg"].isna().sum(), "NaNs")
print(sub.head())

print("\n=== RESULTS ===")
print("  geometry solved: {} / {}".format((df.geom != "FALLBACK").sum(), len(df)))
print("  fallback images:", df[df.geom == "FALLBACK"]["name"].tolist()[:12])
print("\n  by calibration method:")
print(df.groupby("method")[["pa", "fl", "mt"]].agg(["count", "mean"]).round(1).to_string())
print("\n  cross-family table (PA scale-free; MT/FL should match across families):")
print(df.groupby("family")[["pa", "fl", "mt"]].mean().round(1).to_string())
print("\n  PNG calibration validation (span vs 2cm-marker spacing):")
for name in ["IMG_00252.png", "IMG_00258.png", "IMG_00275.png"]:
    row = cal_by_name.get(name)
    if row is None: continue
    dep = parse_depth_mm(row.depth_txt)
    gray = np.asarray(Image.open(next(x for x in pools["test_imgs"]
                                      if x.name == name)).convert("L"))
    t = ruler_table(gray, 6)
    if not t or dep is None: continue
    span = max(q[2] for q in t) - min(q[2] for q in t)
    lns = [q[3] for q in t]
    big = [q[2] for q in t if q[3] >= max(6, int(0.6 * max(lns)))]
    sp = float(np.median(np.diff(big))) if len(big) > 1 else -1
    print("    {}: depth={}mm | span={} -> {:.4f} | markers: {} -> {:.4f}".format(
        name, dep, round(span, 1), dep / span, round(sp, 1),
        20.0 / sp if sp > 0 else -1))
print("\n  saved submission_v16.csv ({} rows)".format(len(sub)))
print(df[["name", "pa", "fl", "mt", "method", "geom"]].head(8).to_string())