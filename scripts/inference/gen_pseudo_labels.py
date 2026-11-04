# gen_pseudo_labels.py -- pipeline on fasc training images -> pseudo-label CSV
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
OUTPUTS_DIR = PROJECT / "outputs"

ROOT = DATA_DIR
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
                        vx=float(v[0]), vy=float(v[1]), fit=fit,
                        elong=float(np.sqrt(evals[-1] / max(evals[0], 1e-9)))))
    return out

def angle_between(v, d):
    c = min(1.0, abs(float(np.dot(v, d))))
    return np.degrees(np.arccos(c))

def geometry(fasc_m, apo_m):
    ac = sorted(components(apo_m, 500), key=lambda c: -c["area"])
    if len(ac) < 2 or abs(ac[0]["cy"] - ac[1]["cy"]) < 60: return None
    sup, deep = sorted(ac[:2], key=lambda c: c["cy"])
    bd, ad = deep["fit"]; bs, as_ = sup["fit"]
    d = np.array([1.0, bd]);  d /= np.linalg.norm(d)
    ds = np.array([1.0, bs]); ds /= np.linalg.norm(ds)
    fc = []
    for c in components(fasc_m, 80):
        a = angle_between(np.array([c["vx"], c["vy"]]), d)
        if 6 <= a <= 60 and c["elong"] >= 4 and sup["cy"] + 20 < c["cy"] < deep["cy"] - 20:
            fc.append((c, a))
    if not fc: return None
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
for k in [0, 1]:
    mm = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                  in_channels=1, classes=1).to(DEV)
    mm.load_state_dict(torch.load(MODELS_DIR / ("fascnet_v3_fold{}.pt".format(k)),
                                  map_location=DEV))
    mm.eval()
    models_f.append(mm)

# collect training images: nonempty, unique, not twins
hashes = pd.read_csv(ARTIFACTS_DIR / "content_hashes.csv")
twins = pd.read_csv(ARTIFACTS_DIR / "twins.csv")
twin_fasc = {Path(p).stem for p in twins.fasc_img}
inv = pd.read_csv(ARTIFACTS_DIR / "fasc_mask_inventory.csv")
hmap = hashes[hashes.pool == "fasc_imgs"].set_index("stem")["hash"]
seen, train = set(), []
for stem in inv[inv.frac > 0.001].stem:
    if stem in twin_fasc: continue
    h = hmap.get(stem)
    if h is None or h in seen: continue
    seen.add(h)
    train.append(stem)
print("generating pseudo-labels for {} images".format(len(train)))

fasc_by_stem = {f.stem: f for f in pools["fasc_imgs"]}
rows = []
for k, stem in enumerate(train):
    img_path = fasc_by_stem[stem]
    img = Image.open(img_path).convert("L")
    g = geometry(predict(models_f, img, size=640), predict(model_a, img))
    src = "ens"
    if g is None:
        g = geometry(predict(model_f1, img, tta=True), predict(model_a, img)); src = "v1"
    if g is None:
        g = geometry(predict(model_f1, img, tta=False), predict(model_a, img, tta=False)); src = "v1s"
    rows.append(dict(stem=stem, pa=g["pa"] if g else None,
                     fl_px=g["fl_px"] if g else None,
                     mt_px=g["mt_px"] if g else None, src=src if g else "FAIL"))
    if (k + 1) % 200 == 0:
        print("  ...{}/{}".format(k + 1, len(train)))

df = pd.DataFrame(rows)
df.to_csv(ARTIFACTS_DIR / "pseudo_labels.csv", index=False)
ok = df[df.pa.notna()]
print("DONE: {}/{} images labeled ({} failed)".format(len(ok), len(df), df.pa.isna().sum()))
print(ok[["pa", "fl_px", "mt_px"]].describe().round(1).to_string())