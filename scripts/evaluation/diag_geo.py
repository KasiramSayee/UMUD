# diag_geo.py -- stage-by-stage diagnosis of the strict fascicle selection
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
                        elong=float(np.sqrt(evals[-1] / max(evals[0], 1e-9))),
                        vh=int(ys.max() - ys.min() + 1)))
    return out

def angle_between(v, d):
    c = min(1.0, abs(float(np.dot(v, d))))
    return np.degrees(np.arccos(c))

print("loading models...")
model_a = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                   in_channels=1, classes=1).to(DEV)
model_a.load_state_dict(torch.load(MODELS_DIR / "aponet_v1.pt", map_location=DEV))
model_a.eval()
models_f = []
for k in range(3):
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=1, classes=1).to(DEV)
    m.load_state_dict(torch.load(MODELS_DIR / ("fascnet_v2_fold{}.pt".format(k)),
                                 map_location=DEV))
    m.eval()
    models_f.append(m)

def predict(models, img, tta=True):
    if not isinstance(models, list): models = [models]
    W0, H0 = img.size
    def forward(im):
        x = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), np.float32)
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
NAMES = ["image_2219.tif", "image_2110.tif", "image_2109.tif",
         "image_2112.tif", "image_2111.tif", "image_2196.tif"]

for nm in NAMES:
    r = next(t for _, t in twins.iterrows() if Path(t.fasc_img).name == nm)
    img = Image.open(r.fasc_img).convert("L")
    pf = predict(models_f, img)
    pa_ = predict(model_a, img)
    ac = sorted(components(pa_, 500), key=lambda c: -c["area"])
    if len(ac) < 2 or abs(ac[0]["cy"] - ac[1]["cy"]) < 60:
        print("\n{}: APO STAGE FAIL".format(nm)); continue
    sup, deep = sorted(ac[:2], key=lambda c: c["cy"])
    MH = deep["cy"] - sup["cy"]
    bd, ad = deep["fit"]; bs, as_ = sup["fit"]
    d = np.array([1.0, bd]);  d /= np.linalg.norm(d)
    ds = np.array([1.0, bs]); ds /= np.linalg.norm(ds)
    comps = sorted(components(pf, 80), key=lambda c: -c["area"])
    print("\n=== {} | muscle_height={:.0f}px | {} components ===".format(nm, MH, len(comps)))
    for c in comps[:10]:
        a = angle_between(np.array([c["vx"], c["vy"]]), d)
        ok_band = sup["cy"] + 20 < c["cy"] < deep["cy"] - 20
        print("  area={:5d} vh={:4d} ang={:5.1f} elong={:5.1f} band={} | "
              "ang/el/vh/band = {}/{}/{}/{}".format(
                  c["area"], c["vh"], a, c["elong"], ok_band,
                  int(5 <= a <= 60), int(c["elong"] >= 4),
                  int(c["vh"] >= 0.35 * MH), int(ok_band)))
    for thr in [0.35, 0.25, 0.15, 0.0]:
        n = sum(1 for c in comps
                if 5 <= angle_between(np.array([c["vx"], c["vy"]]), d) <= 60
                and c["elong"] >= 4 and c["vh"] >= thr * MH and
                sup["cy"] + 20 < c["cy"] < deep["cy"] - 20)
        print("  survivors at vh>={:.2f}*MH: {}".format(thr, n))
    fc = [c for c in comps
          if 5 <= angle_between(np.array([c["vx"], c["vy"]]), d) <= 60
          and c["elong"] >= 4 and c["vh"] >= 0.25 * MH and
          sup["cy"] + 20 < c["cy"] < deep["cy"] - 20]
    if fc:
        angs = np.array([angle_between(np.array([c["vx"], c["vy"]]), d) for c in fc])
        areas = np.array([c["area"] for c in fc], float)
        mode, best = float(angs[0]), -1.0
        for a0 in angs:
            s = float(areas[np.abs(angs - a0) <= 3.5].sum())
            if s > best: best, mode = s, float(a0)
        sel = [c for c, a in zip(fc, angs) if abs(a - mode) <= 3.5]
        sel = sorted(sel, key=lambda c: -c["area"])[:3]
        W = pf.shape[1]
        gaps = [abs((as_ + bs * x) - (ad + bd * x)) for x in (0.25 * W, 0.5 * W, 0.75 * W)]
        mt = float(np.mean(gaps) * np.cos(np.arctan((bs + bd) / 2)))
        print("  mode={:.1f} | selected angles: {}".format(mode,
              [round(angle_between(np.array([c["vx"], c["vy"]]), d), 1) for c in sel]))
        for c in sel:
            a = angle_between(np.array([c["vx"], c["vy"]]), d)
            v = np.array([c["vx"], c["vy"]]); cpt = np.array([c["cx"], c["cy"]])
            pts = []
            for P, dd in [(np.array([0.0, ad]), d), (np.array([0.0, as_]), ds)]:
                den = v[0] * dd[1] - v[1] * dd[0]
                if abs(den) < 1e-6: pts.append(None); continue
                t = ((P[0] - cpt[0]) * dd[1] - (P[1] - cpt[1]) * dd[0]) / den
                pts.append(cpt + t * v)
            if pts[0] is None or pts[1] is None:
                print("    fascicle ang={:.1f}: near-parallel, skipped".format(a)); continue
            fl = float(np.linalg.norm(pts[1] - pts[0]))
            print("    fascicle ang={:.1f} FL_px={:.0f} ratio={:.2f}".format(
                a, fl, fl * np.sin(np.radians(a)) / max(mt, 1e-6)))