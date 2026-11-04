import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from PIL import Image

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
for d in sorted([d for d in ROOT.rglob("*") if d.is_dir()],
                key=lambda x: len(x.parts)):
    c = classify(d)
    if c and c not in pools:
        pools[c] = sorted(f for f in d.rglob("*")
                          if f.is_file() and f.suffix.lower() in IMG_EXTS)

print("=== Q. COMMA SUBMISSION (submit this one) ===")
s = pd.DataFrame({"image_id": [f.name for f in pools["test_imgs"]]})
s["pa_deg"], s["fl_mm"], s["mt_mm"] = 18.0, 80.0, 23.0
s.to_csv(SUBMISSIONS_DIR / "submission_constants_comma.csv", index=False)
print("  saved submission_constants_comma.csv (comma-separated, 309 rows)")

df = pd.read_csv(ARTIFACTS_DIR / "content_hashes.csv")
fi = df[df.pool == "apo_imgs"]
am = df[df.pool == "apo_masks"].set_index("stem")["hash"]
apo_img = {f.stem: f for f in pools["apo_imgs"]}
apo_msk = {f.stem: f for f in pools["apo_masks"]}
img_size = {f.stem: Image.open(f).size for f in pools["apo_imgs"]}

def bands(mask_path, size):
    m = Image.open(mask_path).convert("L")
    if m.size != size: m = m.resize(size, Image.NEAREST)
    prof = (np.asarray(m) > 127).mean(axis=1)
    out, start = [], None
    for i, v in enumerate(prof > 0.02):
        if v and start is None: start = i
        elif not v and start is not None: out.append([start, i]); start = None
    if start is not None: out.append([start, len(prof)])
    merged = []
    for b in out:
        if merged and b[0] - merged[-1][1] < 20: merged[-1][1] = b[1]
        else: merged.append(b)
    return merged

print("\n=== R. THE 19 PAIRS: WHERE ARE THE BANDS? ===")
for h, g in fi[fi.duplicated("hash", keep=False)].groupby("hash"):
    a, b = g.stem.iloc[0], g.stem.iloc[1]
    if am.get(a) == am.get(b): continue
    size = img_size[a]
    fa = ", ".join(f"rows {x}-{y}" for x, y in bands(apo_msk[a], size))
    fb = ", ".join(f"rows {x}-{y}" for x, y in bands(apo_msk[b], size))
    print(f"  {a}: [{fa}]  VS  {b}: [{fb}]   (H={size[1]})")

print("\n=== R2. BAND COUNTS ACROSS ALL APO MASKS (300 sampled) ===")
counts = Counter()
for f in pools["apo_masks"][::max(1, len(pools["apo_masks"])//300)][:300]:
    counts[len(bands(f, img_size[f.stem]))] += 1
print("  bands per mask:", dict(sorted(counts.items())))

print("\n=== S. CROSS-POOL TWINS (r>0.9995) ===")
z = np.load(ARTIFACTS_DIR / "thumbs32.npz")
F, names = z["F"].astype(np.float32), z["names"].astype(str)
C = (F @ F.T) / F.shape[1]
np.fill_diagonal(C, -2.0)
path_by_name = {}
for key, tag in [("fasc_imgs","fasc"), ("apo_imgs","apo"), ("test_imgs","test")]:
    for f in pools[key]: path_by_name[tag + "|" + f.name] = f
ii, jj = np.where(np.triu(np.ones_like(C, dtype=bool), 1) & (C > 0.9995))
twins = [(C[a,b], names[a], names[b]) for a, b in zip(ii, jj)
         if names[a].split("|")[0] != names[b].split("|")[0]]
n_test = sum(1 for _, a, b in twins if a.startswith("test") or b.startswith("test"))
print(f"  cross-pool twins: {len(twins)} | involving TEST images: {n_test}")
same = 0
for r, a, b in sorted(twins, reverse=True):
    sa, sb = Image.open(path_by_name[a]).size, Image.open(path_by_name[b]).size
    if sa == sb: same += 1
    print(f"  r={r:.4f}  {a} <-> {b}  [{'same size' if sa==sb else str(sa)+' vs '+str(sb)}]")
print(f"  same-size twins: {same}")

fasc_msk = {f.stem: f for f in pools["fasc_masks"]}
n_saved = 0
for r, a, b in sorted(twins, reverse=True):
    if n_saved >= 6: break
    if not (a.startswith("fasc") and b.startswith("apo")): continue
    if Image.open(path_by_name[a]).size != Image.open(path_by_name[b]).size: continue
    im = np.asarray(Image.open(path_by_name[a]).convert("L"))
    A = np.asarray(Image.open(fasc_msk[Path(a.split("|")[1]).stem]).convert("L")
                   .resize((im.shape[1], im.shape[0]), Image.NEAREST)) > 127
    B = np.asarray(Image.open(apo_msk[Path(b.split("|")[1]).stem]).convert("L")
                   .resize((im.shape[1], im.shape[0]), Image.NEAREST)) > 127
    rgb = np.stack([im]*3, -1).astype(np.uint8)
    rgb[A, 0] = 255; rgb[A, 1] = 60   # fasc mask red
    rgb[B, 2] = 255; rgb[B, 1] = 60   # apo mask blue
    Image.fromarray(rgb).save(OUT / f"twin_{Path(a.split('|')[1]).stem}_{Path(b.split('|')[1]).stem}.png")
    n_saved += 1
print(f"  saved {n_saved} twin overlays to checks/ (fasc mask red, apo mask blue)")

print("\n=== T. TEST BURST RUNS ===")
tidx = [i for i, n in enumerate(names) if n.startswith("test|")]
r_adj = [C[tidx[i], tidx[i+1]] for i in range(len(tidx)-1)]
runs, cur = [], [0]
for i in range(len(r_adj)):
    if r_adj[i] > 0.99: cur.append(i+1)
    else: runs.append(cur); cur = [i+1]
runs.append(cur)
bursts = [r for r in runs if len(r) >= 2]
print(f"  burst runs: {len(bursts)} | images covered: {sum(len(r) for r in bursts)}")
print("  run lengths:", sorted([len(r) for r in bursts], reverse=True))