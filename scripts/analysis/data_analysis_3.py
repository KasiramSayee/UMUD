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

print("=== K. HASH-VERIFIED INTERSECTIONS ===")
df = pd.read_csv(ARTIFACTS_DIR / "content_hashes.csv")
fh = set(df[df.pool == "fasc_imgs"].hash)
ah = set(df[df.pool == "apo_imgs"].hash)
th = set(df[df.pool == "test_imgs"].hash)
print(f"  fasc {len(fh)} uniq | apo {len(ah)} uniq | fasc&apo: {len(fh & ah)} | "
      f"fasc&test: {len(fh & th)} | apo&test: {len(ah & th)}")

print("\n=== L. 19 DIFFERING-MASK APO PAIRS (rater noise floor) ===")
fi = df[df.pool == "apo_imgs"]
am = df[df.pool == "apo_masks"].set_index("stem")["hash"]
apo_img = {f.stem: f for f in pools["apo_imgs"]}
apo_msk = {f.stem: f for f in pools["apo_masks"]}
ious = []
for h, g in fi[fi.duplicated("hash", keep=False)].groupby("hash"):
    a, b = g.stem.iloc[0], g.stem.iloc[1]
    if am.get(a) == am.get(b): continue
    im = Image.open(apo_img[a]).convert("L")
    A = np.asarray(Image.open(apo_msk[a]).convert("L").resize(im.size, Image.NEAREST)) > 127
    B = np.asarray(Image.open(apo_msk[b]).convert("L").resize(im.size, Image.NEAREST)) > 127
    iou = (A & B).sum() / max(1, (A | B).sum())
    ious.append(iou)
    rgb = np.stack([np.asarray(im)]*3, -1).astype(np.uint8)
    rgb[A, 0] = 255; rgb[A, 1] = 60     # rater 1 red
    rgb[B, 2] = 255; rgb[B, 1] = 60     # rater 2 blue
    Image.fromarray(rgb).save(OUT / f"apo_raters_{a}_vs_{b}.png")
    print(f"  {a} vs {b}: IoU = {iou:.3f}")
print(f"  mean IoU = {np.mean(ious):.3f}  (1.0 = raters identical)")

print("\n=== M. FIXED CORRELATIONS (true Pearson) ===")
z = np.load(ARTIFACTS_DIR / "thumbs32.npz")
F, names = z["F"].astype(np.float32), z["names"].astype(str)
C = (F @ F.T) / F.shape[1]
np.fill_diagonal(C, -2.0)
off = C[np.triu(np.ones_like(C, dtype=bool), 1)]
print(f"  all pairs: mean r={off.mean():.3f}  p99={np.percentile(off,99):.3f}  "
      f"p99.9={np.percentile(off,99.9):.3f}  max={off.max():.4f}")

tidx = [i for i, n in enumerate(names) if n.startswith("test|")]
r_adj = np.array([C[tidx[i], tidx[i+1]] for i in range(len(tidx)-1)])
print(f"  adjacent test: r>0.99: {(r_adj>0.99).sum()} | 0.95-0.99: "
      f"{((r_adj>0.95)&(r_adj<=0.99)).sum()} | <0.95: {(r_adj<=0.95).sum()}")
for i in np.where(r_adj > 0.99)[0][:25]:
    print(f"    burst: {names[tidx[i]]} <-> {names[tidx[i+1]]}  r={r_adj[i]:.4f}")

ii, jj = np.where(np.triu(np.ones_like(C, dtype=bool), 1) & (C > 0.99))
cross = sorted(((C[a, b], names[a], names[b]) for a, b in zip(ii, jj)
                if names[a].split("|")[0] != names[b].split("|")[0]), reverse=True)
print(f"  near-dupe pairs r>0.99 (all pools): {len(ii)} | cross-pool: {len(cross)}")
for r, a, b in cross[:15]:
    print(f"    r={r:.4f}  {a}  <->  {b}")

parent = list(range(len(names)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b in zip(ii, jj):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
sizes = Counter(Counter(find(i) for i in range(len(names))).values())
print("  cluster sizes (r>0.99 = our CV groups):", dict(sorted(sizes.items())))

print("\n=== N. CALIBRATION EVIDENCE SHEETS (OPEN THESE) ===")
def edge_sheet(key, out, k=8):
    fs = pools[key][::max(1, len(pools[key])//k)][:k]
    rows = []
    for f in fs:
        im = Image.open(f).convert("RGB")
        w, h = im.size
        L = im.crop((0, 0, min(70, w), h)).resize((210, 1200), Image.NEAREST)
        R = im.crop((max(0, w-110), 0, w, h)).resize((330, 1200), Image.NEAREST)
        row = np.concatenate([np.asarray(L), np.full((1200, 10, 3), 255, np.uint8),
                              np.asarray(R)], axis=1)
        rows.append(row)
    Image.fromarray(np.concatenate(rows, axis=0)).save(OUT / out)
    print(f"  saved {OUT / out}")
edge_sheet("test_imgs", "edges_test.png")
edge_sheet("apo_imgs", "edges_apo.png")
edge_sheet("fasc_imgs", "edges_fasc.png")

print("\n=== O. TEST SUBGROUPS ===")
sub = Counter()
for f in pools["test_imgs"]:
    with Image.open(f) as im:
        sub[(f.suffix.lower(), im.mode)] += 1
print("  (ext, mode):", dict(sub))

print("\n=== P. CONSTANT SUBMISSION (SUBMIT THIS) ===")
s = pd.DataFrame({"image_id": [f.name for f in pools["test_imgs"]]})
s["pa_deg"], s["fl_mm"], s["mt_mm"] = 18.0, 80.0, 23.0
s.to_csv(SUBMISSIONS_DIR / "submission_constants.csv", sep=";", index=False)
print(f"  saved submission_constants.csv ({len(s)} rows)")
print(s.head(3).to_string()) 
