import numpy as np, pandas as pd, hashlib
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

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

print("=== G. HASH AUDIT (reads ~6 GB, few minutes, one-time) ===")
recs = [{"pool": c, "file": f.name, "stem": f.stem, "hash": sha(f)}
        for c, fs in pools.items() for f in fs]
df = pd.DataFrame(recs)
df.to_csv(ARTIFACTS_DIR / "content_hashes.csv", index=False)

img = df[df.pool.isin(["fasc_imgs", "apo_imgs", "test_imgs"])]
for c in ["fasc_imgs", "apo_imgs", "test_imgs"]:
    sub = img[img.pool == c]
    dup_files = sub.duplicated("hash", keep=False).sum()
    n_groups = sub[sub.duplicated("hash", keep=False)].hash.nunique()
    print(f"  {c}: {len(sub)} files -> {sub.hash.nunique()} unique "
          f"({dup_files} files in {n_groups} dup groups)")

test_h  = set(img[img.pool == "test_imgs"].hash)
train_h = set(img[img.pool.isin(["fasc_imgs", "apo_imgs"])].hash)
print("  test/train content overlap:", len(test_h & train_h))

print("\n=== H. DO DUPLICATE IMAGES CARRY IDENTICAL MASKS? ===")
for ic, mc in [("fasc_imgs", "fasc_masks"), ("apo_imgs", "apo_masks")]:
    fi = df[df.pool == ic]
    dupg = fi[fi.duplicated("hash", keep=False)]
    if dupg.empty:
        print(f"  {ic}: no duplicate groups"); continue
    fm = df[df.pool == mc].set_index("stem")["hash"]
    same = diff = odd = 0
    for h, g in dupg.groupby("hash"):
        if len(g) != 2: odd += 1; continue
        a, b = g.stem.iloc[0], g.stem.iloc[1]
        if fm.get(a) is not None and fm.get(a) == fm.get(b): same += 1
        else: diff += 1
    print(f"  {ic}: pairs w/ identical masks: {same} | differing: {diff} | >2 copies: {odd}")

print("\n=== I. UNIQUE IMAGE -> MASK COVERAGE (our label inventory) ===")
apo_stems  = set(df[df.pool == "apo_imgs"].stem)
fasc_stems = set(df[df.pool == "fasc_imgs"].stem)
rows = []
for h, g in img[img.pool.isin(["fasc_imgs", "apo_imgs"])].groupby("hash"):
    stems = set(g.stem)
    rows.append((len(g), bool(stems & apo_stems), bool(stems & fasc_stems)))
cov = pd.DataFrame(rows, columns=["n_copies", "apo", "fasc"])
n_both = int((cov.apo & cov.fasc).sum())
n_apo  = int((cov.apo & ~cov.fasc).sum())
n_fasc = int((~cov.apo & cov.fasc).sum())
print(f"  unique train images: {len(cov)} | BOTH masks: {n_both} | apo-only: {n_apo} | fasc-only: {n_fasc}")
print("  copies-per-image distribution:", cov.n_copies.value_counts().to_dict())

print("\n=== J. NEAR-DUPLICATE SCAN (32x32 thumbnails, cached) ===")
CACHE = ARTIFACTS_DIR / "thumbs32.npz"
if CACHE.exists():
    z = np.load(CACHE); F, names = z["F"], z["names"].astype(str)
else:
    F, names = [], []
    for tag, key in [("fasc","fasc_imgs"), ("apo","apo_imgs"), ("test","test_imgs")]:
        for f in pools[key]:
            g = np.asarray(Image.open(f).convert("L").resize((32,32)), np.float32).ravel()
            F.append((g - g.mean()) / (g.std() + 1e-9)); names.append(tag + "|" + f.name)
    F, names = np.stack(F).astype(np.float32), np.array(names)
    np.savez_compressed(CACHE, F=F, names=names)
C = F @ F.T
np.fill_diagonal(C, -2)
top1 = C.max(axis=1)
for lo, hi in [(0.999, 1.01), (0.99, 0.999), (0.97, 0.99), (0.93, 0.97), (0.0, 0.93)]:
    band = ((top1 >= lo) & (top1 < hi)).sum()
    print(f"  images whose best match has r in [{lo},{hi}): {band}")
upper = np.triu(np.ones_like(C, dtype=bool), 1)
ii, jj = np.where(upper & (C > 0.97))
print(f"  pairs r>0.97: {len(ii)}")

parent = list(range(len(names)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b in zip(ii, jj):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
sizes = Counter(Counter(find(i) for i in range(len(names))).values())
print("  cluster size distribution (r>0.97):", dict(sorted(sizes.items())))
shown = 0
for a, b in zip(ii, jj):
    if C[a, b] < 0.999 and shown < 12:
        print(f"    r={C[a,b]:.3f}  {names[a]}  <->  {names[b]}"); shown += 1