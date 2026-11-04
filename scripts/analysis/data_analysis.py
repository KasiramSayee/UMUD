import re, sqlite3
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from PIL import Image

# New universal path structure
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data"
MODELS_DIR = PROJECT / "models"
ARTIFACTS_DIR = PROJECT / "artifacts"
SUBMISSIONS_DIR = PROJECT / "submissions"
OUTPUTS_DIR = PROJECT / "outputs"

ROOT = DATA_DIR
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# ---- 0. discover real structure (typo & nesting proof) ----
def classify(d):
    s = str(d.relative_to(ROOT)).lower()
    m = "mask" in s
    if "apo"  in s: return "apo_masks" if m else "apo_imgs"
    if "fasc" in s: return "fasc_masks" if m else "fasc_imgs"
    if "test" in s: return "test_imgs"
    return None

pools = {}
for d in sorted([d for d in ROOT.rglob("*") if d.is_dir()],
                key=lambda x: len(x.parts)):          # shallowest first
    c = classify(d)
    if c and c not in pools:
        pools[c] = d

def images_under(d):
    return sorted(f for f in d.rglob("*")
                  if f.is_file() and f.suffix.lower() in IMG_EXTS)

pool_files = {c: images_under(d) for c, d in pools.items()}
print("=== 0. DISCOVERED STRUCTURE ===")
for c, d in pools.items():
    print(f"  {c:11s} <- {d.relative_to(ROOT)}   ({len(pool_files[c])} images)")

# ---- 1. inventory ----
all_files = [f for f in ROOT.rglob("*") if f.is_file()]
print("\n=== 1. INVENTORY (expect ~7930 total per Kaggle page) ===")
for ext, n in Counter(f.suffix.lower() or "<none>" for f in all_files).most_common():
    print(f"  {ext:10s} {n}")
strays = [f for f in all_files if f.suffix.lower() not in IMG_EXTS | {".csv"}]
print("  non-image/non-csv files:", [str(f.relative_to(ROOT)) for f in strays] or "NONE")

# ---- 2. sqlite hunt (backslash-free queries) ----
print("\n=== 2. .db / SQLITE HUNT ===")
db = [f for f in all_files if f.suffix.lower() in {".db", ".sqlite", ".sqlite3"}]
if not db:
    print("  NO local .db file  <-- re-check the Kaggle Data page download!")
for p in db:
    print(f"found: {p} ({p.stat().st_size/1e6:.2f} MB)")
    try:
        con = sqlite3.connect(str(p))
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        print("  tables:", tables)
        for t in tables:
            q_count = 'SELECT COUNT(*) FROM "{}"'.format(t)
            cnt = con.execute(q_count).fetchone()[0]
            print(f"  -- {t}: {cnt} rows")
            q_rows = 'SELECT * FROM "{}" LIMIT 3'.format(t)
            for row in con.execute(q_rows):
                print("     ", row)
    except Exception as e:
        print("  not SQLite:", e)

# ---- 3. pool sizes & overlap ----
print("\n=== 3. OVERLAP BETWEEN MASK POOLS ===")
apo  = pool_files.get("apo_imgs", [])
fasc = pool_files.get("fasc_imgs", [])
print(f"  apo imgs: {len(apo)}   fasc imgs: {len(fasc)}")
if apo and fasc:
    a_n, f_n = {f.name for f in apo}, {f.name for f in fasc}
    a_s, f_s = {f.stem for f in apo}, {f.stem for f in fasc}
    print(f"  overlap by exact filename: {len(a_n & f_n)}   by stem: {len(a_s & f_s)}")

# ---- 4. dimensions & modes (header-only, ~1-2 min) ----
print("\n=== 4. DIMENSIONS / MODES / MULTI-FRAME ===")
for c, fs in pool_files.items():
    sizes, modes, frames, bad = Counter(), Counter(), Counter(), []
    for p in fs:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
                frames[getattr(im, "n_frames", 1)] += 1
        except Exception as e:
            bad.append((p.name, str(e)[:50]))
    print(f"  {c}: n={sum(sizes.values())}  modes={dict(modes)}")
    extra = f"   *** MULTI-FRAME: {dict(frames)}" if max(frames, default=1) > 1 else ""
    print(f"    top sizes: {sizes.most_common(4)}{extra}")
    if bad: print(f"    UNREADABLE ({len(bad)}):", bad[:3])

# ---- 5. mask semantics ----
print("\n=== 5. MASK SEMANTICS (40 evenly-spaced samples) ===")
def scan_masks(c, n=40):
    fs = pool_files.get(c, [])
    if not fs:
        print(f"  {c}: NOT FOUND"); return
    uq, dt, vals = Counter(), Counter(), Counter()
    for p in fs[::max(1, len(fs)//n)][:n]:
        m = np.array(Image.open(p))
        dt[str(m.dtype)] += 1
        u = np.unique(m); uq[len(u)] += 1
        for v in u: vals[v] += 1
    print(f"  {c}: dtypes={dict(dt)} | #unique per mask={dict(uq)}")
    print(f"    all observed pixel values: {dict(vals)}")
scan_masks("apo_masks"); scan_masks("fasc_masks")

# ---- 6. tiff tags (calibration hunt) ----
print("\n=== 6. TIFF TAGS ===")
for c in ["apo_imgs", "fasc_imgs", "test_imgs"]:
    for p in [f for f in pool_files.get(c, [])
              if f.suffix.lower() in {".tif", ".tiff"}][:3]:
        with Image.open(p) as im:
            print(f"  {c}/{p.name} {im.size} {im.mode}")
            print(f"    tags: {dict(getattr(im, 'tag_v2', {}) or {})}")

# ---- 7. sample submission ----
print("\n=== 7. SAMPLE SUBMISSION ===")
ss = pd.read_csv(ROOT / "sample_submission.csv")
print(f"  shape={ss.shape}  cols={ss.columns.tolist()}")
print(ss.head(5).to_string())
n_test = len(pool_files.get("test_imgs", []))
print(f"  test images locally: {n_test}")
if len(ss) < 300:
    print("  *** sample is a stub -> we must build the full submission ourselves")

# ---- 8. burst detection ----
print("\n=== 8. TEST FILENAME PATTERNS ===")
names = sorted(f.name for f in pool_files.get("test_imgs", []))
print("  histogram:", Counter(re.sub(r"\d+", "#", n) for n in names).most_common(10))
print("  first 15:", names[:15])

# ---- 9. contact sheets ----
print("\n=== 9. CONTACT SHEETS ===")
def sheet(c, out, k=12):
    fs = pool_files.get(c, [])
    if not fs:
        print(f"  {c}: NOT FOUND"); return
    fs = fs[::max(1, len(fs)//k)][:k]
    T, cols = 320, 4
    s = Image.new("L", (cols*T, ((len(fs)+cols-1)//cols)*T))
    for i, p in enumerate(fs):
        with Image.open(p) as im:
            s.paste(im.convert("L").resize((T, T)), ((i % cols)*T, (i//cols)*T))
    s.save(OUTPUTS_DIR / out)
    print(f"  saved {OUTPUTS_DIR / out}")
sheet("test_imgs", "test_sheet.png")
sheet("fasc_imgs", "fasc_sheet.png")
sheet("apo_imgs", "apo_sheet.png")