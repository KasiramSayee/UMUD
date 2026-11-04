import re
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

def load_gray(p):
    return np.asarray(Image.open(p).convert("L"), np.float32)

def center_align(mask, h, w):
    out = np.zeros((h, w), np.uint8)
    mh, mw = mask.shape
    sy, sx = (mh - h) // 2, (mw - w) // 2
    src = mask[max(sy,0):max(sy,0)+min(mh,h), max(sx,0):max(sx,0)+min(mw,w)]
    dy, dx = max(0, -sy), max(0, -sx)
    out[dy:dy+src.shape[0], dx:dx+src.shape[1]] = src
    return out

def letterbox_mask(mask_pil, h, w):
    mw, mh = mask_pil.size
    s = min(w / mw, h / mh)
    nw, nh = max(1, round(mw*s)), max(1, round(mh*s))
    small = mask_pil.resize((nw, nh), Image.NEAREST)
    out = Image.new("L", (w, h), 0)
    out.paste(small, ((w-nw)//2, (h-nh)//2))
    return (np.asarray(out) > 127).astype(np.uint8)

print("=== A. IMAGE -> MASK SIZE CROSSTAB ===")
for imgc, maskc in [("apo_imgs","apo_masks"), ("fasc_imgs","fasc_masks")]:
    mfiles = {f.stem: f for f in pools[maskc]}
    combos = {}
    for f in pools[imgc]:
        m = mfiles.get(f.stem)
        if m is None: continue
        with Image.open(f) as im, Image.open(m) as mk:
            combos.setdefault((im.size, mk.size), []).append((f, m))
    print(f"  [{imgc}]")
    for (isz, msz), pairs in sorted(combos.items(), key=lambda kv: -len(kv[1])):
        tag = "SAME" if isz == msz else "MISMATCH"
        print(f"    {isz} -> {msz}: {len(pairs)}  [{tag}]")

print("\n=== B. ALIGNMENT HYPOTHESES (contrast = intensity under mask minus outside) ===")
for imgc, maskc in [("apo_imgs","apo_masks"), ("fasc_imgs","fasc_masks")]:
    mfiles = {f.stem: f for f in pools[maskc]}
    combos = {}
    for f in pools[imgc]:
        m = mfiles.get(f.stem)
        if m is None: continue
        with Image.open(f) as im, Image.open(m) as mk:
            combos.setdefault((im.size, mk.size), []).append((f, m))
    viz_done = 0
    for (isz, msz), pairs in sorted(combos.items(), key=lambda kv: -len(kv[1])):
        f, m = pairs[0]
        img = load_gray(f)
        w, h = isz
        mask_arr = (np.asarray(Image.open(m).convert("L")) > 127).astype(np.uint8)
        if mask_arr.sum() == 0:
            print(f"    {imgc} {isz}->{msz}: EMPTY MASK for {f.name}!"); continue
        if isz == msz:
            c = img[mask_arr==1].mean() - img[mask_arr==0].mean()
            print(f"    {imgc} SAME {isz} (n={len(pairs)})  identity contrast={c:+.1f}")
            continue
        m1 = (np.asarray(Image.open(m).convert("L").resize((w,h), Image.NEAREST)) > 127).astype(np.uint8)
        m2 = center_align(mask_arr, h, w)
        m3 = letterbox_mask(Image.open(m).convert("L"), h, w)
        sc = {n: img[mk==1].mean() - img[mk==0].mean()
              for n, mk in [("stretch",m1), ("center",m2), ("letterbox",m3)]}
        print(f"    {imgc} {isz}->{msz} (n={len(pairs)})  "
              + "  ".join(f"{k}={v:+.1f}" for k, v in sc.items()))
        if viz_done < 2:
            panels = [np.stack([img]*3, -1).astype(np.uint8)]
            for mk in (m1, m2, m3):
                p = np.stack([img]*3, -1).astype(np.uint8).copy()
                p[mk==1] = (255, 0, 0)
                panels.append(p)
            Image.fromarray(np.concatenate(panels, axis=1)).save(OUT / f"align_{imgc}_{f.stem}.png")
            print(f"      saved checks/align_{imgc}_{f.stem}.png")
            viz_done += 1

print("\n=== C. BURNED-IN COLOR OVERLAYS (ruler hunt) ===")
def color_check(f, tag):
    im = Image.open(f)
    if im.mode != "RGB":
        print(f"  {tag}/{f.name}: mode {im.mode} -> skipped"); return
    a = np.asarray(im, np.int16)
    diff = np.abs(a - a.mean(axis=2, keepdims=True)).max(axis=2)
    colored = diff > 12
    if colored.sum() == 0:
        print(f"  {tag}/{f.name}: pure grayscale, no colored overlays"); return
    colhits, rowhits = colored.sum(0), colored.sum(1)
    print(f"  {tag}/{f.name}: {colored.mean():.4f} colored px | "
          f"hot cols {np.argsort(colhits)[::-1][:6].tolist()}/{im.size[0]} | "
          f"hot rows {np.argsort(rowhits)[::-1][:6].tolist()}/{im.size[1]}")
    vis = np.concatenate([np.asarray(im),
                          np.stack([colored*255]*3, -1).astype(np.uint8),
                          np.asarray(im) * colored[..., None]], axis=1)
    Image.fromarray(vis).save(OUT / f"color_{tag}_{f.stem}.png")

for c in ["test_imgs", "fasc_imgs", "apo_imgs"]:
    fs = pools[c]
    for f in fs[::max(1, len(fs)//8)][:8]:
        color_check(f, c)

print("\n=== D. EDGE CROPS 3x ZOOM (ruler reading) ===")
fs = pools["test_imgs"]
for f in fs[::max(1, len(fs)//4)][:4]:
    im = Image.open(f).convert("RGB")
    w, h = im.size
    for nm, box in [("L",(0,0,min(90,w),h)), ("R",(max(0,w-90),0,w,h)),
                    ("T",(0,0,w,min(70,h))), ("B",(0,max(0,h-70),w,h))]:
        c = im.crop(box)
        c.resize((c.width*3, c.height*3), Image.NEAREST).save(OUT / f"edge_{f.stem}_{nm}.png")
print("  saved edge crops for 4 test images")

print("\n=== E. BURST DETECTION (adjacent-frame correlation) ===")
feats = []
for f in pools["test_imgs"]:
    g = np.asarray(Image.open(f).convert("L").resize((32,32)), np.float32).ravel()
    feats.append((g - g.mean()) / (g.std() + 1e-9))
feats = np.stack(feats)
r = (feats[:-1] * feats[1:]).sum(1)
print(f"  consecutive corr: mean={r.mean():.3f}  >0.9: {(r>0.9).sum()}  >0.8: {(r>0.8).sum()}")
for i in np.argsort(r)[::-1][:15]:
    print(f"    r={r[i]:.3f}  {pools['test_imgs'][i].name} <-> {pools['test_imgs'][i+1].name}")

print("\n=== F. SAMPLE SUBMISSION & FILENAMES ===")
ss = pd.read_csv(ROOT / "sample_submission.csv")
print(f"  shape={ss.shape} cols={ss.columns.tolist()}")
print(ss.head(5).to_string())
names = [f.name for f in pools["test_imgs"]]
print("  patterns:", Counter(re.sub(r"\d+", "#", n) for n in names).most_common())
print(f"  count={len(names)}  first={names[0]}  last={names[-1]}")