# train_regressor.py -- direct regression on pseudo-labels
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd, time
from pathlib import Path
from PIL import Image, ImageEnhance
import torchvision.models as tvm

# New universal path structure
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data"
MODELS_DIR = PROJECT / "models"
ARTIFACTS_DIR = PROJECT / "artifacts"
OUTPUTS_DIR = PROJECT / "outputs"

ROOT = DATA_DIR
SIZE, BATCH, EPOCHS, LR = 640, 12, 50, 3e-4
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

fasc_by_stem = {f.stem: f for f in pools["fasc_imgs"]}

pl = pd.read_csv(ARTIFACTS_DIR / "pseudo_labels.csv")
pl = pl[pl.pa.notna()].reset_index(drop=True)
print("pseudo-labels: {} images".format(len(pl)))

# normalize FL and MT by image dimensions (scale-invariant targets)
dims = []
for stem in pl.stem:
    with Image.open(fasc_by_stem[stem]) as im:
        dims.append(im.size)
pl["W"] = [d[0] for d in dims]
pl["H"] = [d[1] for d in dims]
pl["fl_norm"] = pl.fl_px / pl.W
pl["mt_norm"] = pl.mt_px / pl.H

STATS = {
    "pa": (float(pl.pa.mean()), float(pl.pa.std())),
    "fl": (float(pl.fl_norm.mean()), float(pl.fl_norm.std())),
    "mt": (float(pl.mt_norm.mean()), float(pl.mt_norm.std())),
}
print("target stats:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in STATS.items()})

rng = np.random.default_rng(42)
idx = rng.permutation(len(pl))
n_val = max(1, len(pl) // 10)
val_idx = set(idx[:n_val].tolist())

class RegDS(Dataset):
    def __init__(self, df, train=True):
        self.df, self.tr = df.reset_index(drop=True), train
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(fasc_by_stem[r.stem]).convert("L")
        if self.tr:
            ang = float(np.random.uniform(-15, 15))
            img = img.rotate(ang, resample=Image.BILINEAR)
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.2:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = ImageEnhance.Contrast(img).enhance(float(np.random.uniform(0.8, 1.2)))
            img = ImageEnhance.Brightness(img).enhance(float(np.random.uniform(0.8, 1.2)))
        x = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), np.float32)
        x = (x / 255.0 - 0.5) / 0.5
        x = np.stack([x, x, x], axis=0)
        y = np.array([
            (r.pa - STATS["pa"][0]) / STATS["pa"][1],
            (r.fl_norm - STATS["fl"][0]) / STATS["fl"][1],
            (r.mt_norm - STATS["mt"][0]) / STATS["mt"][1],
        ], dtype=np.float32)
        if self.tr and np.random.rand() < 0.3:
            x = x + np.random.normal(0, 0.03, x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)

if __name__ == "__main__":
    train_df = pl[~pl.index.isin(val_idx)]
    val_df = pl[pl.index.isin(val_idx)]
    print("train={} val={}".format(len(train_df), len(val_df)))

    tr_dl = DataLoader(RegDS(train_df, True), batch_size=BATCH, shuffle=True,
                    num_workers=2, drop_last=True)
    va_dl = DataLoader(RegDS(val_df, False), batch_size=BATCH, num_workers=2)

    model = tvm.resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(2048, 3)
    model = model.to(DEV)

    lossf = nn.L1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    ckpt = MODELS_DIR / "regressor_v1.pt"
    best = 1e9

    for ep in range(EPOCHS):
        t0 = time.time(); model.train(); tot = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
                pred = model(xb)
                loss = lossf(pred, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss.detach()) * len(xb)
        sch.step()
        model.eval(); vsum, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
                    pr = model(xb.to(DEV)).cpu()
                vsum += float((pr - yb).abs().mean()) * len(xb); vn += len(xb)
        vmae = vsum / vn
        print("epoch {:2d} loss={:.4f} val_mae={:.4f} ({:.0f}s){}".format(
            ep + 1, tot / len(train_df), vmae, time.time() - t0,
            "  *BEST*" if vmae < best else ""))
        if vmae < best:
            best = vmae
            torch.save(model.state_dict(), ckpt)
            pd.DataFrame([STATS]).to_csv(ARTIFACTS_DIR / "regressor_stats.csv", index=False)

    print("\nbest val MAE (z-scored): {:.4f}".format(best))
    print("real-unit interpretation:")
    print("  PA MAE ~ {:.2f} deg  (pipeline: 2.96, corrected: 2.60)".format(best * STATS["pa"][1]))
    print("  FL relative MAE ~ {:.3f} of image width".format(best * STATS["fl"][1]))
    print("  MT relative MAE ~ {:.3f} of image height".format(best * STATS["mt"][1]))