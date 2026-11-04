# train_aponet_ens.py -- 2-seed ApoNet ensemble (~1h total)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd, time
from pathlib import Path
from PIL import Image, ImageEnhance
from scipy import ndimage
import segmentation_models_pytorch as smp

# New universal path structure
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data"
MODELS_DIR = PROJECT / "models"
ARTIFACTS_DIR = PROJECT / "artifacts"
OUTPUTS_DIR = PROJECT / "outputs"

ROOT = DATA_DIR
SIZE, BATCH, EPOCHS, LR = 512, 12, 40, 3e-4
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

class DS(Dataset):
    def __init__(self, pairs, train):
        self.pairs, self.tr = pairs, train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("L")
        msk = Image.open(mp).convert("L")
        if msk.size != img.size:
            msk = msk.resize(img.size, Image.BILINEAR)
        m = np.asarray(msk) > 127
        if self.tr:
            ang = float(np.random.uniform(-15, 15))
            img = img.rotate(ang, resample=Image.BILINEAR)
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                           .rotate(ang, resample=Image.NEAREST)) > 127
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT); m = m[:, ::-1]
            if np.random.rand() < 0.2:
                img = img.transpose(Image.FLIP_TOP_BOTTOM); m = m[::-1, :]
            img = ImageEnhance.Contrast(img).enhance(float(np.random.uniform(0.8, 1.2)))
            img = ImageEnhance.Brightness(img).enhance(float(np.random.uniform(0.8, 1.2)))
        img = img.resize((SIZE, SIZE), Image.BILINEAR)
        mm = Image.fromarray((m * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255.0 - 0.5) / 0.5
        y = (np.asarray(mm) > 127).astype(np.float32)
        if self.tr and np.random.rand() < 0.3:
            x = x + np.random.normal(0, 0.03, x.shape).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x[None])),
                torch.from_numpy(y[None]))

def main():
    hashes = pd.read_csv(ARTIFACTS_DIR / "content_hashes.csv")
    twins = pd.read_csv(ARTIFACTS_DIR / "twins.csv")
    twin_apo = {Path(p).stem for p in twins.apo_img}
    fi = hashes[hashes.pool == "apo_imgs"]
    mhash = hashes[hashes.pool == "apo_masks"].set_index("stem")["hash"]
    drop = set()
    for h, g in fi[fi.duplicated("hash", keep=False)].groupby("hash"):
        if len(g) == 2:
            a, b = g.stem.iloc[0], g.stem.iloc[1]
            if mhash.get(a) != mhash.get(b): drop.update([a, b])
    hmap = fi.set_index("stem")["hash"]
    img_by = {f.stem: f for f in pools["apo_imgs"]}
    msk_by = {f.stem: f for f in pools["apo_masks"]}
    seen, train = set(), []
    for stem, h in hmap.items():
        if stem in drop or stem in twin_apo: continue
        if h in seen: continue
        seen.add(h)
        train.append((img_by[stem], msk_by[stem]))
    val = [(Path(r.fasc_img), Path(r.apo_mask)) for _, r in twins.iterrows()]
    print("train={} val={} dev={}".format(len(train), len(val), DEV))

    for s in range(2):
        torch.manual_seed(200 + s)
        tr_dl = DataLoader(DS(train, True), batch_size=BATCH, shuffle=True,
                           num_workers=2, drop_last=True)
        va_dl = DataLoader(DS(val, False), batch_size=BATCH, num_workers=2)
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                         in_channels=1, classes=1).to(DEV)
        lossb = nn.BCEWithLogitsLoss()
        lossd = smp.losses.DiceLoss(mode="binary", from_logits=True)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
        ckpt = MODELS_DIR / ("aponet_ens_{}.pt".format(s))
        best = 0.0
        for ep in range(EPOCHS):
            t0 = time.time(); model.train(); tot = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
                    logits = model(xb)
                    loss = lossb(logits, yb) + lossd(logits, yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += float(loss.detach()) * len(xb)
            sch.step()
            model.eval(); dsum, n = 0.0, 0
            with torch.no_grad():
                for xb, yb in va_dl:
                    with torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
                        pr = torch.sigmoid(model(xb.to(DEV))).cpu()
                    prb = (pr > 0.5).float()
                    inter = (prb * yb).sum(dim=(1, 2, 3))
                    union = prb.sum(dim=(1, 2, 3)) + yb.sum(dim=(1, 2, 3))
                    dsum += float(((2 * inter + 1) / (union + 1)).sum()); n += len(prb)
            vd = dsum / n
            print("seed {} epoch {:2d} loss={:.4f} val_dice={:.4f} ({:.0f}s){}".format(
                s, ep + 1, tot / len(train), vd, time.time() - t0,
                "  *BEST*" if vd > best else ""))
            if vd > best:
                best = vd; torch.save(model.state_dict(), ckpt)
        print("seed {} best: {:.4f} -> {}".format(s, best, ckpt))

if __name__ == "__main__":
    main()