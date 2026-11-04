# eval_regressor_twins.py -- the distillation hypothesis meets ground truth
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch, torch.nn as nn
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.models as tvm

# New universal path structure
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data"
MODELS_DIR = PROJECT / "models"
ARTIFACTS_DIR = PROJECT / "artifacts"
SUBMISSIONS_DIR = PROJECT / "submissions"
OUTPUTS_DIR = PROJECT / "outputs"

ROOT = DATA_DIR
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 640

# normalization stats (from training output — hardcode for now)
PA_MEAN, PA_STD = 18.226, 5.997
FL_MEAN, FL_STD = 1.032, 0.388
MT_MEAN, MT_STD = 0.385, 0.077

model = tvm.resnet50(weights=None)
model.fc = nn.Linear(2048, 3)
model.load_state_dict(torch.load(MODELS_DIR / "regressor_v1.pt", map_location=DEV))
model = model.to(DEV)
model.eval()

def predict_reg(img):
    x = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), np.float32)
    x = (x / 255.0 - 0.5) / 0.5
    x = np.stack([x, x, x], axis=0)
    with torch.no_grad():
        out = model(torch.from_numpy(x[None]).to(DEV)).cpu().numpy()[0]
    pa = out[0] * PA_STD + PA_MEAN
    fl_norm = out[1] * FL_STD + FL_MEAN
    mt_norm = out[2] * MT_STD + MT_MEAN
    return pa, fl_norm, mt_norm

twins = pd.read_csv(ARTIFACTS_DIR / "twins.csv")
ev = pd.read_csv(ARTIFACTS_DIR / "twin_eval.csv")

rows = []
for _, r in twins.iterrows():
    img = Image.open(r.fasc_img).convert("L")
    W, H = img.size
    pa, fl_norm, mt_norm = predict_reg(img)
    rows.append(dict(name=Path(r.fasc_img).name,
                     reg_pa=pa, reg_fl_px=fl_norm * W, reg_mt_px=mt_norm * H))

df = pd.DataFrame(rows).merge(
    ev[["name", "gt_pa", "gt_fl_px", "gt_mt_px", "pr_pa"]], on="name", how="left")
ok = df.dropna()

print("=== REGRESSOR vs GROUND TRUTH ({} twins) ===".format(len(ok)))
print("\nPA (the headline number):")
err_reg = ok.reg_pa - ok.gt_pa
print("  regressor: MAE={:.2f} bias={:+.2f}".format(err_reg.abs().mean(), err_reg.mean()))
print("  (pipeline v3: MAE=2.96, bias=-1.12, corrected=2.60)")

print("\nFL (pixel space):")
err_fl = ok.reg_fl_px - ok.gt_fl_px
print("  regressor: MAE={:.2f} bias={:+.2f}".format(err_fl.abs().mean(), err_fl.mean()))
print("  (pipeline v3: MAE=245, bias=+37)")

print("\nMT (pixel space):")
err_mt = ok.reg_mt_px - ok.gt_mt_px
print("  regressor: MAE={:.2f} bias={:+.2f}".format(err_mt.abs().mean(), err_mt.mean()))
print("  (pipeline v3: MAE=24.8, bias=-0.4)")

print("\n=== OPTIMAL BLEND (PA) ===")
for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    blend = w * ok.reg_pa + (1 - w) * ok.pr_pa
    mae = (blend - ok.gt_pa).abs().mean()
    print("  w_reg={:.1f}: PA MAE={:.2f}".format(w, mae))

# correlation between regressor and pipeline errors (independence check)
err_pipe = ok.pr_pa - ok.gt_pa
corr = np.corrcoef(err_reg, err_pipe)[0, 1]
print("\n  error correlation (regressor vs pipeline): {:.2f}".format(corr))
print("  (lower = more independent = better blending)")