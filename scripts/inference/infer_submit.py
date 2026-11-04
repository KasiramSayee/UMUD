#!/usr/bin/env python3
"""
infer_submit.py -- v12_m5 REFERENCE RECONSTRUCTION (leaderboard best, 0.510)
============================================================================

This file reconstructs the exact best-scoring configuration of the UMUD
muscle-architecture pipeline, as specified in the project chronicle
(Ch.11 Table 9 "pipeline stage by stage" + Appendix B Table 11 "complete
constants").  It exists to revert the post-v12_m5 experiments:

    v14  robust Huber fitting + inline calibration + twin-refit PA -> UNDONE
    v15  ApoNet 2-seed ensemble                                   -> UNDONE
    v19  classical structure-tensor PA blend                      -> UNDONE

v12_m5 pipeline (execution order):

    1. Fascicle masks : FascNet v3 folds 0+1, 640 px, probability average,
                        horizontal-flip TTA, threshold 0.5, binary closing
    2. Apo mask       : ApoNet v1 (SINGLE seed, no ensemble), 512 px,
                        threshold 0.5, no closing
    3. Geometry       : polyfit deg-1 least-squares apo lines (NOT Huber),
                        component filters (Table 4), sin-weighted mode
                        clustering (+/-3.5 deg), top-3 by area,
                        line-intersection FL (extrapolation inherent),
                        3-site perpendicular MT, consistency ratio 0.4..2.5
    4. Calibration    : per image from calibration_v1.csv
                        (PNG: 20/marker_px measured; tif priors 0.0656/0.0830)
    5. PA correction  : PA = 7.54 + 0.59 * PA_raw   (twin shrinkage inverse)
    6. FL scale       : FL *= 0.925
    7. Burst averaging: mean over near-duplicate runs (Pearson r > 0.99)
    8. Clipping       : PA 5..45, FL 30..200, MT 10..50
    9. CSV transform  : tif-suffixed rows: FL and MT *= 0.95
   10. Fallback chain : v3 ens + TTA -> v1 + TTA -> v1 -> constants
                        18.0 deg / 80.0 mm / 23.0 mm

RECONSTRUCTION CAVEATS -- the 3 spots that must match YOUR training code
(the checkpoints dictate them; keep your local version of these if in doubt):
    * IN_CHANNELS (config below)      -- 1 or 3, must match training
    * preprocess()                    -- /255.0 normalization; change if your
                                         training used ImageNet mean/std
    * CLOSE_KERNEL                    -- binary-closing kernel size
Everything else is fixed by the documented constants.

Run from the project root with checkpoints, calibration_v1.csv and Data/ in
place:      python infer_submit.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows MKL/OpenCV clash guard
import torch  # MUST be the first heavy import on Windows (WinError 1114)
import cv2
import math
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# New universal path structure
PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "Data"
MODELS_DIR = PROJECT / "models"
ARTIFACTS_DIR = PROJECT / "artifacts"
SUBMISSIONS_DIR = PROJECT / "submissions"
OUTPUTS_DIR = PROJECT / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# v12_m5 CONFIGURATION -- every constant (chronicle Appendix B, Table 11)
# ======================================================================
# --- models -------------------------------------------------------------
APONET_CKPT = MODELS_DIR / "aponet_v1.pt"            # SINGLE seed; v15 ensemble REVERTED
APO_RES = 512                           # ApoNet native resolution
FASC_CKPTS = [MODELS_DIR / "fascnet_v3_fold0.pt",    # production pair (fold 2 dropped)
              MODELS_DIR / "fascnet_v3_fold1.pt"]
FASC_RES = 640                          # v3 native resolution -- never 512
FASC_FALLBACK_CKPT = MODELS_DIR / "fascnet_v1.pt"    # safety net at bottom of chain
FASC_FALLBACK_RES = 512                 # v1 native resolution
IN_CHANNELS = 3                         # CAVEAT: must match training

# --- mask post-processing ------------------------------------------------
MASK_THRESHOLD = 0.5
CLOSE_KERNEL = 5                        # CAVEAT: match your original kernel

# --- geometry (chronicle Table 4) -----------------------------------------
APO_MIN_SIZE = 500          # apo component minimum size, px
APO_MIN_SEPARATION = 60     # apo centroid separation, px
FASC_MIN_SIZE = 80          # fascicle component minimum size, px
ANGLE_WINDOW = (6.0, 60.0)  # candidate angle vs deep apo line, degrees
MIN_ELONGATION = 4.0        # long/short axis ratio of candidate component
MODE_HALF_WIDTH = 3.5       # sin-weighted mode cluster half-width, degrees
N_FASCICLES = 3             # selected per image (raters' 3-fascicle protocol)
MT_SITES = (0.25, 0.50, 0.75)   # fraction of image width, 3-site protocol
CONSISTENCY_BOUNDS = (0.4, 2.5) # FL*sin(PA)/MT plausibility window

# --- calibration (chronicle Table 6) ---------------------------------------
CALIB_CSV = ARTIFACTS_DIR / "calibration_v1.csv"        # standing forensics artifact
TIF_PRIORS = {(1200, 800): 0.0656,      # mm/px, tick-structure priors
              (1088, 644): 0.0830}
PNG_DEPTH_FALLBACK_MM = 60.0            # crude depth/604 fallback (rarely fires)

# --- post-processing (chronicle Ch.8) ---------------------------------------
PA_A = 7.54               # PA = PA_A + PA_B * PA_raw (v14 refit REVERTED)
PA_B = 0.59
FL_SCALE = 0.925
BURST_R = 0.99             # near-duplicate burst threshold (Pearson r)
CLIP_PA = (5.0, 45.0)
CLIP_FL = (30.0, 200.0)
CLIP_MT = (10.0, 50.0)

# --- CSV transform (chronicle Ch.7.4) ----------------------------------------
TIF_SCALE = 0.95           # tif rows: FL and MT *= 0.95 (bisection-proven)

# --- fallback constants (chronicle Ch.3) --------------------------------------
CONST_PA = 18.0           # fallback constants (chronicle Ch.3)
CONST_FL = 80.0
CONST_MT = 23.0

# --- I/O ----------------------------------------------------------------------
DATA_DIR = PROJECT / "Data"
TEST_SUBSTR = "test_images_v2"          # auto-discovery by path substring
OUT_CSV = SUBMISSIONS_DIR / "submission_v12_m5_repro.csv"
N_EXPECTED = 309
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# ======================================================================
# Data discovery and loading
# ======================================================================

def discover_test_images(data_dir):
    """Auto-discover test images by path substring (folders are nested
    irregularly -- never trust the directory tree, chronicle Sec 2.1)."""
    root = Path(data_dir)
    if not root.exists():
        sys.exit(f"ERROR: data directory '{data_dir}' not found. "
                 "Run from the project root.")
    found = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and TEST_SUBSTR in str(p):
            found.append((p.name, p))
    found.sort(key=lambda t: t[0])
    if len(found) != N_EXPECTED:
        sys.exit(f"ERROR: expected {N_EXPECTED} test images, found {len(found)}.")
    return found


def load_image(path):
    """Load as 2-D uint8 grayscale (PIL fallback for stubborn tifs)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(path).convert("L"))
    return img


# ======================================================================
# Models
# ======================================================================

def build_model():
    import segmentation_models_pytorch as smp
    # U-Net / ResNet34, logits out (activation=None) -- sigmoid applied here.
    return smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=IN_CHANNELS, classes=1, activation=None)


def load_model(ckpt):
    m = build_model()
    sd = torch.load(ckpt, map_location="cpu")
    try:
        m.load_state_dict(sd)
    except RuntimeError as e:
        if "in_channels" in str(e) or "conv1" in str(e):
            sys.exit(f"ERROR: checkpoint {ckpt} does not match IN_CHANNELS="
                     f"{IN_CHANNELS}. Set IN_CHANNELS to match your "
                     "training configuration (try 1 or 3).")
        raise
    m.to(DEVICE).eval()
    return m


def preprocess(gray, res):
    """CAVEAT spot: /255.0 normalization, bilinear resize. Match training."""
    if IN_CHANNELS == 3:
        img = np.stack([gray] * 3, axis=-1)
    else:
        img = gray[..., None]
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
    return x


def predict_prob(model, gray, res, tta):
    """Predict at the model's NATIVE resolution (never at another),
    probability map stretched back to original size (plain bilinear,
    chronicle Sec 2.5).  TTA = horizontal flip on the fascicle path only."""
    h, w = gray.shape
    x = preprocess(gray, res).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        if tta:
            xf = torch.flip(x, dims=[3])
            p2 = torch.sigmoid(model(xf))[0, 0].cpu().numpy()
            p2 = np.flip(p2, axis=1)  # flip back along width
            p = 0.5 * (p + p2)
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


_CLOSE_STRUCT = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)


def fascicle_mask_from_prob(prob):
    """Threshold 0.5 + binary closing (chronicle: adopted at submission v9)."""
    m = (prob >= MASK_THRESHOLD).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, _CLOSE_STRUCT)


def apo_mask_from_prob(prob):
    """ApoNet path: threshold 0.5 only (no closing, no TTA, no ensemble)."""
    return (prob >= MASK_THRESHOLD).astype(np.uint8)


# ======================================================================
# Geometry engine (chronicle Ch.5, Table 4) -- least-squares polyfit,
# NOT the v14 Huber variant
# ======================================================================

class GeometryFail(Exception):
    pass


def _fit_line(xs, ys):
    """Degree-1 least-squares polyfit. Apo bands are near-horizontal; a
    degenerate (vertical) component fails loudly rather than silently."""
    if float(np.std(xs)) < 1.0:
        raise GeometryFail("degenerate apo component (no x spread)")
    m, b = np.polyfit(xs, ys, 1)
    return float(m), float(b)


def _line_y(coeffs, x):
    m, b = coeffs
    return m * x + b


def extract_apo_lines(apo_mask):
    """Two largest 8-connected components, >=500 px, centroids >=60 px apart.
    Screen-higher centroid = superficial, lower = deep."""
    n, lab = cv2.connectedComponents(apo_mask.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if len(xs) < APO_MIN_SIZE:
            continue
        comps.append((len(xs), xs, ys))
    if len(comps) < 2:
        raise GeometryFail("fewer than 2 apo components >= %d px" % APO_MIN_SIZE)
    comps.sort(key=lambda c: -c[0])
    _, xs1, ys1 = comps[0]
    _, xs2, ys2 = comps[1]
    if math.hypot(xs1.mean() - xs2.mean(), ys1.mean() - ys2.mean()) < APO_MIN_SEPARATION:
        raise GeometryFail("apo centroids closer than %d px" % APO_MIN_SEPARATION)
    l1 = _fit_line(xs1, ys1)
    l2 = _fit_line(xs2, ys2)
    # smaller mean row = higher on screen = superficial
    if ys1.mean() <= ys2.mean():
        return l1, l2
    return l2, l1


def fascicle_candidates(fasc_mask, apo_sup, apo_deep):
    """Four gates: size >=80 px, angle in [6,60] vs deep line, elongation
    >=4, centroid vertically between the two apo lines (chronicle 5.2)."""
    n, lab = cv2.connectedComponents(fasc_mask.astype(np.uint8), connectivity=8)
    dm, db = apo_deep
    v_deep = np.array([1.0, dm]) / math.sqrt(1.0 + dm * dm)
    out = []
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if len(xs) < FASC_MIN_SIZE:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (_, (rw, rh), _) = cv2.minAreaRect(pts)
        short, long_ = min(rw, rh), max(rw, rh)
        if short <= 0 or long_ / short < MIN_ELONGATION:
            continue
        # principal-axis line through centroid
        pts64 = pts.astype(np.float64)
        pts64 -= pts64.mean(axis=0)
        _, evecs = np.linalg.eigh(np.cov(pts64.T))
        d = evecs[:, -1]
        if abs(d[0]) < 1e-9:
            d = np.array([0.0, 1.0])
        elif d[0] < 0:
            d = -d
        cx, cy = float(xs.mean()), float(ys.mean())
        # pennation angle = acute angle between fascicle line and deep apo
        cosang = min(1.0, abs(float(np.dot(d, v_deep))))
        ang = math.degrees(math.acos(cosang))
        if not (ANGLE_WINDOW[0] <= ang <= ANGLE_WINDOW[1]):
            continue
        ysup, ydep = _line_y(apo_sup, cx), _line_y(apo_deep, cx)
        if not (min(ysup, ydep) <= cy <= max(ysup, ydep)):
            continue
        out.append({"ang": ang, "area": int(len(xs)), "c": (cx, cy), "d": d})
    return out


def select_fascicles(cands):
    """Sin-weighted mode clustering (chronicle 5.3): the mode maximizes
    sum of areas * sin(candidate's own angle) within +/-3.5 deg.  A
    near-horizontal junk component votes with ~sin(6 deg) ~ 0.1 and can
    no longer drag the mode toward zero.  Top 3 by area are selected."""
    if not cands:
        raise GeometryFail("no fascicle candidates passed the gates")
    best_mode, best_score = None, -1.0
    for c in cands:
        score = sum(o["area"] * math.sin(math.radians(o["ang"]))
                    for o in cands
                    if abs(o["ang"] - c["ang"]) <= MODE_HALF_WIDTH)
        if score > best_score:
            best_score, best_mode = score, c["ang"]
    sel = [o for o in cands if abs(o["ang"] - best_mode) <= MODE_HALF_WIDTH]
    sel.sort(key=lambda o: -o["area"])
    if not sel:
        raise GeometryFail("empty mode cluster")
    return sel[:N_FASCICLES]


def _cross_param(pt, d, coeffs):
    """Parameter t along the fascicle's infinite line where it crosses the
    apo line y = m x + b.  None if parallel."""
    m, b = coeffs
    nrm = math.sqrt(m * m + 1.0)
    denom = (m * d[0] - d[1]) / nrm
    if abs(denom) < 1e-9:
        return None
    return -((m * pt[0] - pt[1] + b) / nrm) / denom


def extract_geometry(apo_mask, fasc_mask):
    """Full extraction: returns (pa_raw_deg, fl_px, mt_px).
    Runs identically on predicted masks (production) and real masks (twins
    harness), which is what makes the harness meaningful."""
    apo_sup, apo_deep = extract_apo_lines(apo_mask)

    cands = fascicle_candidates(fasc_mask, apo_sup, apo_deep)
    sel = select_fascicles(cands)

    # PA: mean angle of the selected fascicle lines against the deep apo
    pa_raw = float(np.mean([o["ang"] for o in sel]))

    # FL: mean distance between the two line-line intersections, using
    # infinite lines so extrapolation off-frame is inherent (Table 1)
    fls = []
    for o in sel:
        t_sup = _cross_param(o["c"], o["d"], apo_sup)
        t_dep = _cross_param(o["c"], o["d"], apo_deep)
        if t_sup is None or t_dep is None:
            continue
        fls.append(abs(t_sup - t_dep))
    if not fls:
        raise GeometryFail("no fascicle crosses both aponeuroses")
    fl_px = float(np.mean(fls))

    # MT: perpendicular distance between the apo lines at 25/50/75 % width
    h, w = fasc_mask.shape
    dm, db = apo_deep
    nrm = math.sqrt(dm * dm + 1.0)
    dists = []
    for frac in MT_SITES:
        xi = frac * (w - 1)
        yi = _line_y(apo_sup, xi)
        dists.append(abs(dm * xi - yi + db) / nrm)
    mt_px = float(np.mean(dists))
    if mt_px <= 1e-6:
        raise GeometryFail("degenerate muscle thickness")

    # Consistency filter: plausible fascicle-aponeurosis triangle
    ratio = fl_px * math.sin(math.radians(pa_raw)) / mt_px
    lo, hi = CONSISTENCY_BOUNDS
    if not (lo <= ratio <= hi):
        raise GeometryFail("consistency ratio %.3f outside [%.1f, %.1f]"
                           % (ratio, lo, hi))
    return pa_raw, fl_px, mt_px


# ======================================================================
# Calibration (chronicle Ch.7) -- lookup from calibration_v1.csv; the
# inline fallbacks below are a safety net only.  v14's "inline calibration"
# is REVERTED: the CSV produced by dd_calibration.py is the v12_m5 source
# of truth (the PNG markers were measured there).
# ======================================================================

def load_calibration(path):
    p = Path(path)
    if not p.exists():
        print(f"[warn] {path} not found -- using built-in calibration "
              "fallbacks. Place calibration_v1.csv (dd_calibration.py "
              "output) for exact v12_m5 reproduction.")
        return {}
    df = pd.read_csv(path)
    id_col = next((c for c in df.columns
                   if any(k in c.lower() for k in ("image", "file", "id"))),
                  df.columns[0])
    scale_col = None
    for c in df.columns:
        l = c.lower()
        if l in ("mm_per_px", "mmppx", "mm_per_pixel", "scale"):
            scale_col = c
            break
    if scale_col is None:
        for c in df.columns:
            l = c.lower()
            if "mm" in l and "px" in l:
                scale_col = c
                break
    if scale_col is None:
        for c in df.columns:
            if "marker" in c.lower() and "px" in c.lower():
                df = df.assign(mm_per_px=20.0 / df[c].astype(float))
                scale_col = "mm_per_px"
                break
    if scale_col is None:
        sys.exit(f"ERROR: no mm/px scale column found in {path}")
    lookup = {}
    for _, row in df.iterrows():
        iid = str(row[id_col])
        lookup[iid] = float(row[scale_col])
        lookup[Path(iid).stem] = float(row[scale_col])  # extensionless alias
    print(f"[calib] {path}: {len(df)} rows, scale column '{scale_col}'")
    return lookup


def mm_per_px(image_id, gray, calib):
    """PNG family: 20 / marker_px (measured, from the CSV).
    tif families: tick-structure priors by pixel format."""
    for key in (image_id, Path(image_id).stem):
        if key in calib:
            return calib[key]
    h, w = gray.shape[:2]
    if image_id.lower().endswith((".tif", ".tiff")):
        if (h, w) in TIF_PRIORS:
            return TIF_PRIORS[(h, w)]
        if (w, h) in TIF_PRIORS:
            return TIF_PRIORS[(w, h)]
        print(f"[warn] tif {image_id} ({w}x{h}) not in priors/CSV; "
              "using 0.0656")
        return 0.0656
    # PNG crude fallback: depth_mm / 604 (almost never fires when the CSV
    # is in place)
    print(f"[warn] PNG {image_id} missing from calibration CSV; using "
          f"depth fallback {PNG_DEPTH_FALLBACK_MM}/604")
    return PNG_DEPTH_FALLBACK_MM / 604.0


# ======================================================================
# Burst averaging (chronicle 2.6 / 8.3): 32 px thumbnails, Pearson
# r > 0.99, union-find; mean of predictions within each near-duplicate run
# ======================================================================

def _thumb(gray):
    t = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    sd = t.std()
    if sd < 1e-9:
        return None
    return (t - t.mean()) / sd


def burst_groups(image_ids, grays):
    parent = {i: i for i in image_ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    thumbs = {iid: _thumb(grays[iid]) for iid in image_ids}
    ids = list(image_ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ta, tb = thumbs[ids[i]], thumbs[ids[j]]
            if ta is None or tb is None:
                continue
            r = float(np.corrcoef(ta.ravel(), tb.ravel())[0, 1])
            if r > BURST_R:
                union(ids[i], ids[j])
    groups = defaultdict(list)
    for iid in ids:
        groups[find(iid)].append(iid)
    return list(groups.values())


# ======================================================================
# Main pipeline
# ======================================================================

def main():
    print(f"[device] {DEVICE}")
    test_files = discover_test_images(DATA_DIR)

    # --- load models (single ApoNet; the v15 ensemble is REVERTED) ------
    apo_model = load_model(APONET_CKPT)
    fasc_models = [load_model(c) for c in FASC_CKPTS]
    fallback_model = load_model(FASC_FALLBACK_CKPT)
    print(f"[models] ApoNet v1 x1 @ {APO_RES}, FascNet v3 x{len(fasc_models)} "
          f"@ {FASC_RES}, fallback v1 @ {FASC_FALLBACK_RES}")

    calib = load_calibration(CALIB_CSV)

    # --- retry chain (chronicle Table 9): degrade toward simplicity ------
    def mask_v3_ens_tta(gray):
        probs = [predict_prob(m, gray, FASC_RES, tta=True) for m in fasc_models]
        return fascicle_mask_from_prob(float(np.mean(probs, axis=0)))

    def mask_v1_tta(gray):
        return fascicle_mask_from_prob(
            predict_prob(fallback_model, gray, FASC_FALLBACK_RES, tta=True))

    def mask_v1(gray):
        return fascicle_mask_from_prob(
            predict_prob(fallback_model, gray, FASC_FALLBACK_RES, tta=False))

    chain = [("v3_ens_tta", mask_v3_ens_tta),
             ("v1_tta", mask_v1_tta),
             ("v1", mask_v1)]

    # --- per-image inference ---------------------------------------------
    results = {}   # image_id -> [pa_deg, fl_mm, mt_mm]
    levels = defaultdict(int)
    grays = {}
    failed_ids = []

    for k, (iid, path) in enumerate(test_files):
        gray = load_image(path)
        grays[iid] = gray
        apo_mask = apo_mask_from_prob(
            predict_prob(apo_model, gray, APO_RES, tta=False))

        got = None
        level = None
        for level_name, mask_fn in chain:
            try:
                pa_raw, fl_px, mt_px = extract_geometry(apo_mask, mask_fn(gray))
                got = (pa_raw, fl_px, mt_px)
                level = level_name
                break
            except GeometryFail:
                continue
        if got is None:
            levels["constants"] += 1
            failed_ids.append(iid)
            results[iid] = [CONST_PA, CONST_FL, CONST_MT]  # final values as-is
        else:
            levels[level] += 1
            scale = mm_per_px(iid, gray, calib)
            pa_raw, fl_px, mt_px = got
            pa = PA_A + PA_B * pa_raw          # shrinkage inverse (v9b)
            fl = fl_px * scale * FL_SCALE      # calibration + global FL scale
            mt = mt_px * scale
            results[iid] = [pa, fl, mt]
        if (k + 1) % 25 == 0 or k + 1 == len(test_files):
            print(f"[infer] {k + 1}/{len(test_files)} "
                  f"(levels: {dict(levels)})")

    # --- burst averaging (after calibration & corrections, Table 9) -------
    groups = burst_groups([iid for iid, _ in test_files], grays)
    n_burst = sum(1 for g in groups if len(g) > 1)
    for g in groups:
        if len(g) > 1:
            for col in range(3):
                m = float(np.mean([results[iid][col] for iid in g]))
                for iid in g:
                    results[iid][col] = m
    print(f"[burst] {len(groups)} groups, {n_burst} multi-image bursts averaged")

    # --- range clipping (Table 9: guards, never changes a healthy value) --
    for iid in results:
        pa, fl, mt = results[iid]
        results[iid] = [min(max(pa, CLIP_PA[0]), CLIP_PA[1]),
                        min(max(fl, CLIP_FL[0]), CLIP_FL[1]),
                        min(max(mt, CLIP_MT[0]), CLIP_MT[1])]

    # --- CSV transform: tif rows FL and MT *= 0.95 (the v12_m5 stage) -----
    for iid in results:
        if iid.lower().endswith((".tif", ".tiff")):
            results[iid][1] *= TIF_SCALE
            results[iid][2] *= TIF_SCALE

    # --- write submission --------------------------------------------------
    with open(OUT_CSV, "w", newline="") as f:
        wcsv = csv.writer(f)   # comma-separated (the semicolon episode is over)
        wcsv.writerow(["image_id", "pa_deg", "fl_mm", "mt_mm"])
        for iid, _ in test_files:
            pa, fl, mt = results[iid]
            wcsv.writerow([iid, f"{pa:.6f}", f"{fl:.6f}", f"{mt:.6f}"])

    # --- final assertions ---------------------------------------------------
    chk = pd.read_csv(OUT_CSV)
    assert len(chk) == N_EXPECTED, f"expected {N_EXPECTED} rows, got {len(chk)}"
    assert list(chk.columns) == ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    assert chk["pa_deg"].between(*CLIP_PA).all()
    assert chk["fl_mm"].between(CLIP_FL[0] * TIF_SCALE - 1e-9, CLIP_FL[1]).all()
    assert chk["mt_mm"].between(CLIP_MT[0] * TIF_SCALE - 1e-9, CLIP_MT[1]).all()

    print(f"\n[done] wrote {OUT_CSV}: {len(chk)} rows")
    print(f"[levels] {dict(levels)}")
    if failed_ids:
        print(f"[constants-fallback images] ({len(failed_ids)}): "
              + ", ".join(failed_ids[:10])
              + (" ..." if len(failed_ids) > 10 else ""))
    print("[verify] v12_m5 configuration: single ApoNet v1 @512 | v3 folds 0+1 "
          "@640 | PA=7.54+0.59*raw | FL x0.925 | burst r>0.99 | clip 5-45/"
          "30-200/10-50 | tif FL,MT x0.95 | fallback 18.0/80.0/23.0")


if __name__ == "__main__":
    main()
