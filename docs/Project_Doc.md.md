# UMUD Challenge — Muscle Architecture in Ultrasound: Complete Project Documentation

## 1. Problem

Predict three muscle-architecture parameters from a single 2D B-mode ultrasound image (309 test images):

| **Target**   | **Meaning**                                                                                       | **Unit** | **Tolerance (metric divisor)** |
| ------------ | ------------------------------------------------------------------------------------------------- | -------- | ------------------------------ |
| **`pa_deg`** | Pennation angle — angle between a fascicle and the deep aponeurosis                               | degrees  | 6                              |
| **`fl_mm`**  | Fascicle length — along the fascicle line between aponeuroses (linear extrapolation if off-frame) | mm       | 12                             |
| **`mt_mm`**  | Muscle thickness — perpendicular distance between aponeuroses, averaged at 3 locations            | mm       | 3                              |

**Metric:** **`score = mean(MAE_PA/6, MAE_FL/12, MAE_MT/3)`**, lower is better. Official scoring code was obtained from the competition page and verified locally. 33% public / 67% private split; 5 submissions/day; external UMUD-indexed data and public pretrained models allowed (must be documented); top-3 gated on open-source/FAIR/reproducibility review.

**Reference protocol:** 2 raters × 3 fascicles/3 PAs/3 MTs per image, averaged; inter-rater discrepancies >10mm FL / 4° PA / 1mm MT were re-evaluated.

## 2. Data Forensics (all discoveries, in order)

Data root: **`Data/`** with folders **`apo_imgs_v1/`** (1,048), **`apo_masks_v1/`** (1,048), **`fasc_imgs_v1/`** (2,761), **`fasc_masks_v1/`** (2,761), **`test_images_v2/`** (309), **`sample_submission.csv`** (stub, 2 rows, semicolon-separated; grader requires comma-separated **`image_id,pa_deg,fl_mm,mt_mm`**).

1. **Folder nesting & name typos** are handled by auto-discovery (classify by path substring, shallowest-first).
2. **The "db" files were Windows Thumbs.db thumbnail caches** — not metadata. No hidden data existed.
3. **Masks are binary {0,255}.** Fasc masks = *thin 1px fascicle lines* (\~0.2% of pixels); apo masks = *thick aponeurosis bands* (\~2–4%).
4. **Image↔mask size mismatches** with different aspect ratios: masks are stored at (near-)native annotation resolution; alignment is a plain bilinear stretch (validated by intensity-under-mask contrast tests and visually on twins).
5. **Hash audit (`content_hashes.csv`):** fasc pool 670 exact duplicate image+mask pairs (host-confirmed on the forum; recommended removal before CV); apo pool 19 duplicate image pairs with *different* masks (two alternating annotation conventions — all 38 files dropped); test 309 unique, **zero content overlap with training**.
6. **Zero content overlap between the fasc and apo pools** (2,091 + 1,029 = 3,120 unique; filename overlap of 1,048 was pure name collision from independent indexing). No image carries both mask types.
7. **122 content twins:** fasc images with a byte-near-identical (r=1.0000, pixel diff 0.09) apo-pool counterpart, same size. Each twin = real fascicle mask + real aponeurosis mask on identical content → **the co-labeled validation set** (**`twins.csv`**). Both models' training excludes them (honest holdout).
8. **Near-duplicate clusters** (thumbnail correlation > 0.99, union-find): training contains video sequences (clusters of 576, 273, …) → CV must group by cluster. Test contains **28 burst runs, 138 images** (27×5 + 1×3) → burst-averaging of predictions.
9. **Test UI layout families:** 29 groups by burned-in UI signature (**`calibration_v1.csv`**). Big families: fam 0 (n=84, 1200×800), fam 22 (n=80), fam 26–28 (n=58 PNG), fam 20 (n=40), fam 21 (n=5), fam 24 (n=10).
10. **Test devices/muscles differ from training** (Philips Lumify, Rectus femoris, cerebral palsy cohort unseen in training). Cross-family prediction tables revealed the muscle map: fam 0/22 = VL-like, fam 20/21/24 = RF-like (long FL, low PA, thick MT), PNG family = GM-like.
11. **Training fascicle angle histogram** (from mask components): 0–5°: 2,824; 5–10°: 11,899; 10–15°: 19,527; 15–20°: 12,034; 20–25°: 2,831; >25°: \~552 (**1.1%**) — the steep-angle blind spot, motivating v4's ±30° rotation augmentation.

## 3. Method — the full pipeline

### 3.1 Models

- **ApoNet** (**`aponet_v1.pt`**): U-Net, ResNet34 encoder (ImageNet), 1-channel in, 512×512, BCE+Dice, AdamW 3e-4, cosine, 40 epochs, batch 12, AMP. Training set: 888 unique apo images (hash-dedup, minus 19 corrupted pairs, minus twin apo images). Augment: rotate ±15°, hflip 50%, vflip 20%, contrast/brightness ±20%, noise. Val (twins, cross-pool): **dice 0.82** — the strategy's existential validation.
- **FascNet v1** (**`fascnet_v1.pt`**): same arch, targets = fascicle lines dilated 2×(3×3) at native res, 512×512, 1,838 unique nonempty fasc images (dedup, nonempty via **`fasc_mask_inventory.csv`** frac>0.001, twins held out). Val dice 0.35 (thin-line dice is pessimistic; geometry is the real metric).
- **FascNet v3** (**`fascnet_v3_fold{0,1}.pt`**, 2 folds): 640×640, **dilation-1 targets** (thinner → less merge-smear → truer lines), batch 8, 50 epochs, 2/3-data folds, seeds natural. Fold 2 collapsed twice (early best-checkpoint fluke, then mediocrity) — **dropped permanently; hot mid-schedule checkpoints poison ensemble averages**. Twin marks (2-fold): corr 0.60, PA MAE 2.96, bias −1.12, FL bias +37px (v2's was +110), FL MAE 245px, MT MAE 24.8px.
- **FascNet v4** (in flight): v3 recipe + rotation ±30° (targets the 1.1%-steep blind spot).
- **ApoNet ensemble** (**`train_aponet_ens.py`**, 2 seeds): optional MT refinement.

### 3.2 Prediction

Per model at **its own training resolution** (critical — v3 must predict at 640): resize → normalize ((x/255−0.5)/0.5) → sigmoid → average over model list → **horizontal-flip TTA average** → resize probability to original size → threshold 0.5 → **binary_closing (3×3)** (bridges 1–2px gaps without thickening). Retry chain: v3-ensemble → fascnet_v1+TTA → fascnet_v1 no-TTA → constants (18.0, 80.0, 23.0).

### 3.3 Geometry (the measurement core)

**text**

svg

1\. Aponeuroses: 2 largest components (≥500px, 8-connectivity), centroids ≥60px apart.

   Upper = superficial, lower = deep. Fit y = a + b·x (polyfit) to each band's pixels.

2\. Fascicle candidates: components ≥80px, angle vs deep line in [6°, 60°],

   elongation (sqrt(λ1/λ2)) ≥ 4, centroid between bands (sup_cy+20 < cy < deep_cy−20).

3\. Mode-cluster: fascicles within a muscle are roughly parallel. Find angle A\* maximizing

   Σ(area·sin(angle)) within ±3.5° windows; keep components within ±3.5° of A\*;

   take top 3 by area. (sin-weighting prevents large flat junk from winning.)

4\. PA = mean angle of selected fascicles vs deep line.

5\. FL = mean distance between the fascicle line's intersections with both aponeurosis

   lines (line-through-centroid along PCA direction; solves 2×2 system per aponeurosis).

   Extrapolation is inherent — lines extend to the aponeuroses off-frame.

6\. MT = perpendicular distance between aponeurosis lines, mean at x = 25/50/75% width.

7\. Consistency filter: keep fascicles with 0.4 ≤ FL·sin(PA)/MT ≤ 2.5 (geometric identity

   for straight fascicles; validated median 1.04 on real masks).

**Critical implementation note:** **`scipy.ndimage.label`** default is 4-connectivity — diagonal 1px lines shatter. Always use **`structure=np.ones((3,3))`**.

### 3.4 Calibration (pixels → mm)

- **PNG family (58 images, per-image):** left ruler has labeled 2cm markers. **`mm/px = 20 / marker_spacing`** where markers = long ticks (length ≥ 60% of max) from the tick table at x=6; validity window spacing ∈ [120, 420]px. Fallback: **`depth_mm / 604`** (validated constant ruler span). Depth from OCR ("Tiefe X.X cm", Tesseract, 3× upscale + autocontrast, psm 6).
- **Tif families (251 images, prior):** **`(1200,800) → 0.0656 mm/px`**, **`(1088,644) → 0.0830`**, else 0.070. Derived from fam-0 tick structure (9 ticks / 152.5px long-spacing → 10mm majors) and the De-50mm family; never per-family verified — A/B tests pending.
- PA needs no calibration; FL/MT use the vertical scale isotropically (A/B-tested: anisotropy correction ×0.85 rejected).
- **Known failure:** twin (training) images are cropped clean — no UI text, no calibration (OCR probe reads speckle garbage).

### 3.5 Post-processing

1. **PA correction (twin-fitted, LB-validated):** **`pa = 7.23 + 0.59 · pa_raw`** (v9b experiment; twin shrinkage fit; refit after model changes). Fallback rows keep 18.0.
2. **FL_SCALE = 0.925** (LB-validated, stable across model generations; twin bias maps \~2:1 to test).
3. **Burst averaging:** mean of (PA, FL, MT) within each r>0.99 adjacent-frame run.
4. **Clip** to physiological ranges: PA [5,45], FL [30,200], MT [10,50]. Never NaN/inf (grader rejects).

## 4. Validation methodology — the twins

**`evaluate_twins.py`** is the instrument: for 122 held-out twins, run the full prediction chain, compare against geometry on **real** masks (loose filters: angle [2,70], top-3 by area). Outputs: PA/FL/MT MAE + bias (pixel space), shrinkage fit **`gt = A + B·pr`**, worst-6 with per-component angle lists, QA overlays (real fascicles green, predicted red, real apo cyan, predicted magenta). Current marks to beat listed in §3.1. The twins exaggerate test errors \~2× (hard single-muscle domain) — directionally reliable, magnitudes halve on the LB.

## 5. Experimental log (submissions)

| **#** | **Submission** | **Config**                                                | **Score**   | **Verdict**                              |
| ----- | -------------- | --------------------------------------------------------- | ----------- | ---------------------------------------- |
| 1     | constants      | PA 18, FL 80, MT 23                                       | 1.11370     | floor anchor                             |
| 2     | v2             | first full pipeline (v1 models)                           | 0.928       | —                                        |
| 3     | v3             | + PNG calibration fix (marker/604)                        | 0.736       | +0.19, biggest calibration win           |
| 4     | v4             | 3-fold v2 ensemble + flip TTA                             | 0.71269\*   | marginal — correlated folds              |
| 5     | v6b            | FL×0.925                                                  | 0.68508     | FL bias was real                         |
| 6     | v7             | +no-TTA retry (recovered 5 images)                        | 0.66877     | —                                        |
| 7     | v9             | geometry v3: floor 6°, sin-weighted mode cluster, closing | 0.64848     | —                                        |
| 8     | v9a            | PA mean-centered un-shrink (×1/0.59)                      | 0.76693     | **REJECTED** — test wants lifting        |
| 9     | v9b            | PA = 7.54+0.59·pr (twin fit)                              | 0.59136     | **WINNER** — twin bias transfers         |
| 10    | v10            | FL × sin(PA_old)/sin(PA_new)                              | 0.66694     | **REJECTED** — FL is noise, not coupling |
| 11    | v11            | v3 models + FL_SCALE 0.97                                 | 0.59497     | neutral (bundled-change ambiguity)       |
| 12    | v11b           | v3 models + FL_SCALE 0.925                                | **0.57566** | **current best, rank 160**               |

\*One rejected submission (299 rows, mangled edit) — caught by the in-script **`assert len(recs)==309`** guard added afterward.

A/B discipline throughout: single-variable transforms of the current CSV, decision rules declared before submission.

## 6. Negative results (do not retry)

1. **4-connectivity component analysis** — shatters 1px diagonal lines; the "empty twin masks" was this bug.
2. **vh (vertical-extent ≥ 0.35×muscle-height) fascicle filter** — predictions are fragments, not spanning lines; killed 121/122 images.
3. **Geometric constraint projection** (FL from MT/sin(PA)) — error propagation: 2.5° PA → \~11mm FL; worse than direct.
4. **FL sin-ratio coupling correction** (v10) — FL error is per-image noise; global 0.925 already absorbed the bias.
5. **PA mean-centered un-shrink** (v9a) — −0.12; the twin offset (bias) transfers, the spread hypothesis doesn't.
6. **Correlated-fold ensembling** (v2×3) — predictions moved \~1°; buy diversity, not replicas.
7. **Hot mid-schedule checkpoints** (fold 2, twice) — spiky probabilities poison ensemble averages; only converged (late-cosine) checkpoints.
8. **Evaluation at wrong resolution** (v3 at 512) — fully-convolutional nets accept any size but scale-calibrated confidence breaks; each model predicts at its training resolution.
9. **Test-image pseudo-label fine-tuning** — rejected on paper: reinforces regression-to-mean errors on the domain-shifted images.
10. **Anisotropy-corrected FL** (×0.85 global test) — neutral; isotropic vertical scale matches the raters.

## 7. Artifact manifest (project root)

**Dataframes:** **`content_hashes.csv`**, **`twins.csv`**, **`fasc_mask_inventory.csv`**, **`calibration_v1.csv`**, **`twin_eval.csv`**, **`geometry_v2_twins.csv`**, **`thumbs32.npz`**.
**Checkpoints:** **`fascnet_v1.pt`**, **`fascnet_v2_fold{0,1,2}.pt`**, **`fascnet_v3_fold{0,1,2}.pt`** (fold2 broken), **`aponet_v1.pt`**; v4/aponet_ens pending.
**Scripts:** **`data_analysis*.py`** (forensics), **`dd_calibration.py`**, **`training_seg.py`** (v1), **`train_fascnet_v2/v3/v4.py`**, **`train_aponet_ens.py`**, **`infer_submit.py`** (the pipeline), **`evaluate_twins.py`** (the instrument), **`diag_geo.py`**, **`classical_pa.py`**, **`make_*.py`** (A/B transforms).
**Environment:** Python 3.11, torch 2.10+cu128 (RTX 5050 — import torch FIRST on this machine: DLL load-order), smp, scipy, pytesseract (+ UB-Mannheim Tesseract engine), pandas, PIL.

## 8. Replication order

1. Forensics: hash audit → twins.csv → inventory → calibration_v1.csv (scripts above).
2. Train ApoNet (training_seg.py, MODE="apo") → 0.82 cross-pool val.
3. Train FascNet v1 → v3 2-fold (640, dil-1) → verify with evaluate_twins.py against §3.1 marks.
4. Calibration: dd_calibration.py → get_mm_px (marker/604/prior).
5. Pipeline: infer_submit.py (with PA_A/PA_B from the current twin fit, FL_SCALE 0.925) → submission CSV.
6. A/Bs: CSV transforms per §5, single-variable.
7. Iterate models against the twin harness; submit only twin-validated improvements.

## 9. Standing insights

- **Instrument before submitting.** The 122 twins cost zero submissions and predicted every LB verdict (v9b won exactly as the fit implied; v9a/v10 died as the error structure implied).
- **Post-processing has a ceiling** — reached at FL 0.925 + PA correction. Beyond it: model precision only.
- **Error decomposition drives priority:** current ≈ PA 2.8° / FL 9.5mm / MT 1.5mm; all three are model-scatter now.
- **Dice is vanity, geometry is sanity** — 0.35-dice masks produced consistency ratios of 0.98.
- **The LB is 33% public** — global constants transfer; per-family tuning is where overfitting risk lives.

## 10. Open items / roadmap

v4 harness verdict → v12; distilled direct regressor (image→PA/FL/MT on pipeline pseudo-labels, twins-held-out, blended — the centerpiece of the 0.35 push); classical structure-tensor PA second opinion; calibration A/Bs on the 251 prior-family images; ApoNet ensemble; external labeled-data hunt via UMUD (the only 0.35-adjacent route); FAIR repo assembly.