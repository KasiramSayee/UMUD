# UMUD Muscle Architecture Challenge - Kaggle Competition Repository

This repository contains the complete solution for the **UMUD Muscle Architecture Challenge** (Kaggle), which predicts three muscle-architecture parameters from 2D B-mode ultrasound images:
- **PA** (Pennation Angle) - degrees
- **FL** (Fascicle Length) - mm
- **MT** (Muscle Thickness) - mm

## Repository Structure

```
UMUD/
├── models/                          # All .pt model checkpoints (13 files)
│   ├── aponet_v1.pt
│   ├── aponet_ens_0.pt
│   ├── aponet_ens_1.pt
│   ├── fascnet_v1.pt
│   ├── fascnet_v2_fold0.pt
│   ├── fascnet_v2_fold1.pt
│   ├── fascnet_v2_fold2.pt
│   ├── fascnet_v3_fold0.pt
│   ├── fascnet_v3_fold1.pt
│   ├── fascnet_v3_fold2.pt
│   ├── fascnet_v4_fold0.pt
│   ├── fascnet_v4_fold1.pt
│   └── regressor_v1.pt
│
├── scripts/
│   ├── training/                    # Model training scripts
│   │   ├── training_seg.py          # ApoNet v1 / FascNet v1
│   │   ├── train_fascnet_v2.py
│   │   ├── train_fascnet_v3.py
│   │   ├── train_fascnet_v4.py
│   │   ├── train_aponet_ens.py
│   │   └── train_regressor.py
│   │
│   ├── inference/                   # Inference & submission generation
│   │   ├── infer_submit.py          # Main pipeline (v12_m5 - LB best 0.510)
│   │   ├── infer_submit_original.py
│   │   ├── submission_5.py
│   │   ├── submission_6_b.py
│   │   ├── submission_11b.py
│   │   └── gen_pseudo_labels.py
│   │
│   ├── analysis/                    # Data forensics & exploration
│   │   ├── data_analysis.py
│   │   ├── data_analysis_2.py
│   │   ├── data_analysis_3.py
│   │   ├── data_analysis_4.py
│   │   ├── data_correction.py
│   │   ├── geometry.py
│   │   ├── sample.py
│   │   └── sample_2.py
│   │
│   ├── calibration/                 # Pixel-to-mm calibration
│   │   ├── dd_calibration.py
│   │   ├── bisection_calibration.py
│   │   ├── bisection_calibration_v2.py
│   │   └── calibration_ab.py
│   │
│   ├── evaluation/                  # Validation & diagnostics
│   │   ├── evaluate_twins.py
│   │   ├── eval_regressor_twins.py
│   │   ├── diag_geo.py
│   │   └── AA-BB.py
│   │
│   └── postprocessing/              # A/B transforms & blending
│       ├── make_v9ab.py
│       ├── make_v10.py
│       ├── apply_classical_blend.py
│       ├── classical_pa.py
│       └── audit_revert.py
│
├── submissions/                     # All submission CSVs (23 files)
│   ├── submission_v2.csv through submission_v16.csv
│   ├── submission_v9a.csv, submission_v9b.csv
│   ├── submission_v10.csv
│   ├── submission_v11b.csv
│   ├── submission_v12_m*.csv
│   ├── submission_v15_blended.csv
│   └── submission_constants*.csv
│
├── artifacts/                       # Data artifacts & intermediate CSVs
│   ├── content_hashes.csv
│   ├── twins.csv
│   ├── twin_eval.csv
│   ├── fasc_mask_inventory.csv
│   ├── calibration_v1.csv
│   ├── geometry_v0_twins.csv
│   ├── geometry_v1_twins.csv
│   ├── geometry_v2_twins.csv
│   ├── pseudo_labels.csv
│   ├── regressor_stats.csv
│   └── thumbs32.npz
│
├── docs/                            # Documentation & reference images
│   ├── project.md
│   ├── Project_Doc.md.md
│   ├── UMUD_Muscle_Architecture_Challenge_Project_Chronicle.pdf
│   ├── apo_sheet.png
│   ├── fasc_sheet.png
│   └── test_sheet.png
│
├── outputs/                         # QA overlays (predQA_*, twinQA_*, geomQA_*, etc.)
│
├── Data/                            # Training/test images & masks (unchanged)
│   ├── apo_imgs_v1/
│   ├── apo_masks_v1/
│   ├── fasc_imgs_v1/
│   ├── fasc_masks_v1/
│   └── test_images_v2/
│
└── Competition Question Images/     # Challenge reference images
```

## Quick Start

### Requirements
- Python 3.11+
- PyTorch 2.10+ with CUDA (RTX 5050 tested)
- segmentation_models_pytorch (smp)
- scipy, pandas, PIL, pytesseract (for calibration)
- Tesseract OCR engine (UB-Mannheim installer)

### Environment Setup
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install segmentation_models_pytorch scipy pandas pillow pytesseract opencv-python
# Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
```

### Running the Best Pipeline (v12_m5 - LB 0.510)
```bash
# From project root
python scripts/inference/infer_submit.py
```
This generates `submissions/submission_v12_m5_repro.csv` - the reference best submission.

### Training Models
```bash
# Train ApoNet v1 (aponeurosis segmentation)
python scripts/training/training_seg.py --mode apo

# Train FascNet v1 (fascicle segmentation)  
python scripts/training/training_seg.py --mode fasc

# Train FascNet v2 (3-fold, 512px)
python scripts/training/train_fascnet_v2.py

# Train FascNet v3 (3-fold, 640px, dil=1) - production folds 0,1
python scripts/training/train_fascnet_v3.py

# Train FascNet v4 (3-fold, 640px, ±30° rotation, dil=1) - folds 0,1
python scripts/training/train_fascnet_v4.py

# Train ApoNet ensemble (2 seeds)
python scripts/training/train_aponet_ens.py

# Train direct regressor (image → PA/FL/MT)
python scripts/training/train_regressor.py
```

### Data Forensics & Calibration
```bash
# Hash audit, twin detection, duplicate analysis
python scripts/analysis/data_analysis.py
python scripts/analysis/data_analysis_2.py
python scripts/analysis/data_analysis_3.py
python scripts/analysis/data_analysis_4.py

# Calibration (OCR + mechanical)
python scripts/calibration/dd_calibration.py

# Generate geometry CSV on twins
python scripts/analysis/geometry.py
python scripts/analysis/sample.py
python scripts/analysis/sample_2.py
```

### Evaluation
```bash
# Evaluate pipeline on twins (main diagnostic)
python scripts/evaluation/evaluate_twins.py

# Evaluate direct regressor on twins
python scripts/evaluation/eval_regressor_twins.py

# Diagnose geometry pipeline
python scripts/evaluation/diag_geo.py
```

### A/B Post-Processing
```bash
# Generate v9a/v9b from v9
python scripts/postprocessing/make_v9ab.py

# Generate v10 from v9/v9b
python scripts/postprocessing/make_v10.py

# Classical PA blend
python scripts/postprocessing/apply_classical_blend.py

# Audit for post-v12_m5 remnants
python scripts/postprocessing/audit_revert.py
```

## Best Configuration (v12_m5)

From the project chronicle (Ch.11 Table 9 + Appendix B Table 11):

| Component | Configuration |
|-----------|---------------|
| **Fascicle masks** | FascNet v3 folds 0+1, 640px, prob avg, H-flip TTA, thresh 0.5, binary closing |
| **Apo mask** | ApoNet v1 (single seed), 512px, thresh 0.5, no closing |
| **Geometry** | polyfit deg-1 LSQ, component filters, sin-weighted mode clustering (±3.5°), top-3 by area, line-intersection FL, 3-site MT, consistency 0.4-2.5 |
| **Calibration** | Per-image from calibration_v1.csv (PNG: 20/marker_px; TIF priors 0.0656/0.0830) |
| **PA correction** | PA = 7.54 + 0.59 × PA_raw (twin shrinkage inverse) |
| **FL scale** | FL ×= 0.925 |
| **Burst averaging** | Mean over near-duplicate runs (Pearson r > 0.99) |
| **Clipping** | PA 5-45°, FL 30-200mm, MT 10-50mm |
| **CSV transform** | TIF rows: FL, MT ×= 0.95 |
| **Fallback** | v3 ens+TTA → v1+TTA → v1 → constants (18.0°/80.0mm/23.0mm) |

## Documentation

- **project.md** - Complete project documentation with all discoveries, methods, and results
- **Project_Doc.md.md** - Formatted version of the above
- **UMUD_Muscle_Architecture_Challenge_Project_Chronicle.pdf** - Full competition chronicle
- **docs/*.png** - Reference annotation sheets (apo, fasc, test)

## Outputs

The `outputs/` directory contains QA overlay images generated during training/evaluation:
- `predQA_*` - Model prediction overlays
- `twinQA_*` - Twin evaluation overlays  
- `geomQA_*` - Geometry visualization overlays
- `calibQA_*` - Calibration overlays

## License

This repository is for the UMUD Kaggle competition. See competition rules for submission requirements.

---

*Repository organized per the UMUD reorganization plan. All scripts use universal path constants (PROJECT, DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, SUBMISSIONS_DIR, OUTPUTS_DIR) for portability.*