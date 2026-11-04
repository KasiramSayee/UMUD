#!/usr/bin/env python3
"""
audit_revert.py -- scan your local infer_submit.py for post-v12_m5 remnants
============================================================================

The best configuration is v12_m5 (leaderboard 0.510).  Three change-sets
were made AFTER it and must be fully undone:

    v14 (0.523):  robust Huber apo fitting + inline calibration + PA refit
    v15 (0.628):  ApoNet 2-seed ensemble (aponet_ens_0/1.pt)
    v19 (0.628):  classical structure-tensor PA blend (circular mean)

v16 (0.521) only reverted the ensemble -- the v14 changes appear to have
lingered, which is the most likely reason it did not recover to 0.510.

Scanning is code-aware: comments and docstrings are stripped first, so
prose that merely *mentions* a reverted experiment is not flagged.  Only
live code can be a remnant.

Usage (pure stdlib, runs anywhere):
    python audit_revert.py                 # scans ./infer_submit.py
    python audit_revert.py path/to/file.py

Exit codes: 0 = clean, 1 = remnants/missing constants found.
Lines are printed with numbers so you can jump straight to them.
"""

import re
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# REMNANTS: patterns that must NOT appear in live v12_m5 code
# ----------------------------------------------------------------------
REMNANTS = [
    ("v15/v18 ApoNet ensemble",
     [r"aponet_ens", r"apo_ens_\d", r"ens_0\.pt", r"ens_1\.pt",
      r"aponet.{0,12}ensemble", r"two[-_ ]?seed"]),
    ("v14 robust/Huber apo fitting",
     [r"HuberRegressor", r"[Hh]uber[_A-Za-z]*\s*\(", r"huber",
      r"robust_fit", r"IRLS", r"irls", r"TheilSen"]),
    ("v19 classical structure-tensor blend",
     [r"structure[_ ]?tensor", r"classical[_ ]?pa", r"circular[_ ]?mean",
      r"classical_blend", r"blend_pa", r"apply_classical"]),
]

# Soft patterns: worth eyeballing when present in live code
SOFT = [
    ("possible v14 built-in-calibration path",
     [r"inline[_ ]cal", r"calibrate_inline", r"measure_markers"]),
    ("possible blend leftovers",
     [r"\bblend", r"\bclassical\b"]),
]

# ----------------------------------------------------------------------
# v12_m5 constants (chronicle Appendix B, Table 11).  Each entry lists
# (regex, capture-group) attempts so both single-line and tuple-style
# assignments are recognized, e.g.  PA_A, PA_B = 7.54, 0.59
# ----------------------------------------------------------------------
CONST_EQ = [
    ("PA_A (PA = PA_A + PA_B*raw)", 7.54,
     [(r"PA_A\s*,\s*PA_B\s*=\s*([0-9.]+)\s*,", 1),
      (r"PA_A\s*=\s*([0-9.]+)", 1)]),
    ("PA_B", 0.59,
     [(r"PA_A\s*,\s*PA_B\s*=\s*[0-9.]+\s*,\s*([0-9.]+)", 1),
      (r"PA_B\s*=\s*([0-9.]+)", 1)]),
    ("FL_SCALE", 0.925,
     [(r"FL_SCALE\s*=\s*([0-9.]+)", 1)]),
    ("TIF_SCALE (tif FL/MT multiplier)", 0.95,
     [(r"TIF_SCALE\s*=\s*([0-9.]+)", 1)]),
    ("Burst threshold r", 0.99,
     [(r"BURST_R\s*=\s*([0-9.]+)", 1),
      (r"\b0\.99\b", 0)]),
    ("Mode cluster half-width", 3.5,
     [(r"MODE_HALF_WIDTH\s*=\s*([0-9.]+)", 1),
      (r"\b3\.5\b", 0)]),
    ("Apo min size", 500,
     [(r"APO_MIN_SIZE\s*=\s*([0-9]+)", 1)]),
    ("Apo centroid separation", 60,
     [(r"APO_MIN_SEPARATION\s*=\s*([0-9]+)", 1)]),
    ("Fascicle min size", 80,
     [(r"FASC_MIN_SIZE\s*=\s*([0-9]+)", 1)]),
    ("Min elongation", 4.0,
     [(r"MIN_ELONGATION\s*=\s*([0-9.]+)", 1)]),
    ("Fallback const PA", 18.0,
     [(r"CONST_PA\s*,\s*CONST_FL\s*,\s*CONST_MT\s*=\s*([0-9.]+)\s*,", 1),
      (r"CONST_PA\s*=\s*([0-9.]+)", 1)]),
    ("Fallback const FL", 80.0,
     [(r"CONST_PA\s*,\s*CONST_FL\s*,\s*CONST_MT\s*=\s*[0-9.]+\s*,\s*([0-9.]+)", 1),
      (r"CONST_FL\s*=\s*([0-9.]+)", 1)]),
    ("Fallback const MT", 23.0,
     [(r"CONST_PA\s*,\s*CONST_FL\s*,\s*CONST_MT\s*=\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*([0-9.]+)", 1),
      (r"CONST_MT\s*=\s*([0-9.]+)", 1)]),
    ("ApoNet resolution", 512,
     [(r"APO_RES\s*=\s*([0-9]+)", 1)]),
    ("FascNet v3 resolution", 640,
     [(r"FASC_RES\s*=\s*([0-9]+)", 1)]),
]

MUST_HAVE = [
    ("Single ApoNet v1 checkpoint",       r"aponet_v1\.pt"),
    ("FascNet v3 fold 0 checkpoint",      r"fascnet_v3_fold0\.pt"),
    ("FascNet v3 fold 1 checkpoint",      r"fascnet_v3_fold1\.pt"),
    ("FascNet v1 fallback checkpoint",    r"fascnet_v1\.pt"),
    ("Least-squares polyfit (not Huber)", r"polyfit"),
    ("tif prior 0.0656 mm/px",            r"0\.0656"),
    ("tif prior 0.0830 mm/px",            r"0\.0830"),
    ("calibration_v1.csv lookup",         r"calibration_v1\.csv"),
    ("8-connectivity labeling",           r"connectivity\s*=\s*8|structure\s*=\s*np\.ones\(\(3,\s*3\)"),
    ("Binary closing",                    r"MORPH_CLOSE|morphologyEx"),
    ("Consistency lower bound 0.4",       r"\b0\.4\b"),
    ("Consistency upper bound 2.5",       r"\b2\.5\b"),
    ("Angle floor 6 deg",                 r"\b6\.0?\b"),
    ("Angle ceiling 60 deg",              r"\b60\b"),
    ("Top-3 fascicle selection",          r"\[:3\]|\[:N_FASC|N_FASCICLES\s*=\s*3"),
    ("MT sites 25/50/75%",                r"0\.25|MT_SITES"),
    ("Clip PA 5..45",                     r"\b45\b"),
    ("Clip FL 30..200",                   r"\b200\b"),
    ("Clip MT 10..50",                    r"\b50\b"),
    ("309-row assertion",                 r"\b309\b"),
    ("Horizontal-flip TTA",               r"flip"),
    ("Depth/604 PNG fallback",            r"604"),
]


def code_lines_of(text):
    """Return [(lineno, code_text)] with comments and docstrings stripped.
    Only live code survives, so prose mentioning a reverted experiment in a
    docstring is never mistaken for the experiment itself."""
    out = []
    in_doc = False
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw
        if in_doc:
            toggles = line.count('"""') + line.count("'''")
            if toggles % 2 == 1:
                in_doc = False
            continue  # inside a docstring: drop the line
        toggles = line.count('"""') + line.count("'''")
        if toggles % 2 == 1:
            in_doc = True
            line = line.split('"""')[0].split("'''")[0]
        # strip trailing comment (naive: no # inside strings on these files
        # matters for remnant scanning)
        code = line.split("#")[0].rstrip()
        if code.strip():
            out.append((i, code))
    return out


def scan(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    clines = code_lines_of(text)
    issues = 0

    print(f"=== v12_m5 revert audit: {path} ===")
    print(f"    ({len(clines)} live code lines scanned, comments/docstrings "
          "stripped)\n")

    print("--- REMNANTS (post-v12_m5 changes that must be undone) ---")
    for label, pats in REMNANTS:
        hits = {}
        for pat in pats:
            for i, ln in clines:
                if re.search(pat, ln):
                    hits.setdefault(i, ln.strip())
        if hits:
            issues += 1
            print(f"  [X REMNANT] {label}")
            for i, ln in sorted(hits.items())[:6]:
                print(f"      line {i}: {ln[:100]}")
            if len(hits) > 6:
                print(f"      ... and {len(hits) - 6} more")
        else:
            print(f"  [ok] none: {label}")

    print("\n--- SOFT PATTERNS (in live code? eyeball these) ---")
    any_soft = False
    for label, pats in SOFT:
        hits = {}
        for pat in pats:
            for i, ln in clines:
                if re.search(pat, ln):
                    hits.setdefault(i, ln.strip())
        if hits:
            any_soft = True
            print(f"  [~ REVIEW] {label}")
            for i, ln in sorted(hits.items())[:4]:
                print(f"      line {i}: {ln[:100]}")
    if not any_soft:
        print("  [ok] none")

    print("\n--- v12_m5 CONSTANTS (chronicle Appendix B) ---")
    for label, expected, attempts in CONST_EQ:
        found = None
        for pat, grp in attempts:
            for i, ln in clines:
                m = re.search(pat, ln)
                if m:
                    val = m.group(grp) if grp else m.group(0)
                    found = (i, val)
                    break
            if found:
                break
        if found is None:
            issues += 1
            print(f"  [X MISSING] {label}: expected {expected}, not found")
        else:
            try:
                val = float(found[1])
                if abs(val - float(expected)) < 1e-9:
                    print(f"  [ok] {label} = {val}")
                else:
                    issues += 1
                    print(f"  [X WRONG]   {label}: found {val} at line "
                          f"{found[0]}, expected {expected}")
            except ValueError:
                issues += 1
                print(f"  [X WRONG]   {label}: unparseable '{found[1]}' "
                      f"at line {found[0]}")

    print("\n--- v12_m5 MUST-HAVE TOKENS ---")
    for label, pat in MUST_HAVE:
        hit = next((i for i, ln in clines if re.search(pat, ln)), None)
        if hit is None:
            issues += 1
            print(f"  [X MISSING] {label}  (pattern: {pat})")
        else:
            print(f"  [ok] {label} (line {hit})")

    print("\n" + "=" * 60)
    if issues == 0:
        print("VERDICT: CLEAN -- file matches the v12_m5 constant set and "
              "carries no post-v12_m5 remnants.")
    else:
        print(f"VERDICT: {issues} issue group(s) found. Undo the [X] items "
              "above (see REVERT_CHECKLIST.md), or replace the file with "
              "the reference reconstruction infer_submit.py shipped in "
              "this folder.")
    print("=" * 60)
    return 0 if issues == 0 else 1


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "infer_submit.py"
    if not Path(target).exists():
        sys.exit(f"ERROR: '{target}' not found. Pass the path to your "
                 "current infer_submit.py as the first argument.")
    sys.exit(scan(target))
