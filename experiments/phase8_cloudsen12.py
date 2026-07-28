#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8 (★ REAL Sentinel-2) — band-as-modality missing-band robustness measured as SAMPLED
PER-PIXEL SPECTRAL CLASSIFICATION on the expert labels of CloudSEN12-high. This is the
real-world validation the reviewers demanded (replacing the synthetic PoC / Indian-Pines-only
evidence).

*** WHAT THIS IS, AND WHAT IT IS NOT ***
CloudSEN12 is a SEMANTIC SEGMENTATION dataset. This experiment is NOT a segmentation benchmark
and must never be described as one. Every sample here is ONE PIXEL's 13-band spectrum; the
models have no receptive field, no neighbours, no texture, no boundary. Concretely: we draw
`--px-test` pixels uniformly from each test patch's real 509x509 region (default 300 of 259,081
= 0.116%), and each becomes an independent length-13 vector.
  * WHY pixelwise is the RIGHT design here, not a shortcut: the claim under test is that a
    SPECTRAL mechanism (band-as-modality tokens + wavelength PE) survives losing a band. A
    spatial encoder can inpaint a missing band from its neighbours, which would confound the
    spectral effect with spatial redundancy. Removing spatial context is what isolates the
    quantity the paper is about.
  * WHY the mIoU number is still a legitimate estimate: the model is pixelwise, so its
    full-image per-pixel IoU is a function of per-pixel tp/fp/fn rates alone. Uniform pixel
    sampling estimates those rates without bias, so the reported mIoU is a consistent estimator
    of the SAME pixelwise model's full-image per-pixel mIoU (a ratio estimator; at ~292k test
    pixels the bias is negligible against the ROI-bootstrap CI we report).
  * WHAT IT CANNOT BE COMPARED TO: published CloudSEN12 UNet/segmentation leaderboards, boundary
    IoU, object-level recall, connected-component or thin-cloud-continuity metrics. None of those
    are measurable from independent pixels, and no claim here may reference them.
Say "sampled per-pixel spectral classification on CloudSEN12 labels", never "cloud segmentation".

Data: CloudSEN12 HIGH subset (csaybar/CloudSEN12-high, CC-BY-NC-4.0 — NON-COMMERCIAL, which is a
licence the submission venue must tolerate). CloudSEN12+ (isp-uv-es) is a LATER, CC0-licensed
EXTENSION/REVISION with additional data and revised annotations — it is NOT a byte-identical
relicensed twin of CloudSEN12-high, so results are not transferable between them without a re-run.
13 Sentinel-2 L1C bands (incl. B10 cirrus) + expert manual labels (LABEL_manual_hq): 0=clear,
1=thick cloud, 2=thin cloud, 3=cloud shadow. Official disjoint-ROI split (train 8490 / val 535 /
test 975 patches, 512x512, verified on disk). We TRAIN on the train split (default: ALL 8490
patches, 300 sampled pixels each — not the full imagery) and evaluate on the geographically
disjoint TEST split — no spatial leakage.

*** PHYSICAL S2 GROUPING (fixes the group-granularity bug) ***
The paper's atomic unit is a spectral GROUP ("band-as-modality"), but the operational missing
bands (B1 coastal, B9 water-vapour, B10 cirrus) are SINGLE bands. A naive equal-size grouping
put B9 and B10 in the same group, so "drop B10" silently dropped B9 too. We instead group along
the REAL S2 spectral design and ISOLATE the operationally-relevant bands as singleton groups:
    G0=[B1]  G1=[B2,B3,B4]  G2=[B5,B6,B7]  G3=[B8,B8A]  G4=[B9]  G5=[B10]  G6=[B11,B12]
so "drop B10" == drop group {B10} == exactly one band. A runtime assertion (_assert_singleton)
GUARDS this: any regrouping that de-isolates B1/B9/B10 fails loudly instead of silently
dropping neighbours.

Missing-band scenarios evaluated (per-class IoU + mIoU, mean over >=5 seeds — ENFORCED, see
_preflight):
  clean        : all 13 L1C bands.
  dropB10      : CONTROLLED L1C->L2A — remove ONLY B10 (cirrus) from L1C reflectance.
  dropB1B9B10  : CONTROLLED atmospheric loss — remove exactly B1 + B9 + B10.
  L2A_real     : OPERATIONAL L1C->L2A — real Sen2Cor L2A surface reflectance (B10 genuinely
                 absent), fed to the L1C-trained model (real missing-band + TOA->BOA shift).
                 CloudSEN12 test ships real L2A bands; this is the true operational anchor, not
                 a simulation.

*** HOW THE TWO L1C->L2A CAUSES SPLIT (state this precisely or not at all) ***
  clean -> dropB10   : band set 13 -> 12, product held at L1C   = the missing-band effect.
  dropB10 -> L2A_real: product L1C -> L2A, band set held at 12   = the TOA->BOA shift.
The two differences SUM EXACTLY to the total clean->L2A_real drop — it is an algebraic identity
on the same fixed pixels and the same trained model, so no interaction term is unaccounted for.
But it is a SEQUENTIAL (path-ordered) decomposition, NOT a symmetric/Shapley attribution: the
reverse path needs "L2A with B10", a product that does not exist (Sen2Cor emits no B10), so the
other ordering is not merely unmeasured, it is undefined. Report it as "removing B10 costs X;
moving to real BOA reflectance costs a further Y", never as "B10 accounts for X of the damage".

Reuses the Phase-2 training primitives (SGMAE pretrain + wavelength-PE cross-band attention +
band-group dropout) via import.

*** TWO ASYMMETRIES THAT MUST BE STATED WHEN THESE NUMBERS ARE INTERPRETED ***
1. TRAIN/EVAL CORRUPTION MISMATCH. phase2's band-group dropout and SGMAE masking draw 0-3 (resp.
   1-3) missing groups per sample, but the degradation curve is evaluated out to `--max-missing`
   (default 5 of 7). The right tail of the curve is EXTRAPOLATION beyond the corruption level any
   method was trained for; it is a stress test, not an in-distribution measurement.
2. B6 -> Proposed IS A TWO-FACTOR STEP by default. B6 = learned per-group embedding + SGMAE,
   group_dropout OFF. Proposed = wavelength PE + SGMAE, group_dropout ON. Without a matched arm
   the gap cannot be attributed to the wavelength PE — the paper's headline mechanism — at all.
   `--pe-ablation` adds exactly that arm, B7 = Proposed's recipe with a learned embedding, so
       B6 -> B7       (weight-matched: same seed, same pe_type)  = the group-dropout effect
       B7 -> Proposed (same SGMAE, same dropout)                 = the wavelength-PE effect
   Costs one extra pretrain+finetune per seed. WITHOUT --pe-ablation, do not write "the wavelength
   PE gives +N mIoU" from this table; nothing in the default run supports it.

Outputs (../paper/). Every CSV and the figure is written ATOMICALLY (temp file in the same
directory + os.replace), so a killed run cannot leave a half-updated, internally inconsistent set
of deliverables. The `.provenance.json` sidecars are NOT atomic — bandsim.provenance.stamp writes
them in place, deliberately never raising, since losing a finished experiment to a failed
provenance write would be the worse outcome:
  figs/fig_degradation_cloudsen12.pdf        (mIoU vs #missing GROUPS — the paper's unit)
  figs/fig_degradation_cloudsen12_bands.pdf  (mIoU vs #missing BANDS — the information-loss axis)
  results_phase8_cloudsen12_curve.csv        (aggregate: mean +/- seed SD per #missing groups)
  results_phase8_cloudsen12_scenarios.csv    (per state: mIoU, seed SD, ROI-bootstrap CI,
                                              retention, AUDC, params, optimizer steps)
  results_phase8_cloudsen12_perclass.csv     (per-class IoU per method per state + class support)
  results_phase8_cloudsen12_paired.csv       (Proposed - baseline, paired ROI-bootstrap 95% CI)
  results_phase8_cloudsen12_bandcurve.csv    (the same evaluations on the missing-BAND axis)
  results_phase8_cloudsen12_dropsets.csv     (per m: worst/best drop set, and the spread ACROSS
                                              drop sets against the seed noise floor — i.e. does
                                              it matter WHICH groups go missing, or only how many)
  results_phase8_cloudsen12_raw_curve.csv    (RAW: seed x method x EVERY drop set, full precision)
  results_phase8_cloudsen12_raw_scen.csv     (RAW: seed x method x state x per-class IoU + support)
The two RAW files are the canonical record: every aggregate above is recomputable from them, which
is what makes the tables auditable and paired significance tests possible after the fact without
retraining anything.

*** WHAT THE TWO UNCERTAINTIES MEAN — they are not interchangeable ***
  seed SD (`*_seedsd`, and the shaded band in the figure): spread over --seeds, i.e. model init
      plus the per-seed 80% training subsample, on a FIXED test set. It contains NO test-sampling
      and NO geographic uncertainty, and it is NOT a generalisation interval.
  ROI bootstrap (`*_roi_lo/_roi_hi`, and results_..._paired.csv): resamples the 195 test ROIs, the
      only defensible independent geographic unit here — the ~292k test pixels are not independent
      (300 come from one 509x509 patch, and 5 patches share one ROI footprint on 5 dates).
Method comparisons use the PAIRED difference on a shared resample, never two separate intervals:
ROI-to-ROI mIoU varies by tens of points while methods differ by a few, so unpaired intervals
overlap almost completely and would hide a real and consistent difference.

Usage:
  python experiments/phase8_cloudsen12.py --smoke
  python experiments/phase8_cloudsen12.py --seeds 0 1 2 3 4 --epochs 40
"""
import os, sys, csv, json, argparse, hashlib, itertools, tempfile, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase2_degradation as P2
from bandsim.grouping import group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention, count_params
from bandsim.metrics import miou, per_class_iou, audc
from bandsim import hw, parallel
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))


def P(rel):
    """Resolve `rel` under ../paper, creating its directory LAZILY (on first request).

    The makedirs used to run at IMPORT time, so merely importing this module — which
    phase8R/phase8D/phase8E and three test modules all do, only ever for `load_split` — created
    paper/figs as a side effect of an import. Attach the side effect to the caller that actually
    intends to write instead."""
    out = os.path.join(PAPER_DIR, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out


_UMASK = os.umask(0); os.umask(_UMASK)   # read-and-restore: umask has no getter


def _atomic_write(path, write_fn):
    """Run `write_fn(tmp_path)`, then os.replace(tmp_path, path) — an all-or-nothing publish.

    This experiment writes five CSVs, a PDF and five provenance sidecars that are only meaningful
    AS A SET (the aggregates must be recomputable from the raw files). Written in place, a run
    killed midway leaves this-run's curve beside last-run's scenarios, with matching mtimes and no
    way to tell from the files that they disagree. Same-directory temp + os.replace makes each
    artefact's update atomic on POSIX, so a killed run leaves every file either fully old or fully
    new. On failure the temp is removed rather than left behind as a mystery dotfile."""
    d = os.path.dirname(path) or "."
    # Keep the real filename as the temp SUFFIX, not the prefix: matplotlib picks its output format
    # from the extension, so a temp named `.tmp_ab12cd` would be written as PNG bytes into a file
    # subsequently renamed to *.pdf.
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix="_" + os.path.basename(path))
    os.close(fd)
    try:
        write_fn(tmp)
        # mkstemp creates 0600. Without this every regenerated result file would silently become
        # owner-only — readable by the run that wrote it and by nothing else, on a shared box.
        # Inherit the mode the artefact already had if it exists, so a deliberate chmod survives.
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        except FileNotFoundError:
            os.chmod(tmp, 0o644 & ~_UMASK)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _write_csv(path, header, rows):
    """Atomically write one CSV. Rows are written verbatim — format numbers at the call site so
    the RAW files can keep full precision while the paper tables round."""
    def _w(tmp):
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    return _atomic_write(path, _w)

DATA = os.path.join(os.path.dirname(_HERE), "data", "cloudsen12")
# Sentinel-2 L1C band order in CloudSEN12 + centre wavelengths (nm)
L1C_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
# NOMINAL centre wavelengths, common to S2A and S2B. CloudSEN12 mixes both platforms and their
# published centres differ (e.g. B1 443.9 vs 442.3 nm) — at most ~3 nm, i.e. <0.7% of the narrowest
# band centre and far below the 60-350 nm spacing between the GROUP centres that actually reach the
# positional encoding. Sensor-specific axes are therefore deliberately NOT modelled; what is
# modelled is spectral ORDER and SPACING. For the same reason the group "wavelength" below is the
# arithmetic mean of member band centres and ignores SRF width/shape: call it a wavelength-
# conditioned encoding, never a full sensor-physics or SRF-aware encoding.
S2_WL_NM  = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]
# L2A bands present in CloudSEN12 test (Sen2Cor removes B10 -> 12 bands, mapped to L1C indices)
L2A_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
# The ONLY two products on disk. An explicit enum, because the loader used to select bands with
# `L1C_BANDS if product == "L1C" else L2A_BANDS`: anything that was not the literal "L1C" took the
# L2A band list, and only the subsequent filename lookup stopped it. That is the wrong thing to be
# relying on -- it happens to raise today because no `<typo>_B1.dat` exists, not because the loader
# checked. Name the contract instead of inheriting it from the filesystem.
PRODUCTS = {"L1C": L1C_BANDS, "L2A": L2A_BANDS}
SPLITS = ("train", "val", "test")
B1_IDX, B9_IDX, B10_IDX = L1C_BANDS.index("B1"), L1C_BANDS.index("B9"), L1C_BANDS.index("B10")
# Physical S2 grouping — atmospheric/operational bands (B1,B9,B10) isolated as singletons.
S2_PHYSICAL_GROUPS = [[0], [1, 2, 3], [4, 5, 6], [7, 8], [9], [10], [11, 12]]
NUM_CLASSES = 4
CLASS_NAMES = ["clear", "thick", "thin", "shadow"]
MIN_FORMAL_SEEDS = 5   # the docstring's ">=5 seeds" claim, enforced in _preflight (was only prose:
                       # the shipped results_phase8_cloudsen12_*.csv were produced by a 2-seed run)
# EXPLICIT little-endian int16. `np.int16` means NATIVE byte order: the .dat files were written
# little-endian on x86, so on a big-endian host every reflectance would be byte-swapped into a
# plausible-looking wrong number (e.g. 0x0064=100 -> 0x6400=25600) with nothing raising. Pin the
# on-disk contract instead of inheriting it from whatever CPU happens to read it.
DAT_DTYPE = "<i2"
SIDE = 512          # on-disk storage side (memmap reshape)
VALID_SIDE = 509    # REAL CloudSEN12-high content (metadata proj_shape, uniform across all splits).
VALID_OFF = 1       # EMPIRICALLY VERIFIED padding layout of THESE .dat files (mean over 200 patches):
                    #   padding (value 0) is at row/col 0 AND rows/cols 510-511; the real 509x509 is
                    #   rows[1:510], cols[1:510]. So crop = [VALID_OFF:VALID_OFF+509]. (An earlier
                    #   [:509,:509] crop was WRONG: it kept the zero row0/col0 and dropped real
                    #   row509/col509. NOT [0:509] and NOT the [3:512] some docs imply — the .dat
                    #   conversion re-centred the pad. Re-checked directly on L1C_B1.dat 2026-07-19.)


def validate_partition(groups, n_bands, what="grouping"):
    """Require `groups` to be an EXACT, gap-free, non-overlapping partition of range(n_bands).

    `_assert_singleton` proves only that B1/B9/B10 each sit alone; it says nothing about the other
    ten bands. A regrouping that duplicated B4 into two groups, or dropped B12 from every group,
    passes every singleton guard and every existing test: the dropB10 scenario stays band-exact
    while the degradation curve is silently computed on a model that never sees B12 (or whose
    group-dropout mask double-counts B4). Wrong numbers, no crash — the class this repo fails
    loudly on. Cheap enough (7 groups) to run on every call, so it guards phase8R/8D/8E too, which
    all obtain their grouping from `s2_physical_groups()`."""
    if len(groups) == 0:
        raise ValueError(f"{what}: no groups at all")
    sizes = [np.asarray(g, int).ravel().size for g in groups]
    if any(s == 0 for s in sizes):
        raise ValueError(f"{what}: empty group(s) at index "
                         f"{[i for i, s in enumerate(sizes) if s == 0]} — an empty group has no "
                         f"centre wavelength and contributes a content-free token")
    flat = np.concatenate([np.asarray(g, int).ravel() for g in groups])
    if flat.size != n_bands or not np.array_equal(np.sort(flat), np.arange(n_bands)):
        dup, cnt = np.unique(flat, return_counts=True)
        missing = np.setdiff1d(np.arange(n_bands), flat)
        raise ValueError(
            f"{what} is not an exact partition of {n_bands} bands: "
            f"{flat.size} entries, missing bands {missing.tolist()[:8]}, "
            f"duplicated bands {dup[cnt > 1].tolist()[:8]}, "
            f"out-of-range {flat[(flat < 0) | (flat >= n_bands)].tolist()[:8]}. Every band must "
            f"belong to exactly one group or the missing-band accounting is meaningless.")
    return groups


def s2_physical_groups():
    """Physical S2 grouping as list[np.ndarray] (band-as-modality along real spectral design).

    Partition-validated on every call: the grouping is the unit the whole experiment counts in, so
    an invalid one must never reach a model."""
    return validate_partition([np.asarray(g, int) for g in S2_PHYSICAL_GROUPS], len(L1C_BANDS),
                              what="S2_PHYSICAL_GROUPS")


def _group_of_band(groups, band_idx):
    gs = [g for g, idx in enumerate(groups) if band_idx in list(idx)]
    if len(gs) != 1:
        raise ValueError(f"band {band_idx} found in {len(gs)} groups (expected exactly 1)")
    return gs[0]


def _assert_singleton(groups, band_idx, name):
    """GUARD against the group-granularity bug: the scenario band MUST be its own group, else
    dropping its group would silently remove neighbour bands (the original B9+B10 blocker).

    Raise, not assert: this guard's whole job is to stop a regrouping from quietly changing what
    "dropB10" means, and `python -O` strips asserts — under which the guard is gone and the scenario
    silently drops B9 alongside B10 again, which is the bug it was written for."""
    g = _group_of_band(groups, band_idx)
    if len(groups[g]) != 1:
        raise ValueError(
            f"SCENARIO GUARD FAILED: {name} (band idx {band_idx}) is in group {g}="
            f"{list(groups[g])} which is NOT a singleton — dropping it would remove neighbour "
            f"bands and corrupt the '{name}' scenario. Fix the grouping (isolate this band).")
    return g


def _memmap_checked(path, dtype, n):
    """np.memmap(...).reshape(n, SIDE, SIDE) with the size checked FIRST, by name.

    reshape() is already an exact size check (a truncated .dat raises rather than reshaping into
    garbage — verified), but it raises `cannot reshape array of size 320 into shape (6,8,8)`, which
    names neither the file nor the shortfall. On a 500 MB memmap that message costs an hour of
    bisecting. Say which file, how many patches it actually holds, and how many the metadata
    promised."""
    itemsize = np.dtype(dtype).itemsize
    want = n * SIDE * SIDE * itemsize
    got = os.path.getsize(path)
    if got != want:
        raise ValueError(
            f"{path}: size {got} B != expected {want} B (= {n} patches x {SIDE}x{SIDE} x "
            f"{itemsize} B). The file holds {got / (SIDE * SIDE * itemsize):.3f} patches' worth of "
            f"bytes while metadata.csv lists {n} — truncated, half-written, or written with a "
            f"different SIDE/dtype. Refusing to reshape.")
    return np.memmap(path, dtype=dtype, mode="r").reshape(n, SIDE, SIDE)


def patch_roi_ids(split):
    """Per-PATCH ROI id (e.g. 'ROI_0001') for `split`, in on-disk patch order.

    CloudSEN12 ships SEVERAL patches per geographic ROI — the test split is exactly 195 ROIs x 5
    patches, and the 5 patches of an ROI are the IDENTICAL 512x512 footprint on 5 different dates
    (verified: proj_geometry is constant within an roi_id). So a calibration/evaluation split that
    is disjoint by PATCH INDEX is NOT disjoint by location: at the phase8E/phase8R default
    (max_patches=300, calib_frac=0.5) ~51% of calibration patches have a same-ROI sibling in the
    evaluation set, and over all 975 test patches that rises to ~92%. Callers that need
    location-disjointness must group on THIS, not on the patch index."""
    import pandas as pd
    return _roi_from_meta(pd.read_csv(os.path.join(DATA, split, "metadata.csv")), split)


def _roi_from_meta(meta, split):
    if "roi_id" not in meta.columns:
        raise ValueError(f"{split}/metadata.csv has no 'roi_id' column (has {list(meta.columns)[:8]}...) "
                         f"— cannot group patches by location")
    return meta["roi_id"].to_numpy()


def scene_component_ids(split):
    """Per-PATCH SCENE-CONNECTED-COMPONENT id for `split`, in on-disk patch order.

    roi_id-disjoint is NOT scene-disjoint: some Sentinel-2 `s2_id` products appear under TWO different
    roi_ids (on CloudSEN12 test, 12 of them, collapsing 195 ROIs -> 184 components), so two ROIs a
    conformal split treats as independent can be the same acquisition/tile/atmosphere -- a correlation
    CRC exchangeability forbids. Union roi_ids that share ANY s2_id (connected components over the
    ROI-scene graph) and use the COMPONENT as the exchangeable unit on both the split and the CRC
    grouping. The component label is the lexicographically smallest roi_id it contains (stable).
    Callers that certify on CloudSEN12 must group on THIS, not on patch_roi_ids."""
    import pandas as pd
    meta = pd.read_csv(os.path.join(DATA, split, "metadata.csv"))
    for col in ("roi_id", "s2_id"):
        if col not in meta.columns:
            raise ValueError(f"{split}/metadata.csv has no '{col}' column (has {list(meta.columns)[:8]}"
                             f"...) — scene grouping needs both roi_id and s2_id")
        if meta[col].isna().any():                                  # fail closed: a NaN s2_id would union
            raise ValueError(f"{split}/metadata.csv has {int(meta[col].isna().sum())} NaN {col!r}; a NaN "
                             f"would collapse all unknown scenes into ONE false component -- resolve provenance")
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)          # attach to the smaller label -> stable root
    rois = meta["roi_id"].astype(str).to_numpy()
    scenes = meta["s2_id"].astype(str).to_numpy()
    for r in np.unique(rois):
        find(r)
    by_scene = {}
    for r, s in zip(rois, scenes):
        by_scene.setdefault(s, []).append(r)
    for members in by_scene.values():
        for other in members[1:]:
            union(members[0], other)
    comp_of_roi = {r: find(r) for r in np.unique(rois)}
    return np.array([comp_of_roi[r] for r in rois], dtype=object)


def label_histogram(split, force=False):
    """PER-PATCH label counts over the whole split's valid 509x509 region: (n_patches, 256). Cached.

    load_split can only vouch for the ~0.1% of pixels it SAMPLED: its label-domain check runs on
    what it returns. One 255 nodata pixel anywhere else is invisible to seed 54321 and fatal to
    seed 54322 — a run whose success depends on its seed. Scanning once settles that, and answers
    the question a reviewer asks the moment they see "300 of 259,081 pixels per patch": whether
    that sample reproduces the split's real class balance. The POPULATION balance exists nowhere
    else, so without this the sampling can only be asserted to be representative, never shown.

    Kept PER PATCH rather than summed: `.sum(0)` gives the split total, and indexing by the patch
    ids a run actually loaded gives the population for exactly that subset. Without it a run using
    --patches-test could only be compared against the whole split, which is a different population,
    so the check would have to be skipped precisely when subsetting made it most worth doing. The
    cost is 975x256 (test) or 8490x256 (train) int64 — 2 MB and 17 MB.

    Cached under data/cloudsen12/.cache because it reads ~255 MB (test) to ~2.2 GB (train) of
    memmap. The cache key carries the label file's size and mtime plus the crop geometry, so a
    re-exported dataset or a changed VALID_OFF invalidates it rather than serving a stale answer.
    """
    import pandas as pd
    root = os.path.join(DATA, split)
    lab_path = os.path.join(root, "LABEL_manual_hq.dat")
    st = os.stat(lab_path)
    key = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
           "valid_side": VALID_SIDE, "valid_off": VALID_OFF, "layout": "per_patch"}
    cache = os.path.join(DATA, ".cache", f"label_hist_{split}.npz")
    if not force and os.path.exists(cache):
        try:
            with np.load(cache) as z:
                if json.loads(str(z["meta"].item())) == key:
                    return z["hist"]
        except (ValueError, KeyError, OSError):
            pass                                   # unreadable/stale cache -> just rescan
    n = len(pd.read_csv(os.path.join(root, "metadata.csv")))
    lab = _memmap_checked(lab_path, np.uint8, n)
    o, V = VALID_OFF, VALID_SIDE
    hist = np.zeros((n, 256), np.int64)
    for p in range(n):                             # crop first: the 0-valued PADDING is not "clear"
        hist[p] = np.bincount(np.asarray(lab[p][o:o + V, o:o + V]).ravel(), minlength=256)
    try:
        os.makedirs(os.path.join(DATA, ".cache"), exist_ok=True)
        # Atomic like every other artefact here. np.savez writes in place, so two runs racing on a
        # cold cache can leave a truncated .npz that a later run reads as authoritative — and a
        # silently wrong POPULATION is worse than no population at all.
        _atomic_write(cache, lambda tmp: np.savez_compressed(
            tmp, hist=hist, meta=np.array(json.dumps(key))))
    except OSError:
        pass                                       # a read-only data mount must not fail the run
    return hist


def dataset_manifest(split, products=("L1C",)):
    """SHA-256 of the split's metadata.csv plus the byte size of every .dat that will be read.

    _memmap_checked already refuses a .dat whose size disagrees with metadata.csv, so truncation
    is covered. What no size check can see is the dataset being REPLACED between runs by a
    re-export with identical shapes: every size still matches, every number moves, and the
    provenance records the same command. Hashing the 547 KB metadata and recording the .dat sizes
    puts a fingerprint in the run record for a few milliseconds. The .dat files are 511 MB each
    and are deliberately NOT hashed — that is a minutes-long read for a marginal gain over the
    size check plus the metadata hash.
    """
    root = os.path.join(DATA, split)
    h = hashlib.sha256()
    with open(os.path.join(root, "metadata.csv"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sizes = {"LABEL_manual_hq.dat": os.path.getsize(os.path.join(root, "LABEL_manual_hq.dat"))}
    for product in products:
        for b in PRODUCTS[product]:
            name = f"{product}_{b}.dat"
            p = os.path.join(root, name)
            # null, not omitted: a manifest whose job is to say what will be read must not quietly
            # drop the entry for a file that is not there.
            sizes[name] = os.path.getsize(p) if os.path.exists(p) else None
    return {"metadata_sha256": h.hexdigest(), "dat_bytes": sizes}


def _sampling_se_pp(n):
    """Standard error, in percentage points, of a class SHARE estimated from `n` sampled pixels.

    Worst case p=0.5, hence sqrt(0.25/n). The design is self-weighting — the same number of pixels
    from every patch, and every patch is the same 509x509 — so the sample share is an unbiased
    estimate of the population share and this is the right scale for its noise."""
    return float(np.sqrt(0.25 / max(1, int(n))) * 100.0)


def _drift_alarm_pp(n):
    """Threshold (pp) above which population-vs-sample class-balance drift is not sampling noise.

    Five sampling standard errors, floored at 0.5 pp. MEASURED on this dataset, the observed
    max-over-classes drift tracks _sampling_se_pp closely: 0.09 pp at 292,500 sampled pixels
    (SE 0.09), 0.37 at 12,000 (SE 0.46), 0.87 at 2,000 (SE 1.12). A CONSTANT 1.0 pp therefore
    alarms on nothing but sampling noise once --px-test or --patches-test is small — which is
    exactly when someone is running a quick check and least wants a spurious alarm."""
    return max(0.5, 5.0 * _sampling_se_pp(n))


def _reflectance_profile(X, band_idx):
    """Per-band distribution summary for the bands a product actually read.

    The dead-band guard catches a column that is EXACTLY constant. It says nothing about the
    softer corruptions that also survive every size, dtype and label check: a band that is 90%
    zeros because a nodata fill was written as 0 reflectance, one clipped at the int16 ceiling,
    one whose scale factor was applied twice. None of those raise anywhere — they just move the
    numbers. Putting the quantiles in the provenance means two runs that disagree can be compared
    on their INPUTS instead of only on their outputs.
    """
    out = {}
    for i in band_idx:
        col = X[:, int(i)]
        q = np.percentile(col, [0.1, 50.0, 99.9])
        out[L1C_BANDS[int(i)]] = {
            "min": float(col.min()), "p00_1": float(q[0]), "median": float(q[1]),
            "p99_9": float(q[2]), "max": float(col.max()), "sd": float(col.std()),
            "zero_frac": float((col == 0).mean()), "neg_frac": float((col < 0).mean())}
    return out


def load_split(split, product="L1C", pixels_per_patch=400, n_patches=None, seed=0,
               patch_ids=None, return_patch_id=False, return_roi_id=False,
               return_pixel_index=False):
    """Load CloudSEN12 pixels -> (X [N,13] reflectance in the L1C layout, y [N] in 0..3).

    product='L1C' reads the 13 TOA bands; product='L2A' reads the 12 real Sen2Cor surface-
    reflectance bands (B10 absent) and places them at their L1C indices, leaving B10 (index 10)
    as zero. RNG draws depend ONLY on (seed, n_patches, pixels_per_patch) — NOT on product — so
    load_split('test','L1C',seed=s) and load_split('test','L2A',seed=s) sample the SAME pixels,
    giving a 1:1 L1C<->L2A correspondence. Train/test are disjoint ROIs (no spatial leakage).

    patch_ids: explicit patch-index array to load (overrides n_patches sampling) — used by the
    reliability experiment to build DISJOINT calibration/evaluation patch sets (patch-level, not
    pixel-level, split preserves conformal exchangeability).

    Returns (X, y) plus, IN THIS FIXED ORDER, whichever extras were requested:
        patch_id (return_patch_id) -> roi_id (return_roi_id) -> pixel_index (return_pixel_index)
    patch_id is the source patch INDEX; roi_id is the source LOCATION (see patch_roi_ids — several
    patches share one ROI, so grouping by patch_id does NOT group by place); pixel_index is the
    FLAT offset inside that patch's 509x509 valid region, i.e. (row, col) = divmod(idx, 509).
    (patch_id, pixel_index) is the full sample key: it identifies the exact pixel a row came from,
    which is what lets a caller PROVE two loads sampled the same pixels rather than infer it from
    the labels happening to match.

    EVERY check below is an explicit raise, never an assert: `python -O` deletes assert statements
    outright, and a data-validation guard that evaporates under an optimisation flag is worse than
    no guard — it reads as protection in the source while the run proceeds on bad data. (Verified:
    under -O the old proj_shape assert did not fire on a metadata file declaring proj_shape=999.)"""
    import pandas as pd
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product!r}; expected one of {sorted(PRODUCTS)}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    root = os.path.join(DATA, split)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CloudSEN12 split directory not found: {root}")
    if not (isinstance(pixels_per_patch, (int, np.integer)) and pixels_per_patch >= 1):
        raise ValueError(f"pixels_per_patch must be a positive int, got {pixels_per_patch!r}")
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    n = len(meta)
    if n == 0:
        raise ValueError(f"{root}/metadata.csv lists 0 patches")
    # proj_shape is REQUIRED, not best-effort. The [1:510] crop is justified ONLY by every patch
    # declaring a 509x509 valid region. With the column absent the old code skipped the check and
    # cropped anyway — reading, in source, as though it had verified the padding layout it did not
    # verify. (Checked on disk: train/val/test metadata all declare proj_shape=509 uniformly, so
    # requiring it costs nothing today and refuses to guess if the conversion ever changes.)
    if "proj_shape" not in meta.columns:
        raise ValueError(
            f"{root}/metadata.csv has no 'proj_shape' column, so the {VALID_SIDE}x{VALID_SIDE} "
            f"valid-region assumption behind the [{VALID_OFF}:{VALID_OFF + VALID_SIDE}] crop "
            f"cannot be verified (has {list(meta.columns)[:8]}...). Refusing to crop on an "
            f"unchecked padding assumption.")
    shp = set(meta["proj_shape"].unique())
    if not shp <= {VALID_SIDE}:
        raise ValueError(f"proj_shape {sorted(shp)} != VALID_SIDE {VALID_SIDE} — re-check padding")
    rng = np.random.default_rng(seed)
    if patch_ids is not None:
        # A patch index list is the one input a CALLER composes by hand (disjoint calib/eval sets),
        # so it is the one most worth policing. Each of these silently produced a wrong number:
        #   -1  -> numpy wrap-around served the LAST patch while the returned patch_id said "-1";
        #   1.9 -> asarray(...,int) truncated to 1 with no complaint;
        #   [2,2] -> the same patch twice, i.e. duplicated pixels in a conformal calibration set,
        #            which inflates n while adding no information;
        #   []  -> "need at least one array to concatenate", from numpy, pointing nowhere useful.
        pid_in = np.asarray(patch_ids)
        if pid_in.ndim != 1:
            raise ValueError(f"patch_ids must be 1-D, got shape {pid_in.shape}")
        if pid_in.size == 0:
            raise ValueError("patch_ids is empty — nothing to load")
        if pid_in.dtype == np.bool_:
            #   [True, False] -> asarray keeps dtype bool, which is NOT np.integer, so it fell to
            #   the "integral value" branch, passed (1.0 and 0.0 are integral) and became indices
            #   [1, 0]. A caller handing in a boolean SELECTION MASK therefore loaded patches 0
            #   and 1 instead of the masked set. Masks longer than 3 usually trip the duplicate
            #   check below — by accident, not by design, and a 2-element mask never does.
            raise ValueError(
                f"patch_ids must be integer INDICES, got a boolean array of length {pid_in.size} "
                f"— numpy coerces it to 0/1 indices, so a boolean selection MASK silently becomes "
                f"'load patches 0 and 1'. Pass np.flatnonzero(mask) instead.")
        if not np.issubdtype(pid_in.dtype, np.integer):
            if not np.all(np.equal(np.mod(pid_in.astype(float), 1), 0)):
                raise ValueError(f"patch_ids must be integers, got non-integral values "
                                 f"{pid_in[np.mod(pid_in.astype(float), 1) != 0][:5]}")
        pidx = pid_in.astype(int)
        if pidx.min() < 0 or pidx.max() >= n:
            bad = pidx[(pidx < 0) | (pidx >= n)]
            raise IndexError(f"patch_ids out of range for split {split!r} ({n} patches): {bad[:8]}"
                             f"{' ...' if bad.size > 8 else ''}")
        if np.unique(pidx).size != pidx.size:
            dup, cnt = np.unique(pidx, return_counts=True)
            raise ValueError(f"patch_ids contains duplicates {dup[cnt > 1][:8]} — the same patch "
                             f"would contribute its pixels more than once")
    else:
        if n_patches is not None and not (isinstance(n_patches, (int, np.integer)) and n_patches >= 1):
            raise ValueError(f"n_patches must be a positive int or None, got {n_patches!r}")
        if n_patches is not None and n_patches > n:
            # `min(n_patches, n)` clamped silently: a run configured with --patches-test 5000
            # evaluated on the 975 that exist, and neither the console nor the provenance record
            # said the requested count was never met. None already means "all patches", so an
            # oversized explicit count is a configuration error, not a request to be interpreted.
            raise ValueError(
                f"n_patches={n_patches} exceeds the {n} patches in split {split!r} — this used to "
                f"clamp silently. Pass n_patches=None to mean 'all patches'.")
        pidx = np.arange(n) if n_patches is None else rng.choice(n, size=n_patches, replace=False)
    bands = PRODUCTS[product]
    tgt = [L1C_BANDS.index(b) for b in bands]
    roi = _roi_from_meta(meta, split) if return_roi_id else None    # reuse the metadata already read
    need_pid = return_patch_id or return_pixel_index
    label = _memmap_checked(os.path.join(root, "LABEL_manual_hq.dat"), np.uint8, n)
    bmm = [_memmap_checked(os.path.join(root, f"{product}_{b}.dat"), DAT_DTYPE, n) for b in bands]
    X, Y, PID, ROI, PIX = [], [], [], [], []
    V = VALID_SIDE; o = VALID_OFF                                 # real region = rows/cols [o:o+509]
    for p in pidx:
        lab = np.asarray(label[p][o:o + V, o:o + V]).ravel()     # crop real 509x509 (drop padding row0/col0 + 510-511)
        sel = rng.choice(lab.size, size=min(pixels_per_patch, lab.size), replace=False)
        spec = np.zeros((sel.size, len(L1C_BANDS)), np.float32)   # B10 col stays 0 for L2A
        for j, bm in enumerate(bmm):
            spec[:, tgt[j]] = np.asarray(bm[p][o:o + V, o:o + V]).ravel()[sel].astype(np.float32) * 1e-4
        X.append(spec); Y.append(lab[sel].astype(int))
        if need_pid:
            PID.append(np.full(sel.size, int(p)))
        if return_pixel_index:
            PIX.append(sel.astype(np.int32))          # flat offset in the 509x509 crop
        if roi is not None:
            ROI.append(np.full(sel.size, roi[p]))
    Xc, Yc = np.concatenate(X), np.concatenate(Y)
    # DEAD BAND. A .dat that is present, correctly sized, and entirely CONSTANT (all-zero from a
    # truncated-then-repadded export, a stuck detector, a byte-order mistake that mapped everything
    # to one value) passes every check above — the size guard checks bytes, the label guard checks
    # labels. Downstream `sd = X.std(0) + 1e-8` then divides ~0 by 1e-8 and the dead column becomes
    # a constant-0 feature: the model silently trains on 12 bands while every table, figure and
    # provenance record says 13. Only the bands THIS product actually read are checked (L2A leaves
    # B10 zero by design, which is not a defect), and only when the sample is large enough that
    # exact constancy cannot be real data — a genuinely dark SWIR band over a few hundred ocean
    # pixels can legitimately be all-zero, and failing a smoke test on that would be a false alarm.
    if Xc.shape[0] >= 1000:
        filled = np.asarray(tgt, int)
        dead = filled[Xc[:, filled].std(0) == 0.0]
        if dead.size:
            raise ValueError(
                f"{root}: band(s) {[L1C_BANDS[i] for i in dead]} are EXACTLY constant "
                f"(value {Xc[0, dead[0]]:.6g}) across all {Xc.shape[0]} sampled pixels of product "
                f"{product!r}. A dead band survives the file-size check and is then flattened to a "
                f"constant 0 feature by train-set standardisation, so the run would report on "
                f"{len(filled) - dead.size} effective bands while claiming {len(filled)}.")
    # Label domain, checked on what we actually RETURN. Out-of-range labels do eventually surface
    # (metrics._check_labels raises; CrossEntropyLoss raises "Target N is out of bounds") but only
    # after training has been set up, and NOT in the plain `pred == y` accuracy/coverage arithmetic
    # the reliability scripts do, where a 255 nodata sentinel is simply counted as "wrong" forever.
    # Fail here, where the file that produced it is still in scope.
    if Yc.size and (Yc.min() < 0 or Yc.max() >= NUM_CLASSES):
        bad = np.unique(Yc[(Yc < 0) | (Yc >= NUM_CLASSES)])
        raise ValueError(f"{root}/LABEL_manual_hq.dat produced labels outside [0,{NUM_CLASSES}): "
                         f"{bad[:8]} — expected {CLASS_NAMES}")
    out = [Xc, Yc]
    if return_patch_id:
        out.append(np.concatenate(PID))
    if return_roi_id:
        out.append(np.concatenate(ROI))
    if return_pixel_index:
        out.append(np.concatenate(PIX))
    return tuple(out)


# ---- band-exact evaluation (returns predictions so we can report per-class IoU) ----
PREDICT_BATCH = 65536      # rows per forward pass; see _predict
# Which method keys are GROUPED-ATTENTION models (missing groups ride a present-mask) rather than
# MLP baselines (missing groups are zero-filled or interpolated). Named once: this used to be a
# literal tuple inside _predict, so adding an arm and forgetting to extend it would silently
# evaluate a cross-band attention model down the MLP path — the mask never applied, the raw B10
# column fed in as if present, and a plausible number out.
GROUPED_KINDS = ("proposed", "b4", "b6", "b7")


def _predict(kind, model, X, groups, drop_group_ids, wl, X_raw=None, mu=None, sd=None,
             batch_size=PREDICT_BATCH):
    """Predicted class per row, evaluated in BATCHES and with eval mode asserted, not assumed.

    BATCHING. This used to push the entire test set through the model in a single call. At the
    default 975 patches x 300 px that is 292,500 rows at once, and the grouped models materialise
    (N, G, d_model) activations per encoder layer — the MEASURED peak was 7.6 GB per worker at
    EVAL (not at training), which is what caps how many seeds fit on a 32 GB V100 and it grows
    linearly with --px-test. Batching bounds that transient independently of test-set size. It
    changes no number: rows are independent, the model is in eval mode, and argmax is per row.

    EVAL MODE. Every phase-2 train_* returns an eval()-mode model and both architectures default
    to dropout=0.0, so forcing it here changes nothing today. It is here because `_predict` is a
    module-level helper reachable by any caller: the day a dropout>0 config or a caller that left
    the model in .train() shows up, the failure is silent — dropout masks sampled at evaluation
    time, giving a slightly wrong and irreproducible number rather than an error. Restore the
    caller's mode afterwards so this stays a pure function of its arguments."""
    import torch
    if not (isinstance(batch_size, (int, np.integer)) and batch_size >= 1):
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    grouped = kind in GROUPED_KINDS
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dev = next(model.parameters()).device
            if grouped:                                           # grouped attention models
                Xc = X                                            # missing groups ride the mask
            else:                                                 # MLP baselines
                if kind == "b3":
                    # B3 = spectral interpolation, and the ORDER matters: interpolate RAW
                    # reflectance, THEN standardize. Interpolating in standardized space is a
                    # different (wrong) quantity -- phase2's own selfcheck pins the divergence at
                    # 1.0 vs 5.05 on a fixed example. This used to fall back to the wrong order
                    # whenever raw inputs were absent, which is silent: B3 simply scores
                    # differently with no indication why. Raise instead.
                    if X_raw is None or mu is None or sd is None:
                        raise ValueError(
                            "B3 needs RAW reflectance plus the train mu/sd to interpolate before "
                            "standardising. Falling back to standardized-space interpolation would "
                            "silently compute a different quantity (1.0 vs 5.05 on phase2's "
                            "selfcheck).")
                    Xc = P2.b3_impute(X_raw, groups, drop_group_ids, wl, mu, sd)
                else:
                    Xc = P2.zero_missing(X, groups, drop_group_ids)
            out = np.empty(Xc.shape[0], np.int64)
            for s in range(0, Xc.shape[0], batch_size):
                xb = torch.from_numpy(np.ascontiguousarray(Xc[s:s + batch_size])).to(dev)
                if grouped:
                    pm = P2.group_present_mask(xb.shape[0], groups, drop_group_ids)
                    logits = model(xb, torch.from_numpy(pm).to(dev))
                else:
                    logits = model(xb)
                out[s:s + batch_size] = logits.argmax(1).cpu().numpy()
            return out
    finally:
        model.train(was_training)


# ---- drop-set enumeration + ROI-level statistics -------------------------------------------
def _drop_sets(n_groups, max_missing):
    """EVERY subset of spectral groups of size 0..max_missing, as (m, tuple_of_group_ids).

    The physical S2 grouping has only 7 groups, so sizes 0..5 come to 1+7+21+35+35+21 = 120
    subsets — cheap enough to evaluate ALL of them (one small forward pass each). The Monte-Carlo
    version (`--trials 8` random draws per m) was estimating exactly this mean with added noise:
    it left most 2- and 3-subsets untested, made the curve depend on the RNG, and could not answer
    "which combination of missing bands hurts most" at all. Enumeration ELIMINATES that estimator
    variance rather than averaging it down, and targets the SAME estimand (the mean over
    uniformly-chosen size-m subsets), so an enumerated curve is comparable to an old MC one —
    exactly, rather than up to noise."""
    return [(m, c) for m in range(max_missing + 1)
            for c in itertools.combinations(range(n_groups), m)]


def _mc_drop_sets(n_groups, max_missing, trials, rng):
    """Legacy Monte-Carlo drop sets (--drop-policy mc), kept so the pre-enumeration curve stays
    reproducible. m=0 has no subset to randomise and is drawn once; each m>0 draws `trials`."""
    if int(trials) < 1:
        raise ValueError(f"--trials must be >= 1 under --drop-policy mc, got {trials}")
    out = [(0, ())]
    for m in range(1, max_missing + 1):
        for _ in range(int(trials)):
            out.append((m, tuple(sorted(rng.choice(n_groups, size=m, replace=False).tolist()))))
    return out


def _miou_from_counts(agg):
    """mIoU (%) from aggregated (..., K, 3) [tp, fp, fn] counts.

    Deliberately the SAME definition as bandsim.metrics.miou: average over classes PRESENT IN
    GROUND TRUTH (tp+fn > 0), a present class with an empty union scoring 1.0. Two subtly
    different definitions of the paper's headline metric living in one file would be a slow-
    burning defect, so run_seed CHECKS this against metrics.miou on real predictions instead of
    trusting the comment."""
    tp, fp, fn = agg[..., 0], agg[..., 1], agg[..., 2]
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), 1.0)
    present = (tp + fn) > 0
    den = present.sum(-1)
    return np.where(den > 0, np.where(present, iou, 0.0).sum(-1) / np.maximum(den, 1), 0.0) * 100.0


def _roi_class_counts(y, pred, roi_code, n_roi, num_classes=NUM_CLASSES):
    """(n_roi, num_classes, 3) per-ROI [tp, fp, fn] — the sufficient statistic for IoU.

    Storing COUNTS per ROI rather than per-ROI IoUs is what makes an honest ROI bootstrap
    possible: IoU is a ratio, so a resample must sum tp/fp/fn over the drawn ROIs and divide ONCE.
    Averaging per-ROI IoUs instead would weight a 300-pixel ROI like a 3000-pixel one, i.e.
    estimate a different quantity from the one the paper reports."""
    out = np.zeros((n_roi, num_classes, 3), np.int64)
    for c in range(num_classes):
        t = y == c
        p = pred == c
        out[:, c, 0] = np.bincount(roi_code[p & t], minlength=n_roi)
        out[:, c, 1] = np.bincount(roi_code[p & ~t], minlength=n_roi)
        out[:, c, 2] = np.bincount(roi_code[~p & t], minlength=n_roi)
    return out


def _build_grouped(groups, cwl, seed, pe_type="sinusoidal"):
    """GroupedCrossBandAttention constructed under an ISOLATED, explicitly-seeded RNG.

    The three grouped models used to be constructed with whatever global torch RNG state the
    PREVIOUS method's training loop happened to leave behind — train_hcs/finetune_proposed seed
    themselves, but only AFTER their model already exists. Results stayed deterministic for a
    fixed script, which is precisely what made it easy to miss: inserting a method, reordering the
    training calls, or changing --epochs (more batches consumed => different leftover state)
    silently re-initialised B4/B6/Proposed. It also left B6 and Proposed — whose contrast is the
    paper's PE ablation — starting from unrelated random weights for no stated reason. B4 and B6
    share `seed` and the same pe_type, so they now start from IDENTICAL weights and differ only in
    training recipe. Proposed cannot be weight-matched to them: with pe_type='sinusoidal' the PE
    is a buffer rather than a Parameter, so its parameter set genuinely differs. fork_rng keeps
    this draw from perturbing the caller's stream in either direction."""
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GroupedCrossBandAttention(groups, cwl, NUM_CLASSES, pe_type=pe_type)


def _opt_steps(n, epochs, bs=256):
    """Adam steps one phase-2 training loop takes: epochs x ceil(n/bs) (no drop_last)."""
    return int(epochs) * int(np.ceil(n / bs))


def run_seed(seed, Xtr, ytr, Xte, yte, Xte_l2a, subsample_frac, max_missing, epochs,
             drop_policy="exhaustive", trials=8, roi_code=None, n_roi=0, pe_ablation=False,
             Xte_raw=None, Xte_l2a_raw=None, mu=None, sd=None):
    """Train all methods on real S2 pixels + evaluate degradation curve + band-exact scenarios.

    Per-seed data variance: each seed trains on an independent `subsample_frac` subsample of the
    shared training-pixel pool (not just a different init), so the spread across seeds reflects
    data sampling too. The test set is held FIXED across seeds, which makes method comparisons
    PAIRED — but also means this spread contains NO test-set and NO geographic uncertainty. That
    is what the ROI bootstrap in main() is for; the two must never be conflated in the write-up.
    """
    # `P2.NUM_CLASSES = NUM_CLASSES` used to sit here and is deliberately GONE. Rewriting another
    # module's global is not reentrant, and bandsim.parallel.run_jobs executes its SERIAL path
    # (--jobs 1, or a one-seed run) inside THIS process, so the assignment escaped into whatever
    # ran next; phase2's own module docstring records the measured consequence (phase 6 left it at
    # 9, phase 4 then built a 9-output head and scored it with miou(..., 16), reporting a
    # plausible, silently deflated number).
    # DELETING THE LINE ALONE WOULD HAVE BEEN WORSE THAN LEAVING IT: train_mlp reads
    # P2.NUM_CLASSES at CALL time, so without the explicit num_classes= below it would build
    # 16-output MLP heads for this 4-class problem. The metrics happen to be immune (miou averages
    # over GT-present classes, so miou(y,p,16) == miou(y,p,4) on labels in 0..3 — verified), which
    # is exactly why the defect would not have announced itself. The models are not immune.
    if not (0 < float(subsample_frac) <= 1):
        # `max(1, ...)` below is a floor, not a validation: subsample_frac<=0 quietly collapses to
        # k=1, i.e. every method trains on ONE pixel and the run still writes a full results
        # table. (frac>1 does raise, but from rng.choice's "cannot take a larger sample than
        # population".) Bound it here so the failure names the knob.
        raise ValueError(f"subsample_frac must be in (0, 1], got {subsample_frac!r} — "
                         f"<=0 would silently train on a single pixel")
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs} — 0 epochs returns an UNTRAINED "
                         f"model and the run would still write a full, plausible results table")
    # per-seed training subsample (data variance, not just init variance)
    rs = np.random.default_rng(seed)
    ntr = Xtr.shape[0]
    k = max(1, int(round(subsample_frac * ntr)))
    sub = rs.choice(ntr, size=k, replace=False)
    Xtr_s, ytr_s = Xtr[sub], ytr[sub]

    groups = s2_physical_groups()                 # partition-validated inside
    wl = np.array(S2_WL_NM, float)
    cwl = group_center_wavelengths(wl, groups)
    G = len(groups)
    gsize = [int(len(g)) for g in groups]

    # scenario drop-group ids, each GUARDED to be a singleton (band-exact)
    g_b1 = _assert_singleton(groups, B1_IDX, "B1 coastal")
    g_b9 = _assert_singleton(groups, B9_IDX, "B9 water-vapour")
    g_b10 = _assert_singleton(groups, B10_IDX, "B10 cirrus (L1C->L2A)")
    DROP = {"clean": [], "dropB10": [g_b10], "dropB1B9B10": [g_b1, g_b9, g_b10]}

    # ---- train the 6 methods (identical recipe to phase2, on real S2 pixels) ----
    # bs=auto_bs, NOT the P2 default of 256. That default is sized for Indian Pines' 21k pixels;
    # on this 2.0M-pixel subsample it means 9,950 launch-bound optimizer steps per epoch and ~7h
    # per seed for models whose forward pass is microseconds (measured 2026-07-21: 112 steps/s,
    # 8.9 ms/step, GPU "97-99%" the whole time -- that metric counts kernel-resident time, not
    # throughput). auto_bs targets ~200 steps/epoch (2.0M -> 8192) and by construction leaves
    # every small-data phase at 256. Recorded in provenance below: batch size is a hyperparameter,
    # and these results are not comparable with a bs=256 run of the same phase.
    pre = max(1, epochs // 2)
    bs = P2.auto_bs(Xtr_s.shape[0])
    t_tr = time.time()

    def _prog(name):
        # Within-seed progress (task #43): the bs=256 pathology ran 7h with a silent log, and the
        # only sign anything was wrong was a seed counter stuck at 0/5. One line per model bounds
        # the blind window at one model's training time instead of one seed's.
        print(f"    [seed {seed}] {name} trained ({time.time() - t_tr:.0f}s cum, bs={bs})",
              flush=True)

    m_b1 = P2.train_mlp(Xtr_s, ytr_s, groups, seed, group_dropout=False, epochs=epochs,
                        num_classes=NUM_CLASSES, bs=bs)
    _prog("b1")
    m_b2 = P2.train_mlp(Xtr_s, ytr_s, groups, seed, group_dropout=True, epochs=epochs,
                        num_classes=NUM_CLASSES, bs=bs)
    _prog("b2")
    m_b4 = _build_grouped(groups, cwl, seed, pe_type="learned")
    P2.train_hcs(m_b4, Xtr_s, ytr_s, groups, seed, epochs=epochs, bs=bs)
    _prog("b4")
    m_b6 = _build_grouped(groups, cwl, seed, pe_type="learned")
    P2.pretrain_sgmae(m_b6, Xtr_s, groups, seed, epochs=pre, bs=bs)
    P2.finetune_proposed(m_b6, Xtr_s, ytr_s, groups, seed, epochs=epochs, group_dropout=False,
                         bs=bs)
    _prog("b6")
    m_prop = _build_grouped(groups, cwl, seed)
    P2.pretrain_sgmae(m_prop, Xtr_s, groups, seed, epochs=pre, bs=bs)
    P2.finetune_proposed(m_prop, Xtr_s, ytr_s, groups, seed, epochs=epochs, bs=bs)
    _prog("proposed")

    models = {"b1": m_b1, "b2": m_b2, "b3": m_b1, "b4": m_b4, "b6": m_b6, "proposed": m_prop}
    if pe_ablation:
        # B7 = Proposed's EXACT recipe with a learned per-group embedding instead of the
        # wavelength PE. It is the only arm that isolates the paper's headline mechanism:
        #   B6  -> B7        : same learned PE, same SGMAE, dropout ON vs OFF   = the dropout effect
        #   B7  -> Proposed  : same SGMAE, same dropout, learned PE vs wavelength = the PE effect
        # HOW CLOSELY MATCHED, precisely: B6 and B7 share an initialisation (same seed, same
        # pe_type) AND, because pretrain_sgmae reseeds from `seed` and neither arm has diverged
        # yet, identical SGMAE-pretrained weights. They then diverge in finetuning by more than
        # the dropout itself — drawing dropout masks consumes RNG, so the batch permutation
        # differs from the second epoch on. That is inherent to the ablation, not a defect, but it
        # is not a strict single-factor contrast and must not be described as one. (On CUDA the
        # attention backward has no deterministic kernel, so all of this holds exactly only on the
        # CPU reference path — see bandsim.parallel's determinism note.)
        # Without B7, B6 -> Proposed moves two factors at once and NOTHING in this experiment
        # licenses "the wavelength PE contributes N mIoU".
        m_b7 = _build_grouped(groups, cwl, seed, pe_type="learned")
        P2.pretrain_sgmae(m_b7, Xtr_s, groups, seed, epochs=pre, bs=bs)
        P2.finetune_proposed(m_b7, Xtr_s, ytr_s, groups, seed, epochs=epochs, group_dropout=True,
                             bs=bs)
        models["b7"] = m_b7
        _prog("b7")

    # Compute actually spent, reported rather than assumed equal: B6 and Proposed get a
    # `pre`-epoch SGMAE pretrain on top of the same fine-tuning budget, i.e. ~1.5x the optimizer
    # steps of B1/B2/B4 at the default --epochs. That is a defensible design (the pretrain is
    # label-free) but it is NOT a compute-matched comparison, and a table that does not say so
    # invites the reviewer to assume it was.
    # Derived from `models` so a new arm cannot be added without its compute being reported. B3 is
    # B1's model plus a test-time imputation rule, so it counts the same parameters and steps.
    n_params = {name: int(count_params(m)) for name, m in models.items()}
    # bs=bs, not the helper's 256 default: the models above were trained at auto_bs, and a step
    # count computed at 256 would record 32x the Adam updates any model actually received.
    _sgmae = _opt_steps(k, pre, bs=bs) + _opt_steps(k, epochs, bs=bs)     # pretrain + finetune
    steps = {"b1": _opt_steps(k, epochs, bs=bs), "b2": _opt_steps(k, epochs, bs=bs),
             "b3": _opt_steps(k, epochs, bs=bs), "b4": _opt_steps(k, epochs, bs=bs),
             "b6": _sgmae, "proposed": _sgmae}
    if pe_ablation:
        steps["b7"] = _sgmae
    if set(steps) != set(models):
        raise RuntimeError(f"optimizer-step accounting misses {set(models) - set(steps)} — every "
                           f"arm in the table must report the compute it was given")

    # ---- degradation curve over EVERY drop set (all methods see the SAME sets -> paired) ----
    if drop_policy == "exhaustive":
        dsets = _drop_sets(G, max_missing)
    elif drop_policy == "mc":
        dsets = _mc_drop_sets(G, max_missing, trials, np.random.default_rng(seed + 999))
    else:
        raise ValueError(f"unknown drop_policy {drop_policy!r}; expected 'exhaustive' or 'mc'")
    curve_rows, curves = [], {}
    for kk in models:
        per_m = {m: [] for m in range(max_missing + 1)}
        for m, ds in dsets:
            pred = _predict(kk, models[kk], Xte, groups, list(ds), wl,
                            X_raw=Xte_raw, mu=mu, sd=sd)
            v = float(miou(yte, pred, NUM_CLASSES))
            per_m[m].append(v)
            curve_rows.append([int(seed), kk, int(m),
                               "+".join(str(g) for g in ds),
                               "+".join(L1C_BANDS[b] for g in ds for b in groups[g]),
                               int(sum(gsize[g] for g in ds)), v])
        curves[kk] = np.array([float(np.mean(per_m[m])) for m in range(max_missing + 1)])

    # ---- band-exact scenarios: mIoU + per-class IoU + per-ROI counts for the bootstrap ----
    scen_miou = {kk: {} for kk in models}
    scen_pc = {kk: {} for kk in models}
    roi_counts = {kk: {} for kk in models}
    for kk in models:
        todo = [(name, drop, Xte, Xte_raw) for name, drop in DROP.items()]
        if Xte_l2a is not None:                               # OPERATIONAL real L2A (B10 absent)
            todo.append(("L2A_real", [g_b10], Xte_l2a, Xte_l2a_raw))
        for name, drop, Xin, Xin_raw in todo:
            pred = _predict(kk, models[kk], Xin, groups, drop, wl, X_raw=Xin_raw, mu=mu, sd=sd)
            scen_miou[kk][name] = float(miou(yte, pred, NUM_CLASSES))
            scen_pc[kk][name] = per_class_iou(yte, pred, NUM_CLASSES)
            if roi_code is not None:
                roi_counts[kk][name] = _roi_class_counts(yte, pred, roi_code, n_roi)

    # ---- self-checks. Four identities that MUST hold, each of which was silently breakable ----
    for kk in models:
        # (1) the curve's m=0 point and the "clean" scenario are the same evaluation. They are
        #     computed by separate code paths, so a standardisation or API drift between them
        #     shows up here instead of as an unexplained offset between two published tables.
        if not np.isclose(scen_miou[kk]["clean"], float(curves[kk][0]), atol=1e-6):
            raise RuntimeError(f"{kk}: clean scenario mIoU {scen_miou[kk]['clean']:.6f} != curve "
                               f"m=0 {float(curves[kk][0]):.6f} — the two eval paths disagree")
        # (2) the per-class table and the scenario table must agree, since metrics.miou is by
        #     definition the nanmean of metrics.per_class_iou. They are written to two separate
        #     CSVs that readers compare.
        if not np.isclose(float(_nanmean(scen_pc[kk]["clean"])), scen_miou[kk]["clean"], atol=1e-6):
            raise RuntimeError(f"{kk}: nanmean(per-class IoU) != scenario mIoU — the per-class and "
                               f"scenario CSVs would disagree with each other")
        # (3) the ROI counts must reproduce the pixel-level metric exactly when summed over ALL
        #     ROIs. This is what licenses using them for the bootstrap: if they did not, every
        #     confidence interval would be for a different statistic than the reported point.
        for name, cnt in roi_counts[kk].items():
            if not np.isclose(float(_miou_from_counts(cnt.sum(0))), scen_miou[kk][name], atol=1e-6):
                raise RuntimeError(f"{kk}/{name}: ROI counts give mIoU "
                                   f"{float(_miou_from_counts(cnt.sum(0))):.6f} but the pixel-level "
                                   f"metric gives {scen_miou[kk][name]:.6f} — the bootstrap would "
                                   f"be quantifying a different statistic from the reported one")
    # (4) B3 IS B1's model with a test-time imputation rule, and with nothing dropped that rule is
    #     the identity. Their clean scores must therefore be bit-identical; any drift means B3's
    #     raw-reflectance path and main()'s standardisation have diverged (the 1.0-vs-5.05 bug).
    if not np.isclose(scen_miou["b1"]["clean"], scen_miou["b3"]["clean"], rtol=0, atol=1e-9):
        raise RuntimeError(f"b1 clean {scen_miou['b1']['clean']:.9f} != b3 clean "
                           f"{scen_miou['b3']['clean']:.9f}: B3 shares B1's model and imputes "
                           f"nothing when no group is dropped, so these must be identical — the "
                           f"raw-vs-standardised interpolation order has drifted")

    xs = np.arange(0, max_missing + 1)
    return {"curves": curves, "curve_rows": curve_rows, "scen_miou": scen_miou,
            "scen_pc": scen_pc, "roi_counts": roi_counts, "n_params": n_params, "steps": steps,
            "audc": {kk: float(audc(xs, curves[kk])) for kk in models},
            "n_train_sub": int(k), "n_drop_sets": len(dsets), "train_bs": int(bs)}


def _sd(a, axis=0):
    """SAMPLE standard deviation (ddof=1) across seeds.

    numpy's default ddof=0 is the population SD of the seeds that happened to run, which is not
    what "mean +/- std over N seeds" means in a paper: it understates the sample SD by 10.6% at
    n=5 and by 29% at n=2. Returns NaN rather than 0 for a single seed — a 0 would read as
    "perfectly reproducible" when it actually means "never measured"."""
    a = np.asarray(a, float)
    if a.shape[axis] < 2:
        return np.full(np.delete(np.asarray(a.shape), axis), np.nan)
    return np.std(a, axis=axis, ddof=1)


def _nanmean(a, axis=None):
    """np.nanmean without the "Mean of empty slice" RuntimeWarning.

    That warning fires when a class is absent from the test labels in EVERY seed — a state
    _preflight already refuses for a formal run and which --smoke already announces on its own
    class-support line. Raised again from inside CSV writing it points at the writer rather than
    at the cause, so here it is noise, not information. The VALUE is untouched: an all-NaN column
    still yields NaN and is written as `nan`, never as a fabricated 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _roi_bootstrap(counts_by_key, rng, n_boot):
    """{key: (n_boot,) mIoU samples} from ONE SHARED resample of ROIs.

    The same resampled ROI list is used for every key, so differences between keys are PAIRED on
    identical geography. That is not a refinement, it is the difference between a usable and an
    unusable test here: ROI-to-ROI mIoU varies by tens of points while the gap between methods is
    a few, so two independently drawn CIs overlap almost completely and suggest "no difference"
    where the paired difference is decisive. Resampling ROIs (not pixels) is the point — the 195
    test ROIs are the independent geographic units; the ~292k pixels are not, since 300 of them
    come from the same 509x509 patch and 5 patches share one ROI footprint."""
    keys = list(counts_by_key)
    n = counts_by_key[keys[0]].shape[0]
    idx = rng.integers(0, n, size=(int(n_boot), n))
    out = {}
    for k in keys:
        agg = counts_by_key[k][idx].sum(1)                      # (n_boot, K, 3)
        # HOMOGENEITY OF THE ESTIMAND ACROSS REPLICATES. miou averages over classes PRESENT IN
        # GROUND TRUTH, so a resample that happened to exclude every ROI containing a rare class
        # would score a 3-class mean while its neighbours score a 4-class one — and the bootstrap
        # distribution would silently mix two different quantities, exactly the failure phase2
        # pins down for its block splits with an explicit class_set. The fix here cannot be to
        # FORCE the missing class into the average: with no ground truth and no predictions its
        # union is empty and it would score a free 1.0, which is the defect metrics.miou was
        # written to avoid. So detect it instead. With 195 test ROIs this is unreachable in
        # practice; if it ever fires, the interval is not a like-for-like one and must not be
        # quoted.
        full = (counts_by_key[k][..., 0] + counts_by_key[k][..., 2]).sum(0) > 0   # (K,) GT-present
        bad = int(((agg[..., 0] + agg[..., 2] > 0) != full[None]).any(1).sum())
        if bad:
            print(f"  WARNING: {bad}/{int(n_boot)} ROI resamples changed the present-class set "
                  f"for '{k}' (classes {[CLASS_NAMES[c] for c in np.flatnonzero(full)]} expected) "
                  f"— those replicates average over a different class set from the reported "
                  f"point estimate, so this CI is not like-for-like")
        out[k] = _miou_from_counts(agg)
    return out


def _ci(v, alpha=0.05):
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def _preflight(args, n_groups):
    """Reject configurations that would produce a plausible-looking but unsupportable table.

    Every check here corresponds to a run that ALREADY completed successfully and wrote official
    filenames. The shipped results_phase8_cloudsen12_*.csv carry a provenance record reading
    `--seeds 0 1 --patches-train 3000` while this file's docstring claimed "mean over >=5 seeds"
    and reproduce.sh used seven: a two-seed "std" is the half-range of two numbers, and nothing in
    the CSV said so. Prose in a docstring is not a guard."""
    if len(args.seeds) != len(set(args.seeds)):
        dup = sorted({s for s in args.seeds if list(args.seeds).count(s) > 1})
        raise SystemExit(
            f"--seeds contains duplicates {dup}. Every training function here reseeds from the "
            f"seed value, so a repeated seed reproduces the same model bit-for-bit: it adds no "
            f"information while shrinking the reported spread toward zero and inflating n.")
    if not args.smoke and len(args.seeds) < MIN_FORMAL_SEEDS:
        raise SystemExit(
            f"a formal run writes the paper's flagship tables and must average over at least "
            f"{MIN_FORMAL_SEEDS} seeds (got {len(args.seeds)}: {list(args.seeds)}). With one seed "
            f"the reported std is exactly 0 and with two it is a half-range, in both cases "
            f"printed into a column named '_std' that a reader will take for a 5+-seed estimate. "
            f"Use --smoke for a quick check (it writes *_smoke artefacts), or pass at least "
            f"{MIN_FORMAL_SEEDS} distinct seeds.")
    if args.epochs < 1:
        raise SystemExit(f"--epochs must be >= 1, got {args.epochs}: 0 epochs leaves every model "
                         f"at its random initialisation and still writes a full results table.")
    if not (0 <= args.max_missing < n_groups):
        raise SystemExit(
            f"--max-missing must satisfy 0 <= m < {n_groups} (the number of spectral groups), got "
            f"{args.max_missing}: dropping all {n_groups} leaves B3 spectral interpolation with "
            f"no observed band to interpolate from, and the attention models with no token.")
    if not (0 < args.subsample_frac <= 1):
        raise SystemExit(f"--subsample-frac must be in (0, 1], got {args.subsample_frac}")
    if args.drop_policy == "mc" and args.trials < 1:
        raise SystemExit(f"--trials must be >= 1 under --drop-policy mc, got {args.trials}")
    if args.jobs is not None and args.jobs < 1:
        raise SystemExit(f"--jobs must be >= 1 if given, got {args.jobs} (0 silently meant 'auto')")
    if args.boot < 100:
        raise SystemExit(f"--boot must be >= 100, got {args.boot}: a percentile CI from fewer "
                         f"replicates is dominated by resampling noise in its own endpoints.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--train-split", default="train", choices=["val", "train"],
                    help="split to TRAIN on (train=all 8490 patches, --px-train pixels sampled "
                         "from each [default, headline]; val=fast 535)")
    ap.add_argument("--max-missing", type=int, default=5)
    ap.add_argument("--drop-policy", default="exhaustive", choices=["exhaustive", "mc"],
                    help="exhaustive (default): evaluate EVERY drop set of size 0..max-missing "
                         "(120 sets at 7 groups) — no Monte-Carlo noise, and the worst-case "
                         "combination becomes reportable. mc: legacy --trials random draws per m.")
    ap.add_argument("--trials", type=int, default=8,
                    help="random drop sets per m; ONLY used by --drop-policy mc")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--subsample-frac", type=float, default=0.8,
                    help="per-seed fraction of the training-pixel pool (data variance across seeds)")
    ap.add_argument("--px-train", type=int, default=300, help="pixels sampled per train patch")
    ap.add_argument("--px-test", type=int, default=300, help="pixels sampled per test patch")
    ap.add_argument("--patches-train", type=int, default=None)
    ap.add_argument("--patches-test", type=int, default=None)
    ap.add_argument("--boot", type=int, default=2000,
                    help="ROI bootstrap replicates PER SEED (geographic uncertainty)")
    ap.add_argument("--pe-ablation", action="store_true",
                    help="add arm B7 = Proposed's exact recipe with a LEARNED per-group embedding "
                         "instead of the wavelength PE. This is the only arm that isolates the "
                         "paper's headline mechanism (B6->B7 = group dropout, B7->Proposed = the "
                         "wavelength PE); without it B6->Proposed moves two factors at once. "
                         "Costs one extra pretrain+finetune per seed (~20% more training).")
    ap.add_argument("--label-scan", default="test", choices=["test", "all", "none"],
                    help="scan WHOLE label files for out-of-domain values and the population "
                         "class balance (cached). 'test' (default) reads ~255 MB once; 'all' also "
                         "reads the ~2.2 GB train labels; 'none' trusts the sampled pixels only.")
    ap.add_argument("--no-l2a", action="store_true", help="skip the real-L2A operational scenario")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    ap.add_argument("--nondeterministic", action="store_true")
    args = ap.parse_args()
    # A smoke run must NOT write the real deliverables. It used to: a 2-minute 2-seed sanity check
    # overwrote results_phase8_cloudsen12_*.csv, which is what the paper's flagship numbers come
    # from, leaving nothing in the file to say it was a smoke. Suffix every artefact instead.
    sfx = ""
    if args.smoke:
        forced = dict(seeds=[0, 1], epochs=10, trials=3, patches_train=60, patches_test=60,
                      px_train=200, px_test=200, boot=200)
        clob = [f"--{f.replace('_', '-')} {getattr(args, f)!r}" for f in forced
                if getattr(args, f) != ap.get_default(f)]
        if clob:
            # Silently discarding a flag the operator typed is how a "smoke" run gets mistaken for
            # a configured one. Name what is being thrown away.
            print(f"[smoke] NOTE: --smoke OVERRIDES the {', '.join(clob)} you passed — smoke "
                  f"settings are fixed so a *_smoke artefact means the same thing every run")
        for _f, _v in forced.items():
            setattr(args, _f, _v)
        sfx = "_smoke"
        print("[smoke] reduced settings; writing *_smoke artefacts, NOT the real deliverables")
    _preflight(args, len(S2_PHYSICAL_GROUPS))
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device); print("HW:", hw.info())

    # ---- load real S2 pixels (once; shared across seeds) ----
    print(f"loading CloudSEN12: train='{args.train_split}' test='test' ...")
    Xtr, ytr = load_split(args.train_split, "L1C", pixels_per_patch=args.px_train,
                          n_patches=args.patches_train, seed=12345)
    Xte, yte, pid_te, roi_te, pix_te = load_split(
        "test", "L1C", pixels_per_patch=args.px_test, n_patches=args.patches_test, seed=54321,
        return_patch_id=True, return_roi_id=True, return_pixel_index=True)
    # real L2A test pixels (same pixels as L1C test -> 1:1); test-only product
    Xte_l2a = None
    if not args.no_l2a:
        # ALL twelve L2A bands, checked by name. The old test was `os.path.exists(L2A_B2.dat)`
        # alone, which failed two ways: with B2 absent the flagship operational scenario silently
        # vanished from the results table (a run that never evaluated real L2A, reported as if L2A
        # were simply not part of the design), and with B2 present but B12 not, the failure came
        # from deep inside the sampling loop after minutes of loading.
        need = [os.path.join(DATA, "test", f"L2A_{b}.dat") for b in L2A_BANDS]
        gone = [os.path.basename(p) for p in need if not os.path.exists(p)]
        if gone:
            raise SystemExit(
                f"the real-L2A operational scenario is ON by default but {len(gone)} of "
                f"{len(L2A_BANDS)} L2A band files are missing from "
                f"{os.path.join(DATA, 'test')}: {gone[:6]}{' ...' if len(gone) > 6 else ''}. "
                f"Pass --no-l2a to run WITHOUT the operational anchor deliberately, rather than "
                f"losing it silently.")
        Xte_l2a, yte_l2a, pid_l2a, pix_l2a = load_split(
            "test", "L2A", pixels_per_patch=args.px_test, n_patches=args.patches_test, seed=54321,
            return_patch_id=True, return_pixel_index=True)
        # The 1:1 L1C<->L2A pixel correspondence is the whole basis of the L2A_real scenario: if
        # the two loads drift apart, every "same pixel, different product" comparison below is
        # computed on different ground. Compare the FULL SAMPLE KEY (patch index, offset within
        # the 509x509 crop), not just the labels: label equality is necessary, not sufficient —
        # two different pixel sets can carry identical label vectors, and "clear" alone is ~60% of
        # this dataset. An assert would be deleted by `python -O`, leaving the drift silent.
        if not (np.array_equal(pid_te, pid_l2a) and np.array_equal(pix_te, pix_l2a)):
            nbad = int(np.sum((pid_te != pid_l2a) | (pix_te != pix_l2a)))
            raise RuntimeError(
                f"L1C/L2A test pixels misaligned: {nbad} of {pid_te.size} rows come from a "
                f"different (patch, pixel) than their L1C counterpart. load_split's RNG stream is "
                f"supposed to depend only on (seed, n_patches, pixels_per_patch) and NOT on "
                f"product — that contract has broken, and the L2A_real scenario would compare "
                f"different ground.")
        if not np.array_equal(yte, yte_l2a):
            raise RuntimeError("L1C/L2A labels differ although the sample keys match — the label "
                               "memmap read is not deterministic across products")

    # Fingerprint the inputs AT READ TIME. Computed after training instead, a manifest describes
    # whatever the files became by then, not what this run actually consumed.
    manifest = {args.train_split: dataset_manifest(args.train_split, ("L1C",)),
                "test": dataset_manifest("test", ("L1C", "L2A") if Xte_l2a is not None
                                         else ("L1C",))}

    # Class support, reported and guarded. metrics.miou averages over classes PRESENT IN GT, so a
    # class missing from the test sample does not crash and does not score 0 — it silently turns
    # the headline number from a 4-class mean into a 3-class one, typically RAISING it, and
    # nothing in the CSV records which classes were averaged.
    support = np.bincount(yte, minlength=NUM_CLASSES)
    print("  test class support: " + "  ".join(f"{c}={int(v)}" for c, v in zip(CLASS_NAMES, support))
          + f"  ({int(np.unique(roi_te).size)} ROIs, {int(np.unique(pid_te).size)} patches)")
    if np.any(support == 0):
        absent = [CLASS_NAMES[c] for c in range(NUM_CLASSES) if support[c] == 0]
        msg = (f"class(es) {absent} have ZERO pixels in the test sample, so mIoU would be a "
               f"{int((support > 0).sum())}-class mean reported under a 4-class name")
        if args.smoke:
            print(f"[smoke] WARNING: {msg} — expected at this size, but *_smoke mIoU is NOT "
                  f"comparable to a formal run's")
        else:
            raise SystemExit(f"{msg}. Raise --px-test/--patches-test.")

    # WHOLE-FILE label scan + is the 0.116% sample actually representative?
    _off = "disabled (--label-scan none)"                  # recorded as such, never as an empty {}
    label_scan, balance, hists = _off, _off, {}
    if args.label_scan != "none":
        label_scan = {}
        for sp in (("test",) if args.label_scan == "test" else ("test", args.train_split)):
            hists[sp] = label_histogram(sp)
            hist = hists[sp].sum(0)                        # per-patch -> split total
            bad = np.flatnonzero(hist[NUM_CLASSES:]) + NUM_CLASSES
            if bad.size:
                raise SystemExit(
                    f"{sp}/LABEL_manual_hq.dat contains label value(s) {bad.tolist()[:8]} outside "
                    f"[0,{NUM_CLASSES}) over {int(hist[bad].sum())} pixels. load_split checks only "
                    f"the ~0.1% of pixels it samples, so this defect surfaces as a crash on some "
                    f"seeds and not others — mask the sentinel or fix the export.")
            label_scan[sp] = {c: int(hist[i]) for i, c in enumerate(CLASS_NAMES)}
        # The population is the patches THIS RUN read, not the whole split: with --patches-test the
        # two are different populations, and comparing against the wrong one would either raise a
        # false alarm or force the check to be skipped exactly when subsetting made it worth doing.
        pop = hists["test"][np.unique(pid_te)].sum(0)[:NUM_CLASSES].astype(float)
        pop_share, samp_share = pop / pop.sum() * 100, support / support.sum() * 100
        # The alarm threshold SCALES WITH THE SAMPLE: the drift it watches for has to beat the
        # sampling noise of the run it is watching (see _drift_alarm_pp for the measured basis).
        se_pp, thresh = _sampling_se_pp(support.sum()), _drift_alarm_pp(support.sum())
        balance = {"population_pct": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, pop_share)},
                   "sampled_pct": {c: round(float(s), 4) for c, s in zip(CLASS_NAMES, samp_share)},
                   "max_abs_drift_pp": float(np.abs(pop_share - samp_share).max()),
                   "sampling_se_pp": round(se_pp, 4), "alarm_threshold_pp": round(thresh, 4),
                   "population_is": "the patches this run actually read"}
        print("  test class balance %, population vs sampled: "
              + "  ".join(f"{c}={p:.2f}/{s:.2f}" for c, p, s in zip(CLASS_NAMES, pop_share, samp_share))
              + f"  | max drift {balance['max_abs_drift_pp']:.3f} pp "
                f"(sampling SE {se_pp:.3f} pp, alarm at {thresh:.2f})")
        # Sampling the same number of pixels from every patch, all of which are 509x509, is a
        # self-weighting design, so the sample share is an unbiased estimate of the population
        # share. Drift beyond five sampling standard errors is therefore not sampling noise — it
        # means the crop, the patch selection or the label file disagrees with what the scan read.
        if balance["max_abs_drift_pp"] > thresh:
            print(f"  WARNING: the sampled class balance is off the balance of the patches it was "
                  f"drawn from by {balance['max_abs_drift_pp']:.2f} pp, more than 5x the "
                  f"{se_pp:.3f} pp sampling error — the crop, the patch selection or the label "
                  f"file disagree, and the reported mIoU describes THIS sample, not those patches")

    mu = Xtr.mean(0); sd_raw = Xtr.std(0)                      # normalise by TRAIN (L1C) stats
    # `sd + 1e-8` is an epsilon for numerical safety, not a licence to standardise a dead band: a
    # constant column divided by 1e-8 becomes a constant 0 feature, i.e. the band silently leaves
    # the experiment while every table still says 13 bands. load_split already refuses a dead band
    # on a large sample; this catches the train pool specifically, which is what mu/sd come from.
    _dead = np.flatnonzero(sd_raw < 1e-6)
    if _dead.size:
        raise SystemExit(
            f"train band(s) {[L1C_BANDS[i] for i in _dead]} have standard deviation "
            f"{sd_raw[_dead].max():.3g} < 1e-6 across {Xtr.shape[0]} pixels. Standardising them "
            f"divides by the 1e-8 epsilon and flattens them to a constant feature — the run would "
            f"train on {Xtr.shape[1] - _dead.size} effective bands while reporting {Xtr.shape[1]}.")
    sd = sd_raw + 1e-8
    # Reflectance distribution of what was actually read, on RAW values before standardisation.
    # Only the bands the product supplies are profiled: L2A's B10 is zero BY DESIGN and profiling
    # it would raise a permanent false alarm on the one column that is legitimately empty.
    _l1c_idx = np.arange(len(L1C_BANDS))
    _l2a_idx = np.array([L1C_BANDS.index(b) for b in L2A_BANDS])
    refl_profile = {"train_L1C": _reflectance_profile(Xtr, _l1c_idx),
                    "test_L1C": _reflectance_profile(Xte, _l1c_idx)}
    if Xte_l2a is not None:
        refl_profile["test_L2A"] = _reflectance_profile(Xte_l2a, _l2a_idx)
    for _tag, _prof in refl_profile.items():
        for _b, _s in _prof.items():
            if _s["zero_frac"] > 0.5:
                print(f"  WARNING: {_tag} band {_b} is {_s['zero_frac'] * 100:.1f}% exactly zero — "
                      f"a nodata fill written as 0 reflectance survives every size and dtype check "
                      f"and simply shifts the numbers")
            if _s["neg_frac"] > 0.02:
                print(f"  WARNING: {_tag} band {_b} is {_s['neg_frac'] * 100:.1f}% negative — "
                      f"TOA/BOA reflectance should not be, so suspect a sentinel or a sign error")
    Xte_raw = Xte.astype(np.float32)                           # keep RAW reflectance for physically-correct B3
    Xtr = ((Xtr - mu) / sd).astype(np.float32); Xte = ((Xte - mu) / sd).astype(np.float32)
    Xte_l2a_raw = None
    if Xte_l2a is not None:
        Xte_l2a_raw = Xte_l2a.astype(np.float32)               # raw L2A for B3 on the operational path
        Xte_l2a = ((Xte_l2a - mu) / sd).astype(np.float32)     # L1C-trained model sees L2A (shift)
        # sanity: L2A surface reflectance should differ from L1C TOA (else scale mismatch).
        # The old line printed |mean(L1C) - mean(L2A)| under the label "mean |L1C-L2A|" — a mean of
        # signed differences, which cancels: a band that is +0.5 on half the pixels and -0.5 on the
        # other half reported 0.000 while every pixel differed. Both quantities are useful and they
        # are different, so print both under their own names.
        _present = np.array([i for i in range(len(L1C_BANDS)) if i != B10_IDX])
        _d = Xte[:, _present] - Xte_l2a[:, _present]
        print(f"  L2A domain shift over {_present.size} common bands (norm units): "
              f"mean per-pixel |L1C-L2A| = {float(np.abs(_d).mean()):.3f}, "
              f"mean |per-band bias| = {float(np.abs(_d.mean(0)).mean()):.3f}")
    print(f"train {Xtr.shape[0]} px / test {Xte.shape[0]} px | {Xtr.shape[1]} L1C bands | "
          f"physical groups {[[L1C_BANDS[i] for i in g] for g in s2_physical_groups()]} | classes {CLASS_NAMES}")

    # ROI codes: the independent geographic unit for the bootstrap. Passed to the workers as small
    # ints rather than the string ids, which would pickle ~292k Python strings per worker.
    roi_names, roi_code = np.unique(roi_te, return_inverse=True)
    roi_code = np.ascontiguousarray(roi_code.ravel().astype(np.int32))
    n_roi = int(roi_names.size)
    if n_roi < 20:
        # A percentile CI from a handful of resampled units is not a confidence interval, it is a
        # restatement of those few units. Say so rather than printing a narrow-looking bound.
        print(f"  WARNING: only {n_roi} distinct test ROIs — the ROI bootstrap CI below is not "
              f"meaningful at this size and must not be quoted as a geographic interval")

    keys = ["b1", "b2", "b3", "b4", "b6"] + (["b7"] if args.pe_ablation else []) + ["proposed"]
    states = ["clean", "dropB10", "dropB1B9B10"] + (["L2A_real"] if Xte_l2a is not None else [])
    results = parallel.run_jobs(
        run_seed, args.seeds,
        shared=dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xte_l2a=Xte_l2a,
                    Xte_raw=Xte_raw, Xte_l2a_raw=Xte_l2a_raw, mu=mu, sd=sd,
                    subsample_frac=args.subsample_frac, max_missing=args.max_missing,
                    drop_policy=args.drop_policy, trials=args.trials, epochs=args.epochs,
                    roi_code=roi_code, n_roi=n_roi, pe_ablation=args.pe_ablation),
        prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
        label="phase8/seed")

    n = len(args.seeds)
    # The arms the workers actually built must be the arms this report is about. `keys` and
    # run_seed's `models` are derived from the same --pe-ablation flag, so a mismatch means the
    # flag did not survive the trip through parallel.run_jobs' `shared` dict — which would
    # otherwise surface as a bare KeyError from inside a comprehension.
    _got = set(results[0]["curves"])
    if _got != set(keys):
        raise RuntimeError(f"workers returned arms {sorted(_got)} but the report expects "
                           f"{sorted(keys)} — --pe-ablation did not reach the worker")
    curves = {k: np.stack([r["curves"][k] for r in results]) for k in keys}       # (n_seed, m+1)
    scen = {k: {s: np.array([r["scen_miou"][k][s] for r in results]) for s in states} for k in keys}
    scen_pc = {k: {s: np.stack([r["scen_pc"][k][s] for r in results]) for s in states} for k in keys}
    audc_s = {k: np.array([r["audc"][k] for r in results]) for k in keys}         # per-seed AUDC
    n_params, steps = results[0]["n_params"], results[0]["steps"]
    for r in results[1:]:            # identical by construction; if not, the seeds are not comparable
        if r["n_params"] != n_params or r["steps"] != steps:
            raise RuntimeError("parameter or optimizer-step counts differ across seeds — the "
                               "per-seed models are not the same size and cannot be averaged")
    xs = np.arange(0, args.max_missing + 1)
    stats = {k: (curves[k].mean(0), _sd(curves[k])) for k in keys}

    # ---- RAW results, full precision. The canonical record: every aggregate below is
    # recomputable from these two files, which is what makes the tables auditable and lets a
    # paired significance test be run after the fact without re-training anything. ----
    _write_csv(P(f"results_phase8_cloudsen12_raw_curve{sfx}.csv"),
               ["seed", "method", "n_missing_groups", "drop_group_ids", "drop_bands",
                "n_missing_bands", "miou"],
               [row for r in results for row in r["curve_rows"]])
    _write_csv(P(f"results_phase8_cloudsen12_raw_scen{sfx}.csv"),
               ["seed", "method", "state"] + [f"IoU_{c}" for c in CLASS_NAMES]
               + ["miou", "audc"] + [f"support_{c}" for c in CLASS_NAMES],
               [[int(sd_i), k, s] + [("" if np.isnan(v) else repr(float(v)))
                                     for v in scen_pc[k][s][i]]
                + [repr(float(scen[k][s][i])), repr(float(audc_s[k][i]))]
                + [int(v) for v in support]
                for i, sd_i in enumerate(args.seeds) for k in keys for s in states])

    # ---- views that only EXHAUSTIVE enumeration makes available -------------------------------
    # (method, drop_group_ids) -> [n_groups, n_bands, drop_bands, [miou per seed]]
    per_set = {}
    for r in results:
        for _sd_i, _k, _m, _g, _b, _nb, _v in r["curve_rows"]:
            per_set.setdefault((_k, _g), [_m, _nb, _b, []])[3].append(float(_v))

    # (a) the missing-BAND-count axis. "One missing group" is not "one missing band": the physical
    # S2 groups hold 1, 3, 3, 2, 1, 1 and 2 bands, so dropping one group removes between 1 and 3
    # bands and the group-count axis cannot be read as an information-loss axis. Same evaluations,
    # re-marginalised onto the axis a reader has intuition for. `_spread` mixes drop-set choice
    # with seed noise ON PURPOSE — at a fixed band count the drop sets are the population of
    # interest, not a nuisance.
    _band_counts = sorted({v[1] for v in per_set.values()})
    _by_band = {(k, nb): [x for (kk_, _g), vv in per_set.items() if kk_ == k and vv[1] == nb
                          for x in vv[3]]
                for k in keys for nb in _band_counts}
    _write_csv(P(f"results_phase8_cloudsen12_bandcurve{sfx}.csv"),
               ["n_missing_bands"] + sum([[f"{k}_mean", f"{k}_spread", f"{k}_n"] for k in keys], []),
               [[nb] + sum([[f"{np.mean(_by_band[(k, nb)]):.4f}",
                             f"{np.std(_by_band[(k, nb)], ddof=1):.4f}"
                             if len(_by_band[(k, nb)]) > 1 else "nan",
                             len(_by_band[(k, nb)])] for k in keys], [])
                for nb in _band_counts])

    # (b) does it matter WHICH groups go missing, or only HOW MANY? Exhaustive enumeration answers
    # this directly on real S2: at each m, compare the spread of mIoU ACROSS drop sets with the
    # seed-to-seed noise. (phase9 asks the same question by weighting a SAMPLER on Indian Pines;
    # this is the complete answer on this dataset, not a replacement for that experiment.) The
    # worst set is also the operationally interesting one — it names the combination to worry about.
    _ds_rows = []
    for k in keys:
        for m in range(args.max_missing + 1):
            sets_m = [(g, v) for (kk_, g), v in per_set.items() if kk_ == k and v[0] == m]
            if not sets_m:
                continue
            means = np.array([np.mean(v[3]) for _g, v in sets_m])
            seed_sd = np.array([np.std(v[3], ddof=1) if len(v[3]) > 1 else np.nan
                                for _g, v in sets_m])
            lo, hi = int(means.argmin()), int(means.argmax())
            _ds_rows.append([k, m, len(sets_m),
                             sets_m[lo][0] or "-", sets_m[lo][1][2] or "-", f"{means[lo]:.4f}",
                             sets_m[hi][0] or "-", sets_m[hi][1][2] or "-", f"{means[hi]:.4f}",
                             f"{means.max() - means.min():.4f}",
                             f"{np.std(means, ddof=1):.4f}" if means.size > 1 else "nan",
                             f"{float(_nanmean(seed_sd)):.4f}"])
    _write_csv(P(f"results_phase8_cloudsen12_dropsets{sfx}.csv"),
               ["method", "n_missing_groups", "n_drop_sets", "worst_group_ids", "worst_bands",
                "worst_miou", "best_group_ids", "best_bands", "best_miou", "range",
                "sd_across_drop_sets", "mean_seed_sd"], _ds_rows)

    # ---- ROI bootstrap: geographic uncertainty, and PAIRED method differences ----
    # One shared ROI resample per (seed, state) across all methods, then samples concatenated over
    # seeds. Element-wise alignment is preserved (same seed order, same n_boot each), which is what
    # keeps the differences paired.
    # WHAT THIS INTERVAL IS, PRECISELY: a mixture over (uniformly chosen observed seed) x (ROI
    # resample). It covers geographic variability fully and the seed component only as far as the
    # seeds actually run — it does NOT resample seeds, so it treats the observed seed set as the
    # population and mildly understates that second source. It is also MARGINAL and UNADJUSTED:
    # results_..._paired.csv reports len(states) x (len(keys)-1) = up to 20 intervals, so
    # `significant_95` is a per-comparison 5% statement, not a family-wise one. Quote it as
    # "the 95% paired interval excludes zero for this comparison", never as "p < 0.05 overall".
    brng = np.random.default_rng(20260720)
    boot = {}
    for s in states:
        per_key = {k: [] for k in keys}
        for r in results:
            samp = _roi_bootstrap({k: r["roi_counts"][k][s] for k in keys}, brng, args.boot)
            for k in keys:
                per_key[k].append(samp[k])
        for k in keys:
            boot[(s, k)] = np.concatenate(per_key[k])

    _write_csv(P(f"results_phase8_cloudsen12_paired{sfx}.csv"),
               ["state", "method_a", "method_b", "delta_miou_mean", "roi_boot_ci_lo",
                "roi_boot_ci_hi", "frac_resamples_a_gt_b", "significant_95"],
               [[s, "proposed", k,
                 f"{float((boot[(s, 'proposed')] - boot[(s, k)]).mean()):.4f}",
                 f"{_ci(boot[(s, 'proposed')] - boot[(s, k)])[0]:.4f}",
                 f"{_ci(boot[(s, 'proposed')] - boot[(s, k)])[1]:.4f}",
                 f"{float((boot[(s, 'proposed')] > boot[(s, k)]).mean()):.4f}",
                 int(_ci(boot[(s, 'proposed')] - boot[(s, k)])[0] > 0
                     or _ci(boot[(s, 'proposed')] - boot[(s, k)])[1] < 0)]
                for s in states for k in keys if k != "proposed"])

    # ---- degradation curve csv ----
    _write_csv(P(f"results_phase8_cloudsen12_curve{sfx}.csv"),
               ["missing_groups"] + sum([[f"{k}_mean", f"{k}_std"] for k in keys], []),
               [[int(i)] + sum([[f"{stats[k][0][i]:.2f}", f"{stats[k][1][i]:.2f}"] for k in keys], [])
                for i in xs])

    # ---- scenarios csv (mIoU mean +/- seed SD, ROI bootstrap CI, retention, per-seed AUDC) ----
    _scen_rows = []
    for k in keys:
        row = [k]
        means = {s: float(scen[k][s].mean()) for s in states}
        for s in states:
            lo, hi = _ci(boot[(s, k)])
            row += [f"{means[s]:.2f}", f"{float(_sd(scen[k][s])):.2f}", f"{lo:.2f}", f"{hi:.2f}"]
        c = means["clean"]
        row += [f"{means['dropB10'] / c * 100:.1f}" if c > 0 else "0"]
        if "L2A_real" in states:
            row += [f"{means['L2A_real'] / c * 100:.1f}" if c > 0 else "0"]
        row += [f"{float(audc_s[k].mean()):.2f}", f"{float(_sd(audc_s[k])):.2f}",
                n_params[k], steps[k]]
        _scen_rows.append(row)
    _head = ["method"] + sum([[f"{s}_miou", f"{s}_seedsd", f"{s}_roi_lo", f"{s}_roi_hi"]
                              for s in states], [])
    _head += ["dropB10_retention%"] + (["L2A_real_retention%"] if "L2A_real" in states else [])
    _head += ["AUDC_mean", "AUDC_seedsd", "params", "optimizer_steps"]
    _write_csv(P(f"results_phase8_cloudsen12_scenarios{sfx}.csv"), _head, _scen_rows)

    # ---- per-class IoU csv (method x state -> per-class IoU) ----
    _write_csv(P(f"results_phase8_cloudsen12_perclass{sfx}.csv"),
               ["method", "state"] + [f"IoU_{c}" for c in CLASS_NAMES] + ["mIoU"]
               + [f"support_{c}" for c in CLASS_NAMES],
               [[k, s] + [f"{v:.2f}" for v in _nanmean(scen_pc[k][s], 0)]
                + [f"{_nanmean(_nanmean(scen_pc[k][s], 0)):.2f}"] + [int(v) for v in support]
                for k in keys for s in states])

    # ---- figure ----
    labels = {"b1": "B1 MLP + zero-fill", "b2": "B2 band-group dropout", "b3": "B3 interpolation",
              "b4": "B4 ChannelViT-style", "b6": "B6 SatMAE-style",
              "b7": "B7 learned PE + dropout", "proposed": "Proposed"}
    colors = {"b1": "#c0392b", "b2": "#e67e22", "b3": "#8e44ad", "b4": "#16a085",
              "b6": "#2980b9", "b7": "#7f8c8d", "proposed": "#1f6f3a"}
    styles = {"b1": "-o", "b2": "-s", "b3": "-D", "b4": "-v", "b6": "-P", "b7": "-X",
              "proposed": "-^"}
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    for k in keys:
        me, st = stats[k]
        ax.plot(xs, me, styles[k], color=colors[k], lw=1.8, ms=4, label=labels[k])
        ax.fill_between(xs, me - st, me + st, color=colors[k], alpha=0.15, lw=0)
    ax.set_xlabel("Number of missing spectral groups")
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"Missing-band robustness — CloudSEN12 real S2 ({n} seeds)", fontsize=8.5)
    # Say what the band IS. It is the seed-to-seed SD (model init + 80% training subsample) on a
    # FIXED test set — it contains no geographic or test-sampling uncertainty, which lives in the
    # ROI bootstrap CI in the scenarios CSV and is several times wider.
    ax.grid(alpha=0.3); ax.legend(fontsize=6.0, frameon=False, loc="upper right",
                                  title="shaded: $\\pm$1 SD over seeds", title_fontsize=5.5)
    fig.tight_layout()
    _atomic_write(P(f"figs/fig_degradation_cloudsen12{sfx}.pdf"), fig.savefig)
    plt.close(fig)

    # Companion figure on the missing-BAND axis. The group-count curve above is the paper's unit
    # of analysis, but its x-axis is not an information-loss axis (groups hold 1-3 bands), so a
    # reader comparing "3 missing groups" here with "3 missing bands" elsewhere is comparing
    # different corruptions. Same evaluations, honest axis, separate file so the main figure is
    # untouched.
    fig2, ax2 = plt.subplots(figsize=(3.7, 2.8))
    for k in keys:
        me2 = np.array([np.mean(_by_band[(k, nb)]) for nb in _band_counts])
        ax2.plot(_band_counts, me2, styles[k], color=colors[k], lw=1.8, ms=4, label=labels[k])
    ax2.set_xlabel("Number of missing spectral bands")
    ax2.set_ylabel("mIoU (%)")
    ax2.set_title(f"Missing-band robustness by BAND count ({n} seeds)", fontsize=8.5)
    ax2.grid(alpha=0.3); ax2.legend(fontsize=6.0, frameon=False, loc="upper right")
    fig2.tight_layout()
    _atomic_write(P(f"figs/fig_degradation_cloudsen12_bands{sfx}.pdf"), fig2.savefig)
    plt.close(fig2)

    # ---- console ----
    print(f"\n===== Phase 8 CloudSEN12 (sampled per-pixel spectral classification, {n} seeds) =====")
    print(f"drop-policy={args.drop_policy} ({results[0]['n_drop_sets']} drop sets/method/seed)  "
          f"train subsample={results[0]['n_train_sub']} px/seed")
    print("miss  " + "  ".join(f"{k:>9}" for k in keys))
    for i in xs:
        print(f"{i:4d}  " + "  ".join(f"{stats[k][0][i]:6.1f}" for k in keys))
    print("\nAUDC (mean+/-SD over seeds): "
          + "  ".join(f"{k}={audc_s[k].mean():.1f}+/-{float(_sd(audc_s[k])):.1f}" for k in keys))
    for s in states:
        print(f"{s:>12} mIoU: " + "  ".join(f"{k}={scen[k][s].mean():.1f}" for k in keys))
    print("\nper-class IoU (proposed):")
    for s in states:
        pc = _nanmean(scen_pc["proposed"][s], 0)
        print(f"  {s:>12}: " + "  ".join(f"{c}={v:.1f}" for c, v in zip(CLASS_NAMES, pc)))
    # The old line printed a bare `proposed best AUDC? True/False` from a single mean, with no
    # uncertainty attached to a headline claim. Report the margin AND whether it survives a paired
    # ROI bootstrap, so a 0.1-point lead inside a 1.5-point spread cannot read as a win.
    print("\npaired ROI bootstrap, Proposed - baseline (95% CI; * = excludes 0):")
    for s in states:
        parts = []
        for k in keys:
            if k == "proposed":
                continue
            d = boot[(s, "proposed")] - boot[(s, k)]
            lo, hi = _ci(d)
            parts.append(f"{k}={d.mean():+.1f}[{lo:+.1f},{hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}")
        print(f"  {s:>12}: " + "  ".join(parts))
    # Does it matter WHICH groups go missing, or only HOW MANY? Exhaustive enumeration is what
    # makes this answerable: at each m the spread across drop sets is the effect of the CHOICE,
    # and the seed SD is the noise floor it has to beat.
    print("\nwhich groups vs how many (proposed) — spread across drop sets vs seed noise:")
    for _r in [r_ for r_ in _ds_rows if r_[0] == "proposed" and r_[1] > 0]:
        _ss, _sn = float(_r[10]), float(_r[11])
        _ratio = (_ss / _sn) if _sn > 0 else float("inf")
        print(f"  m={_r[1]}  {_r[2]:>3} sets   worst={_r[4]:<18}{float(_r[5]):5.1f}   "
              f"best={float(_r[8]):5.1f}   sd(sets)={_ss:5.2f} vs sd(seeds)={_sn:5.2f}"
              f"   ratio={_ratio:.1f}x")
    if args.pe_ablation:
        # Sequential again, and exact for the same reason the L1C->L2A split is: the two steps are
        # differences of the same three measured arms, so they sum to the total by construction.
        # What it does NOT give is a symmetric attribution — B7 is the only ordering available,
        # since "wavelength PE without dropout" is B6's recipe with a different PE, not measured.
        print("\nmechanism decomposition, B6 -> B7 -> Proposed (mean over seeds):")
        for s in states:
            d_drop = scen["b7"][s].mean() - scen["b6"][s].mean()
            d_pe = scen["proposed"][s].mean() - scen["b7"][s].mean()
            print(f"  {s:>12}: group-dropout {d_drop:+6.2f}   wavelength-PE {d_pe:+6.2f}"
                  f"   total {d_drop + d_pe:+6.2f}")
    _best = max(keys, key=lambda k: audc_s[k].mean())
    print(f"\nhighest mean AUDC: {_best} ({audc_s[_best].mean():.2f}); proposed="
          f"{audc_s['proposed'].mean():.2f}+/-{float(_sd(audc_s['proposed'])):.2f}")
    print("params/optimizer-steps: " + "  ".join(f"{k}={n_params[k] / 1e3:.1f}k/{steps[k]}"
                                                 for k in keys))
    # Stamp the flagship's provenance beside its numbers: a --smoke here used to overwrite these
    # exact files, so "which run produced this table" was unanswerable from the table alone.
    # EVERY deliverable, not just the curve. Stamping one of them left the others in exactly the
    # state this stamp exists to prevent: cited tables whose run is unrecoverable from the file.
    # They come from one run, so they share one record.
    _prov = {"task": "sampled per-pixel spectral classification on CloudSEN12 labels "
                     "(NOT spatial semantic segmentation)",
             "n_train_px": int(Xtr.shape[0]), "n_test_px": int(Xte.shape[0]),
             "n_train_px_per_seed": int(results[0]["n_train_sub"]),
             "n_test_roi": n_roi, "n_test_patches": int(np.unique(pid_te).size),
             "class_support": {c: int(v) for c, v in zip(CLASS_NAMES, support)},
             "l2a_test_available": Xte_l2a is not None, "groups": len(s2_physical_groups()),
             "drop_policy": args.drop_policy,
             "n_drop_sets_per_method_per_seed": int(results[0]["n_drop_sets"]),
             "arms": list(keys), "pe_ablation": bool(args.pe_ablation),
             "n_seeds": n, "seed_sd_ddof": 1, "roi_boot_replicates_per_seed": int(args.boot),
             "loader_seeds": {"train": 12345, "test": 54321},
             "params": n_params, "optimizer_steps": steps,
             # The batch the models were ACTUALLY trained at (P2.auto_bs of the per-seed
             # subsample). A hyperparameter, not an implementation detail: results at this bs are
             # not comparable with a bs=256 run of the same phase, and a sidecar that omitted it
             # would present the two as the same experiment.
             # .get, not [...]: this line runs AFTER every seed has trained, and phase2gabl's
             # enumeration_cap precedent applies verbatim -- a stamp-time KeyError over a metadata
             # field would throw away the whole run to record a fact about it. None in the sidecar
             # says "not recorded" (it is exactly how a stubbed run_seed in the reporting-guard
             # test presents); a traceback here says nothing and costs every model.
             "train_bs": results[0].get("train_bs"),
             # Fingerprint the INPUTS, not just the code: a dataset re-exported with identical
             # shapes passes every size check while moving every number, and the provenance would
             # otherwise record the same command for two different experiments.
             "dataset_manifest": manifest,
             "label_scan_full_split": label_scan, "class_balance_pop_vs_sample": balance,
             "reflectance_profile": refl_profile}
    _names = ("curve", "scenarios", "perclass", "paired", "bandcurve", "dropsets",
              "raw_curve", "raw_scen")
    for _n in _names:
        stamp(P(f"results_phase8_cloudsen12_{_n}{sfx}.csv"), args, extra=_prov)
    # Print the paths ACTUALLY written. These were once hardcoded without `sfx`, so a --smoke run
    # announced that it had written the real deliverables while correctly writing *_smoke files --
    # an operator reading the log would believe the paper's tables had just been regenerated.
    print(f"wrote: {P(f'figs/fig_degradation_cloudsen12{sfx}.pdf')}")
    print(f"       {P(f'figs/fig_degradation_cloudsen12_bands{sfx}.pdf')}")
    for _n in _names:
        print(f"       {P(f'results_phase8_cloudsen12_{_n}{sfx}.csv')}")


if __name__ == "__main__":
    main()
