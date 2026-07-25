#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 supporting panel — Design A separability of one scene under different SENSOR BAND-SETS.

WHAT THIS IS. Take the Indian Pines cube and project it, through measured (or Gaussian) spectral
response functions, onto the band-sets of Sentinel-2 MSI (12 L2A surface-reflectance bands) and
Landsat-8 OLI (7 L2 surface-reflectance bands). Train a separate MLP per band-set on a spatially
disjoint train split and evaluate it on that same band-set's held-out test split. OLI physically
lacks S2's red-edge (705/740/783 nm) and narrow water-vapour (945 nm) bands, so on an agricultural
scene it is expected to score below S2. Every band-set sees the SAME spatial split per seed, and
normalization uses each band-set's own TRAIN-split statistics only.

WHAT THIS IS NOT — three limits, each of which used to be overstated here:

1. NOT cross-sensor transfer. No model is trained on one sensor and tested on another (the band-sets
   do not even share a dimensionality) and there is no source->target domain shift. This measures the
   IN-DISTRIBUTION separability achievable from each band-set. For a real source->shift transfer
   protocol see phase5_ab_flagship.py, which trains once on a clean source and evaluates under shift.

2. NOT "the scene as Sentinel-2/Landsat would see it", and NOT an HLS-equivalent simulation. This is
   a SPECTRAL-ONLY bandpass projection AT FIXED SPATIAL SUPPORT. Every band-set stays on Indian
   Pines' own ~20 m AVIRIS pixel grid and shares one pixel-level ground truth, so nothing here models
   ground sample distance (S2 is 10/20/60 m by band, OLI 30 m), PSF/MTF, mixed pixels, spatial
   resampling or co-registration, and nothing models sensor SNR, quantization, stray light,
   calibration uncertainty, BRDF or atmospheric-correction differences. Real HLS additionally applies
   BRDF normalization, atmospheric correction and a common-band bandpass adjustment onto a shared
   30 m grid; none of that happens here. The GSD omission does not have a clean direction: all 7 OLI
   bands are given a finer grid than they have (over-crediting OLI), but so are S2's 60 m B1 and B9
   (over-crediting S2, and B9 is one of the bands S2's advantage is attributed to). Treat the spatial
   term as UNMODELLED, not as a conservative margin.

3. NOT an isolation of the red-edge. S2 and OLI differ in band COUNT, red-edge presence, the narrow
   945 nm water-vapour band, NIR band placement, band widths and visible/SWIR response shapes all at
   once. The measured S2-OLI gap is a BAND-SET effect; attributing it specifically to the red-edge
   requires the nested ablation this script provides under --red-edge-ablation (S2 full vs the SAME
   S2 minus B5/B6/B7 only), which changes exactly three bands and nothing else.

BAND-SET CONTRACT (the defect that made --srf incomparable). `bandsim.srf.pyspectral_srf` returns
whatever the RSR store holds -- 13 bands for S2 (including B10 cirrus, which is not in the L2A
product) and 9 for OLI (including the 15 m panchromatic and cirrus) -- while `gaussian_srf` driven by
this repo's centre tables returns the 12/7 product bands. This script used to pass no exclusion, so
`--srf pyspectral` and `--srf gaussian` silently ran on DIFFERENT band sets, different input
dimensions and different parameter counts: they were never the "measured vs synthetic SRF"
sensitivity analysis the flag advertises. Worse, S2 B10 / OLI cirrus sit at ~1373 nm, right on the
water-vapour gap the Indian Pines 'corrected' cube removes (~1378-1436 nm), so those bands were being
synthesized from a surviving TAIL and renormalized back to row-sum 1 -- a finite, plausible-looking
channel measuring roughly 1362 nm and calling itself cirrus. Both paths now go through
`bandsim.srf.sensor_bandset`, which pins one canonical band set matched BY CENTRE WAVELENGTH (band
NAMES are not comparable across the two sources: both spell an OLI band 'B6' and mean 1373 vs 1609
nm) and fails closed on any band the axis does not resolve.

CAPACITY CONFOUND. Swapping the sensor swaps the spectral information AND the MLP's input dimension,
hence its parameter count (the first Linear is bands x hidden):
    full-HSI      200 bands   121,360 params
    Sentinel-2     12 bands    73,232 params
    Landsat-8 OLI   7 bands    71,952 params
The full-HSI-vs-multispectral gap is therefore NOT attributable to spectral content alone. Use
--match-capacity to shrink every model to the smallest band-set's budget (71,952 params: full-HSI
h=256->180, S2 256->253, OLI unchanged at 256) so the parameter counts land within 1.005x. Note what
that does and does not control: it equalises the PARAMETER COUNT, which is one axis of capacity, not
hidden width, input dimension, optimization conditioning or effective function class.

Do NOT argue "S2 and OLI differ by only 1.8% of parameters, which is small next to a several-mIoU
gap" -- a percentage of parameters and a number of mIoU points are different units and their relative
sizes carry no inference. The argument that DOES work is the matched run: under --match-capacity the
S2 model is strictly SMALLER than the OLI model (71,615 vs 71,952 params), so if S2 still wins, the
win cannot be bought with capacity. Run --paired-capacity to get both arms on the IDENTICAL seeds and
splits in one process and read the per-seed paired difference; comparing a matched run against an
unmatched run with a DIFFERENT seed set confounds the capacity effect with split sampling, which is
the error the previously-recorded control here made (its own built-in scale check, an OLI arm whose
configuration was byte-identical across the two columns, moved by more than the effect being bounded).

RESULTS LIVE IN THE CSV, NOT IN THIS DOCSTRING. Every run writes a per-seed long-form table plus a
summary, both stamped with provenance; read those. The previously-quoted mIoU table was INVALIDATED
on 2026-07-20 by the band-set contract fix above (it was measured on the 13/9-band store sets, with a
cirrus band synthesized across a data gap) and has been removed rather than left to be re-quoted.

MACRO ESTIMAND. mIoU and AA are averaged over a FIXED class set -- the classes present in the test
split of EVERY requested offset, from bandsim.metrics.common_class_set -- and OA/kappa stay over all
labelled pixels. Without that, each seed's macro average is taken over whichever classes its own
split happens to contain, and their mean is not an average of any one quantity: measured here, the
guard band removes Oats (20 px in the scene) for offsets 0/7/8/9 and Grass-pasture-mowed (28 px) for
1/2, giving 14 of 16 for every offset list from 0..1 through 0..9. That is not a rounding detail -- a
class with ~10 test pixels carries 1/15 of the average with an essentially random IoU, which can move
the headline by more than the band-set effect being measured. The average-over-present variant is
still written to every CSV as mIoU_present/AA_present, and per-class IoU per seed is in the raw
table, so any other convention can be recomputed without re-running. Same convention, and the same
single definition, as phase1_indian_pines and phase2_degradation.
    The exclusion is NOT obviously neutral for this panel's claim, and that should be said plainly:
both dropped classes (Grass-pasture-mowed, Oats) are grass/vegetation, which is exactly the kind of
discrimination the red-edge argument is about. The defence is that neither can be measured on ~10
test pixels, not that they are irrelevant -- so before quoting a red-edge conclusion, check their
per-class IoU columns in the raw table rather than assuming the exclusion cut nothing.

Honesty: a FIRST-ORDER radiometric simulator, and Indian Pines 'corrected' has already had 20
water-vapour/low-SNR bands removed. Error bars are the spread over `--seeds`, which are one-pixel
shifts of the same checkerboard, NOT independent scene replicates: seeds 0-4 share a mean pairwise
test-set IoU of 0.448 (adjacent seeds 0.632) against 0.333 for two genuinely independent half-splits,
so the spread UNDERSTATES sampling uncertainty and no confidence interval is quoted from it. Training
minimises unweighted cross-entropy while the headline metric is macro mIoU, so large classes dominate
the objective. See docs/guide/03_physical_simulation.md.

Outputs (../paper/), where [_tag] is the experiment variant (_paired / _matched / _rededge) and
[_smoke] marks a --smoke run, which is NEVER a deliverable:
  figs/fig_cross_sensor[_tag][_smoke].pdf            - mIoU bar chart (mean +/- spread)
  results_phase2_cross_sensor[_tag][_smoke].csv      - one row per condition (summary)
  results_phase2_cross_sensor[_tag][_smoke]_raw.csv  - one row per (seed, condition): paired analysis

Usage:
  python experiments/phase2_cross_sensor.py --seeds 0 1 2 3 4
  python experiments/phase2_cross_sensor.py --seeds 0 1 2 3 4 --paired-capacity
  python experiments/phase2_cross_sensor.py --seeds 0 1 --epochs 60 --match-capacity
  python experiments/phase2_cross_sensor.py --seeds 0 1 2 3 4 --red-edge-ablation --match-capacity
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch  # threads/device configured adaptively by bandsim.hw / bandsim.parallel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.io import load_mat_cube, disjoint_block_split, AVIRIS_WL_NM, axis_sha256
from bandsim.srf import sensor_bandset, apply_srf
from bandsim.metrics import (miou, overall_accuracy, average_accuracy, cohen_kappa,
                             per_class_iou, common_class_set, macro_over)
from bandsim.model import MLPBaseline
from bandsim import hw, parallel
from bandsim.provenance import stamp, file_sha256

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)

DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "indian_pines")
NUM_CLASSES = 16

# Effective-centre window (nm) defining "red-edge" for --red-edge-ablation. Selected by WAVELENGTH,
# never by band name: the two SRF sources name bands differently (see bandsim.srf), so a name-based
# ablation would drop different bands depending on --srf. S2 B5/B6/B7 sit at 705/740/783 nm; the
# window excludes B4 (665) and B8 (833) with >20 nm of margin on both sides.
RED_EDGE_NM = (690.0, 800.0)
# ONE guard width. It governs BOTH the split every seed trains on and the class set the macro metric
# is defined over, and those two must be the same split: a `guard` changed in one place and not the
# other silently redefines the estimand against splits it was not derived from. (run_seed's
# self-check would catch the mismatch, but only after training everything.)
GUARD = 1
FULL, S2, OLI = "full-HSI", "Sentinel-2", "Landsat-8 OLI"
S2RE = "Sentinel-2 minus red-edge"


def synth_sensor(cube, wl, sensor, srf_source="pyspectral", fwhm=30.0):
    """Synthesize a sensor's band cube (H, W, B) through ONE canonical, grid-checked band-set.

    Thin wrapper over `bandsim.srf.sensor_bandset`, which owns the band contract: it selects the
    product band-set by centre wavelength (identically for measured and Gaussian SRFs), and fails
    closed on any band this wavelength axis cannot resolve. Returns (cube_out, info) where `info`
    carries the band names, effective centres, sampling diagnostics, selection report and W hash --
    all of which go into provenance, because "which bands were these, exactly" turned out to be the
    question this panel's numbers depended on most.
    """
    # B1 EXCLUDED BY DECISION, not by silent drop (decided with the 2026-07-20 axis correction):
    # the AVIRIS-based synthesis axis starts at 437.7 nm;
    # S2/OLI B1 (443 nm) has 16-24% of its response below the first sampled wavelength, and synthesizing a truncated, renormalised B1 would fabricate signal the sensor never measured on this cube. The exclusion and its
    # reason are recorded in info["selection"]["excluded"], so provenance names the subset.
    info = sensor_bandset(wl, sensor, source=srf_source, fwhm_nm=fwhm, exclude=("B1",),
                          exclude_reason=("the AVIRIS-based synthesis axis starts at 437.7 nm; S2/OLI B1 (443 nm) has 16-24% of its response below the first sampled wavelength, and synthesizing a truncated, renormalised B1 would fabricate signal the sensor never measured on this cube"))
    return apply_srf(cube, info["W"]), info


def drop_bands(cube, info, lo_nm, hi_nm):
    """Drop every band whose EFFECTIVE centre lies in [lo_nm, hi_nm] -> (cube, info, dropped_names).

    The ablation arm: same scene, same sensor, same SRFs, same spatial split, N fewer bands. Removing
    bands from the ALREADY-SYNTHESIZED cube (rather than re-running the SRF over a subset) guarantees
    the surviving bands are bit-identical to the full arm's, so the only difference between the two
    conditions is the presence of the dropped bands.
    """
    centers = np.asarray(info["centers_nm"], float)
    keep = ~((centers >= float(lo_nm)) & (centers <= float(hi_nm)))
    if keep.all():
        raise ValueError(f"no band has an effective centre in [{lo_nm}, {hi_nm}] nm; centres are "
                         f"{[round(c, 1) for c in centers]} -- the ablation would be a no-op")
    if not keep.any():
        raise ValueError(f"[{lo_nm}, {hi_nm}] nm would drop EVERY band")
    dropped = [f"{n}@{c:.0f}nm" for n, c, k in zip(info["names"], centers, keep) if not k]
    sub = dict(info)
    sub["names"] = [n for n, k in zip(info["names"], keep) if k]
    sub["centers_nm"] = [float(c) for c, k in zip(centers, keep) if k]
    sub["W"] = info["W"][keep]
    # The diagnostics must describe the bands this arm ACTUALLY has. Carrying the parent's full dict
    # would put the ablated-away bands into this arm's provenance, i.e. the record would claim the
    # red-edge was present in the condition built to not have it.
    sub["diagnostics"] = {n: d for n, d in info["diagnostics"].items() if n in set(sub["names"])}
    sub["ablated_out"] = dropped
    return cube[..., keep], sub, dropped


def mlp_param_count(n_bands, num_classes=NUM_CLASSES, hidden=256):
    """Trainable parameter count of MLPBaseline(n_bands, num_classes, hidden) WITHOUT building it.

    Closed-form so the confound can be quantified and equalised before any training starts (and so
    the CSV can carry it even for a band-set that was never run). The band count enters only through
    the first Linear -- which is exactly why the panel is capacity-confounded: bands*hidden + hidden
    is 51,456 of full-HSI's 121,360 params but only 2,048 of OLI's 71,952.
    Pinned against the real module in tests/test_experiment_guards.py, AND re-checked against the
    live module inside train_mlp so a model-architecture change cannot silently invalidate it.
    """
    return ((n_bands * hidden + hidden)          # Linear(n_bands, hidden)
            + (hidden * hidden + hidden)         # Linear(hidden, hidden)
            + (hidden * num_classes + num_classes))   # Linear(hidden, num_classes)


def hidden_for_budget(n_bands, budget, num_classes=NUM_CLASSES):
    """Largest `hidden` whose MLPBaseline(n_bands, ...) fits within `budget` trainable parameters.

    Used by --match-capacity to remove the input-dimension confound: the parameter count is
    monotonic in `hidden` (quadratic with positive coefficients), so a plain search is exact and
    the result is the tightest common budget rather than an approximation. A band count too large
    for the budget RAISES instead of returning 0 or a negative width, which would silently build a
    model with a degenerate hidden layer and still report a mIoU for it.

    Linear search on purpose: `hidden` is bounded by 256 here, so this is a few hundred integer
    operations run once per band-set before any training -- a closed-form quadratic root would need
    its own floor/rounding correctness argument to save time that is not being spent.
    """
    if n_bands < 1 or num_classes < 2 or budget < 1:
        raise ValueError(f"hidden_for_budget needs n_bands>=1, num_classes>=2, budget>=1; got "
                         f"n_bands={n_bands}, num_classes={num_classes}, budget={budget}")
    if mlp_param_count(n_bands, num_classes, hidden=1) > budget:
        raise ValueError(
            f"parameter budget {budget:,} is too small for a {n_bands}-band model: even hidden=1 "
            f"needs {mlp_param_count(n_bands, num_classes, hidden=1):,} params. Raise the budget "
            f"(or drop the band-set) rather than training a width-0 hidden layer.")
    h = 1
    while mlp_param_count(n_bands, num_classes, hidden=h + 1) <= budget:
        h += 1
    return h


def train_mlp(Xtr, ytr, seed, epochs=60, hidden=256, lr=1e-3, bs=256):
    """Train ONE MLPBaseline on already-normalized SOURCE features and return it in ``.eval()`` mode.

    Returned in eval mode, NOT frozen: the parameters keep ``requires_grad=True``. That is deliberate
    -- the mode is what evaluation depends on, and freezing would stop a caller fine-tuning. Training
    and evaluation are split into two functions on purpose so a caller can train ONCE on a source
    condition and then evaluate the SAME model under several shifted target conditions without
    re-training (source->shift transfer; see phase5_ab_flagship.py).
    """
    if epochs < 1 or bs < 1:
        raise ValueError(f"epochs and batch size must be >= 1; got epochs={epochs}, bs={bs}")
    Xtr = np.asarray(Xtr)
    if Xtr.ndim != 2 or Xtr.shape[0] == 0 or Xtr.shape[1] == 0:
        raise ValueError(f"train features must be a non-empty 2-D array, got shape {Xtr.shape}")
    ytr = np.asarray(ytr)
    if ytr.shape[0] != Xtr.shape[0]:
        raise ValueError(f"train features/labels length mismatch: {Xtr.shape[0]} vs {ytr.shape[0]}")
    if not np.isfinite(Xtr).all():
        raise ValueError(f"train features contain {int((~np.isfinite(Xtr)).sum())} non-finite values; "
                         f"training would return a model whose predictions are NaN-driven")
    if ytr.min() < 0 or ytr.max() >= NUM_CLASSES:
        raise ValueError(f"train labels outside [0, {NUM_CLASSES}): range [{ytr.min()}, {ytr.max()}] "
                         f"-- CrossEntropyLoss would index out of bounds, or silently train a class "
                         f"the metrics never score")
    dev = hw.device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLPBaseline(Xtr.shape[1], NUM_CLASSES, hidden=hidden).to(dev)
    # The closed-form count above picks the --match-capacity widths BEFORE anything is built, so a
    # change to MLPBaseline would silently make every reported `params` a fiction and every "matched"
    # run unmatched. Check the formula against the object that was actually constructed.
    built = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expect = mlp_param_count(Xtr.shape[1], NUM_CLASSES, hidden=hidden)
    if built != expect:
        raise ValueError(f"mlp_param_count is stale: MLPBaseline({Xtr.shape[1]}, {NUM_CLASSES}, "
                         f"hidden={hidden}) has {built:,} trainable params but the formula says "
                         f"{expect:,}. Every reported parameter count and every capacity-matched "
                         f"width is wrong until the formula is updated to match the module.")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr.astype(np.float32)).to(dev); yt = torch.from_numpy(ytr).long().to(dev)
    n = Xt.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        for s in range(0, n, bs):
            bi = perm[s:s + bs]
            opt.zero_grad()
            lossf(model(Xt[bi]), yt[bi]).backward()
            opt.step()
    model.eval()
    return model


def eval_mlp(model, Xte, yte, class_set=None):
    """Evaluate a model on already-normalized TARGET features. Runs under ``no_grad`` and never
    updates the model (MLPBaseline has no BN/running stats), so it is safe to call repeatedly on
    different shifted conditions with the identical model + scaler.

    The device comes from the MODEL, not from hw.device(): this function is imported by phase5, which
    trains once and evaluates many times, and a caller that moved the model would otherwise hit a
    device mismatch. Eval mode is enforced rather than assumed -- it costs nothing today (dropout=0)
    but stops a train-mode model returning stochastic metrics the day the architecture gains dropout
    or normalization, and the caller's mode is restored so this stays a read-only operation.

    `class_set` (0-based, from bandsim.metrics.common_class_set) fixes WHICH classes the macro
    metrics average over, so every split reports the same estimand. `mIoU`/`AA` are that fixed-set
    macro when it is given; `mIoU_present`/`AA_present` always carry the average-over-whatever-is-
    present variant beside it, so neither convention has to be recovered by re-running. Default None
    keeps the legacy behaviour with `mIoU` unchanged, which is the contract phase5 imports.
    """
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(torch.from_numpy(np.asarray(Xte).astype(np.float32)).to(dev))
            pred = logits.argmax(1).cpu().numpy()
    finally:
        model.train(was_training)
    out = {
        # OA and kappa are NOT macro averages -- an absent class does not change what they estimate
        # -- so they stay over every labelled pixel regardless of the class set.
        "OA": overall_accuracy(yte, pred),
        "kappa": cohen_kappa(yte, pred, NUM_CLASSES) * 100,
        "mIoU_present": miou(yte, pred, NUM_CLASSES),
        "AA_present": average_accuracy(yte, pred, NUM_CLASSES),
        # NaN for classes absent from y_true, so ANY class-set convention can be recomputed from the
        # CSV without a re-run -- which matters because the guard band really does eliminate a
        # different tiny class for different split offsets on this scene.
        "per_class_iou": per_class_iou(yte, pred, NUM_CLASSES),
    }
    if class_set is None:
        out["mIoU"], out["AA"] = out["mIoU_present"], out["AA_present"]
    else:
        out["AA"], out["mIoU"] = macro_over(yte, pred, class_set, NUM_CLASSES)
    return out


def train_eval_mlp(Xtr, ytr, Xte, yte, seed, epochs=60, hidden=256, lr=1e-3, bs=256, class_set=None):
    """Convenience wrapper: train on a condition's TRAIN split and evaluate on that SAME
    condition's TEST split — i.e. IN-DISTRIBUTION separability of one band-set (used by the
    per-sensor panel below). For source->shift transfer, call ``train_mlp`` once on the source
    then ``eval_mlp`` per shifted target instead (see phase5_ab_flagship.py)."""
    model = train_mlp(Xtr, ytr, seed, epochs=epochs, hidden=hidden, lr=lr, bs=bs)
    return eval_mlp(model, Xte, yte, class_set=class_set)


def run_seed(seed, sensors, conditions, gt, block, epochs, model_seed_offset=0, class_set=None):
    """One split seed: one spatially-disjoint split, then train+evaluate EVERY condition on it.

    A "condition" is (band-set, hidden width), so the sensor comparison and the capacity comparison
    are the same loop. That matters: it is what lets --paired-capacity put a matched and an unmatched
    model on the BYTE-IDENTICAL split, which is the only way the capacity delta is identified. Running
    the two arms as separate invocations with different seed sets confounds capacity with split
    sampling, and that confound is larger than the effect being bounded.

    `split_seed` and `model_seed` are recorded separately so the two variance sources can be told
    apart later; with the default offset of 0 they are equal and every number is unchanged.
    """
    split_seed = int(seed)
    model_seed = int(seed) + int(model_seed_offset)
    tr_mask, te_mask = disjoint_block_split(gt, block=block, guard=GUARD, offset=split_seed)
    ytr = gt[tr_mask].astype(int) - 1
    yte = gt[te_mask].astype(int) - 1
    # The mIoU estimand depends on WHICH classes reach the test split: bandsim.metrics.miou averages
    # over ground-truth-present classes, so a seed whose guard band eliminates a small class averages
    # over a different set than its neighbours. Record it per seed instead of assuming it is 16.
    split = {"split_seed": split_seed, "model_seed": model_seed,
             "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
             "n_train_classes": int(np.unique(ytr).size), "n_test_classes": int(np.unique(yte).size),
             "test_classes": sorted(int(c) for c in np.unique(yte))}
    out = {}
    for cond in conditions:
        sc = sensors[cond["sensor"]]
        Xtr = sc[tr_mask]; Xte = sc[te_mask]
        mu = Xtr.mean(0); sdv = Xtr.std(0)
        # A band with (near-)zero train variance would be amplified by 1/sdv into pure noise. Neutralise
        # it and COUNT it, rather than adding an epsilon that turns 1e-12 of noise into a unit-scale
        # feature. `constant_bands` reaches the CSV so "this band-set had a dead channel" stays visible.
        dead = sdv < 1e-8
        sdv = np.where(dead, 1.0, sdv)
        Xtr = (Xtr - mu) / sdv; Xte = (Xte - mu) / sdv
        if dead.any():
            Xtr[:, dead] = 0.0; Xte[:, dead] = 0.0
        m = train_eval_mlp(Xtr, ytr, Xte, yte, model_seed, epochs=epochs, hidden=cond["hidden"],
                           class_set=class_set)
        m["constant_bands"] = int(dead.sum())
        out[cond["label"]] = m
    return {"split": split, "metrics": out}


def sample_sd(values):
    """Cross-seed dispersion as the SAMPLE standard deviation (ddof=1); NaN below two samples.

    `np.std` defaults to the POPULATION formula, which understates a sample's spread by
    sqrt((n-1)/n) -- 10.6% at n=5. Every `+/-` reported by this script is a handful of seeds treated
    as a sample, so every one of them needs ddof=1. One definition, used by the CSV, the figure and
    the printed summary alike, so the three cannot drift apart.

    NaN rather than 0.0 at n=1: a single run has no dispersion to report, which is not the same
    statement as zero dispersion, and 0.00 in a table reads as the latter.

    NOT for feature normalization. `Xtr.std(0)` in run_seed is the spread of the training pixels
    themselves -- there the population formula is the right one, and a sweep that changed every
    `.std(` in this file to ddof=1 would corrupt the scaler.
    """
    v = np.asarray(values, float)
    return float(v.std(ddof=1)) if v.size > 1 else float("nan")


def build_conditions(band_n, arms):
    """[{label, sensor, arm, bands, hidden, params}] for every (band-set x capacity arm) to run.

    The matched arm's budget is the SMALLEST band-set's default-width cost, so the widest model gives
    up width to pay for its input layer while the narrowest is left UNTOUCHED (its matched width is
    still 256). Under --paired-capacity that untouched band-set is a SELF-CHECK, not a noise estimate:
    both of its arms are the same configuration on the same split with the same model seed, so their
    difference must be exactly 0. A non-zero value means the two arms are not actually paired, which
    would silently invalidate every other capacity delta in the table.

    (This is a stronger property than the control it replaces. The previous control compared a 2-seed
    matched run against a 5-seed unmatched one and read the untouched band-set's shift as a bound on
    seed noise -- but that shift WAS the seed-set difference, so it bounded nothing about capacity.)
    """
    budget = min(mlp_param_count(nb) for nb in band_n.values())
    conds = []
    for arm in arms:
        for sensor, nb in band_n.items():
            hidden = 256 if arm == "unmatched" else hidden_for_budget(nb, budget)
            label = sensor if len(arms) == 1 else f"{sensor} [{arm}]"
            conds.append({"label": label, "sensor": sensor, "arm": arm, "bands": nb,
                          "hidden": hidden, "params": mlp_param_count(nb, hidden=hidden)})
    return conds, budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--srf", default="pyspectral", choices=["pyspectral", "gaussian"],
                    help="real measured RSR (pyspectral) or synthetic gaussian; BOTH now resolve to "
                         "the same canonical band-set, so this varies band SHAPE only")
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent seed workers (default: adaptive; also BANDSIM_WORKERS)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                    help="device for the workers (default: auto; also BANDSIM_DEVICE)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="disable deterministic algorithms for a small speedup")
    ap.add_argument("--match-capacity", action="store_true",
                    help="equalise trainable parameters across band-sets (shrink each model's hidden "
                         "width to the smallest band-set's budget); writes to *_matched.csv")
    ap.add_argument("--paired-capacity", action="store_true",
                    help="run the matched AND unmatched arms on the SAME seeds and splits in one "
                         "process and report the per-seed paired capacity delta; writes *_paired.csv")
    ap.add_argument("--red-edge-ablation", action="store_true",
                    help="add an arm that is Sentinel-2 with ONLY B5/B6/B7 removed, so the red-edge "
                         "contribution is measured against the same sensor instead of inferred from "
                         "the S2-OLI gap; writes *_rededge.csv")
    ap.add_argument("--model-seed-offset", type=int, default=0,
                    help="model_seed = split_seed + offset. Default 0 (identical, as before); set it "
                         "to vary model initialisation independently of the spatial split")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed / 12 epochs, written to *_smoke artefacts. NOT a deliverable")
    args = ap.parse_args()
    if args.epochs < 1 or args.block < 2:
        ap.error(f"--epochs must be >= 1 and --block >= 2; got {args.epochs} and {args.block}")
    if len(set(args.seeds)) != len(args.seeds):
        ap.error(f"--seeds must be unique; duplicates would be counted as extra samples: {args.seeds}")
    if args.match_capacity and args.paired_capacity:
        ap.error("--match-capacity and --paired-capacity are alternatives: --paired-capacity already "
                 "runs the matched arm, alongside the unmatched one on the same splits")
    # Applied AFTER validation, so --smoke overrides whatever the caller asked for rather than
    # validating a set of arguments that is then thrown away.
    sfx = ""
    if args.smoke:
        args.seeds = [0]; args.epochs = 12
        sfx = "_smoke"
        print("[smoke] 1 seed / 12 epochs — writing *_smoke artefacts, NOT the deliverables")
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device)
    print('HW:', hw.info(), '| SRF:', args.srf)

    cube_path = os.path.join(DATA_DIR, "Indian_pines_corrected.mat")
    gt_path = os.path.join(DATA_DIR, "Indian_pines_gt.mat")
    cube = load_mat_cube(cube_path, key="indian_pines_corrected").astype(np.float64)
    gt = load_mat_cube(gt_path, key="indian_pines_gt").astype(int)
    wl = AVIRIS_WL_NM
    # Data contract. Each of these would otherwise produce a NUMBER rather than an error: a transposed
    # cube trains on the wrong axis, a wavelength/channel mismatch mis-assigns every SRF row, and a
    # stray label silently changes which classes the metrics average over.
    if cube.ndim != 3 or gt.ndim != 2:
        raise ValueError(f"expected a 3-D cube and a 2-D label map, got {cube.shape} and {gt.shape}")
    if cube.shape[:2] != gt.shape:
        raise ValueError(f"cube/ground-truth spatial shape mismatch: {cube.shape[:2]} vs {gt.shape}")
    if cube.shape[-1] != wl.size:
        raise ValueError(f"cube has {cube.shape[-1]} channels but the AVIRIS axis has {wl.size} "
                         f"wavelengths; every synthesized band would integrate the wrong channels")
    if not np.isfinite(cube).all():
        raise ValueError(f"cube has {int((~np.isfinite(cube)).sum())} non-finite values")
    labels = set(int(v) for v in np.unique(gt))
    if not labels <= set(range(NUM_CLASSES + 1)):
        raise ValueError(f"ground truth has labels outside 0..{NUM_CLASSES}: {sorted(labels)}")

    # THE MACRO ESTIMAND, fixed before anything is trained. miou/average_accuracy average over the
    # classes PRESENT in y_true, and on this scene the guard band really does eliminate a different
    # tiny class for different offsets (Oats, 20 px, for offsets 0/7/8/9; Grass-pasture-mowed, 28 px,
    # for 1/2), so a per-seed macro would average over different class sets and their mean would not
    # be an average of any one quantity. Pinning the set makes every seed report the same estimand --
    # the convention phase1 and phase2_degradation already adopted, sharing this one definition.
    class_set, split_count = common_class_set(gt, args.block, args.seeds, guard=GUARD,
                                              num_classes=NUM_CLASSES)
    excluded = [c for c in range(NUM_CLASSES) if c not in class_set]
    print(f"MACRO CLASS SET: {len(class_set)}/{NUM_CLASSES} classes present in the test split of "
          f"EVERY offset {list(args.seeds)} -- mIoU/AA average over those; OA/kappa stay over all "
          f"labelled pixels.")
    if excluded:
        print("  excluded: " + ", ".join(
            f"GT label {c + 1} (in {split_count[c]}/{len(args.seeds)} splits, "
            f"{int((gt == c + 1).sum())} px in the scene)" for c in excluded))

    s2_cube, s2_info = synth_sensor(cube, wl, "sentinel2", srf_source=args.srf)
    oli_cube, oli_info = synth_sensor(cube, wl, "landsat_oli", srf_source=args.srf)
    sensors = {FULL: cube, S2: s2_cube, OLI: oli_cube}
    band_info = {S2: s2_info, OLI: oli_info}
    if args.red_edge_ablation:
        sensors[S2RE], band_info[S2RE], dropped_re = drop_bands(s2_cube, s2_info, *RED_EDGE_NM)
        print(f"red-edge ablation: dropped {len(dropped_re)} band(s) from {S2}: {', '.join(dropped_re)}")
    for name, info in band_info.items():
        diag = info["diagnostics"]
        cov = [d["coverage"] for d in diag.values() if "coverage" in d]
        ratio = max(d["grid_dlambda_ratio"] for d in diag.values())
        print(f"{name}: {len(info['names'])} bands {info['names']}")
        print(f"    effective centres (nm): {[round(c) for c in info['centers_nm']]}")
        if info["selection"].get("dropped"):
            print(f"    NOT in the canonical surface-reflectance set, dropped: "
                  f"{info['selection']['dropped']}")
        # dlambda ratio is the gate (an interior gap); coverage is reported but not gated, because on
        # a ~9.6 nm axis it cannot separate "unresolved" from "narrower than the grid" -- see
        # bandsim.srf.grid_sampling_diagnostics.
        print(f"    worst grid_dlambda_ratio {ratio:.2f} (gate 2.0)"
              + (f"  |  quadrature coverage {min(cov):.2f}-{max(cov):.2f}" if cov else "")
              + f"  |  W sha256 {info['W_sha256'][:12]}")

    band_n = {name: sc.shape[-1] for name, sc in sensors.items()}
    arms = (["unmatched", "matched"] if args.paired_capacity
            else ["matched"] if args.match_capacity else ["unmatched"])
    conditions, budget = build_conditions(band_n, arms)
    spread = max(c["params"] for c in conditions) / min(c["params"] for c in conditions)
    print("capacity: " + "  ".join(f"{c['label']}={c['params']:,}p(h={c['hidden']})"
                                   for c in conditions))
    if args.match_capacity:
        print(f"  budget {budget:,} params, max/min = {spread:.3f}x  [PARAMETER-COUNT MATCHED]")
    elif args.paired_capacity:
        print(f"  both arms on identical splits; matched budget {budget:,} params  [PAIRED]")
    else:
        print(f"  max/min = {spread:.2f}x  [CONFOUNDED: capacity scales with band count, so the "
              f"full-HSI-vs-multispectral gap is not spectral-only. Rerun with --paired-capacity.]")

    results = parallel.run_jobs(
        run_seed, args.seeds,
        shared=dict(sensors=sensors, conditions=conditions, gt=gt, block=args.block,
                    epochs=args.epochs, model_seed_offset=args.model_seed_offset,
                    class_set=class_set),
        prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
        label="crosssensor/seed")
    if len(results) != len(args.seeds):
        raise RuntimeError(f"run_jobs returned {len(results)} results for {len(args.seeds)} seeds")

    METRICS = ("mIoU", "OA", "AA", "kappa", "mIoU_present")
    agg = {c["label"]: {m: [] for m in METRICS} for c in conditions}
    per_seed = {c["label"]: {} for c in conditions}
    splits = {}
    for sd, res in zip(args.seeds, results):
        splits[sd] = res["split"]
        for c in conditions:
            r = res["metrics"][c["label"]]
            for m in METRICS:
                agg[c["label"]][m].append(r[m])
            per_seed[c["label"]][sd] = r
        print(f"seed {sd}: " + " | ".join(f"{c['label']} {res['metrics'][c['label']]['mIoU']:.1f}"
                                          for c in conditions))
    # Self-check on the estimand: every class in the fixed set must actually be present in every
    # split that was RUN. It is derived from the same offsets, so this can only fail if the offsets
    # diverged from the seeds -- in which case macro_over would return NaN rather than a wrong
    # number, but failing here says WHY, while the two lists are still in hand.
    absent = {sd: sorted(c + 1 for c in set(class_set) - set(sp["test_classes"]))
              for sd, sp in splits.items()}
    absent = {sd: m for sd, m in absent.items() if m}
    if absent:
        raise RuntimeError(f"the macro class set is not present in every split that ran: GT labels "
                           f"{absent} are missing. The offsets the set was derived from must be the "
                           f"offsets that were run.")

    tag = ("_paired" if args.paired_capacity else "_matched" if args.match_capacity else "")
    tag += "_rededge" if args.red_edge_ablation else ""

    # RAW, one row per (seed, condition): a summary alone cannot support a paired comparison, a
    # bootstrap, or the detection of one outlier seed -- all of which this panel's claims need.
    raw_path = P(f"results_phase2_cross_sensor{tag}{sfx}_raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        # mIoU/AA are the FIXED-class-set macro; mIoU_present/AA_present are the average-over-
        # whatever-is-present variant, carried so neither convention needs a re-run to recover.
        w.writerow(["seed", "split_seed", "model_seed", "condition", "sensor", "arm", "srf",
                    "bands", "hidden", "params", "n_train", "n_test", "n_train_classes",
                    "n_test_classes", "macro_classes", "constant_bands",
                    "mIoU", "AA", "OA", "kappa", "mIoU_present", "AA_present"]
                   + [f"iou_c{k + 1}" for k in range(NUM_CLASSES)])
        for sd in args.seeds:
            sp = splits[sd]
            for c in conditions:
                r = per_seed[c["label"]][sd]
                # Blank, not "nan", for a class absent from this seed's test split: an empty cell is
                # unambiguously "not evaluated here", whereas a 0 would be read as a total miss and a
                # "nan" string silently poisons any column mean taken over it.
                pci = ["" if not np.isfinite(v) else f"{v:.4f}" for v in r["per_class_iou"]]
                w.writerow([sd, sp["split_seed"], sp["model_seed"], c["label"], c["sensor"],
                            c["arm"], args.srf, c["bands"], c["hidden"], c["params"], sp["n_train"],
                            sp["n_test"], sp["n_train_classes"], sp["n_test_classes"],
                            len(class_set), r["constant_bands"],
                            f"{r['mIoU']:.4f}", f"{r['AA']:.4f}", f"{r['OA']:.4f}",
                            f"{r['kappa']:.4f}", f"{r['mIoU_present']:.4f}",
                            f"{r['AA_present']:.4f}"] + pci)

    summary_path = P(f"results_phase2_cross_sensor{tag}{sfx}.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        # `params`/`hidden`/`capacity_matched` travel WITH the mIoU they qualify: changing the
        # band-set changes the input dimension, so without them a reader cannot tell how much of a
        # gap is spectral content and how much is a bigger model. n_seeds/epochs for the same reason.
        w.writerow(["sensor", "arm", "bands", "params", "hidden", "capacity_matched", "srf",
                    "n_seeds", "epochs", "macro_classes", "mIoU_mean", "mIoU_sd",
                    "mIoU_min", "mIoU_max", "AA_mean", "OA_mean", "kappa_mean",
                    "mIoU_present_mean", "n_test_classes_min", "n_test_classes_max"])
        # n_test_classes_{min,max} sit in the SUMMARY too, not only in the raw table: a reader who
        # never opens the per-seed file must still be able to see that the seeds being averaged did
        # not all evaluate the same classes, because that decides whether the mean means anything.
        ncmin = min(s["n_test_classes"] for s in splits.values())
        ncmax = max(s["n_test_classes"] for s in splits.values())
        for c in conditions:
            d = agg[c["label"]]
            v = np.asarray(d["mIoU"], float)
            w.writerow([c["label"], c["arm"], c["bands"], c["params"], c["hidden"],
                        int(c["arm"] == "matched"), args.srf, len(args.seeds), args.epochs,
                        # ONE dispersion column, and it is the sample SD. Shipping both formulas
                        # side by side only invites the smaller one being quoted.
                        len(class_set), f"{v.mean():.2f}",
                        "" if np.isnan(sd := sample_sd(v)) else f"{sd:.2f}",
                        f"{v.min():.2f}", f"{v.max():.2f}",
                        f"{np.mean(d['AA']):.2f}", f"{np.mean(d['OA']):.2f}",
                        f"{np.mean(d['kappa']):.2f}", f"{np.mean(d['mIoU_present']):.2f}",
                        ncmin, ncmax])

    # ---- figure --------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    labels = [c["label"] for c in conditions]
    fig, ax = plt.subplots(figsize=(max(3.4, 1.15 * len(labels)), 2.8))
    means = [float(np.mean(agg[l]["mIoU"])) for l in labels]
    # No error bar at all below two seeds: a 0-length bar is a claim of zero dispersion.
    stds = [sample_sd(agg[l]["mIoU"]) for l in labels] if len(args.seeds) > 1 else None
    palette = {FULL: "#1f6f3a", S2: "#2980b9", OLI: "#c0392b", S2RE: "#8e44ad"}
    bars = ax.bar(range(len(labels)), means, yerr=stds, width=0.6, capsize=3,
                  color=[palette.get(c["sensor"], "#7f8c8d") for c in conditions])
    for b, c in zip(bars, conditions):
        if c["arm"] == "matched":
            b.set_hatch("//")
    ax.set_xticks(range(len(labels)))
    # The parameter count goes ON the x tick, under the band count: the two move together, and that
    # is precisely what a reader must see before reading the bar heights as a spectral effect.
    ticklabels = [f"{c['sensor'].replace(' ', chr(10))}\n{c['bands']}b "
                  f"{c['params'] / 1000:.0f}k" + ("\n[matched]" if c["arm"] == "matched" else "")
                  for c in conditions]
    ax.set_xticklabels(ticklabels, fontsize=6.5)
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Separability by sensor band-set\nspectral-only, at fixed spatial support",
                 fontsize=8)
    for i, (mn, st) in enumerate(zip(means, stds or [0.0] * len(means))):
        ax.text(i, mn + st + 1, f"{mn:.1f}", ha="center", fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    caveat = (f"error bars = sample SD (ddof=1) over {len(args.seeds)} one-pixel checkerboard shifts "
              f"(mean pairwise test IoU 0.448), NOT independent replicates and NOT a CI"
              if stds is not None else "single seed: no dispersion is reportable")
    if arms == ["unmatched"]:
        caveat += f"\ncapacity varies with band count (max/min {spread:.2f}x)"
    # Offset scales with the tallest tick label: "Sentinel-2 minus red-edge [matched]" wraps to five
    # lines, and a fixed -0.30 would drop the caveat on top of it.
    n_lines = max(lbl.count("\n") for lbl in ticklabels) + 1
    ax.text(0.5, -(0.14 + 0.055 * n_lines), caveat, transform=ax.transAxes, ha="center", va="top",
            fontsize=5.6, color="#b03030")
    # bbox_inches="tight" is REQUIRED, not cosmetic: the caveat above sits OUTSIDE the axes and
    # tight_layout() does not reserve space for arbitrary text artists, so without it the warning is
    # cropped out of the PDF entirely and the figure ships claiming more than it shows.
    fig.tight_layout()
    fig_path = P(f"figs/fig_cross_sensor{tag}{sfx}.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # ---- reporting -----------------------------------------------------------------------------
    print(f"\n===== Separability by band-set, in-distribution ({len(args.seeds)} split offsets) =====")
    for c in conditions:
        d = agg[c["label"]]
        v = np.asarray(d["mIoU"], float)
        sd = sample_sd(v)
        print(f"{c['label']:32s} {c['params']:>8,}p  mIoU={v.mean():5.1f}"
              + (f"+-{sd:4.1f}" if not np.isnan(sd) else "  (n=1)")
              + f"  OA={np.mean(d['OA']):5.1f}  AA={np.mean(d['AA']):5.1f}  "
                f"kappa={np.mean(d['kappa']):5.1f}")

    def paired(a_label, b_label):
        a = np.array([per_seed[a_label][sd]["mIoU"] for sd in args.seeds])
        b = np.array([per_seed[b_label][sd]["mIoU"] for sd in args.seeds])
        return a - b
    by_arm = {}
    for c in conditions:
        by_arm.setdefault(c["arm"], {})[c["sensor"]] = c
    for arm, m in by_arm.items():
        if S2 in m and OLI in m:
            d = paired(m[S2]["label"], m[OLI]["label"])
            note = ("the S2 model is SMALLER than OLI's here, so this gap cannot be bought with "
                    "capacity" if m[S2]["params"] < m[OLI]["params"] else
                    f"S2 carries {m[S2]['params'] - m[OLI]['params']:+,} params vs OLI")
            print(f"\n[{arm}] Sentinel-2 - OLI, PAIRED per seed: {np.round(d, 2).tolist()}  "
                  f"mean {d.mean():+.2f}  ({note})")
        if S2 in m and S2RE in m:
            d = paired(m[S2]["label"], m[S2RE]["label"])
            print(f"[{arm}] red-edge contribution (S2 minus the same S2 without B5/B6/B7), PAIRED: "
                  f"{np.round(d, 2).tolist()}  mean {d.mean():+.2f} mIoU over {len(d)} splits")
    if args.paired_capacity:
        print("\nCAPACITY DELTA, paired per seed (unmatched - matched, same split, same model seed):")
        for sensor in band_n:
            u, mm = by_arm["unmatched"][sensor], by_arm["matched"][sensor]
            d = paired(u["label"], mm["label"])
            tail = ""
            if u["params"] == mm["params"]:
                # Identical configuration, identical split, identical model seed -> this is the SAME
                # computation twice and must be exactly 0. It is a self-check that the pairing is
                # real, NOT a noise estimate: there is no seed variation between paired arms to
                # estimate. If it is non-zero, every other delta in this table is meaningless.
                worst = float(np.abs(d).max())
                tail = "   <- SELF-CHECK: same config + same split + same model seed, must be 0.000"
                if worst > 0 and not args.nondeterministic:
                    tail += (f"\n     !! IT IS {worst:.4f} -- THE ARMS ARE NOT ACTUALLY PAIRED, so "
                             f"every capacity delta above is confounded with run-to-run variation")
            print(f"  {sensor:28s} {u['params']:>8,} -> {mm['params']:>8,}p  "
                  f"delta {np.round(d, 2).tolist()} mean {d.mean():+.2f}{tail}")
    print("\nNo confidence interval is quoted: the seeds are one-pixel shifts of the same "
          "checkerboard (mean pairwise test-set IoU 0.448), so they are positively correlated "
          "replicates and any CI computed from them would be too narrow.")

    prov = {
        "srf_source": args.srf,
        "band_contract": {n: {"names": i["names"],
                              "centers_nm": [round(c, 2) for c in i["centers_nm"]],
                              "product": i.get("product"), "platform": i.get("platform"),
                              "instrument": i.get("instrument"), "detectors": i.get("detectors"),
                              "fwhm_nm": i.get("fwhm_nm"), "W_sha256": i["W_sha256"],
                              "selection": i["selection"], "diagnostics": i["diagnostics"],
                              "ablated_out": i.get("ablated_out")} for n, i in band_info.items()},
        "conditions": conditions,
        "arms": arms,
        "bands_by_sensor": band_n,
        "capacity_spread_max_over_min": float(spread),
        "matched_budget_params": int(budget),
        "macro_estimand": {
            "class_set_0based": list(class_set),
            "class_set_gt_labels": [c + 1 for c in class_set],
            "excluded_gt_labels": [c + 1 for c in excluded],
            "splits_containing_each_class": {c + 1: int(split_count[c]) for c in range(NUM_CLASSES)},
            # Interpolated, not written out: a prose description of the estimand that can disagree
            # with the code is the same defect as two copies of the code.
            "definition": f"bandsim.metrics.common_class_set(gt, block={args.block}, "
                          f"offsets=seeds, guard={GUARD}): classes present in the TEST split of "
                          f"EVERY offset. mIoU/AA average over this set; OA/kappa stay over all "
                          f"labelled pixels. mIoU_present/AA_present carry the "
                          f"average-over-present variant.",
        },
        "wavelength_axis_sha256": axis_sha256(wl),
        "inputs": {"cube": cube_path, "cube_sha256": file_sha256(cube_path),
                   "gt": gt_path, "gt_sha256": file_sha256(gt_path)},
        "splits_by_seed": splits,
        "split": f"disjoint block-checkerboard, block={args.block}, guard={GUARD}, "
                 f"offset=split_seed",
        "split_independence": "seeds are one-pixel shifts of the SAME checkerboard; mean pairwise "
                              "test-set IoU 0.448 (adjacent seeds 0.632) vs 0.333 for independent "
                              "half-splits -- the reported spread understates sampling uncertainty",
        "raw_csv": os.path.basename(raw_path),
        "unmodelled": ["ground sample distance (S2 10/20/60 m, OLI 30 m; all kept at AVIRIS ~20 m)",
                       "PSF/MTF, mixed pixels, spatial resampling, co-registration",
                       "sensor SNR, quantization, stray light, calibration uncertainty",
                       "BRDF and atmospheric-correction differences", "HLS bandpass adjustment"],
    }
    # Both artefacts get their own stamp, and both are stamped by NAME rather than in a loop: the
    # AST guard in tests/test_smoke_isolation.py resolves `x = P(...)` one level, and a loop variable
    # ranging over a tuple is not resolvable -- a write it cannot see is a write it cannot protect.
    stamp(summary_path, args, extra=prov)
    stamp(raw_path, args, extra=prov)
    print(f"\nwrote: {fig_path}\n       {summary_path}\n       {raw_path}")
    print("band contract: " + json.dumps({n: i["names"] for n, i in band_info.items()}))


if __name__ == "__main__":
    main()
