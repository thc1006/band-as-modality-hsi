#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 (★core) — Design A missing-band DEGRADATION CURVE on real Indian Pines.

Reproduces the paper's signature claim on real HSI: dropping whole spectral GROUPS
("band-as-modality" = a group is a modality), the proposed grouped cross-band attention
(with SGMAE group-masked pretraining + band-group dropout) degrades more gracefully than
channel-stack baselines. See docs/guide/EXP_PHASE2_PLAN.md for the plan / self-critique.

Methods compared:
  B1  MLP + test-time band zeroing (no defense, lower bound)
  B2  MLP + band-group dropout training (ModDrop generalization)
  B3  MLP(B1) + test-time piecewise-linear imputation with CONSTANT ENDPOINT FILLING.
      Not "spectral interpolation" without qualification, and not physically correct as such:
      np.interp holds the nearest observed value beyond the observed range rather than
      extrapolating, so a missing EDGE group is filled with its neighbour's value (measured on a
      4-band toy: dropping the first band gives [2,2,3,4], not the linear [1,2,3,4]). With only one
      group observed the whole spectrum becomes that constant ([4,4,4,4]) -- at that point it is not
      interpolation at all. What IS physically correct is the ORDER: imputation happens in raw
      reflectance space and standardisation follows, not the reverse.
  Proposed  grouped cross-band attention + SGMAE pretrain + band-group dropout

Parameter counts, measured at the defaults (200 bands, 10 groups, 16 classes) and written to
results_phase2_summary.tex by every run:
  MLPBaseline (B1/B2/B3)      121,360      GroupedCrossBandAttention   70,692 (Proposed)
                                           learned-PE variant (B4/B6)  71,332
This line used to read "all <100k params", which was false for the very first baseline it listed:
the MLP's input layer is (bands x hidden) = 200x256, so it scales with the FULL band count, while
the grouped model embeds one token per GROUP and does not. The attention model being the SMALLER of
the two is the point worth making — with the measured numbers, not with a bound that does not hold.

Anti-leakage: disjoint block-checkerboard split; SGMAE pretrains on TRAIN-REGION pixels only.

Outputs (../paper/):
  figs/fig_degradation_real.pdf     - mIoU vs #missing groups, mean+/-std (KEY FIGURE)
  results_phase2_curve.csv          - per-#missing mean/std for the SIX methods (+ n_seeds/epochs)
  results_phase2_raw.csv            - RAW: one row per (seed, method, m, drop set)
  results_phase2_summary.tex        - AUDC / retention / param counts (LaTeX macros)
--smoke writes the SAME artefacts under a `_smoke` suffix instead. It used to write to the real
paths, so a 15-second 1-seed sanity check silently replaced the 5-seed deliverables that
experiments/make_paper_tables.py reads straight into the paper's baselines table.

Usage:
  python experiments/phase2_degradation.py --smoke                 # 1 seed, quick (_smoke outputs)
  python experiments/phase2_degradation.py --seeds 0 1 2 3 4       # full
  python experiments/phase2_degradation.py --seeds 0 1 2 3 4 --groups 10 --max-missing 6
"""
import os
import sys
import csv
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch  # threads/device are configured adaptively by bandsim.hw / bandsim.parallel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.io import load_mat_cube, disjoint_block_split, AVIRIS_WL_NM
from bandsim.grouping import contiguous_groups, group_center_wavelengths, build_group_matrix
from bandsim.metrics import miou, overall_accuracy, average_accuracy, cohen_kappa, audc, retention
from bandsim.model import GroupedCrossBandAttention, MLPBaseline, count_params
from bandsim import hw, parallel
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)

DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "indian_pines")

# NUM_CLASSES is the DEFAULT for this module's own dataset (Indian Pines). Every function that
# needs it also takes an explicit `num_classes=` argument — use that from other scripts rather than
# assigning `P2.NUM_CLASSES = n` from the outside. Rewriting another module's global is not
# reentrant, and `bandsim.parallel.run_jobs` runs its SERIAL path (BANDSIM_WORKERS=1, or a
# single-item job list) inside the CALLING process, so the assignment escapes the caller. Measured
# consequence: with phase 6 having set it to 9, phase 4 — which imports NUM_CLASSES BY VALUE (16)
# but calls train_mlp, which reads the global at CALL time — built a 9-output head and scored it
# with miou(..., 16). Classes 9..15 then cannot be predicted at all and each contributes IoU 0, so
# the run reports a plausible, silently deflated mIoU and raises nothing.
NUM_CLASSES = 16

# Enumerate every drop set of a given size when the space is at most this large; sample beyond it.
# At G=10 the whole m=0..6 space is 848 sets, so the default enumerates everything and the curve
# stops depending on which positions 12 random draws happened to pick.
ENUMERATION_CAP = 1024

# The checkerboard block size. A module constant, not a literal repeated in run_seed's
# default and again where the macro class set is computed: if those two ever disagreed the
# class set would be derived from a different split than the one actually evaluated.
SPLIT_BLOCK = 10


def common_class_set(gt, block, offsets, guard=1, num_classes=None):
    """Classes present in the TEST split of EVERY seed -- the set the macro metric averages over.

    `miou` ignores classes absent from y_true (measured: it returns 100.00 when 2 of 16 classes are
    missing and everything present is correct). Each seed shifts the checkerboard, so each averages
    over whichever classes ITS split happens to contain, and the mean over seeds averages DIFFERENT
    estimands -- the same defect fixed in phase1. Fixing the set once makes every seed report the
    same quantity. On Indian Pines at block=10 that set holds 14 of 16 classes; Grass-pasture-mowed
    and Oats fall out, and they are exactly the two whose per-class numbers are already
    uninterpretable (9.9 and 3.6 mean test pixels)."""
    K = NUM_CLASSES if num_classes is None else num_classes
    present = np.zeros(K, int)
    for off in offsets:
        _, te = disjoint_block_split(gt, block=block, guard=guard, offset=off)
        y = gt[te].astype(int) - 1
        for c in range(K):
            if np.any(y == c):
                present[c] += 1
    keep = [c for c in range(K) if present[c] == len(offsets)]
    if len(keep) < 2:
        raise ValueError(f"only {len(keep)} class(es) appear in every split -- no common macro set")
    return keep, present


def miou_over(y_true, y_pred, num_classes, class_set):
    """mIoU over a FIXED class set, so every seed reports the same estimand (see common_class_set).

    class_set=None keeps the legacy behaviour (average over whatever is present), which is what
    other modules importing this function still expect."""
    if class_set is None:
        return miou(y_true, y_pred, num_classes)
    from bandsim.metrics import per_class_iou
    return float(np.mean(per_class_iou(y_true, y_pred, num_classes)[np.asarray(class_set, int)]))


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_data():
    cube = load_mat_cube(os.path.join(DATA_DIR, "Indian_pines_corrected.mat"),
                         key="indian_pines_corrected").astype(np.float64)
    gt = load_mat_cube(os.path.join(DATA_DIR, "Indian_pines_gt.mat"),
                       key="indian_pines_gt").astype(int)
    return cube, gt


def prep(cube, gt, block=10, guard=1, offset=0, return_raw=False):
    # guard=1: leakage-hardened by default (drop test pixels adjacent to train across seams);
    # offset: per-seed grid shift so the data split varies with the seed (split variance in std)
    tr_mask, te_mask = disjoint_block_split(gt, block=block, guard=guard, offset=offset)
    Xtr = cube[tr_mask]; ytr = gt[tr_mask].astype(int) - 1
    Xte_raw = cube[te_mask]; yte = gt[te_mask].astype(int) - 1
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xte = ((Xte_raw - mu) / sd).astype(np.float32)
    if return_raw:
        # RAW test reflectance + (mu, sd) so the B3 spectral-interpolation baseline can impute in
        # RAW reflectance space and THEN standardize (physically correct order; see b3_impute).
        return Xtr, ytr, Xte, yte, Xte_raw, mu, sd
    return Xtr, ytr, Xte, yte


# --------------------------------------------------------------------------------------
# corruption helpers
# --------------------------------------------------------------------------------------
def group_present_mask(n, groups, drop_group_ids):
    """(n, G) bool present-mask with the given group ids dropped for all rows."""
    G = len(groups)
    m = np.ones((n, G), bool)
    for g in drop_group_ids:
        m[:, g] = False
    return m


def zero_missing(X, groups, drop_group_ids):
    Xc = X.copy()
    for g in drop_group_ids:
        Xc[:, groups[g]] = 0.0
    return Xc


def _check_wavelength_axis(wl, n_bands):
    """np.interp requires a STRICTLY INCREASING x-axis and does not enforce it.

    numpy's own docs say "The x-coordinate sequence is expected to be increasing, but this is not
    explicitly enforced" — a non-increasing `xp` returns nonsense values with NO exception. Measured
    on a 3-band spectrum [1, 10, 100] with the axis stored DESCENDING (500, 450, 400 nm), imputing
    the middle band returns 1.0 instead of the correct 50.5: a 98% error, silently, straight into
    the B3 baseline. Descending axes are not hypothetical — several L1 products ship them — and
    `interp_missing` is dataset-agnostic (phase6 passes whatever axis a registry entry supplies).
    Ties are rejected too: with duplicate wavelengths np.interp picks one side by binary-search
    order, so the imputed value depends on array order rather than on physics.
    The LENGTH is checked as well: a longer axis silently interpolates against the wrong band
    centres (same failure mode as bandsim.grouping.group_center_wavelengths).
    """
    wl = np.asarray(wl, float)
    if wl.ndim != 1 or wl.size != n_bands:
        raise ValueError(f"wavelength axis must be 1-D with one entry per band: got shape "
                         f"{wl.shape} for {n_bands} bands (a mismatched axis interpolates against "
                         f"the wrong band centres without raising)")
    if not np.isfinite(wl).all():
        raise ValueError("wavelength axis must be finite (np.interp silently propagates NaN)")
    if not np.all(np.diff(wl) > 0):
        bad = int(np.argmin(np.diff(wl)))
        raise ValueError(
            f"wavelength axis must be STRICTLY INCREASING for np.interp (numpy does not enforce "
            f"this and returns wrong values instead of raising): wl[{bad}]={wl[bad]:.4g} >= "
            f"wl[{bad+1}]={wl[bad+1]:.4g}. Sort the axis (and the band dimension with it) first.")
    return wl


def interp_missing(X, groups, drop_group_ids, wl):
    """np.interp does NOT extrapolate: outside the observed range it returns the nearest endpoint
    value (documented behaviour, `left`/`right` default to fp[0]/fp[-1]). A missing edge group is
    therefore CONSTANT-FILLED from its neighbour, and if only one group survives the entire spectrum
    collapses to that constant. Both are measured, not inferred. Report edge-gap and interior-gap
    results separately if the distinction matters to a claim."""
    """Spectral linear interpolation over missing bands from observed neighbours."""
    Xc = X.copy()
    wl = _check_wavelength_axis(wl, X.shape[1])
    miss = np.concatenate([groups[g] for g in drop_group_ids]) if drop_group_ids else np.array([], int)
    if miss.size == 0:
        return Xc
    keep = np.setdiff1d(np.arange(X.shape[1]), miss)
    # per-sample interpolation across wavelength using observed bands
    for i in range(X.shape[0]):
        Xc[i, miss] = np.interp(wl[miss], wl[keep], X[i, keep])
    return Xc


def b3_impute(Xte_raw, groups, drop_group_ids, wl, mu, sd):
    """B3 imputation done PHYSICALLY: linear spectral interpolation in RAW reflectance space,
    THEN train-set standardization ((x-mu)/sd). Interpolating AFTER standardization is wrong:
    the per-band z-score scales differ across bands, so they leak into the interpolant and change
    the imputed value (and the B3 ranking) — see selfcheck_b3 for a worked counterexample.
    """
    Xc_raw = interp_missing(Xte_raw, groups, drop_group_ids, wl)   # interpolate RAW reflectance
    return ((Xc_raw - mu) / sd).astype(np.float32)                 # THEN standardize (train mu/sd)


def selfcheck_b3():
    """Numeric guard for the B3 fix: physical spectral interpolation must happen in RAW reflectance
    space and THEN be standardized — not in the standardized feature space. Worked counterexample:
    raw spectrum [1, 10, 100], per-band scale sd=[1, 10, 100], drop the middle band. Standardized-
    space interp -> 1.0; raw-space interp then standardize -> 5.05. The two orders disagree, which
    can flip the B3 ranking. This also proves the eval path (b3_impute, used by eval_mlp) takes the
    raw-space branch (it calls b3_impute directly)."""
    wl = np.array([400.0, 450.0, 500.0])                          # 3 bands, equal wavelength spacing
    raw = np.array([[1.0, 10.0, 100.0]])                          # one pixel, non-trivial spectrum
    mu = np.array([0.0, 0.0, 0.0]); sd = np.array([1.0, 10.0, 100.0])
    groups = [np.array([0]), np.array([1]), np.array([2])]
    drop = [1]                                                    # drop middle band (its own group)

    std = (raw - mu) / sd                                         # WRONG order: standardize first,
    wrong = float(interp_missing(std, groups, drop, wl)[0, 1])    #   then interpolate in std space
    right = float(b3_impute(raw, groups, drop, wl, mu, sd)[0, 1]) # eval path: raw interp -> stdize

    print(f"[selfcheck B3] standardized-space interp       = {wrong:.4f}  (WRONG order)")
    print(f"[selfcheck B3] raw-space interp -> standardize  = {right:.4f}  (b3_impute / eval path)")
    print(f"[selfcheck B3] |difference|                     = {abs(right - wrong):.4f}")
    assert abs(wrong - 1.0) < 1e-5, f"standardized-space interp expected 1.0, got {wrong}"
    assert abs(right - 5.05) < 1e-4, f"raw-space interp->standardize expected 5.05, got {right}"
    assert abs(right - wrong) > 1.0, f"raw vs standardized interp must differ: {right} vs {wrong}"
    print("[selfcheck B3] OK: eval path uses RAW-space interpolation; the two orders provably differ.")
    return wrong, right


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def _vec_group_subset(rng, b, G, lo, hi, leave_one=False):
    """Vectorized per-sample random group subset: for each of b samples pick a subset of the G
    groups whose size is uniform in [lo, hi). Returns (b, G) bool (True = selected).

    SAME DISTRIBUTION as a per-sample `rng.integers(lo,hi)` + `rng.choice(G,size,replace=False)`
    loop — a uniform random size, then a uniform random subset of that size (obtained here by
    ranking the groups with uniform random scores and taking the lowest `count`). This is the
    training-speed fix: O(b*G log G) array ops (two argsorts, not O(b*G) as this line used to claim) replace the Python double loop. It consumes the RNG
    differently, so results are STATISTICALLY EQUIVALENT to the old loop, not bit-identical.

    leave_one: when True the per-sample count is clamped to <= G-1, so at least one group is ALWAYS
    left UNSELECTED. Use it for the MASK/DROP callers (where selected == removed/masked): the size
    draw [lo, hi) can exceed G at small G (e.g. hi=4 with G<=3), and without the clamp `ranks<count`
    marks EVERY group -> a fully-masked training row with no reconstruction/classification context.
    The rate differs per caller because they draw different sizes: the band-group DROPOUT caller
    (lo=0, counts ~ U{0..3}) loses ~75%/50%/25% of rows at G=1/2/3, while the SGMAE MASK caller
    (lo=1, counts ~ U{1..3}) degenerates harder at 100%/67%/33%.
    The HCS "keep 1..G" caller leaves leave_one=False,
    since selecting all G is a valid keep-all sample (and lo>=1 already keeps >=1). The clamp is a
    no-op -- hence distribution-IDENTICAL -- whenever hi <= G (e.g. the default G=10 groups), so it
    only alters the pathological small-G regime it is meant to fix.

    G=1 is REJECTED under leave_one: the clamp becomes min(count, 0) = 0 for every row, so NOTHING
    is ever selected. Measured at G=1: 100% of rows get zero masked groups, the SGMAE loss weight
    w.sum() is 0, the loss is identically 0.0, and 5 epochs of pretraining move the weights by
    exactly 0.0 -- pretraining silently no-ops instead of failing. Band-group dropout degenerates
    the same way (0 groups dropped on 100% of rows, so B2's defense is off while still LOOKING
    trained: it differs from B1 only because the extra RNG draws shift the shuffle order). Asking
    to leave one of one group unselected while selecting >= lo of them is simply unsatisfiable, so
    say so rather than returning an all-False mask."""
    if leave_one and G < 2:
        raise ValueError(
            f"leave_one=True needs at least 2 groups, got G={G}: with one group the 'leave >=1 "
            f"unselected' clamp forces every count to 0, so no group is ever masked/dropped — the "
            f"SGMAE reconstruction loss becomes identically 0 and pretraining silently does "
            f"nothing. Use n_groups >= 2.")
    counts = rng.integers(lo, hi, size=b)                    # subset size per sample
    if leave_one:
        counts = np.minimum(counts, G - 1)                   # never select all G -> >=1 stays unselected
    ranks = rng.random((b, G)).argsort(1).argsort(1)         # uniform random rank per group
    return ranks < counts[:, None]                           # the `counts[r]` lowest-ranked groups


def auto_bs(n_train, target_steps=200, floor=256, cap=32768):
    """Batch size scaled to the dataset: nearest power of two to n_train/target_steps, clipped.

    bs=256 below is sized for Indian Pines (21k labelled pixels -> 82 steps/epoch). Handing the
    same functions CloudSEN12's 2.0M-pixel subsample without rescaling gives 9,950 steps/epoch --
    121x the steps for the same epochs -- and each step on these d_model=64 models is microseconds
    of GPU work inside ~9ms of Python/launch overhead, so the run is launch-bound and a phase that
    should take an hour takes a day. (nvidia-smi shows 97-99% "utilization" throughout, because
    that metric only reports the fraction of time SOME kernel was resident; it is not evidence of
    healthy throughput on this workload.)

    The floor keeps every historical small-data result byte-identical: nearest-power-of-two
    rounding holds the batch at 256 up to n = 72,600 (the exact 256->512 edge, verified in
    test_auto_bs), and every single-scene train fold in this repo -- Indian Pines 21k, Pavia 43k,
    EMIT 50k, Salinas 54k -- is comfortably below it, so phases 1/2/3/4/4R/6/7/9 are unchanged (and
    in any case they never call this; the floor is belt-and-suspenders). Only genuinely large
    datasets move: phase8D 600k -> 4096, phase8R 720k -> 4096, phase8 2.0M -> 8192.

    Batch size is a HYPERPARAMETER: changing it changes the optimisation trajectory, so results
    produced under auto_bs are not comparable with bs=256 results on the SAME large dataset, and
    callers must record the value they used (stamp it into provenance). Adam's per-parameter
    scaling absorbs much of the difference and each epoch still visits every pixel; the phase-8
    calibration run (one seed, new recipe vs the committed curve) is the empirical check that the
    recipe change moves absolute numbers by noise, not by regime.
    """
    if n_train < 1:
        raise ValueError(f"n_train must be >= 1, got {n_train}")
    raw = max(1, n_train // int(target_steps))
    p = 2 ** round(math.log2(raw))
    return int(min(cap, max(floor, p)))


def train_mlp(Xtr, ytr, groups, seed, group_dropout, epochs=60, hidden=256, lr=1e-3, bs=256,
              num_classes=None):
    """`num_classes=None` means "this module's NUM_CLASSES" (Indian Pines, 16). Pass it EXPLICITLY
    from any script running a different dataset: callers used to reconfigure this module by
    assigning `P2.NUM_CLASSES = n` from the outside, which is not reentrant and leaks into whatever
    runs next in the same process — see the note on NUM_CLASSES above."""
    if epochs < 1 or bs < 1 or hidden < 1 or not (lr > 0 and np.isfinite(lr)):
        raise ValueError(f"invalid hyperparameters: epochs={epochs} bs={bs} hidden={hidden} lr={lr}"
                         f" -- epochs=0 used to evaluate an UNTRAINED model and report the result")
    if Xtr.shape[0] == 0:
        raise ValueError("empty training set")
    dev = hw.device()
    torch.manual_seed(seed)
    # TWO streams. One RNG served both the epoch shuffle and the band-group dropout, and only the
    # dropout arm draws from it -- so after the first epoch B1 (no dropout) and B2 (dropout) were
    # seeing DIFFERENT minibatch orderings, and the B1-vs-B2 gap carried that on top of the dropout
    # it is supposed to isolate.
    shuffle_rng = np.random.default_rng(seed)
    mask_rng = np.random.default_rng(seed + 1009)
    model = MLPBaseline(Xtr.shape[1], NUM_CLASSES if num_classes is None else num_classes,
                        hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(ytr).long().to(dev)
    n = Xt.shape[0]; G = len(groups); C = Xt.shape[1]
    Mgb = build_group_matrix(C, groups).astype(np.float32)   # (G,C) group->band membership
    model.train()
    for _ in range(epochs):
        perm = torch.from_numpy(shuffle_rng.permutation(n)).to(dev)
        for s in range(0, n, bs):
            bi = perm[s:s + bs]
            xb = Xt[bi]; yb = yt[bi]
            if group_dropout:
                # zero 0-3 random whole groups per sample, VECTORIZED (was a Python double loop):
                # drop (b,G) @ membership (G,C) -> dropped bands (groups are disjoint), 1-that=keep.
                # leave_one=True: never drop ALL groups (keep >=1 group as classification context).
                drop = _vec_group_subset(mask_rng, xb.shape[0], G, 0, 4, leave_one=True).astype(np.float32)
                # keep a band iff NO dropped group covers it (== 1-drop@M for the disjoint
                # groupings used here, but robust: never a negative mask if a grouping overlaps).
                bmask = (drop @ Mgb == 0.0).astype(np.float32)  # (b,C) 0/1 band keep-mask
                xb = xb * torch.from_numpy(bmask).to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
    model.eval()
    return model


def pretrain_sgmae(model, Xtr, groups, seed, epochs=25, lr=1e-3, bs=256):
    """Spectral-group masked reconstruction on TRAIN-REGION pixels only (no labels)."""
    dev = hw.device()
    model.to(dev)
    torch.manual_seed(seed + 7)
    rng = np.random.default_rng(seed + 7)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(Xtr).to(dev)
    n = Xt.shape[0]; G = len(groups)
    valid = model.group_valid.cpu().numpy()  # (G,S)
    model.train()
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        for s in range(0, n, bs):
            bi = perm[s:s + bs]
            xb = Xt[bi]
            b = xb.shape[0]
            # mask 1..3 groups/sample (vectorized); leave_one=True keeps >=1 group as SGMAE context.
            masked = _vec_group_subset(rng, b, G, 1, 4, leave_one=True)
            masked_t = torch.from_numpy(masked).to(dev)
            pred = model.reconstruct(xb, masked_t)                     # (b,G,S)
            target = xb[:, model.group_idx] * model.group_valid[None]  # (b,G,S)
            w = torch.from_numpy((masked[:, :, None] * valid[None]).astype(np.float32)).to(dev)
            loss = ((pred - target) ** 2 * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def _dcs_present_mask(rng, b, group_emb, tau=0.1):
    """DiChaViT Diverse Channel Sampling (Pham & Plummer, NeurIPS 2024, Algorithm 1) over GROUPS.

    Per sample: (1) keep-count n ~ Uniform{1..G} (identical to HCS's count draw); (2) a uniform
    anchor group a; (3) cosine similarity s_i = <f_a, f_i> between the anchor's group embedding and
    every other group's; (4) p = softmax((1 - s)/tau); (5) draw the remaining n-1 groups distinct,
    without replacement, by p; (6) keep = {a} ∪ those n-1. tau=0.1 is the paper's default (Table 5);
    "DCS reduces to HCS as tau -> inf". `group_emb` is (G, d) — here the model's per-group PE, which
    is DiChaViT's "channel token embedding" analog (learned for pe_type='learned', the fixed
    wavelength PE for 'sinusoidal'; read fresh each batch so a learned embedding's diversity tracks
    training). Faithful to Algorithm 1: a single anchor-derived p, not an iterative re-selection.
    Returns (b, G) bool present-mask (True = kept).

    SCOPE (do not overclaim): this is DiChaViT's DCS SAMPLING component ONLY. DiChaViT also adds a
    Channel Diversity Loss and a Token Diversity Loss that push the channel tokens apart; we do NOT
    add those, because the experiment isolates the training-time SAMPLING distribution with the
    architecture held fixed. So this arm answers "does diverse sampling beat HCS/our-dropout for
    missing-band robustness on redundant HSI", not "does full DiChaViT win". Report it as
    "DCS-sampling", never as "DiChaViT"."""
    E = np.asarray(group_emb, dtype=np.float64)
    G = E.shape[0]
    norm = np.linalg.norm(E, axis=1, keepdims=True)
    En = E / np.clip(norm, 1e-12, None)
    S = En @ En.T                                   # (G, G) cosine similarity
    # VECTORIZED via the Gumbel-top-k trick, which is EXACTLY weighted sampling WITHOUT replacement
    # (Vieira 2014; Kool, Van Hoof, Welling, ICML 2019): adding i.i.d. Gumbel noise to log-weights
    # ell_i and taking the top-k is distributionally identical to k sequential draws proportional to
    # exp(ell_i). Here ell_i = (1 - s_i)/tau, so exp(ell) ∝ softmax((1 - s)/tau) = the paper's p.
    # This replaces a per-sample rng.choice(replace=False, p=...) loop that was 1.2e8 Python calls
    # per seed on CloudSEN12 (79 min for the DCS arm) with a few array ops -- SAME distribution over
    # kept groups, a different RNG realisation than the loop.
    counts = rng.integers(1, G + 1, size=b)          # n ~ Uniform{1..G} keep-count (as in HCS)
    anchors = rng.integers(0, G, size=b)             # uniform anchor group per sample
    logw = (1.0 - S[anchors]) / tau                  # (b, G) log-weights from each sample's anchor row
    u = np.clip(rng.random((b, G)), 1e-12, 1.0)
    keys = logw + (-np.log(-np.log(u)))              # Gumbel-perturbed log-weights
    keys[np.arange(b), anchors] = np.inf             # anchor is forced kept (not part of the weighting)
    # keep the top `counts[i]` columns per row: a column is kept iff its descending-key rank < count.
    # anchor (key=inf) is always rank 0, so keep = {anchor} ∪ top-(count-1) diverse groups. count==G
    # keeps all; count==1 keeps only the anchor -- matching Algorithm 1's boundary cases.
    rank = np.argsort(np.argsort(-keys, axis=1, kind="stable"), axis=1, kind="stable")
    return rank < counts[:, None]


def finetune_proposed(model, Xtr, ytr, groups, seed, epochs=60, lr=1e-3, bs=256, group_dropout=True,
                      sampling=None):
    """`sampling` overrides `group_dropout` ONLY when set, so every existing caller (sampling=None)
    is byte-identical. Modes, all training-time GROUP present-masking augmentations for a fair
    dropout-vs-channel-sampling comparison (task: is our band-group dropout ~= ChannelViT-HCS on
    redundant HSI bands?):
      None  -> use group_dropout (legacy: drop 0-3 groups/sample, leave_one) — DEFAULT, unchanged.
      'none'-> no augmentation (all groups present every step).
      'drop'-> the same band-group dropout as group_dropout=True (explicit name).
      'hcs' -> ChannelViT Hierarchical Channel Sampling: keep k~Uniform{1..G} groups, uniform subset.
      'dcs' -> DiChaViT Diverse Channel Sampling (see _dcs_present_mask), over the model's group PE.
    """
    dev = hw.device()
    model.to(dev)
    torch.manual_seed(seed + 11)
    rng = np.random.default_rng(seed + 11)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(ytr).long().to(dev)
    n = Xt.shape[0]; G = len(groups)
    mode = ("drop" if group_dropout else "none") if sampling is None else sampling
    if mode not in ("none", "drop", "hcs", "dcs"):
        raise ValueError(f"sampling must be none/drop/hcs/dcs, got {sampling!r}")
    model.train()
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        for s in range(0, n, bs):
            bi = perm[s:s + bs]
            xb = Xt[bi]; yb = yt[bi]
            b = xb.shape[0]
            if mode == "drop":
                # drop 0-3 groups/sample (vectorized); leave_one=True -> the present-mask ~drop
                # always has >=1 group present (never a fully-missing training row).
                pm = ~_vec_group_subset(rng, b, G, 0, 4, leave_one=True)
            elif mode == "hcs":
                # ChannelViT HCS: keep 1..G groups/sample (same call train_hcs uses, so this arm is
                # HCS applied to OUR architecture rather than to B4's learned-PE/no-SGMAE model).
                pm = _vec_group_subset(rng, b, G, 1, G + 1)
            elif mode == "dcs":
                pm = _dcs_present_mask(rng, b, model.pe.detach().cpu().numpy())
            else:                                    # "none"
                pm = np.ones((b, G), bool)
            opt.zero_grad()
            logits = model(xb, torch.from_numpy(pm).to(dev))
            loss = lossf(logits, yb)
            loss.backward(); opt.step()
    model.eval()
    return model


def train_hcs(model, Xtr, ytr, groups, seed, epochs=60, lr=1e-3, bs=256):
    """B4 baseline (ChannelViT-style): supervised training with Hierarchical Channel Sampling.

    Two-step per-sample sampling of PRESENT groups: (1) k ~ Uniform{1..G}, (2) choose which k
    groups uniformly. Unlike band-group dropout (drop 0-3 independently), HCS covers channel
    COUNTS uniformly (insitro/ChannelViT, ICLR'24). Uses a learned per-group embedding
    (pe_type='learned') and NO masked pretraining -> the mechanism competitor to our method.
    """
    dev = hw.device(); model.to(dev)
    torch.manual_seed(seed + 13); rng = np.random.default_rng(seed + 13)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(ytr).long().to(dev)
    n = Xt.shape[0]; G = len(groups)
    model.train()
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        for s in range(0, n, bs):
            bi = perm[s:s + bs]; xb = Xt[bi]; yb = yt[bi]; b = xb.shape[0]
            # HCS: keep 1..G groups/sample (uniform count, vectorized; was a Python loop)
            pm = _vec_group_subset(rng, b, G, 1, G + 1)
            opt.zero_grad()
            loss = lossf(model(xb, torch.from_numpy(pm).to(dev)), yb)
            loss.backward(); opt.step()
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def eval_mlp(model, Xte, yte, groups, drop_ids, wl, impute, Xte_raw=None, mu=None, sd=None,
             class_set=None,
             num_classes=None):
    # The B3 argument check runs BEFORE the model is touched: a caller that cannot supply raw
    # reflectance has a configuration error, and finding that out before a forward pass (rather
    # than after) is what makes the failure legible.
    if impute and (Xte_raw is None or mu is None or sd is None):
        # There used to be a back-compat branch here that interpolated the ALREADY-STANDARDIZED
        # features whenever a caller omitted Xte_raw/mu/sd. That is the exact order selfcheck_b3
        # exists to prove WRONG (the per-band z-score scales leak into the interpolant), and it
        # silently returned a DIFFERENT number: on the selfcheck's worked example the raw-space
        # path imputes 5.05 and the standardized-space path 1.0. Which B3 baseline got evaluated
        # then depended on which keyword arguments a caller happened to pass — so refuse instead.
        raise ValueError(
            "B3 imputation needs RAW reflectance: pass Xte_raw, mu and sd (got "
            f"Xte_raw={'set' if Xte_raw is not None else 'None'}, "
            f"mu={'set' if mu is not None else 'None'}, sd={'set' if sd is not None else 'None'}). "
            "Interpolating in standardized space instead is physically wrong and returns a "
            "different value (5.05 vs 1.0 on selfcheck_b3's example) — see b3_impute.")
    dev = next(model.parameters()).device
    if impute:
        # B3 (correct): spectral interpolation in RAW reflectance space, THEN standardize.
        Xc = b3_impute(Xte_raw, groups, drop_ids, wl, mu, sd)
    else:
        Xc = zero_missing(Xte, groups, drop_ids)               # zero-fill in standardized space
    pred = model(torch.from_numpy(Xc).to(dev)).argmax(1).cpu().numpy()
    return miou_over(yte, pred, NUM_CLASSES if num_classes is None else num_classes, class_set)


@torch.no_grad()
def eval_proposed(model, Xte, yte, groups, drop_ids, num_classes=None, class_set=None):
    dev = next(model.parameters()).device
    pm = group_present_mask(Xte.shape[0], groups, drop_ids)
    pred = model(torch.from_numpy(Xte).to(dev), torch.from_numpy(pm).to(dev)).argmax(1).cpu().numpy()
    # miou_over, not miou: this took the class_set argument and then ignored it, while eval_mlp
    # routed through miou_over. B1/B2/B3 were therefore averaged over the FIXED macro class set and
    # Proposed/B4/B6 over whichever classes each seed's checkerboard split happened to contain --
    # two different estimands for the two arms of the comparison, plus a per-seed drift that
    # common_class_set exists to remove. It never looked wrong in the output: the classes that fall
    # out of the fixed set (Grass-pasture-mowed, Oats) hold ~10 and ~4 test pixels and score IoU~0,
    # so including them DEFLATED the attention models -- the bug ran against the proposed method.
    return miou_over(yte, pred, NUM_CLASSES if num_classes is None else num_classes, class_set)


def degradation_curve(kind, model, Xte, yte, groups, wl, max_missing, trials, rng, class_set=None,
                      record=None,
                      Xte_raw=None, mu=None, sd=None, num_classes=None):
    G = len(groups)
    if not (0 <= max_missing < G):
        # Dropping ALL G groups (max_missing == G) leaves B3 spectral interpolation with no
        # observed support bands -> np.interp on an empty sample crashes. Requiring
        # max_missing < G keeps at least one group present at every point on the curve.
        raise ValueError(
            f"max_missing must satisfy 0 <= max_missing < len(groups)={G}, got {max_missing}: "
            f"dropping >= {G} groups leaves no bands for B3 spectral interpolation.")
    # trials<=0 used to be clamped to 1 by max(1, trials), which quietly turned "--trials 0" into a
    # single-draw run: the curve still came out, still looked like an averaged Monte-Carlo estimate,
    # and nothing recorded that the requested averaging never happened. A run configured with a bad
    # trial count must fail, not silently produce a 1-sample curve labelled as a `trials`-sample one.
    if int(trials) < 1:
        raise ValueError(f"trials must be >= 1, got {trials}: each m>0 point averages `trials` "
                         f"random drop-set draws, and clamping 0 to 1 would report an unaveraged "
                         f"single draw as if it were a {trials}-draw mean.")
    curve = []
    from itertools import combinations
    from math import comb
    for m in range(0, max_missing + 1):
        # ENUMERATE every drop set at this size when that is affordable, instead of drawing `trials`
        # of them. At G=10 the whole space is 848 sets for m=0..6, and 12 random draws covered about
        # 7.2 of the 10 singletons and 5.6% of the 210 six-subsets -- so the curve was "performance
        # at a handful of randomly chosen missing positions", not "mean over all sets of that size",
        # and which spectral regions happened to be drawn moved it. Enumeration also removes the
        # need for the paired-RNG trick: every method provably sees the identical set list.
        # Beyond the cap the space is too large to enumerate and it falls back to sampling, which is
        # recorded so a reader can tell which regime produced a point.
        vals = []
        # m==0 has no drop-set to randomise, so it is deterministic and needs exactly one draw;
        # every m>0 point averages `trials` independent draws of which groups go missing.
        n_sets = comb(G, m)
        if m == 0:
            drop_sets = [[]]
        elif n_sets <= max(int(trials), ENUMERATION_CAP):
            drop_sets = [list(c) for c in combinations(range(G), m)]
        else:
            drop_sets = [rng.choice(G, size=m, replace=False).tolist() for _ in range(int(trials))]
        for drop in drop_sets:
            if kind in ("proposed", "b4", "b6"):        # grouped cross-band attention models
                vals.append(eval_proposed(model, Xte, yte, groups, drop, num_classes=num_classes,
                                          class_set=class_set))
            else:
                vals.append(eval_mlp(model, Xte, yte, groups, drop, wl, impute=(kind == "b3"),
                                     Xte_raw=Xte_raw, mu=mu, sd=sd, num_classes=num_classes,
                                     class_set=class_set))
            if record is not None:
                # One row per (method, m, drop set). Optional so phase6/phase8/verify_guardband keep
                # their existing call signature. Without it only the MEAN per m survives, and the
                # questions that matter cannot be asked afterwards: which spectral groups the model
                # actually depends on, what the WORST set of that size costs (a mean hides it), and
                # whether a difference between two methods holds set-by-set or only on average.
                # Now that the sets are ENUMERATED the list is identical across methods, so a
                # per-set paired comparison is exact rather than an approximation.
                record.append({"kind": kind, "missing_groups": m,
                               "drop_set": "|".join(str(g) for g in drop),
                               "n_sets_at_m": len(drop_sets),
                               "enumerated": int(m == 0 or n_sets <= max(int(trials), ENUMERATION_CAP)),
                               "miou": float(vals[-1])})
        curve.append(np.mean(vals))
    return np.array(curve)


def run_seed(seed, cube, gt, n_groups, max_missing, trials, epochs, block=SPLIT_BLOCK,
             class_set=None):
    # Gate G >= 2 HERE, before ~2 minutes of training per seed, rather than letting the degenerate
    # case run to completion: at G=1 the SGMAE mask and the band-group dropout are both empty for
    # every row (see _vec_group_subset), so 'Proposed' would report a model whose masked pretraining
    # provably never updated a single weight, and B2 would be B1 with a different RNG stream. The
    # missing-band experiment is also vacuous at G=1 — degradation_curve then permits only m=0.
    if int(n_groups) < 2:
        raise ValueError(
            f"n_groups must be >= 2 for the missing-band experiment, got {n_groups}: with a single "
            f"group there is no group to drop (max_missing would be forced to 0), SGMAE group "
            f"masking has nothing to mask (loss identically 0 -> pretraining is a no-op) and "
            f"band-group dropout degenerates to no dropout at all.")
    Xtr, ytr, Xte, yte, Xte_raw, mu, sd = prep(cube, gt, block=block, offset=seed, return_raw=True)
    wl = AVIRIS_WL_NM
    groups = contiguous_groups(Xtr.shape[1], n_groups)
    cwl = group_center_wavelengths(wl, groups)
    # (no shared eval rng here on purpose -- see dc() below)

    # --- baselines ---
    m_b1 = train_mlp(Xtr, ytr, groups, seed, group_dropout=False, epochs=epochs)
    m_b2 = train_mlp(Xtr, ytr, groups, seed, group_dropout=True, epochs=epochs)
    # --- B4: ChannelViT-INSPIRED ablation (learned group embedding + HCS) ---
    # NOT a ChannelViT baseline. ChannelViT operates on multi-channel IMAGES with channel-aware
    # tokenization over spatial patches; this shares the grouped-attention backbone used by the
    # proposed model and differs only in the PE and the sampling. Borrowed: the learnable channel
    # (here group) embedding and Hierarchical Channel Sampling. Call it what it is.
    # Seed IMMEDIATELY before each constructor. torch.manual_seed was only called inside the
    # training functions, i.e. AFTER these objects were built, so each model's initial weights
    # depended on how much of the global RNG stream the PREVIOUS constructions had consumed. B6 and
    # Proposed are the closest architecture ablation in this table and were not starting from paired
    # weights, so their gap carried optimisation variance on top of the change being studied.
    # NOTE what this can and cannot equalise: B4/B6 use pe_type="learned", which draws an extra
    # nn.init.normal_ for the PE parameter that the sinusoidal Proposed does not, so every parameter
    # constructed after it differs. Seeding here makes B4 and B6 exactly paired and makes Proposed a
    # deterministic function of `seed` instead of of construction order -- it does NOT make the
    # learned-PE and wavelength-PE arms share weights, which no seeding can.
    torch.manual_seed(seed + 101)
    m_b4 = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES, pe_type="learned")
    train_hcs(m_b4, Xtr, ytr, groups, seed, epochs=epochs)
    # --- B6: SatMAE-INSPIRED ablation (learned group embedding + group-masked pretraining) ---
    # NOT a SatMAE baseline. SatMAE patches each spectral group SPATIALLY, gives each its own patch
    # embedding, and uses an asymmetric MAE encoder/decoder with spatial positional encoding. Here
    # there is one spectrum per pixel and no spatial dimension at all. Borrowed: the spectral-group
    # masked reconstruction objective.
    torch.manual_seed(seed + 101)          # same stream as B4 -> B4/B6 are paired
    m_b6 = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES, pe_type="learned")
    pretrain_sgmae(m_b6, Xtr, groups, seed, epochs=max(1, epochs // 2))
    finetune_proposed(m_b6, Xtr, ytr, groups, seed, epochs=epochs, group_dropout=False)
    # --- Proposed: wavelength PE + SGMAE + band-group dropout ---
    torch.manual_seed(seed + 101)
    m_prop = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES)
    pretrain_sgmae(m_prop, Xtr, groups, seed, epochs=max(1, epochs // 2))
    finetune_proposed(m_prop, Xtr, ytr, groups, seed, epochs=epochs)

    params = {"MLP": count_params(m_b1), "Proposed": count_params(m_prop),
              "B4": count_params(m_b4), "B6": count_params(m_b6)}

    raw_rows = []

    def dc(kind, model):
        # B3 gets RAW test reflectance + (mu, sd) so its imputation is done in reflectance space
        # and standardized afterwards; other kinds ignore these (see eval_mlp / degradation_curve).
        # A FRESH generator per method, all from the same seed, so every method is scored on the
        # IDENTICAL set of random band-drop realisations -- a paired comparison. Hoisting this into
        # one shared rng outside dc() would look tidier and silently break that: a single generator
        # advances between calls, so b1 and proposed would face different drops and part of the gap
        # between them would be sampling noise. A dead `rng = default_rng(seed + 999)` used to sit
        # above, inviting exactly that "cleanup".
        return degradation_curve(kind, model, Xte, yte, groups, wl, max_missing, trials,
                                 np.random.default_rng(seed + 999), class_set=class_set,
                                 record=raw_rows, Xte_raw=Xte_raw, mu=mu, sd=sd)
    curves = {
        "b1": dc("b1", m_b1), "b2": dc("b2", m_b2), "b3": dc("b3", m_b1),
        "b4": dc("b4", m_b4), "b6": dc("b6", m_b6), "proposed": dc("proposed", m_prop),
    }
    return curves, params, raw_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--max-missing", type=int, default=6)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true", help="1 seed, few epochs, quick sanity")
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent seed workers (default: adaptive; also BANDSIM_WORKERS)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                    help="device for the workers (default: auto; also BANDSIM_DEVICE)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="disable deterministic algorithms for a small speedup")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the B3 raw-space numeric guard (selfcheck_b3) and exit (no training)")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck_b3(); return
    # --smoke used to write to the SAME paths as a full run, so a 15-second 1-seed sanity check
    # silently replaced the 5-seed paper deliverables — and experiments/make_paper_tables.py reads
    # results_phase2_curve.csv straight into the LaTeX baselines table, with nothing in the file
    # recording how many seeds produced it. Route smoke output to its own names instead.
    out_tag = ""
    if args.smoke:
        args.seeds = [0]; args.epochs = 12; args.trials = 4
        out_tag = "_smoke"
        print("SMOKE RUN: 1 seed / 12 epochs — writing to *_smoke.* so the real deliverables "
              "(results_phase2_curve.csv, results_phase2_summary.tex, figs/fig_degradation_real.pdf) "
              "are NOT overwritten. These numbers are a sanity check, not results.")
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device); print('HW:', hw.info())

    cube, gt = load_data()
    print(f"cube {cube.shape} | groups={args.groups} | seeds={args.seeds} | epochs={args.epochs}")

    keys = ["b1", "b2", "b3", "b4", "b6", "proposed"]
    labels = {
        "b1": "B1 MLP + zero-fill (no defense)",
        "b2": "B2 MLP + band-group dropout",
        "b3": "B3 MLP + spectral interpolation",
        "b4": "B4 ChannelViT-style (learned emb + HCS)",
        "b6": "B6 SatMAE-style (learned emb + group MAE)",
        "proposed": "Proposed (wavelength PE + SGMAE + attn)",
    }
    colors = {"b1": "#c0392b", "b2": "#e67e22", "b3": "#8e44ad",
              "b4": "#16a085", "b6": "#2980b9", "proposed": "#1f6f3a"}
    styles = {"b1": "-o", "b2": "-s", "b3": "-D", "b4": "-v", "b6": "-P", "proposed": "-^"}

    runs = {k: [] for k in keys}
    params = {}
    import time
    t0 = time.time()
    # Each seed is an independent job; fan them across GPUs/CPU cores (serial-identical result).
    # The macro class set, fixed ONCE across every seed. Each seed shifts the checkerboard, so each
    # test split contains a slightly different class list, and `miou` averages over whichever
    # classes are present -- so the mean over seeds was averaging different estimands. Offsets
    # `s` and `s+block` give byte-identical splits (the checkerboard parity is unchanged when both
    # block indices rise by one), so duplicates are collapsed before the set is computed.
    uniq_off = sorted({int(sd) % SPLIT_BLOCK for sd in args.seeds})
    class_set, present = common_class_set(gt, SPLIT_BLOCK, uniq_off)
    excluded = [c for c in range(NUM_CLASSES) if c not in class_set]
    print(f"MACRO CLASS SET: {len(class_set)}/{NUM_CLASSES} classes present in every split "
          f"({len(uniq_off)} distinct offsets). mIoU is averaged over these and ONLY these.")
    if excluded:
        print("  EXCLUDED (absent from at least one split): "
              + ", ".join(f"class {c} ({present[c]}/{len(uniq_off)})" for c in excluded)
              + " -- this CHANGES the metric definition versus a full-16 mIoU; say so where the "
                "numbers appear.")

    results = parallel.run_jobs(
        run_seed, args.seeds,
        shared=dict(cube=cube, gt=gt, n_groups=args.groups, max_missing=args.max_missing,
                    trials=args.trials, epochs=args.epochs, class_set=class_set),
        prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
        label="phase2/seed")
    all_raw, all_params = [], []
    for sd, (curves, params, raw_rows) in zip(args.seeds, results):
        all_params.append(params)
        for r in raw_rows:
            all_raw.append(dict(r, seed=sd))
        for k in keys:
            runs[k].append(curves[k])
        print(f"seed {sd}: " + " ".join(f"{k}[0]={curves[k][0]:.1f},[{args.max_missing}]={curves[k][-1]:.1f}"
              for k in keys))
    print(f"(all {len(args.seeds)} seeds done in {time.time()-t0:.1f}s)")

    xs = np.arange(0, args.max_missing + 1)
    # ddof=1: these seeds are treated as a sample. Note they are NOT independent replicates --
    # the checkerboard offsets they induce overlap heavily -- so read the spread as descriptive.
    stats = {k: (np.mean(np.stack(runs[k]), 0),
                 np.std(np.stack(runs[k]), 0, ddof=1) if len(runs[k]) > 1
                 else np.zeros_like(np.mean(np.stack(runs[k]), 0)) * np.nan) for k in keys}
    n = len(args.seeds)

    # ---- csv ----
    # ---- RAW: one row per (seed, method, m, drop set). Every aggregate is recomputable from it ----
    # The curve alone cannot answer the questions that decide whether the claim holds: which
    # spectral groups the model actually leans on, what the WORST set of a given size costs (a mean
    # hides it), and whether a method's advantage survives set-by-set or only on average. Because the
    # sets are now ENUMERATED, every method sees the identical list, so a per-set paired comparison
    # is exact.
    with open(P(f"results_phase2_raw{out_tag}.csv"), "w", newline="") as f:
        rf = ["seed", "kind", "missing_groups", "drop_set", "miou", "n_sets_at_m", "enumerated"]
        w = csv.DictWriter(f, fieldnames=rf, extrasaction="ignore"); w.writeheader()
        for r in all_raw:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---- what the mean hides: worst-case set, and a PAIRED contrast against B1 ------------------
    ref = "b1"
    by = {}
    for r in all_raw:
        by.setdefault((r["kind"], r["missing_groups"]), {})[(r["seed"], r["drop_set"])] = r["miou"]
    # RETENTION REWARDS A WEAK-BUT-FLAT METHOD, and the two numbers sit side by side below, so say
    # it here rather than letting a reader infer robustness from the ratio. Measured on the smoke
    # run: B4 has the HIGHEST retention (83.7%) and the LOWEST clean mIoU (36.8) -- it barely falls
    # because it barely climbed. Its m=6 value (30.8) does beat Proposed (27.4), but it gives up 14.8
    # points when nothing is missing. Neither number decides anything alone.
    clean_v = {k: float(stats[k][0][0]) for k in keys}
    end_v = {k: float(stats[k][0][-1]) for k in keys}
    print("\nabsolute vs relative -- a high retention can just mean a low starting point:")
    print(f"  {'method':<10} {'clean':>7} {'m=' + str(args.max_missing):>7} {'retention':>10} {'abs drop':>9}")
    for k in keys:
        r = 100.0 * end_v[k] / clean_v[k] if clean_v[k] > 0 else float("nan")
        print(f"  {k:<10} {clean_v[k]:7.1f} {end_v[k]:7.1f} {r:9.1f}% {clean_v[k]-end_v[k]:9.1f}")

    print(f"\nper-m WORST-CASE drop set and PAIRED gain over {ref} (mean over seeds x sets):")
    print(f"  {'m':>2}  " + "  ".join(f"{k:>18}" for k in keys))
    for m in range(0, args.max_missing + 1):
        cells = []
        for k in keys:
            cur = by.get((k, m), {})
            if not cur:
                cells.append(f"{'--':>18}"); continue
            worst = min(cur.values())
            base = by.get((ref, m), {})
            shared = [key for key in cur if key in base]
            d = np.mean([cur[key] - base[key] for key in shared]) if shared and k != ref else float("nan")
            cells.append(f"{worst:6.1f}/{d:+6.1f}" if k != ref else f"{worst:6.1f}/{'ref':>6}")
        print(f"  {m:>2}  " + "  ".join(f"{c:>18}" for c in cells))
    print(f"  (cell = WORST single drop set / mean PAIRED gain over {ref} on the SAME sets; a mean")
    print(f"   curve can look flat while one spectral region is catastrophic, which is what the")
    print(f"   worst-case column exists to expose.)")
    print(f"   NO interval is attached to the paired gain on purpose: drop sets of the same size")
    print(f"   share groups, so they are not independent draws and a naive SE over sets would")
    print(f"   understate. The raw CSV holds every value if a cluster/permutation test is wanted.")

    # Parameter counts must not silently differ between seeds -- the summary quotes ONE of them.
    if any(pp != all_params[0] for pp in all_params):
        raise ValueError(f"parameter counts differ between seeds: {all_params}")

    with open(P(f"results_phase2_curve{out_tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # n_seeds/epochs travel WITH the numbers so a table generator (and a reader) can tell a
        # 5-seed deliverable from a 1-seed smoke run. Trailing columns are ignored by the
        # csv.DictReader consumers (experiments/make_paper_tables.py reads by key).
        w.writerow(["missing_groups"] + sum([[f"{k}_mean", f"{k}_std"] for k in keys], [])
                   + ["n_seeds", "epochs"])
        for i in xs:
            row = ([int(i)] + sum([[f"{stats[k][0][i]:.2f}", f"{stats[k][1][i]:.2f}"] for k in keys], [])
                   + [n, args.epochs])
            w.writerow(row)

    # ---- summary tex (AUDC, retention, params) ----
    with open(P(f"results_phase2_summary{out_tag}.tex"), "w") as f:
        f.write(f"% Phase 2 — Indian Pines, {args.groups} groups, seeds={args.seeds}\n")
        for k in keys:
            mean = stats[k][0]
            a = audc(xs, mean); ret = retention(mean[0], mean[-1]) * 100
            tag = {"b1": "Bone", "b2": "Btwo", "b3": "Bthree",
                   "b4": "Bfour", "b6": "Bsix", "proposed": "Prop"}[k]
            f.write(f"\\newcommand{{\\audc{tag}}}{{{a:.1f}}}\n")
            f.write(f"\\newcommand{{\\ret{tag}}}{{{ret:.1f}}}\n")
        f.write(f"\\newcommand{{\\paramsMlp}}{{{params.get('MLP',0)/1000:.1f}k}}\n")
        f.write(f"\\newcommand{{\\paramsProp}}{{{params.get('Proposed',0)/1000:.1f}k}}\n")

    # ---- figure ----
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    for k in keys:
        mean, std = stats[k]
        ax.plot(xs, mean, styles[k], color=colors[k], lw=1.8, ms=4, label=labels[k])
        ax.fill_between(xs, mean - std, mean + std, color=colors[k], alpha=0.15, linewidth=0)
    ax.set_xlabel("Number of missing spectral groups")
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"Missing-band robustness — Indian Pines ({n} seeds)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=6.0, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(P(f"figs/fig_degradation_real{out_tag}.pdf")); plt.close(fig)

    print("\n===== Phase 2 degradation (mean over {} seeds) =====".format(n))
    print("miss  " + "  ".join(f"{k:>9}" for k in keys))
    for i in xs:
        print(f"{i:4d}  " + "  ".join(f"{stats[k][0][i]:6.1f}+-{stats[k][1][i]:3.1f}" for k in keys))
    print("\nAUDC:     " + "  ".join(f"{k}={audc(xs, stats[k][0]):.1f}" for k in keys))
    print("retention:" + "  ".join(f"{k}={retention(stats[k][0][0], stats[k][0][-1])*100:.1f}%" for k in keys))
    print(f"params: MLP={params.get('MLP',0)/1000:.1f}k  Proposed={params.get('Proposed',0)/1000:.1f}k  "
          f"B4={params.get('B4',0)/1000:.1f}k  B6={params.get('B6',0)/1000:.1f}k")
    # Stamped under the SAME `out_tag` the CSV was written with, so a _smoke sidecar can never end up
    # describing the 5-seed deliverable. The curve CSV's headers are bare keys (b1..b6, proposed):
    # the labels and the parameter counts say what those keys were in THIS run.
    for _nm in ("curve", "raw"):
        stamp(P(f"results_phase2_{_nm}{out_tag}.csv"), args,
          extra={"methods": keys, "method_labels": labels, "params": params,
                 "missing_levels": [int(i) for i in xs]})
    print(f"\nwrote: {P(f'figs/fig_degradation_real{out_tag}.pdf')}")
    print(f"       {P(f'results_phase2_curve{out_tag}.csv')}  {P(f'results_phase2_summary{out_tag}.tex')}")
    print(f"       {P(f'results_phase2_raw{out_tag}.csv')}  (RAW: one row per seed x method x drop set)")


if __name__ == "__main__":
    main()
