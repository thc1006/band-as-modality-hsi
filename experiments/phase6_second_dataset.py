#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S6 — second-dataset TRANSFER panel: run the Phase-2 missing-band degradation, UNCHANGED, on a
second dataset. Dataset-agnostic (registry below): works for any HSI .mat given (#classes,
wavelength axis).

WHAT THIS DOES AND DOES NOT SHOW (it used to say it "proves the conclusion is not an Indian-Pines
artefact"; it does not):
  * It DOES show the pipeline runs end-to-end on a different sensor, band count and class count,
    and that under Indian-Pines settings the method ordering is reproduced there.
  * It does NOT establish dataset generalization. Nothing here is re-tuned: --groups 10, --epochs 60
    and every model hyper-parameter (hidden=256, lr=1e-3, bs=256, SGMAE epochs = epochs//2) are the
    values selected on Indian Pines and transferred verbatim. There is no tuning/validation split at
    all — the split is train/test only, so a fair per-dataset tuning budget was never given to ANY
    method, ours included. A ranking obtained under source-tuned hyper-parameters is a TRANSFER
    result, not a generalization result: the baselines are equally untuned, which makes the
    comparison fair but leaves open whether each method's best configuration on this dataset ranks
    the same way.
  * `--dataset synthetic` is a code-generality check on fabricated data. It carries no evidential
    weight about real scenes and must never be quoted as a second dataset. It is REQUIRED to be
    named explicitly: it used to be the DEFAULT, so a bare `python phase6_second_dataset.py` wrote a
    complete, real-looking `results_phase6_synthetic.csv` + figure out of fabricated data — from the
    one invocation most likely to be typed by accident.
To claim generalization, add a per-dataset tuning split and re-tune every method on it.

Real Pavia/Salinas are on the GIC server (ehu.eus); when it is reachable (it returned HTTP
503 during development), drop the .mat files under data/<name>/ and run:
  python experiments/phase6_second_dataset.py --dataset pavia --seeds 0 1 2 3 4

To prove the code is genuinely dataset-agnostic without that data, a synthetic HSI with a
different band/class count is available:
  python experiments/phase6_second_dataset.py --dataset synthetic --seeds 0 1 2 3 4

Outputs (../paper/), all sharing one <suffix>:
  figs/fig_degradation_<dataset><suffix>.pdf
  results_phase6_<dataset><suffix>.csv          - the curve, strictly rectangular
  results_phase6_<dataset><suffix>_summary.csv  - protocol provenance + PAIRED per-seed AUDC margins
  results_phase6_<dataset><suffix>_raw.csv      - long form, one row per (seed, method, m)
  ... plus a .provenance.json beside each CSV.

<suffix> IS EMPTY ONLY FOR THE CANONICAL CONFIGURATION (see CANONICAL below). `--smoke` writes
`_smoke`; any other departure writes `_nc-<what differed>`. Outputs used to be keyed on the dataset
alone, so `--groups 8` or `--epochs 30` overwrote the committed deliverables in place.
"""
import os, sys, csv, hashlib, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
import torch
import phase2_degradation as P2
from bandsim import hw, parallel
from bandsim.io import (load_mat_cube, disjoint_block_split, AVIRIS_WL_NM, ROSIS_WL_NM,
                        axis_sha256)
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention
from bandsim.metrics import audc, retention
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)
DATA = os.path.join(os.path.dirname(_HERE), "data")

SOURCE_DATASET = "indian_pines"          # where every hyper-parameter here was selected

# The configuration whose outputs ARE the paper's deliverables. The argparse defaults are READ from
# here rather than repeated: as duplicate literals they can drift, and raising a default without
# touching this dict would make a plain no-flag run non-canonical, redirect it to a _nc- path, and
# quietly stop regenerating the deliverables while the stale ones stayed in place.
# seeds = 0..4 because that is what reproduce.sh:72,100,101 and this module's usage lines run --
# both were checked against this dict, and the docstring's synthetic line was corrected to match.
# Setting it to [0,1,2] made every one of those invocations non-canonical: they would have written
# _nc-s01234 artefacts, left the committed deliverables untouched and stale, and exited 0 -- turning
# a harmless docs/config mismatch into a silent failure to regenerate.
CANONICAL = {"seeds": [0, 1, 2, 3, 4], "groups": 10, "max_missing": 6, "trials": 8, "epochs": 60,
             "nondeterministic": False}
_ABBR = {"seeds": "s", "groups": "g", "max_missing": "m", "trials": "t", "epochs": "e",
         "nondeterministic": "nondet"}

# Salinas is AVIRIS like Indian Pines but its corrected cube drops a DIFFERENT band set, so it
# cannot borrow AVIRIS_WL_NM: 224 nominal bands minus [108-112], [154-167] and 224 leaves 204, and
# the water-vapour gaps land in different places than Indian Pines' 220-minus-20 axis. Built here
# rather than in bandsim.io only because that module is under concurrent edit; promote it there.
#
# ** bandsim.io.AVIRIS_WL_NM IS WRONG BY UP TO 37.7 nm, AND THIS IS THE PROOF. **  Not fixed here:
# io.py is under concurrent edit and eighteen files read that axis, so correcting it moves every
# phase's numbers and is a decision for whoever owns it. Escalated instead, with the evidence.
#
# The two removal lists are the SAME LIST offset by exactly +4 at every endpoint:
#     Indian Pines (220-indexed) : [104-108], [150-163], 220
#     Salinas      (224-indexed) : [108-112], [154-167], 224
# A single-valued difference set means one sensor, with Indian Pines' cube distributed as the 224
# acquisition minus its first four bands: IP_band[j] == AVIRIS_band[j+4]. Both corrected counts then
# fall out, 224-4-20 = 200 and 224-20 = 204, which no other reading explains.
#
# io.py builds the axis as linspace(400, 2500, 220), i.e. it stretches 220 bands across the full
# AVIRIS range instead of taking bands 5..224 of a 224-band axis. The error runs +0.2 to +37.7 nm,
# about four band widths, and it is not cosmetic: it shifts the ten group centres by 1.8-36.0 nm,
# and sinusoidal_wavelength_pe computes phase as wl/2500 * div * 100, so group 0 moves by 1.44 rad
# on the div=1 channel -- a quarter cycle of the positional encoding the proposed model reads.
# The check that settles it: rebuilt on the 224 base, Indian Pines' last six group centres are
# [1280.5 1508.9 1704.3 1998.1 2212.8 2401.1] -- EXACTLY Salinas' last six. On the current axis they
# match none of them, which is not something two axes of one sensor can do.
#
# The fix in io.py is one line: AVIRIS_WL_NM = np.delete(linspace(400,2500,224)[4:], removed_220).
_AVIRIS_224 = np.linspace(400.0, 2500.0, 224)
_SALINAS_REMOVED_1BASED = list(range(108, 113)) + list(range(154, 168)) + [224]
SALINAS_WL_NM = np.delete(_AVIRIS_224, [i - 1 for i in _SALINAS_REMOVED_1BASED])   # 204, WITH gaps

# How much a dataset's wavelength axis can be trusted. The proposed model's group-centre positional
# encoding and B3's spectral interpolation both READ this axis, so a fabricated one does not merely
# mislabel metadata -- it changes those two methods' predictions and can move the ranking.
#   measured          : per-band centres shipped with the scene            (none here yet)
#   nominal_with_gaps : uniform nominal spacing MINUS the real removed bands -> true gap positions
#   nominal_uniform   : uniform spacing, sensor is genuinely near-uniform  -> defensible
#   fabricated        : uniform spacing over a sensor whose gaps are unknown/misplaced
_AXIS = {"indian_pines": "nominal_with_gaps", "salinas": "nominal_with_gaps",
         "pavia": "nominal_uniform", "whu_hi": "nominal_uniform", "synthetic": "fabricated"}

# dataset registry: (cube_path, cube_key, gt_path, gt_key, n_classes, wl_lo, wl_hi)
DATASETS = {
    "pavia": ("pavia/PaviaU.mat", "paviaU", "pavia/PaviaU_gt.mat", "paviaU_gt", 9, 430, 860),
    "salinas": ("salinas/Salinas_corrected.mat", "salinas_corrected",
                "salinas/Salinas_gt.mat", "salinas_gt", 16, 400, 2500),
    "indian_pines": ("indian_pines/Indian_pines_corrected.mat", "indian_pines_corrected",
                     "indian_pines/Indian_pines_gt.mat", "indian_pines_gt", 16, 400, 2500),
    # S8: WHU-Hi LongKou (270 bands, 9 classes) — high-band-count "270->N" degradation panel.
    # GPU now makes this feasible; drop the .mat under data/whu_hi/ (Headwall Nano-Hyperspec).
    "whu_hi": ("whu_hi/WHU_Hi_LongKou.mat", "WHU_Hi_LongKou",
               "whu_hi/WHU_Hi_LongKou_gt.mat", "WHU_Hi_LongKou_gt", 9, 400, 1000),
}


def load_dataset(name):
    if name == "synthetic":
        return _synthetic()
    cp, ck, gp, gk, K, lo, hi = DATASETS[name]
    cube = load_mat_cube(os.path.join(DATA, cp), key=ck).astype(np.float64)
    gt = load_mat_cube(os.path.join(DATA, gp), key=gk).astype(int)
    # Validate what was loaded BEFORE any of it reaches a model. A .mat with an unexpected key, a
    # transposed cube or a GT with more classes than the registry declares would otherwise train a
    # head of the wrong size and score it against labels it can never emit -- silently, as a
    # plausible but deflated mIoU. That exact failure is what the num_classes threading below fixed
    # once already; these checks stop it re-entering through the data instead of the globals.
    if cube.ndim != 3:
        raise ValueError(f"{name}: cube must be (H, W, bands), got {cube.shape}")
    if gt.shape != cube.shape[:2]:
        raise ValueError(f"{name}: ground truth {gt.shape} does not match cube {cube.shape[:2]}")
    if not np.isfinite(cube).all():
        raise ValueError(f"{name}: cube contains NaN/inf; normalisation would propagate it silently")
    lab = np.unique(gt)
    if lab.min() < 0:
        raise ValueError(f"{name}: negative ground-truth labels {lab[:5]}")
    # What matters downstream is the MAXIMUM label, not how many distinct ones there are: the head
    # has K outputs and training feeds it `gt - 1`. Counting distinct non-zero labels was wrong in
    # BOTH directions -- it rejected a legitimate scene whose GT happens to lack one class (labels
    # {1..8} against K=9, which cropped scenes and WHU-Hi subsets routinely produce), and it
    # ACCEPTED non-contiguous labels {1,2,4,5} at K=4, which then dies in train_mlp with
    # "IndexError: Target 4 is out of bounds" after load_dataset had certified the data.
    fg = lab[lab > 0]
    if fg.size == 0:
        raise ValueError(f"{name}: ground truth has no labelled pixels")
    if int(fg.max()) != K:
        raise ValueError(f"{name}: registry declares {K} classes but the ground truth's largest "
                         f"label is {int(fg.max())}. Training feeds `gt - 1` to a {K}-output head, "
                         f"so a larger label is an immediate IndexError and a smaller one means the "
                         f"registry over-declares")
    missing = sorted(set(range(1, K + 1)) - set(int(v) for v in fg))
    if missing:
        # NOT fatal: a class can be genuinely absent from a scene. But mIoU averages over classes
        # PRESENT in the split, so an absent class silently changes what the metric means.
        import warnings
        warnings.warn(f"{name}: classes {missing} are absent from the ground truth entirely. mIoU "
                      f"averages over present classes, so this run's metric covers {K - len(missing)} "
                      f"of the {K} the registry declares.")
    nb = cube.shape[-1]
    # The REAL axis when one exists for this band count. A gapless linspace over a sensor whose
    # gaps are elsewhere does not merely mislabel metadata: the proposed model's group-centre PE and
    # B3's spectral interpolation both read it, so it moves their predictions.
    if name == "indian_pines" and nb == len(AVIRIS_WL_NM):
        wl = AVIRIS_WL_NM.copy()
    elif name == "salinas" and nb == len(SALINAS_WL_NM):
        wl = SALINAS_WL_NM.copy()
    elif name == "pavia" and nb == len(ROSIS_WL_NM):
        wl = ROSIS_WL_NM.copy()
    else:
        # The registry's declared status is what is RETURNED here. Hardcoding "fabricated" made
        # whu_hi unreachable from its own declaration: it has no special branch, so it always
        # reported "fabricated" while the registry said "nominal_uniform" -- and the test that
        # checked every dataset declares a status passed on a value the code never read.
        # `nominal_uniform` is the honest label for a sensor that genuinely has no gaps (ROSIS,
        # Headwall Nano-Hyperspec); `fabricated` is for one whose gap structure is unknown.
        status = _AXIS.get(name, "fabricated")
        wl = np.linspace(lo, hi, nb)
        if status == "fabricated":
            import warnings
            warnings.warn(f"{name}: no matched wavelength axis for {nb} bands -> nominal "
                          f"linspace({lo},{hi},{nb}). Band centres are FABRICATED and any gap "
                          f"positions are wrong; wavelength PE and B3 interpolation read this axis.")
        return cube, gt, wl, K, status
    return cube, gt, wl, K, _AXIS[name]


def _synthetic(H=120, W=120, C=103, K=9, seed=0):
    """A synthetic HSI with DIFFERENT band/class counts than Indian Pines.

    It proves the CODE runs on another band/class count. It proves nothing about real scenes, and
    this docstring used to say "proves generality" while the module docstring correctly said the
    opposite -- the overclaim was inside the function generating the fabricated data.
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(430, 860, C)
    centers = np.stack([0.3 + 0.4 * np.sin(np.linspace(0, k + 1, C)) for k in range(K)])  # class spectra
    load = rng.normal(0, 1, (C, 4)) * 0.05
    gt = np.zeros((H, W), int)
    cube = np.zeros((H, W, C))
    for i in range(H):
        for j in range(W):
            k = ((i // 12) + (j // 12)) % K + 1          # blocky class map (spatial structure)
            gt[i, j] = k
            z = rng.normal(0, 1, 4)
            cube[i, j] = centers[k - 1] + z @ load.T + rng.normal(0, 0.02, C)
    return cube, gt, wl, K, "fabricated"


def run_seed(seed, cube, gt, wl, n_classes, n_groups, max_missing, trials, epochs):
    # The class count is THREADED into every phase-2 helper (num_classes=...), not assigned onto
    # the imported module. This used to read `P2.NUM_CLASSES = n_classes`, which rewrites another
    # module's state: not reentrant, and `parallel.run_jobs` runs its SERIAL path (BANDSIM_WORKERS=1
    # or a single-seed job list) in the CALLING process, so the assignment outlived this function.
    # Measured consequence: after this ran on a 9-class dataset, phase4_ablation — which imports
    # NUM_CLASSES BY VALUE (16) but calls P2.train_mlp, which read the global at CALL time — built a
    # 9-output head and then scored it with miou(..., 16). Classes 9..15 became impossible to
    # predict and each contributed IoU 0, so phase 4 reported a plausible, silently deflated mIoU.
    tr, te = disjoint_block_split(gt, block=10, guard=1, offset=seed)
    Xtr = cube[tr]; ytr = gt[tr].astype(int) - 1
    Xte = cube[te]; yte = gt[te].astype(int) - 1
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xte_raw = Xte.astype(np.float32)                      # raw reflectance for physically-correct B3
    Xtr = ((Xtr - mu) / sd).astype(np.float32); Xte = ((Xte - mu) / sd).astype(np.float32)
    groups = contiguous_groups(cube.shape[-1], n_groups)
    cwl = group_center_wavelengths(wl, groups)

    m_b1 = P2.train_mlp(Xtr, ytr, groups, seed, group_dropout=False, epochs=epochs,
                        num_classes=n_classes)
    m_b2 = P2.train_mlp(Xtr, ytr, groups, seed, group_dropout=True, epochs=epochs,
                        num_classes=n_classes)
    # RESEED before constructing the proposed model. train_mlp seeds torch and then CONSUMES the
    # global RNG for the whole of its training, so this constructor was drawing its initial weights
    # from "seed, advanced by however many random ops B2 happened to perform". Editing a baseline --
    # changing its dropout schedule, its batch count, anything -- silently re-initialised the
    # proposed model too, so a proposed-vs-baseline gap could move without the proposed code
    # changing at all. A distinct offset makes the initialisation depend on `seed` and nothing else.
    # +101 is NOT arbitrary: phase2_degradation seeds exactly `seed + 101` immediately before the
    # identical constructor (its line 569). A different offset here would make phase6 initialise the
    # same architecture from a different stream, so `--dataset indian_pines` -- which this file
    # blesses as a self-consistency check against phase2 -- could never reproduce phase2's numbers.
    torch.manual_seed(seed + 101)
    m_prop = GroupedCrossBandAttention(groups, cwl, n_classes)
    # pretrain_sgmae/finetune_proposed need no class count: SGMAE is label-free, and the finetune
    # head size is already fixed by the GroupedCrossBandAttention constructed above.
    P2.pretrain_sgmae(m_prop, Xtr, groups, seed, epochs=max(1, epochs // 2))
    P2.finetune_proposed(m_prop, Xtr, ytr, groups, seed, epochs=epochs)

    def dc(kind, model):
        return P2.degradation_curve(kind, model, Xte, yte, groups, wl, max_missing, trials,
                                    np.random.default_rng(seed + 999),
                                    Xte_raw=Xte_raw, mu=mu, sd=sd, num_classes=n_classes)
    return {"b1": dc("b1", m_b1), "b2": dc("b2", m_b2), "b3": dc("b3", m_b1),
            "proposed": dc("proposed", m_prop)}


def build_argparser():
    """Factored out of main() so the "which dataset, and was it chosen deliberately?" question is
    testable without running anything."""
    ap = argparse.ArgumentParser()
    # --dataset is REQUIRED and has no default. It used to default to "synthetic", so a bare
    # `python experiments/phase6_second_dataset.py` wrote paper/results_phase6_synthetic.csv +
    # figs/fig_degradation_synthetic.pdf — a complete, real-looking "second dataset" deliverable
    # built entirely from fabricated data, produced by the one invocation most likely to be typed
    # by accident. Picking fabricated data must be a deliberate act, so make the caller say it.
    ap.add_argument("--dataset", required=True, default=None,
                    choices=list(DATASETS) + ["synthetic"],
                    help="real HSI from the registry, or 'synthetic' = FABRICATED data, a "
                         "code-generality check only (never quote it as a second dataset)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL["seeds"]),
                    help="each seed is ALSO the checkerboard grid offset, which is periodic in "
                         "block=10 -- so at most 10 distinct splits exist (offsets 0-9) and e.g. "
                         "seed 10 duplicates seed 0. Duplicates are refused rather than averaged. "
                         "Consecutive seeds are also highly correlated (adjacent pairs share ~63% "
                         "of test pixels); a spread-out set decorrelates far better")
    ap.add_argument("--groups", type=int, default=CANONICAL["groups"])
    ap.add_argument("--max-missing", type=int, default=CANONICAL["max_missing"])
    ap.add_argument("--trials", type=int, default=CANONICAL["trials"])
    ap.add_argument("--epochs", type=int, default=CANONICAL["epochs"])
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent seed workers (default: adaptive; also BANDSIM_WORKERS)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                    help="device for the workers (default: auto; also BANDSIM_DEVICE)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="disable deterministic algorithms for a small speedup")
    # The repo-wide pattern for a harness run. experiments/integrity_check.py drove this script at
    # `--seeds 0 --epochs 12` while naming the UNSUFFIXED canonical outputs, so an integrity check
    # replaced the paper's deliverable with a 1-seed 12-epoch result. Declaring --smoke both fixes
    # that and enrolls this script in tests/test_smoke_isolation.py, which then enforces the
    # suffixing automatically for every artefact it writes.
    ap.add_argument("--smoke", action="store_true",
                    help="fast harness run: 1 seed / 12 epochs, writes *_smoke artefacts and NEVER "
                         "the deliverables")
    return ap


def preflight(args, n_bands=None):
    """Reject configurations that cannot produce a meaningful result, BEFORE anything trains.

    The one that matters most is `--epochs <= 0`. It does not fail: `for _ in range(-2)` simply
    runs zero times, so B1 and B2 come out at their random initialisation and the finetune is
    skipped, while SGMAE pretraining still runs once because it is called as `max(1, epochs // 2)`.
    The run completes and writes a curve, a figure, a summary and a verdict.

    What that verdict describes, precisely -- an earlier version of this docstring claimed it was a
    manufactured WIN for the proposed method, and that is not what happens:
      * SGMAE cannot rescue the proposed model either. Its loss flows through `reconstruct` into
        `decoder`; `classifier` is never on that path, so its gradient stays None and the optimiser
        skips it. Measured: classifier weight and bias move 0.000e+00 across every seed.
      * so all four arms sit at chance, and which one "wins" is noise. Measured on synthetic data,
        the proposed model came LAST and the verdict written was "does NOT transfer".
      * worse, three of the four are the SAME model: b1 and b2 are bit-identical with zero epochs
        (same torch.manual_seed(seed), no training to diverge them) and b3 reuses m_b1.
    So the danger is not a flattering result, it is a meaningless one that looks complete. A summary
    CSV from such a run prints b1_audc_mean == b2_audc_mean exactly, which is the tell.

    The rest are cheaper failures that used to surface only from inside degradation_curve, i.e.
    after every model for every seed had already been trained.
    """
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"--seeds must be unique, got {args.seeds}")
    if any(s < 0 for s in args.seeds):
        raise ValueError("--seeds must be >= 0: the seed is also the checkerboard grid offset")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}. At <= 0 nothing is trained: "
                         f"the MLP loops run zero times, SGMAE cannot reach the classifier head, "
                         f"and b1/b2 are bit-identical while b3 reuses b1. The run would still "
                         f"write a curve, a figure and a verdict -- from four models at chance, "
                         f"three of which are the same model")
    if args.groups < 2:
        raise ValueError(f"--groups must be >= 2, got {args.groups} (band-group dropout degenerates "
                         f"to no dropout at all with a single group)")
    if not 0 <= args.max_missing < args.groups:
        raise ValueError(f"--max-missing must lie in [0, --groups) = [0, {args.groups}), got "
                         f"{args.max_missing}; dropping every group leaves nothing to classify")
    if args.trials < 1:
        raise ValueError(f"--trials must be >= 1, got {args.trials}")
    if args.jobs is not None and args.jobs < 1:
        raise ValueError(f"--jobs must be >= 1, got {args.jobs}")
    if n_bands is not None and args.groups > n_bands:
        raise ValueError(f"--groups {args.groups} exceeds the {n_bands} bands available")


def split_overlaps(gt, seeds, block=10, guard=1):
    """Pairwise test-mask IoU between seeds, and any pair that is the SAME split.

    `offset = seed` shifts the checkerboard by whole pixels, and shifting by a full `block` adds 1
    to both the row and the column block index, so the parity that selects train-vs-test is
    unchanged: offset and offset+block are the IDENTICAL partition. Seeds 0 and 10 are one split
    reported as two, and averaging them looks like a two-seed result with suspiciously small spread.
    Measured on a 120x120 grid: IoU 1.000 at offset 10 and 20, 0.623 at offset 1, 0.170 at offset 5.
    """
    masks = {s: disjoint_block_split(gt, block=block, guard=guard, offset=s)[1] for s in seeds}
    pairs, dup = [], []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            ma, mb = masks[a], masks[b]
            iou = float((ma & mb).sum()) / max(1, int((ma | mb).sum()))
            pairs.append((a, b, iou))
            if np.array_equal(ma, mb):
                dup.append((a, b))
    return pairs, dup


def main():
    args = build_argparser().parse_args()
    # Smoke overrides land BEFORE preflight: they dictate seeds/epochs, so validating the user's
    # (about to be discarded) values would reject `--smoke --epochs 0` for a value smoke replaces.
    if args.smoke:
        args.seeds = [0]; args.epochs = 12
    preflight(args)
    # phase6 asks whether the Phase-2 ordering survives a move to ANOTHER dataset. Run on the source
    # dataset it answers itself, and the verdict string would read "the Phase-2 ordering TRANSFERS
    # to indian_pines". The dataset stays in the registry because re-running the source through this
    # pipeline is a genuine self-consistency check; what is refused is calling that a transfer.
    self_check = args.dataset == SOURCE_DATASET
    if self_check:
        print("!" * 96)
        print(f"! --dataset {SOURCE_DATASET} is the SOURCE of every hyper-parameter used here.")
        print("! This is a SELF-CONSISTENCY check, not a second-dataset transfer result, and the")
        print("! verdict below will say so. Do not quote it as evidence of transfer.")
        print("!" * 96)
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device); print("HW:", hw.info())

    # OUTPUT NAMES CARRY THE CONFIG. Every path here was keyed on the dataset alone, so
    # `--groups 8`, `--epochs 30` or a different seed set overwrote the previous run's curve,
    # summary and figure in place -- and paper/results_phase6_pavia.csv and _salinas.csv are
    # committed deliverables. A non-default configuration is exploratory until someone says
    # otherwise, so it is redirected and says on its face what differed.
    # Seeds compare as a SORTED set: `--seeds 2 1 0` is the same experiment as `--seeds 0 1 2`
    # (each seed is independent and run_jobs returns in item order), so ordering alone must not
    # redirect a canonical run to an exploratory path.
    _val = lambda k: (sorted(getattr(args, k)) if isinstance(CANONICAL[k], list)
                      else getattr(args, k))
    _want = lambda k: (sorted(CANONICAL[k]) if isinstance(CANONICAL[k], list) else CANONICAL[k])
    _diff = [(k, _val(k)) for k in sorted(CANONICAL) if _val(k) != _want(k)]
    sfx = ""
    if args.smoke:
        sfx = "_smoke"
        _diff = []
        print("[smoke] 1 seed / 12 epochs — writing *_smoke artefacts, NOT the deliverables")
    if _diff:
        # Values joined with "_", not concatenated: `--seeds 1 23` and `--seeds 1 2 3` both
        # serialised to "s123", so two different runs shared every output path including the
        # provenance sidecar and the second silently overwrote the first -- the exact failure this
        # suffix exists to prevent, one level down.
        _tag = "-".join(f"{_ABBR[k]}"
                        + ("_".join(str(x) for x in v) if isinstance(v, list) else str(v))
                        for k, v in _diff)
        # Past 40 chars the tag becomes a hash, which keeps filenames usable but stops the NAME
        # describing the run -- the console banner below and the provenance sidecar still do.
        sfx = "_nc-" + (_tag if len(_tag) <= 40 else hashlib.sha256(_tag.encode()).hexdigest()[:12])
        print("!" * 96)
        print(f"! NON-CANONICAL CONFIG -> writing '{sfx}' artefacts, NOT the paper deliverables.")
        for k, v in _diff:
            print(f"!   {k}: {v}   (canonical: {CANONICAL[k]})")
        print("!" * 96)

    cube, gt, wl, K, axis_status = load_dataset(args.dataset)
    preflight(args, n_bands=cube.shape[-1])
    print(f"dataset={args.dataset}: cube {cube.shape}, {K} classes, wl {wl[0]:.0f}-{wl[-1]:.0f}nm "
          f"[axis: {axis_status}]")
    if axis_status == "fabricated":
        print("  ! the wavelength axis is FABRICATED (uniform spacing, gap positions unknown).")
        print("  ! The proposed model's group-centre PE and B3's interpolation both read it, so")
        print("  ! this is a nominal-axis sensitivity run, not a physical-wavelength result.")
    _pairs, _dup = split_overlaps(gt, list(args.seeds))
    if _dup:
        raise ValueError(f"seeds {_dup} produce IDENTICAL train/test splits (offset is periodic in "
                         f"block=10, so offset and offset+10 are the same partition). They would be "
                         f"averaged as independent replicates and shrink the reported spread")
    if _pairs:
        _mo = float(np.mean([p[2] for p in _pairs]))
        print(f"  split overlap: mean pairwise test-mask IoU {_mo:.3f} over {len(_pairs)} seed pair(s). "
              f"1.0 = identical; ~0.333 = independent halves; BELOW that = anti-correlated (a "
              f"half-block shift reaches ~0.17). Seeds are correlated, not independent, and the "
              f"across-seed std understates the true variability.")
        if _mo > 0.45:
            print("  ! these seeds are MORE correlated than independent halves. Only offsets 0-9 "
                  "are distinct (block=10); a spread-out set such as 0 2 4 6 8 decorrelates more.")

    keys = ["b1", "b2", "b3", "proposed"]
    xs = np.arange(0, args.max_missing + 1)
    runs = {k: [] for k in keys}
    import time
    t0 = time.time()
    results = parallel.run_jobs(
        run_seed, args.seeds,
        shared=dict(cube=cube, gt=gt, wl=wl, n_classes=K, n_groups=args.groups,
                    max_missing=args.max_missing, trials=args.trials, epochs=args.epochs),
        prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
        label="phase6/seed")
    for sd, c in zip(args.seeds, results):
        for k in keys:
            runs[k].append(c[k])
        print(f"seed {sd}: proposed AUDC={audc(xs, c['proposed']):.1f} b2={audc(xs, c['b2']):.1f} "
              f"b1={audc(xs, c['b1']):.1f}")
    print(f"(all {len(args.seeds)} seeds in {time.time()-t0:.1f}s)")

    stats = {k: (np.mean(np.stack(runs[k]), 0), np.std(np.stack(runs[k]), 0)) for k in keys}
    # per-seed AUDC so the head-to-head can be PAIRED (same seed = same split and init) and can
    # carry a spread; the summary used to be a bare argmax over seed-mean curves.
    seed_audc = {k: np.array([audc(xs, c) for c in runs[k]], float) for k in keys}
    # Margins against EVERY baseline. Reporting only the strongest observed rival selects the
    # comparison after seeing the data, so that one margin carries a winner's-curse bias: the
    # baseline that happened to score highest on these seeds is also the one most likely to have
    # scored high by luck. It stays -- it is the honest worst case among the baselines -- but it is
    # now reported beside the others rather than instead of them, and it is labelled as selected.
    margins = {k: seed_audc["proposed"] - seed_audc[k] for k in keys if k != "proposed"}
    rival = max(margins, key=lambda k: seed_audc[k].mean())
    margin = margins[rival]
    # Computed HERE, beside `margins`, because both the console summary and the verdict consume it.
    wins_by_rival = {k: int((mg > 0).sum()) for k, mg in margins.items()}
    min_wins = min(wins_by_rival.values())
    worst_rival = min(wins_by_rival, key=lambda k: wins_by_rival[k])
    # The curve file stays strictly rectangular (header on row 1) so it keeps parsing with a plain
    # csv/pandas reader; the protocol caveat and the paired summary go in a sidecar rather than as
    # comment rows, which would silently become the header for any reader without comment=='#'.
    with open(P(f"results_phase6_{args.dataset}{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["missing_groups"] + sum([[f"{k}_mean", f"{k}_std"] for k in keys], []))
        for i in xs:
            w.writerow([int(i)] + sum([[f"{stats[k][0][i]:.2f}", f"{stats[k][1][i]:.2f}"] for k in keys], []))
    # Sidecar: records WHAT PROTOCOL produced the curve, so the numbers cannot be lifted into a
    # table as a generalization result. Every hyper-parameter here was chosen on Indian Pines.
    with open(P(f"results_phase6_{args.dataset}{sfx}_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["key", "value"])
        w.writerow(["dataset", args.dataset])
        w.writerow(["claim", "transfer of the Phase-2 ordering under Indian-Pines settings; "
                             "NOT dataset generalization"])
        w.writerow(["retuned_on_this_dataset", "no"])
        w.writerow(["tuning_split", "none (train/test only)"])
        w.writerow(["hyperparams_selected_on", "indian_pines"])
        w.writerow(["groups", args.groups]); w.writerow(["epochs", args.epochs])
        w.writerow(["n_seeds", len(args.seeds)])
        for k in keys:
            w.writerow([f"{k}_audc_mean", f"{seed_audc[k].mean():.3f}"])
            w.writerow([f"{k}_audc_std", f"{seed_audc[k].std():.3f}"])
        w.writerow(["wavelength_axis_status", axis_status])
        w.writerow(["source_dataset", SOURCE_DATASET])
        w.writerow(["is_self_consistency_check", str(self_check)])
        w.writerow(["block_px", 10]); w.writerow(["guard_px", 1])
        w.writerow(["mean_pairwise_test_mask_iou", f"{np.mean([p[2] for p in _pairs]):.3f}"
                    if _pairs else "n/a"])
        # The x axis is a GROUP COUNT, so these say what one group is worth on THIS sensor. They are
        # NECESSARY but NOT SUFFICIENT for comparing an AUDC across datasets: class count, class
        # balance and baseline accuracy all move AUDC too, and none of them is captured here. Read
        # them as a reason to be careful, not as a licence to compare.
        _gspan = [float(wl[g].max() - wl[g].min()) for g in contiguous_groups(len(wl), args.groups)]
        # AUDC's SCALE is set by class count and clean headroom, which no wavelength field captures:
        # two datasets losing 5% of mIoU per group -- identical robustness -- give AUDC 78.2 at K=9,
        # clean 92 and AUDC 51.9 at K=16, clean 61. Retention normalises exactly that away, and it
        # was imported at the top of this file and never called. Record both, and read the
        # normalised one when comparing datasets.
        w.writerow(["n_classes", int(K)])
        w.writerow(["miou_at_zero_missing_proposed", f"{stats['proposed'][0][0]:.2f}"])
        for k in keys:
            w.writerow([f"{k}_retention_at_max_missing",
                        f"{retention(stats[k][0][0], stats[k][0][args.max_missing]):.4f}"])
            w.writerow([f"{k}_audc_normalised_by_clean",
                        f"{seed_audc[k].mean() / stats[k][0][0]:.4f}" if stats[k][0][0] > 0 else "nan"])
        w.writerow(["bands_per_group", f"{cube.shape[-1] / args.groups:.1f}"])
        # span EXCLUDING the axis gaps -- (wl[-1]-wl[0])/groups reports 209.1 for salinas against
        # 209.0 for indian_pines, i.e. nothing, for the one pair most likely to be laid side by side
        w.writerow(["nm_per_group_mean_span", f"{np.mean(_gspan):.1f}"])
        w.writerow(["nm_per_group_min_span", f"{np.min(_gspan):.1f}"])
        w.writerow(["nm_axis_covered", f"{float(np.sum(np.diff(wl)[np.diff(wl) < 3 * np.median(np.diff(wl))])):.0f}"])
        w.writerow(["frac_bands_missing_at_max", f"{args.max_missing / args.groups:.3f}"])
        # every rival, not only the selected one
        for k, mg in margins.items():
            w.writerow([f"paired_margin_vs_{k}_mean", f"{mg.mean():+.3f}"])
            w.writerow([f"paired_margin_vs_{k}_std", f"{mg.std():.3f}"])
            w.writerow([f"paired_wins_vs_{k}", f"{int((mg > 0).sum())}/{len(args.seeds)}"])
        w.writerow(["best_rival", rival])
        w.writerow(["best_rival_selection", "POST-HOC (highest observed mean AUDC) -- carries a "
                                            "winner's-curse bias; see the per-rival rows above"])
        w.writerow(["paired_margin_mean", f"{margin.mean():+.3f}"])
        w.writerow(["paired_margin_std", f"{margin.std():.3f}"])
        w.writerow(["paired_wins", f"{int((margin > 0).sum())}/{len(args.seeds)}"])
        if args.dataset == "synthetic":
            w.writerow(["data_provenance", "FABRICATED — code-generality check only, not a dataset"])

    # LONG-FORM per-seed values. The curve CSV keeps only mean/std, so a paired bootstrap, a
    # different confidence interval, a check of which seed drove an outlier, or an audit of the
    # post-hoc rival selection all needed the whole run repeated. They no longer do.
    with open(P(f"results_phase6_{args.dataset}{sfx}_raw.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "seed", "method", "missing_groups", "miou", "audc"])
        for k in keys:
            for si, sd in enumerate(args.seeds):
                a = float(audc(xs, runs[k][si]))
                for i in xs:
                    w.writerow([args.dataset, sd, k, int(i), f"{runs[k][si][i]:.4f}", f"{a:.4f}"])

    colors = {"b1": "#c0392b", "b2": "#e67e22", "b3": "#8e44ad", "proposed": "#1f6f3a"}
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for k in keys:
        me, st = stats[k]
        ax.plot(xs, me, "-o", color=colors[k], lw=1.7, ms=3, label=k)
        ax.fill_between(xs, me - st, me + st, color=colors[k], alpha=0.15, lw=0)
    ax.set_xlabel("Number of missing spectral groups"); ax.set_ylabel("mIoU (%)")
    ax.set_title(f"Missing-band robustness — {args.dataset}", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(P(f"figs/fig_degradation_{args.dataset}{sfx}.pdf")); plt.close(fig)

    best = max(keys, key=lambda k: seed_audc[k].mean())
    print(f"\nAUDC (mean+-std over {len(args.seeds)} seeds): "
          + " ".join(f"{k}={seed_audc[k].mean():.1f}+-{seed_audc[k].std():.1f}" for k in keys))
    print(f"proposed best AUDC? {best == 'proposed'} | paired margin vs highest-mean rival "
          f"({rival}): {margin.mean():+.2f}+-{margin.std():.2f} mIoU")
    print(f"  per-seed wins against EACH baseline: {wins_by_rival} -> worst is '{worst_rival}' at "
          f"{min_wins}/{len(args.seeds)}. The headline uses this minimum, not the selected rival's.")
    # Deliberately NOT the word "generalizes": even when the ordering IS reproduced, it was
    # reproduced under Indian-Pines hyper-parameters with no per-dataset re-tuning and no tuning
    # split (see the module docstring). Calling that generalization is the overclaim this panel is
    # most likely to be quoted for. And the verdict must follow the DATA -- state the failure just
    # as plainly when proposed does not win, including when it wins on the mean but not per seed.
    # THE WIN COUNT MUST BE THE MINIMUM OVER ALL RIVALS, NOT THE MEAN-SELECTED ONE. The rival is
    # picked by highest MEAN AUDC, but the headline is a PER-SEED win count -- two different
    # orderings. So "proposed wins on every seed" could print while proposed lost a seed to a rival
    # that merely had a lower mean. Constructed and confirmed against these very lines:
    #   b1=[11,2,2] mean 5.00 | b2=[9,9,1] mean 6.33 | proposed=[10,10,10]
    #   -> rival=b2, margin [1,1,9], won_all=True, headline "wins on every seed"
    #   -> truth: margin vs b1 is [-1,8,8]; proposed LOSES seed 0.
    # Simulated over plausible AUDC noise the old headline fired in 15-37% of runs and was false in
    # 25-40% of those. The claim now quantifies over every baseline, which is what it always said.
    won_all = min_wins == len(args.seeds)
    # The middle branch used to assert, unconditionally, that "the paired margin is within its own
    # spread". It never checked. With per-seed margins [+8, +9, -1] it fires (2/3 wins, so not
    # won_all) and prints "margin +5.33 is within its own spread 4.50" -- which is false. Comparing
    # a mean to an SD is not a test either. So the branch now DESCRIBES and the reader infers.
    _within = abs(margin.mean()) <= margin.std()
    _word = "SELF-CONSISTENCY CHECK on the source dataset"      # only read inside `if self_check`
    if self_check:
        verdict = (f"{args.dataset} IS the source dataset: this reproduces the Phase-2 {_word}, it "
                   f"is NOT a transfer result (proposed best AUDC: {best == 'proposed'}, wins "
                   f"{int((margin > 0).sum())}/{len(args.seeds)} seeds vs {rival})")
    elif best == "proposed" and won_all:
        verdict = (f"the Phase-2 ordering TRANSFERS to {args.dataset} (proposed wins on every "
                   f"seed against EVERY baseline: {wins_by_rival}) under Indian-Pines settings")
    elif best == "proposed":
        verdict = (f"the Phase-2 ordering transfers to {args.dataset} ON THE MEAN ONLY: proposed "
                   f"has the highest mean AUDC but wins only {min_wins}/"
                   f"{len(args.seeds)} seeds against its worst rival '{worst_rival}' "
                   f"({wins_by_rival}). Paired margin vs '{rival}' {margin.mean():+.2f}, "
                   f"SD {margin.std():.2f}"
                   + (" (mean is inside one SD)" if _within else " (mean exceeds one SD)")
                   + ". No inferential claim is made: {n} correlated seeds cannot support one"
                   .format(n=len(args.seeds)))
    else:
        verdict = (f"the Phase-2 ordering does NOT transfer to {args.dataset}: '{rival}' has the "
                   f"higher AUDC (paired margin {margin.mean():+.2f}+-{margin.std():.2f}, proposed "
                   f"wins {int((margin > 0).sum())}/{len(args.seeds)} seeds)")
    print(f"=> {verdict} (no re-tuning, no tuning split). "
          f"Either way this is a transfer result, NOT dataset generalization.")
    if args.dataset == "synthetic":
        print("   NOTE: 'synthetic' is fabricated data — a code-generality check only. It is NOT a "
              "second dataset and must not be reported as one.")
    # BOTH files are stamped, with the same facts: the curve and its protocol sidecar are one run, and
    # the curve is the file most likely to be read alone. `fabricated_data` is the one flag that must
    # survive being separated from the summary CSV -- 'synthetic' is a code-generality check, and a
    # curve quoted as a second dataset is the failure this module exists to prevent.
    prov = {"dataset": args.dataset, "n_classes": int(K), "n_bands": int(cube.shape[-1]),
            "wavelength_nm_range": [float(wl[0]), float(wl[-1])], "methods": keys,
            "fabricated_data": args.dataset == "synthetic",
            "wavelength_axis_status": axis_status,
            "wavelength_sha256": axis_sha256(wl),
            "source_dataset": SOURCE_DATASET,
            "is_self_consistency_check": bool(self_check),
            "canonical_config": sfx == "", "config_suffix": sfx,
            "mean_pairwise_test_mask_iou": (float(np.mean([p[2] for p in _pairs]))
                                            if _pairs else None),
            "best_rival_selection": "post-hoc (highest observed mean AUDC)",
            "hyperparams_selected_on": "indian_pines (no re-tuning, no tuning split)"}
    # stamp() is best-effort by design (it must never lose a finished run), so its return value
    # has to be CHECKED -- an unstamped deliverable is one nobody can attribute to a run, which is
    # the failure provenance exists to prevent.
    # Written out one call per file rather than looped: tests/test_smoke_isolation.py finds the
    # stamped artefacts by AST and looks for a ".csv" in the path expression, so a loop variable
    # hides them from the very check that guarantees deliverables carry provenance. Keeping the
    # literals visible costs three lines and keeps that guard working.
    _unstamped = [p for p in (
        stamp(P(f"results_phase6_{args.dataset}{sfx}.csv"), args, extra=prov),
        stamp(P(f"results_phase6_{args.dataset}{sfx}_summary.csv"), args, extra=prov),
        stamp(P(f"results_phase6_{args.dataset}{sfx}_raw.csv"), args, extra=prov),
    ) if p is None]
    if _unstamped:
        print(f"  ! WARNING: {len(_unstamped)} artefact(s) could not be provenance-stamped -- "
              f"stamp() is best-effort by design, so an unstamped deliverable is one nobody can "
              f"attribute to a run. Do not cite them.")
    print(f"wrote: {P(f'figs/fig_degradation_{args.dataset}{sfx}.pdf')}")


if __name__ == "__main__":
    main()
