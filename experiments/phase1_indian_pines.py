#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 — real HSI pipeline on Indian Pines + honest baselines (roadmap Table 1).

Goal: prove the end-to-end pipeline works on REAL hyperspectral data with a leakage-free
split, and produce Table 1 (OA / AA / kappa / mIoU) for an sklearn SVM and a tiny MLP.
This is the "pipeline runs correctly on real data, no leakage" exit gate before Phase 2.

Anti-leakage: uses a DISJOINT block-checkerboard split (bandsim.io.disjoint_block_split),
NOT random per-pixel sampling — random sampling from one scene leaks via spatial adjacency
(reviewers know this; see docs/guide/01_datasets.md). block=10 keeps all 16 classes in both
splits on Indian Pines. VERIFIED by execution (tests/test_experiment_guards.py, seeds 0-4):
train and test pixel sets never intersect, and with the default guard=1 not a single test pixel
is even 8-CONNECTED (diagonals included) to a train pixel or to the train REGION — the guard
dilates the region, not just the labelled train mask, so unlabelled same-object context is
buffered too. Without the guard 34% of test pixels touch a train pixel, so guard>=1 is load-bearing.

CAVEAT ON PER-CLASS NUMBERS: a leakage-free split is not the same as an interpretable one. Four of
the 16 classes have a mean test support under 30 px (Grass-pasture-mowed 3.8, Oats 4.0, Alfalfa
22.2, Stone-Steel-Towers 24.4) and two of them disappear from the test split entirely on some
seeds. Their per-class IoU/accuracy are noise; `results_phase1_perclass.csv` therefore carries a
`test_support_px` row and main() prints an explicit warning listing them.

Data (place under data/indian_pines/, ~6 MB, see docs/guide/01_datasets.md):
  Indian_pines_corrected.mat   (145x145x200, uint16)
  Indian_pines_gt.mat          (145x145, 16 classes + background=0)

Outputs (under ../paper/):
  results_phase1_table1.csv    - per-model OA/AA/kappa/mIoU (mean +/- std over seeds)
  results_phase1_perclass.csv  - per-model, per-class IoU AND per-class accuracy (mean over seeds)
  results_phase1_table1.tex    - LaTeX macros for the paper's Table 1

Usage:
  python experiments/phase1_indian_pines.py                 # SVM + MLP, block=10, seed 0
  python experiments/phase1_indian_pines.py --seeds 0 1 2 3 4
"""
import os
import csv
import argparse
import warnings
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.io import load_mat_cube, disjoint_block_split
# average_accuracy / miou are deliberately NOT imported: they average over whichever classes the
# split happens to contain, which is the estimand defect this file now fixes with a fixed class set
# (see common_class_set / macro_over).
from bandsim.metrics import overall_accuracy, cohen_kappa, confusion, per_class_iou
from bandsim import parallel
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(PAPER_DIR, exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)

DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "indian_pines")
NUM_CLASSES = 16  # excluding background (0)
# Standard AVIRIS Indian Pines class taxonomy in label order (GT label i+1 <-> CLASS_NAMES[i]);
# the canonical 16-class names used throughout the HSI literature (see docs/guide/01_datasets.md).
CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
    "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]
assert len(CLASS_NAMES) == NUM_CLASSES, "CLASS_NAMES must list all NUM_CLASSES labels"


def load_indian_pines():
    cube = load_mat_cube(os.path.join(DATA_DIR, "Indian_pines_corrected.mat"),
                         key="indian_pines_corrected").astype(np.float64)
    gt = load_mat_cube(os.path.join(DATA_DIR, "Indian_pines_gt.mat"),
                       key="indian_pines_gt").astype(int)
    return cube, gt


def make_xy(cube, gt, mask):
    """Flatten labelled pixels under `mask` to (X spectra, y in 0..NUM_CLASSES-1)."""
    X = cube[mask]                 # (n, 200)
    y = gt[mask].astype(int) - 1   # map labels 1..16 -> 0..15
    return X, y


def evaluate(y_true, y_pred, class_set):
    """OA/kappa over all labelled pixels; AA/mIoU over the FIXED `class_set`.

    The two macro metrics take a class set because average_accuracy/miou silently average over
    whichever classes the split happens to contain -- see common_class_set for why that made Table 1
    and the per-class CSV disagree."""
    aa, mi = macro_over(y_true, y_pred, class_set)
    return {
        "OA": overall_accuracy(y_true, y_pred),
        "AA": aa,
        "kappa": cohen_kappa(y_true, y_pred, NUM_CLASSES) * 100.0,
        "mIoU": mi,
    }


MIN_INTERPRETABLE_SUPPORT = 30   # below this, a per-class rate's 95% CI exceeds ~+/-18 points


def per_class_support(y_true):
    """Number of TEST pixels per class — the denominator every per-class number is computed over.

    Published without it, a per-class IoU printed to two decimals looks equally trustworthy for
    every class. It is not. On this split (block=10, guard=1, seeds 0-4) the measured mean test
    support ranges over three orders of magnitude: Soybean-mintill 830.8 px but Grass-pasture-mowed
    3.8 px and Oats 4.0 px. A rate estimated on ~4 pixels has a 95% binomial half-width near +/-50
    percentage points — the number is noise formatted as a measurement, and it moves the per-class
    mean. Two classes even vanish from the test split entirely on some seeds (Oats on seed 0,
    Grass-pasture-mowed on seeds 1 and 2), which is why the aggregation uses np.nanmean.
    """
    yt = np.asarray(y_true).ravel()
    return np.array([int(np.sum(yt == c)) for c in range(NUM_CLASSES)])


def low_support_classes(support, threshold=MIN_INTERPRETABLE_SUPPORT):
    """Indices of classes whose test support is too small for the per-class number to be read."""
    return [c for c in range(NUM_CLASSES) if support[c] < threshold]


def per_class_recall(y_true, y_pred):
    """Per-class recall (%) as a length-NUM_CLASSES array; NaN for classes ABSENT from y_true.

    NaN (not 0.0) for an absent class so a mean over seeds via np.nanmean ignores it instead of
    counting a spurious 0%. np.nanmean over the present classes then reduces to AA. Complements
    per_class_iou (IoU is the paper's primary metric); both go to the per-class CSV.
    """
    m = confusion(y_true, y_pred, NUM_CLASSES)
    denom = m.sum(1)
    out = np.full(NUM_CLASSES, np.nan)
    present = denom > 0
    out[present] = np.diag(m)[present] / denom[present] * 100.0
    return out


def distinct_offsets(block):
    """The offsets that give DIFFERENT checkerboard splits.

    The pattern is `(bi + bj) % 2`, and raising the offset by `block` raises both block indices by
    one, so the parity is unchanged: offset s and s+block are the SAME split, byte for byte
    (verified: offset 3 and offset 13 produce identical masks at block=10). Passing --seeds 0..4 and
    --seeds 0..14 therefore gives the same five splits with five extra duplicates, and any spread
    computed over them would be an average of copies. The period is `block`, not 2*block."""
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    return list(range(block))


def common_class_set(gt, block, offsets, guard=1):
    """Classes present in the TEST split of EVERY offset -- the class set the macro metrics use.

    THE ESTIMAND PROBLEM THIS SOLVES. average_accuracy and miou ignore classes absent from y_true
    (measured: both return 100.00 when 2 of 16 classes are missing and everything present is
    correct). So a per-offset macro average is taken over whichever classes that offset happens to
    contain, and the mean over offsets averages DIFFERENT estimands. Table 1's AA was
    mean_s (1/|C_s|) sum_{c in C_s}, while the per-class CSV column is (1/C) sum_c nanmean_s -- two
    quantities that are not equal and were printed side by side.

    Fixing the class set once makes every offset report the same estimand. On Indian Pines at
    block=10 that set has 14 of 16 members: Grass-pasture-mowed and Oats fall out (present in 8/10
    and 6/10 splits). Those are exactly the two classes whose per-class numbers this file already
    declares uninterpretable -- 3.8 and 4.0 mean test pixels -- so they are excluded from the MACRO
    average and still reported per class, with their support, in the per-class CSV.

    The alternative -- demanding all 16 in every split -- was measured and rejected: only offsets
    3,4,5,6 qualify, they are mutually adjacent (Jaccard 0.62), and there are no others, because
    13,14,15,16 are duplicates of them."""
    present = np.zeros(NUM_CLASSES, int)
    for off in offsets:
        _, te = disjoint_block_split(gt, block=block, guard=guard, offset=off)
        y = gt[te].astype(int) - 1
        for c in range(NUM_CLASSES):
            if np.any(y == c):
                present[c] += 1
    keep = [c for c in range(NUM_CLASSES) if present[c] == len(offsets)]
    if len(keep) < 2:
        raise ValueError(f"only {len(keep)} class(es) appear in every split -- no macro metric can "
                         f"be defined over a common set here")
    return keep, present


def assert_no_8_connected_leak(tr_mask, te_mask, offset):
    """No test pixel may touch a train pixel, DIAGONALS INCLUDED.

    The module docstring claims an 8-connected guarantee. `disjoint_block_split` implements the guard
    with `binary_dilation(..., iterations=guard)` and no `structure=`, and SciPy's default is
    connectivity 1 -- a 4-neighbour cross. The 8-connected property therefore holds by CHECKERBOARD
    GEOMETRY (a diagonal neighbour across a block corner belongs to a same-parity block, hence to the
    same side), not because the code enforces it. Measured: 0 violations at every distinct offset.

    Switching the shared splitter to an 8-neighbour structure was measured too and REJECTED here: it
    shrinks the test set by 176-347 px (5-10%) at every offset, which would move the numbers of every
    other phase that uses this function. Checking the property instead of strengthening the dilation
    keeps the claim honest and every other experiment unchanged -- and fails loudly if the geometry
    ever stops providing it."""
    from scipy.ndimage import binary_dilation
    viol = int(np.sum(te_mask & binary_dilation(tr_mask, structure=np.ones((3, 3), bool))))
    if viol:
        raise RuntimeError(
            f"offset {offset}: {viol} test pixels are 8-connected to a train pixel. The docstring's "
            f"8-connected guarantee does not hold for this geometry; the guard must be widened or "
            f"the claim withdrawn.")


def macro_over(y_true, y_pred, class_set):
    """AA and mIoU over a FIXED class set, so every split reports the same estimand.

    OA and kappa are NOT macro averages -- an absent class does not change what they estimate -- so
    they stay over all labelled pixels and are computed by the shared helpers."""
    rec = per_class_recall(y_true, y_pred)
    iou = per_class_iou(y_true, y_pred, NUM_CLASSES)
    cs = np.asarray(class_set, int)
    return float(np.mean(rec[cs])), float(np.mean(iou[cs]))


def run_seed(job, cube, gt, block, class_set):
    """One (split_offset, model_seed) draw.

    The two were ONE `seed` before, driving the checkerboard offset AND the MLP initialisation and
    batch order at once. The SVM's spread was then pure split variance while the MLP's mixed split,
    initialisation and batch order -- two columns of the same table measuring different things.
    (`SVC(random_state=...)` is a no-op here anyway: sklearn ignores it unless probability=True.)"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    offset, model_seed = job
    tr_mask, te_mask = disjoint_block_split(gt, block=block, guard=1, offset=offset)
    assert_no_8_connected_leak(tr_mask, te_mask, offset)
    Xtr, ytr = make_xy(cube, gt, tr_mask)
    Xte, yte = make_xy(cube, gt, te_mask)
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        raise ValueError(f"offset {offset}: empty split ({Xtr.shape[0]} train / {Xte.shape[0]} test)")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    out = {}

    def _record(name, y_pred):
        pc = {"IoU": per_class_iou(yte, y_pred, NUM_CLASSES),
              "Recall": per_class_recall(yte, y_pred)}
        out[name] = (evaluate(yte, y_pred, class_set), pc)

    svm = SVC(C=100.0, gamma="scale", kernel="rbf")
    svm.fit(Xtr_s, ytr)
    _record("SVM (RBF)", svm.predict(Xte_s))
    _record("MLP (1D spectral)", train_predict_mlp(Xtr_s, ytr, Xte_s, seed=model_seed))

    return out, int(tr_mask.sum()), int(te_mask.sum()), per_class_support(yte)


def train_predict_mlp(Xtr, ytr, Xte, seed, hidden=128, epochs=80, lr=0.1, bs=256):
    # Explicit contract, not assert: `python -O` strips assert. An empty training set used to run the
    # epoch loop zero times and then predict from the RANDOM initialisation, returning plausible
    # class ids; a label outside [0, K) used to index the one-hot silently (ytr=-1 wrote into the
    # LAST column, i.e. it became class 15). Neither raised anything.
    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ValueError(f"Xtr/Xte must be 2-D, got {Xtr.shape} / {Xte.shape}")
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        raise ValueError("empty training or test set")
    if ytr.shape != (Xtr.shape[0],):
        raise ValueError(f"ytr shape {ytr.shape} does not match Xtr {Xtr.shape}")
    if np.any((ytr < 0) | (ytr >= NUM_CLASSES)):
        raise ValueError(f"training labels outside [0, {NUM_CLASSES}): "
                         f"min={ytr.min()} max={ytr.max()}")
    if epochs < 1 or bs < 1 or hidden < 1 or not (lr > 0 and np.isfinite(lr)):
        raise ValueError(f"invalid hyperparameters: epochs={epochs} bs={bs} hidden={hidden} lr={lr}")
    rng = np.random.default_rng(seed)
    C = Xtr.shape[1]; K = NUM_CLASSES
    W1 = rng.normal(0, np.sqrt(2.0 / C), (C, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, K)); b2 = np.zeros(K)
    Y = np.zeros((ytr.size, K)); Y[np.arange(ytr.size), ytr] = 1
    n = Xtr.shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            bi = idx[s:s + bs]; xb = Xtr[bi]; yb = Y[bi]
            z1 = xb @ W1 + b1; a1 = np.maximum(0, z1)
            z2 = a1 @ W2 + b2
            z2 -= z2.max(1, keepdims=True); e = np.exp(z2); p = e / e.sum(1, keepdims=True)
            g2 = (p - yb) / xb.shape[0]
            gW2 = a1.T @ g2; gb2 = g2.sum(0)
            g1 = (g2 @ W2.T) * (z1 > 0); gW1 = xb.T @ g1; gb1 = g1.sum(0)
            W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1
    a1 = np.maximum(0, Xte @ W1 + b1); z2 = a1 @ W2 + b2
    return z2.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-offsets", type=int, nargs="+", default=None,
                    help="checkerboard offsets; default = every DISTINCT one (period = block)")
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0],
                    help="MLP init / batch order. Separate from the split, which the SVM also uses.")
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent workers (default: adaptive; also BANDSIM_WORKERS)")
    ap.add_argument("--allow-few-splits", action="store_true",
                    help="write the paper filenames even with <5 distinct splits (a spread over "
                         "fewer is not reportable; a single split prints std=0.00)")
    args = ap.parse_args()
    if args.block < 1:
        raise ValueError(f"--block must be >= 1, got {args.block}")
    if not args.model_seeds:
        raise ValueError("need at least one model seed")

    cube, gt = load_indian_pines()
    if cube.ndim != 3 or gt.ndim != 2 or cube.shape[:2] != gt.shape:
        raise ValueError(f"cube {cube.shape} and gt {gt.shape} are not a matching HSI pair")
    if not np.isfinite(cube).all():
        raise ValueError("cube contains NaN or Inf")
    lab = np.unique(gt)
    if lab.min() < 0 or lab.max() > NUM_CLASSES:
        raise ValueError(f"GT labels outside [0, {NUM_CLASSES}]: {lab.tolist()}")

    # Offsets: DISTINCT ones only. offset and offset+block give byte-identical masks (the checkerboard
    # parity (bi+bj)%2 is unchanged when both indices rise by one), so --seeds 0..14 used to be five
    # splits plus ten copies, and any spread over them averaged duplicates.
    allowed = distinct_offsets(args.block)
    if args.split_offsets is None:
        offsets = allowed
    else:
        seen, offsets = set(), []
        for o in args.split_offsets:
            k = o % args.block
            if k in seen:
                print(f"[warn] offset {o} duplicates offset {k % args.block} (period = block = "
                      f"{args.block}); dropped")
                continue
            seen.add(k); offsets.append(o)
    if not offsets:
        raise ValueError("no distinct split offsets")

    class_set, present = common_class_set(gt, args.block, offsets)
    excluded = [CLASS_NAMES[c] for c in range(NUM_CLASSES) if c not in class_set]
    print(f"cube {cube.shape} | gt classes {sorted(np.unique(gt[gt>0]).tolist())}")
    print(f"splits: {len(offsets)} distinct checkerboard offsets {offsets} (period = block = {args.block})")
    print(f"MACRO CLASS SET: {len(class_set)}/{NUM_CLASSES} classes present in EVERY split. "
          f"AA and mIoU are averaged over these and ONLY these, so every split reports the same "
          f"estimand.")
    if excluded:
        print(f"  EXCLUDED from AA/mIoU (absent from at least one split): "
              + ", ".join(f"{CLASS_NAMES[c]} ({present[c]}/{len(offsets)} splits)"
                          for c in range(NUM_CLASSES) if c not in class_set))
        print(f"  They remain in the per-class CSV with their support. Excluding them CHANGES the "
              f"metric definition versus a 16-class AA/mIoU -- say so wherever these numbers appear.")

    jobs = [(o, ms) for o in offsets for ms in args.model_seeds]
    paper_mode = len(offsets) >= 5 or args.allow_few_splits
    sfx = "" if paper_mode else "_fewsplits"
    if not paper_mode:
        print(f"[guard] only {len(offsets)} distinct split(s): a spread over fewer than 5 is not "
              f"reportable and a single split prints std=0.00 as if variance had been measured. "
              f"Writing *{sfx} artefacts. Use --allow-few-splits to override.")

    agg, pcagg = {}, {}
    ntrs, ntes, supports, raw = [], [], [], []
    results = parallel.run_jobs(run_seed, jobs,
                                shared=dict(cube=cube, gt=gt, block=args.block, class_set=class_set),
                                prefer="cpu", jobs=args.jobs, label="phase1/run")
    for (off, ms), (res, n_tr, n_te, sup) in zip(jobs, results):
        ntrs.append(n_tr); ntes.append(n_te); supports.append(sup)
        for model, (metrics, pc) in res.items():
            agg.setdefault(model, {m: [] for m in metrics})
            for m, v in metrics.items():
                agg[model][m].append(v)
            pcm = pcagg.setdefault(model, {k: [] for k in pc})
            for k, arr in pc.items():
                pcm[k].append(arr)
            raw.append(dict({"split_offset": off, "model_seed": ms, "model": model,
                             "n_train_px": n_tr, "n_test_px": n_te}, **metrics))
        line = " | ".join(f"{model}: " + " ".join(f"{m}={metrics[m]:.1f}" for m in metrics)
                          for model, (metrics, _) in res.items())
        print(f"offset {off} / model_seed {ms}: {line}")

    print(f"\nsplit: disjoint block-checkerboard (block={args.block}, guard=1) | "
          f"train={np.mean(ntrs):.0f} test={np.mean(ntes):.0f} labelled px (mean over {len(jobs)} runs)")

    # ---- RAW long-form: every aggregate below is recomputable from this ------------------------
    with open(P(f"results_phase1_raw{sfx}.csv"), "w", newline="") as f:
        fn = ["split_offset", "model_seed", "model", "n_train_px", "n_test_px",
              "OA", "AA", "kappa", "mIoU"]
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore"); w.writeheader()
        for r in raw:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    metric_order = ["OA", "AA", "kappa", "mIoU"]
    with open(P(f"results_phase1_table1{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # `spread` is ddof=1 across runs, and it is NOT a sampling SD: adjacent checkerboard offsets
        # share ~62% of their test pixels (measured), so the runs are correlated by construction.
        # Read it as descriptive split+init sensitivity.
        w.writerow(["model"] + [f"{m}_mean" for m in metric_order]
                   + [f"{m}_spread_ddof1" for m in metric_order] + ["n_runs"])
        for model, md in agg.items():
            means = [np.mean(md[m]) for m in metric_order]
            sds = [(np.std(md[m], ddof=1) if len(md[m]) > 1 else float("nan")) for m in metric_order]
            w.writerow([model] + [f"{x:.2f}" for x in means] + [f"{x:.2f}" for x in sds]
                       + [len(md[metric_order[0]])])

    mean_support = np.mean(np.stack(supports), axis=0)
    with open(P(f"results_phase1_perclass{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # last column is an AGGREGATE, not a mean: for the support row it is the total pixel count.
        w.writerow(["model", "metric"] + list(CLASS_NAMES) + ["aggregate", "in_macro_set"])
        w.writerow(["(all)", "test_support_px"] + [f"{v:.1f}" for v in mean_support]
                   + [f"{mean_support.sum():.1f} (total)", ""])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for model, md in pcagg.items():
                for metric in ("IoU", "Recall"):
                    pc = np.nanmean(np.stack(md[metric]), axis=0)
                    w.writerow([model, metric] + [f"{v:.2f}" for v in pc]
                               + [f"{np.mean(pc[np.asarray(class_set)]):.2f} (macro set)",
                                  " ".join(CLASS_NAMES[c] for c in class_set)])

    with open(P(f"results_phase1_table1{sfx}.tex"), "w") as f:
        f.write(f"% Phase 1 Table 1 — Indian Pines, disjoint block split (block={args.block}, "
                f"guard=1), {len(offsets)} distinct offsets x {len(args.model_seeds)} model seeds\n")
        f.write(f"% AA/mIoU macro set = {len(class_set)}/{NUM_CLASSES} classes present in every "
                f"split; excluded: {excluded}\n")
        for model, md in agg.items():
            tag = "Svm" if model.startswith("SVM") else "Mlp"
            for m in metric_order:
                f.write(f"\\newcommand{{\\ip{tag}{m}}}{{{np.mean(md[m]):.1f}}}\n")
                sd = np.std(md[m], ddof=1) if len(md[m]) > 1 else float("nan")
                f.write(f"\\newcommand{{\\ip{tag}{m}Spread}}{{{sd:.1f}}}\n")

    print(f"\n===== Table 1 (mean over {len(jobs)} runs; AA/mIoU over {len(class_set)} classes) =====")
    hdr = f"{'model':<20} " + " ".join(f"{m:>7}" for m in metric_order)
    print(hdr); print("-" * len(hdr))
    for model, md in agg.items():
        print(f"{model:<20} " + " ".join(f"{np.mean(md[m]):7.1f}" for m in metric_order))

    low = low_support_classes(mean_support)
    print(f"\nper-class TEST SUPPORT (mean px over {len(jobs)} runs): "
          + "  ".join(f"{CLASS_NAMES[c]}={mean_support[c]:.0f}" for c in range(NUM_CLASSES)))
    if low:
        # NOT a confidence interval. 1.96*0.5/sqrt(n) is the WORST-CASE (p=0.5) Wald half-width: it
        # ignores the observed rate, returns no bounds, and assumes independent Bernoulli pixels --
        # which neighbouring HSI pixels from one field are not. Calling it a 95% CI overstated it.
        print(f"WARNING: {len(low)} of {NUM_CLASSES} classes have mean test support < "
              f"{MIN_INTERPRETABLE_SUPPORT} px; their per-class IoU/recall are not interpretable "
              f"(worst-case pixel-i.i.d. half-width ~+/-"
              f"{1.96*0.5/np.sqrt(max(min(mean_support[c] for c in low), 1))*100:.0f} pts at the "
              f"smallest -- an indicative magnitude, NOT a confidence interval):")
        for c in low:
            tag = "" if c in class_set else "  [excluded from AA/mIoU]"
            print(f"  - {CLASS_NAMES[c]}: {mean_support[c]:.1f} px "
                  f"(+/-{1.96*0.5/np.sqrt(max(mean_support[c],1))*100:.0f} pts worst-case){tag}")

    prov = {"n_classes": NUM_CLASSES, "models": sorted(agg),
            "split": f"disjoint block-checkerboard, block={args.block}, guard=1",
            "split_offsets": offsets, "model_seeds": args.model_seeds, "n_runs": len(jobs),
            "offset_period": args.block,
            "macro_class_set": [CLASS_NAMES[c] for c in class_set],
            "macro_class_count": len(class_set),
            "classes_excluded_from_macro": excluded,
            "class_present_in_n_splits": {CLASS_NAMES[c]: int(present[c]) for c in range(NUM_CLASSES)},
            "spread_definition": "ddof=1 across runs; NOT a sampling SD -- adjacent offsets share "
                                 "~62% of their test pixels, so the runs are correlated",
            "eight_connected_guard": "verified per run by assert_no_8_connected_leak; the splitter "
                                     "dilates with SciPy's default 4-neighbour structure, so the "
                                     "property holds by checkerboard geometry and is CHECKED here",
            "mean_train_px": float(np.mean(ntrs)), "mean_test_px": float(np.mean(ntes))}
    for nm in ("table1", "perclass", "raw"):
        stamp(P(f"results_phase1_{nm}{sfx}.csv"), args,
              extra=dict(prov, min_interpretable_support=MIN_INTERPRETABLE_SUPPORT,
                         low_support_classes=[CLASS_NAMES[c] for c in low]))
    for nm in ("table1", "perclass", "raw"):
        print(f"wrote: {P(f'results_phase1_{nm}{sfx}.csv')}")
    print(f"       {P(f'results_phase1_table1{sfx}.tex')}")


if __name__ == "__main__":
    main()
