"""Evaluation metrics: mIoU / OA / AA / kappa, plus robustness summaries (AUDC, retention).

mIoU is the paper's primary metric; OA/AA/kappa are HSI conventions. For robustness,
report the degradation curve and summarise it with AUDC (area under degradation curve)
and retention = mIoU(corrupt) / mIoU(clean). Report mean +/- std over >=5 seeds.
"""
from __future__ import annotations
import numpy as np

# np.trapz was DEPRECATED (not removed) in NumPy 2.0 and renamed np.trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def miou(y_true, y_pred, num_classes):
    """Mean IoU (%) averaged over classes PRESENT in y_true.

    Averaging over ground-truth-present classes (not all `num_classes`) avoids the spurious
    IoU=1.0 a class with no GT and no prediction would otherwise contribute — which would
    inflate mIoU whenever the evaluation split lacks a class. When every class is present
    (e.g. Indian Pines with a class-covering split) this equals the standard all-class mean.
    """
    yt, yp = _check_pair(y_true, y_pred)
    _check_labels(yt, yp, num_classes, "miou")
    ious = []
    for c in range(num_classes):
        if not np.any(yt == c):
            continue                       # class absent from GT -> not evaluated (no free 1.0)
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 1.0)
    return float(np.mean(ious)) * 100.0 if ious else 0.0


def per_class_iou(y_true, y_pred, num_classes):
    """Per-class IoU (%) as a length-`num_classes` array; NaN for classes absent from y_true.

    Complements `miou` (which averages): for cloud segmentation the hard minority classes
    (thin cloud, shadow) must be reported separately — a good mean can hide a class collapse.
    Use np.nanmean over an axis-0 stack to aggregate across seeds while ignoring absent classes.
    """
    yt, yp = _check_pair(y_true, y_pred)
    _check_labels(yt, yp, num_classes, "per_class_iou")
    out = np.full(num_classes, np.nan)
    for c in range(num_classes):
        if not np.any(yt == c):
            continue                       # class absent from GT -> NaN (not a free 100)
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        denom = tp + fp + fn
        out[c] = (tp / denom * 100.0) if denom > 0 else 100.0
    return out


def _check_pair(y_true, y_pred):
    """Require EQUAL shape, THEN flatten — fail loudly instead of silently broadcasting a shape
    mismatch (e.g. (2,1) vs (2,) -> a plausible-looking 50%) or zip-truncating unequal lengths.
    (Checking shape before ravel is essential: (2,1) and (2,) ravel to the same length and would
    slip through a post-ravel length check.)"""
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {yt.shape} vs {yp.shape}")
    return yt.ravel(), yp.ravel()


def _check_labels(yt, yp, num_classes, who):
    """Reject labels outside [0, num_classes) — they are silently INVISIBLE to the class loop.

    `miou`/`per_class_iou` iterate `for c in range(num_classes)`, so a label >= num_classes (or an
    ignore index like -1/255) never forms a class of its own and never contributes tp/fn, while the
    same pixels ARE dropped by `confusion` (-> AA/kappa) and ARE counted by `overall_accuracy`.
    Three metrics over three different pixel populations, with no warning: `miou([0,5],[0,5],3)`
    returned a perfect 100.0 while ignoring half the data. Raising keeps the population identical
    across the metrics. No caller in this repo passes out-of-range labels (every split feeds
    `gt[mask]-1` over labelled pixels only), so this changes no reported number — it only closes the
    hole. Callers that genuinely need an ignore index must MASK it out before calling.
    """
    for name, a in (("y_true", yt), ("y_pred", yp)):
        if a.size and (a.min() < 0 or a.max() >= num_classes):
            raise ValueError(
                f"{who}: {name} has labels outside [0, {num_classes}) "
                f"(range [{a.min()}, {a.max()}]) — such pixels are silently excluded from the "
                f"per-class loop but counted by overall_accuracy; mask them out before calling.")


def overall_accuracy(y_true, y_pred):
    """Plain pixel accuracy (%) over EVERY element passed in.

    Unlike `confusion`/`average_accuracy`/`cohen_kappa` this does NOT drop out-of-range labels — it
    has no `num_classes` to check against — so it is the caller's job to pass only valid pixels.
    `miou`/`per_class_iou` now raise on out-of-range labels, which keeps every metric in this module
    over the same population as long as they are called on the same arrays.
    """
    yt, yp = _check_pair(y_true, y_pred)
    return float(np.mean(yt == yp)) * 100.0


def confusion(y_true, y_pred, num_classes):
    """Confusion matrix (rows=true, cols=pred). Labels OUTSIDE [0, num_classes) (e.g. an ignore
    index like -1 or 255) are DROPPED, not indexed: the old `for t,p in zip(...): m[t,p]+=1`
    truncated on a length mismatch and let a negative label alias the last matrix row."""
    yt, yp = _check_pair(y_true, y_pred)
    valid = (yt >= 0) & (yt < num_classes) & (yp >= 0) & (yp < num_classes)
    m = np.zeros((num_classes, num_classes), int)
    np.add.at(m, (yt[valid], yp[valid]), 1)
    return m


def average_accuracy(y_true, y_pred, num_classes):
    """Mean per-class recall, averaged over classes PRESENT in y_true only.

    Averaging over present classes (rather than all `num_classes`) avoids deflating AA when
    a class is absent from the evaluation labels. When every class is present (e.g. Indian
    Pines with a class-covering split) this is identical to averaging over all classes.
    """
    m = confusion(y_true, y_pred, num_classes)
    row = m.sum(1)
    present = row > 0
    per_class = np.divide(np.diag(m), row, out=np.zeros(num_classes), where=present)
    return float(np.mean(per_class[present])) * 100.0 if present.any() else 0.0


def per_class_recall(y_true, y_pred, num_classes):
    """Per-class recall (%) as a length-`num_classes` array; NaN for classes ABSENT from y_true.

    NaN rather than 0.0 for an absent class, so a mean over seeds via np.nanmean ignores it instead
    of counting a spurious 0%. np.nanmean over the present classes reduces exactly to
    `average_accuracy`. The per-class counterpart of AA, as `per_class_iou` is of mIoU; both feed the
    per-class CSVs and `macro_over`.
    """
    m = confusion(y_true, y_pred, num_classes)
    denom = m.sum(1)
    out = np.full(num_classes, np.nan)
    present = denom > 0
    out[present] = np.diag(m)[present] / denom[present] * 100.0
    return out


def common_class_set(gt, block, offsets, guard=1, num_classes=16):
    """Classes present in the TEST split of EVERY offset -> (class_set, per_class_split_count).

    THE ESTIMAND THIS PINS. `miou` and `average_accuracy` average over classes present in y_true, so
    a per-offset macro average is taken over whichever classes THAT offset's split happens to
    contain, and a mean over offsets then averages DIFFERENT quantities. Fixing the set once makes
    every offset report the same one.

    Measured on Indian Pines at block=10, guard=1: the checkerboard loses Oats (20 px in the whole
    scene) for offsets 0/7/8/9 and Grass-pasture-mowed (28 px) for offsets 1/2, so the common set
    holds 14 of 16 -- and it is 14 for offsets 0..1, 0..4 and 0..9 alike, because both stragglers
    already appear within the first two. Those two are also the classes whose per-class numbers are
    uninterpretable anyway (single-digit mean test pixels).

    Demanding all 16 instead was measured and rejected: only offsets 3,4,5,6 qualify, they are
    mutually adjacent (Jaccard 0.62), and there are no others, because offset s and s+block are the
    same split byte for byte -- `bandsim.io.block_grid`'s parity is unchanged when both block indices
    rise by one. That trade buys a 16-class estimand with four near-duplicate splits.

    `class_set` is 0-BASED, matching the `gt[mask] - 1` convention every caller uses. `num_classes`
    defaults to Indian Pines'/Salinas' 16; passing a value LARGER than a dataset has is harmless (the
    surplus classes appear in no split, so they are excluded), but a smaller one silently truncates.

    One definition on purpose. This was hand-copied into phase1 and phase2_degradation before it had
    a home here, and the two copies already differ in signature -- the drift this repo has been
    bitten by before (`block_grid` copied into phase4R/phase9 moved the conformal units). Those
    copies should import this one; `scripts/doctor.py` reports for as long as they do not.
    """
    from .io import disjoint_block_split          # local: keeps the module's import surface flat
    offsets = list(offsets)
    if not offsets:
        raise ValueError("common_class_set needs at least one offset")
    gt = np.asarray(gt)
    K = int(num_classes)
    present = np.zeros(K, int)
    for off in offsets:
        _, te = disjoint_block_split(gt, block=block, guard=guard, offset=off)
        y = gt[te].astype(int) - 1
        for c in range(K):
            if np.any(y == c):
                present[c] += 1
    keep = [c for c in range(K) if present[c] == len(offsets)]
    if len(keep) < 2:
        raise ValueError(f"only {len(keep)} class(es) appear in the test split of every offset "
                         f"{offsets} -- no macro metric can be defined over a common set here")
    return keep, present


def macro_over(y_true, y_pred, class_set, num_classes):
    """(AA, mIoU) over a FIXED class set, so every split reports the same estimand.

    OA and kappa are deliberately NOT included: they are not macro averages, an absent class does not
    change what they estimate, and they stay over all labelled pixels via `overall_accuracy` /
    `cohen_kappa`.

    A class in `class_set` that is absent from THIS y_true yields NaN, and that NaN is left to
    propagate into the result rather than being nan-meaned away. It can only happen when the caller
    pairs a class set with splits it was not derived from, which is a bug worth seeing, not a gap
    worth filling.
    """
    cs = np.asarray(class_set, int)
    if cs.size == 0:
        raise ValueError("macro_over needs a non-empty class set")
    if cs.min() < 0 or cs.max() >= num_classes:
        raise ValueError(f"class_set entries outside [0, {num_classes}): [{cs.min()}, {cs.max()}]")
    rec = per_class_recall(y_true, y_pred, num_classes)
    iou = per_class_iou(y_true, y_pred, num_classes)
    return float(np.mean(rec[cs])), float(np.mean(iou[cs]))


def cohen_kappa(y_true, y_pred, num_classes):
    m = confusion(y_true, y_pred, num_classes).astype(float)
    n = m.sum()
    if n == 0:
        return 0.0
    po = np.trace(m) / n
    pe = (m.sum(0) @ m.sum(1)) / (n * n)
    return float((po - pe) / (1 - pe)) if pe != 1 else float("nan")   # pe==1 (single class) -> kappa undefined


def audc(missing_counts, miou_curve):
    """Mean mIoU over the tested #missing-band range (normalized area under the degradation curve;
    higher = more robust). SORTED by x so the result is order-independent (an unsorted/reversed x
    otherwise flips the trapezoid sign). Normalized by the x-range -> a mean over that range: only
    compare across runs whose x-range (max #missing) matches."""
    x = np.asarray(missing_counts, float); y = np.asarray(miou_curve, float)
    if x.shape != y.shape:
        raise ValueError(f"missing_counts/miou_curve length mismatch: {x.shape} vs {y.shape}")
    # A single NaN/inf anywhere in the curve makes the whole trapezoid NaN/inf, and that value goes
    # straight into the summary CSV/TeX as the string "nan"/"inf" — a broken evaluation point then
    # silently voids the robustness summary for that method instead of failing the run. Fail here.
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(
            f"audc needs finite inputs: {int((~np.isfinite(x)).sum())} non-finite x, "
            f"{int((~np.isfinite(y)).sum())} non-finite curve values — one bad evaluation point "
            f"would otherwise poison the whole area into nan/inf.")
    o = np.argsort(x); x, y = x[o], y[o]
    return float(_trapezoid(y, x) / (x.max() - x.min())) if x.max() > x.min() else float(y.mean())


def retention(miou_clean, miou_corrupt):
    """Fraction of clean performance retained under corruption."""
    return float(miou_corrupt / miou_clean) if miou_clean > 0 else float("nan")   # clean<=0 -> undefined
