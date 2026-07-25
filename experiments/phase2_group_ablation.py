#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 robustness check — does the core conclusion survive different group counts?

The Phase 2 plan (EXP_PHASE2_PLAN.md, self-critique Q4) committed to verifying the missing-band
conclusion is not an artefact of the G=10 grouping. Here we sweep G in {5,10,20}.

WHAT IS COMPARED. Every method Phase 2 trains: b1, b2, b3, b4, b6, proposed. The list used to be
hard-coded as ["b1","b2","b3","proposed"] while `run_seed` returned six curves, so B4 (learned PE +
HCS) and B6 (learned PE + group-masked pretraining) were TRAINED at every G and every seed and then
dropped on the floor. They are the two CLOSEST architecture ablations to the proposed model —
phase2_degradation's own comment calls B6/Proposed "the closest architecture ablation in this
table" — so the sweep paid their full training cost to answer the question without them, and
"proposed best?" meant "best of the three MLP baselines". The list is now CHECKED against what
run_seed actually returns, so it cannot drift again in silence.

CROSS-G AXIS. Comparison is on the MISSING-FRACTION axis (missing_groups / G in [0,1]), NOT on raw
missing counts: different G sweep different physical fractions of the spectrum (G=5 reaches 4/5,
G=20 reaches 10/20), so a raw-count AUDC has a different x-range per G. m/G is the right axis for a
curve that AVERAGES over drop sets even when groups are unequal in size (contiguous_groups leaves a
remainder whenever G does not divide the band count): by symmetry the expected band count of a
uniformly random m-subset is m*C/G, so m/G IS the expected missing BAND fraction. It is not the
realised fraction of any single drop set.

INTEGRATION. Each curve is integrated EXACTLY as the piecewise-linear function np.interp defines,
over [0, min_G(max fraction)], on that curve's own measured knots clipped to the common maximum.
There is no resampling grid any more. The old code resampled onto a shared np.linspace and guarded
it with "the grid needs at least as many points as the finest curve has knots" — which is not the
property that makes the trapezoid exact: the grid must CONTAIN the knots, not merely match their
count. Measured on 70*exp(-2.5*f) with --groups 6 7, where that guard passes at --grid 6: G=7
summarised to 30.111 where the exact value is 29.730 (+0.38), and the G=6-vs-G=7 gap came out
-0.270 when it is in truth +0.111 — a SIGN FLIP. Misalignment could invert a cross-G statement, not
merely inflate a level. The default {5,10,20} sweep was NOT affected (its linspace happens to
contain every knot: grid and exact agree to every printed digit), so no default number moves.

WHAT THE AUC MEANS. `bandsim.metrics.audc` already divides the trapezoid by the x-range, so every
number here is a MEAN mIoU over the common missing-fraction range, in [0,100]. (Dividing by the
range a second time — which this script did — inflated every value by 1/range, 2.00x for the
default sweep, by a factor that DEPENDS on which G were swept, so two invocations were not even
comparable with each other.)

An absolute mean is NOT a degradation RATE. Phase 2 claims the proposed model "degrades more
gracefully", and a method that starts higher can win the absolute AUC while losing more of what it
started with. Three summaries are therefore reported per method: the absolute AUC, the CLEAN mIoU
at fraction 0, and the RETENTION AUC (the same integral of curve/curve[0], in % of clean). They
satisfy retention_auc = 100 * absolute_auc / clean EXACTLY — curve[0] is a per-seed constant and
integration is linear — which is why retention is computed as that ratio rather than by a second
numerical path that could drift from the first. Drop AUC is clean - absolute_auc.

LIMITS OF THE CROSS-G LEVEL COMPARISON (read before quoting numbers across G).
 1. Coarse-sampling bias. Exact integration removes the GRID error, not the fact that a curve
    measured at G=5 is known only at fractions {0,.2,.4} inside [0,.5] while G=20 is known at
    {0,.05,...,.5}. The trapezoid over-estimates a convex-decreasing curve more when sampling is
    coarser: on an identical synthetic response 70*exp(-2.5*frac) this alone gives G=5 a +0.77 mIoU
    advantage over G=20 — the same order as the between-method margins. That bias is NOT "common to
    all methods" in the sense of cancelling out of a difference (the old wording here): all methods
    at one G share the same knots, but the size of each one's trapezoid error depends on ITS OWN
    curvature, so it cancels only to the extent that two curves bend alike.
 2. Capacity moves with G. GroupedCrossBandAttention pads groups to the largest group size S and
    embeds with nn.Linear(S, d_model), so S, the parameter count and the token-sequence length all
    change with G. Per-G parameter counts are written to the CSV and the sidecar; a cross-G level
    difference is confounded with them.
 3. Sampling regime moves with G. degradation_curve ENUMERATES every drop set when C(G,m) <=
    max(trials, ENUMERATION_CAP) and samples `trials` of them otherwise — so at G=5 every point is
    an exact mean over all sets, while at G=20 the larger m are Monte-Carlo estimates from `trials`
    draws and carry sampling noise the G=5 points do not.
 4. Equal fraction is not equal corruption. One missing group at G=5 is a single 40-band contiguous
    gap; four at G=20 is up to four separated 10-band gaps (that is Phase 9's question).
Compare methods WITHIN each G; treat cross-G LEVEL differences as indicative only.

STATISTICS. The paired margin is reported against EVERY baseline, not only against the
data-dependently chosen strongest one. "Wins on every seed" used to be evaluated against a single
rival picked by highest mean AUC, which does not imply winning against the others: with
proposed=[10,10,10], b1=[9,9,9], b2=[11,0,0] the chosen rival is b1 (mean 9 > 3.67), proposed beats
it 3/3, and the script printed a clean sweep even though b2 beat proposed on seed 0. The headline
boolean is now `beats_every_baseline_every_seed`, and `worst_margin` — the minimum paired margin
over ALL baselines and ALL seeds — is the single number that decides it. That criterion is
deliberately STRICT and is not a hypothesis test: at 5 seeds and 5 baselines it demands 25 out of 25
paired comparisons fall the same way, which noise alone can break. Read it with the per-baseline
`margin_vs_*_mean` / `wins_vs_*` columns, and note that a clean sweep against ONE baseline is
p=(1/2)^n_seeds by an exact one-sided sign test — 0.125 at 3 seeds, 0.031 at 5, and never a JOINT
p-value across baselines that share seeds, splits and drop sets. The per-seed AUCs in the raw CSV
are there so a reader can run whatever paired test they prefer.

EVIDENCE. Aggregates alone cannot be re-analysed. A long-form RAW csv is written next to the
summary with one row per (G, seed, method, missing_groups): the measured mIoU plus that seed's
derived AUC/clean/retention, so a reader can recompute every summary column, or apply different
statistics, without re-running anything.

Outputs (../paper/), named after the sweep so two different --groups can never overwrite each other:
  results_phase2_group_ablation_G5-10-20.csv       (summary: one row per G)
  results_phase2_group_ablation_G5-10-20_raw.csv   (long form: one row per G/seed/method/m)
Usage:
  python experiments/phase2_group_ablation.py --seeds 0 1 2 3 4 --groups 5 10 20
  python experiments/phase2_group_ablation.py --smoke      # 1 seed / 12 epochs -> *_smoke.csv
  python experiments/phase2_group_ablation.py --selfcheck  # numeric guards, no training
"""
import os
import sys
import csv
import inspect
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase2_degradation as P2
from bandsim.metrics import audc
from bandsim import hw, parallel
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
def P(rel):
    return os.path.join(PAPER_DIR, rel)

# Checked against what run_seed actually returns before anything is aggregated (see main). Hard-
# coding a SUBSET is what silently removed B4/B6 from every comparison in this file.
EXPECTED_METHODS = ("b1", "b2", "b3", "b4", "b6", "proposed")


def _sd1(a):
    """SAMPLE standard deviation (ddof=1) over seeds, NaN for a single seed.

    numpy's default ddof=0 is the population SD of the seeds that happened to run, which is not what
    an error bar on a seed mean means; at n=3 it is 82% of the sample SD, i.e. a bar 18% too small.
    NaN at n=1 rather than 0.0: a one-seed run has no spread to report, and writing 0.00 reads as
    "perfectly reproducible" — which is exactly how the last published version of this file's CSV
    read. Matches phase5/phase8. Written to CSV as an EMPTY cell (see _fmt) so integrity_check's
    no-NaN rule keeps meaning what it says."""
    a = np.asarray(a, float)
    return float(a.std(ddof=1)) if a.size > 1 else float("nan")


def _fmt(v, nd=4):
    """NaN -> empty cell. An empty cell is unambiguously "not computed"; "nan" is a value a reader
    (and integrity_check's finite-cell scan) has to special-case."""
    v = float(v)
    return "" if v != v else f"{v:.{nd}f}"


def paired_verdict(aucs, baselines, proposed="proposed"):
    """(worst_margin, beats_every_baseline_every_seed) over EVERY baseline and EVERY seed.

    A module-level function rather than three lines inside main() because it is the sentence the
    whole experiment exists to produce, and it was WRONG in a way no aggregate could reveal. The
    rule it replaces picked ONE rival by highest MEAN auc and asked whether proposed beat that rival
    on every seed — which does not imply beating the others even once. Counterexample, pinned in the
    tests: proposed=[10,10,10], b1=[9,9,9], b2=[11,0,0]. The mean picks b1 as the rival (9 vs 3.67),
    proposed sweeps it 3/3, and the run printed a clean sweep while b2 beat proposed on seed 0. The
    MINIMUM margin over all baselines and all seeds is the one number that selection cannot game.
    """
    if not baselines:
        raise ValueError("no baselines to compare 'proposed' against — a verdict over an empty set "
                         "of rivals is vacuously True, which is not a robustness result.")
    p = np.asarray(aucs[proposed], float)
    for k in baselines:
        # Unequal lengths BROADCAST rather than raise: one seed of a baseline against three of
        # proposed silently becomes three paired differences against the same value, and the verdict
        # comes out of a comparison nobody made.
        if np.shape(aucs[k]) != np.shape(p):
            raise ValueError(f"'{k}' has {np.shape(aucs[k])} per-seed AUCs but 'proposed' has "
                             f"{np.shape(p)}: a paired margin needs one value per seed per method.")
    worst = min(float(np.min(p - np.asarray(aucs[k], float))) for k in baselines)
    return worst, bool(worst > 0)


def _check_fraction_axis(a, who):
    """np.interp's `xp` (the MEASURED axis) must be strictly increasing and numpy does not enforce
    it — numpy's own docs say "The x-coordinate sequence is expected to be increasing, but this is
    not explicitly enforced", and a non-increasing axis returns wrong values with NO exception. (The
    same numpy footgun is guarded on the WAVELENGTH axis by
    `phase2_degradation._check_wavelength_axis`; this is the missing-FRACTION axis, which was not.)

    Measured on the four points [(0,70), (.2,55), (.4,45), (.5,40)]: the correctly ordered curve
    summarises to 53.5, and merely REORDERING the same four pairs to fractions [0, .4, .2, .5]
    returns 54.5 — a normal-looking cross-G number, straight into the results CSV. Ties are rejected
    for the same reason as on the wavelength axis: with a duplicated fraction np.interp picks a side
    by binary-search order, so the summary depends on array order rather than on what was measured.

    The QUERY axis is checked too. numpy does NOT require the query points to be sorted (only `xp`),
    so this is a local invariant of this module rather than a numpy requirement — an earlier version
    of this docstring claimed numpy demanded both, which it does not. The local reason is real: the
    extrapolation guard in frac_auc inspects only x[0] and x[-1], so an unsorted query axis walks
    straight past it — the axis [0.0, 0.9, 0.20] against a curve measured out to 0.20 passes both
    bounds while 0.9 sits far outside the measured range and is flat-extrapolated, which is exactly
    what that guard exists to prevent.
    """
    a = np.asarray(a, float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{who} must be a non-empty 1-D array, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{who} must be finite (np.interp silently propagates NaN)")
    if a.size > 1 and not np.all(np.diff(a) > 0):
        bad = int(np.argmin(np.diff(a)))
        raise ValueError(
            f"{who} must be STRICTLY INCREASING and unique for np.interp (numpy does not enforce "
            f"this and returns wrong values instead of raising): a[{bad}]={a[bad]:.6g} >= "
            f"a[{bad + 1}]={a[bad + 1]:.6g}. Sort the axis (and the curve with it) first.")
    return a


def _check_curve(curve, fracs, who):
    """A curve must be one finite mIoU per measured fraction. np.interp turns a length mismatch into
    a wrong-but-plausible interpolant rather than raising, and audc turns a single NaN into a NaN
    summary that lands in the CSV as the string "nan"."""
    curve = np.asarray(curve, float)
    if curve.shape != np.shape(fracs):
        raise ValueError(f"{who}: curve/fraction-axis shape mismatch {curve.shape} vs "
                         f"{np.shape(fracs)} — np.interp would silently interpolate against the "
                         f"wrong x values.")
    if not np.isfinite(curve).all():
        raise ValueError(f"{who}: curve must be finite, got "
                         f"{int((~np.isfinite(curve)).sum())} non-finite point(s).")
    return curve


def frac_auc(common, fracs, curve):
    """LEGACY grid-resampling summary. NOT the production path any more — see frac_auc_exact.

    Kept because it is the shortest way to DEMONSTRATE the defect that retired it: this returns the
    trapezoid over `common`, which equals the exact piecewise-linear integral only when `common`
    CONTAINS every measured knot inside the range. selfcheck_fraction_axis pins both the case where
    the two agree (the default sweep) and the case where they disagree (--groups 6 7).
    """
    common = np.asarray(common, float); fracs = np.asarray(fracs, float)
    # ORDER MATTERS: the zero-width check runs first because an all-tied grid is ALSO non-monotonic,
    # and "the grid collapsed to zero width" names the actual upstream cause (a G=1 entry dragging
    # frac_max_common, which is a min over G, to 0) where "not strictly increasing" would not.
    if common.size < 2 or not (common[-1] > common[0]):
        raise ValueError(
            f"the common missing-fraction grid must span a positive range with >=2 points, got "
            f"{common.size} point(s) over [{common[0] if common.size else float('nan')}, "
            f"{common[-1] if common.size else float('nan')}] — a zero-width grid makes the mean "
            f"undefined (it used to divide by zero and write `inf` to the CSV).")
    _check_fraction_axis(fracs, "measured missing-fraction axis (np.interp's xp)")
    _check_fraction_axis(common, "common missing-fraction grid (np.interp's x)")
    if common[0] < fracs[0] or common[-1] > fracs[-1]:
        # np.interp EXTRAPOLATES FLAT (it clamps to the end values) instead of raising, so a grid
        # reaching past the measured range would pad the curve with its last measured mIoU and
        # quietly report robustness that was never evaluated.
        raise ValueError(f"grid [{common[0]:.4f}, {common[-1]:.4f}] leaves the measured range "
                         f"[{fracs[0]:.4f}, {fracs[-1]:.4f}]: np.interp would extrapolate flat.")
    return audc(common, np.interp(common, fracs, curve))


def exact_knots(frac_max, fracs, curve):
    """(x, y) that reproduce the measured piecewise-linear curve on [fracs[0], frac_max] EXACTLY.

    The knot set is the measured fractions strictly below frac_max, plus frac_max itself. The
    trapezoid rule is exact on every linear segment, so integrating over these points is the exact
    integral of the interpolant np.interp defines — no grid, no --grid parameter, no way for a
    resampling axis to miss a knot. Adding points BETWEEN knots (what a fine linspace does) cannot
    change the value; omitting a knot can, and did (see the module docstring's 6/7 sign flip).
    """
    fracs = _check_fraction_axis(fracs, "measured missing-fraction axis (np.interp's xp)")
    curve = _check_curve(curve, fracs, "exact_knots")
    frac_max = float(frac_max)
    if not np.isfinite(frac_max) or frac_max <= fracs[0]:
        raise ValueError(
            f"the common missing-fraction maximum must be finite and > the first measured fraction "
            f"({fracs[0]:.4f}), got {frac_max} — a zero-width range makes the mean undefined (it "
            f"used to divide by zero and write `inf` to the CSV for every method at every G).")
    if frac_max > fracs[-1] + 1e-12:
        raise ValueError(f"common maximum {frac_max:.4f} leaves the measured range "
                         f"[{fracs[0]:.4f}, {fracs[-1]:.4f}]: np.interp would extrapolate FLAT and "
                         f"report robustness at fractions that were never evaluated.")
    x = np.unique(np.concatenate([fracs[fracs < frac_max], [frac_max]]))
    return x, np.interp(x, fracs, curve)


def frac_auc_exact(frac_max, fracs, curve):
    """MEAN mIoU over [0, frac_max] for ONE seed's degradation curve — the production summary.

    `audc` ALREADY normalizes by the x-range, so this sits in [0, 100] like a mIoU: a flat mIoU=42
    curve must come back as 42 (it came back as 84 while main() divided by the range a second time).
    """
    x, y = exact_knots(frac_max, fracs, curve)
    return audc(x, y)


def seed_summary(frac_max, fracs, curve, who=""):
    """(absolute_auc, clean_miou, retention_auc) for ONE seed's degradation curve.

    A function rather than four lines inside main() so the arithmetic is reachable by a test: a
    mutation that swapped the retention ratio end for end survived the entire suite while it lived
    in main(), because nothing short of a training run executed that line.

    retention_auc is 100*absolute/clean, which IS the integral of curve/curve[0] over the same range
    — clean is a per-seed constant and integration is linear — computed as the ratio so the two
    cannot disagree, and pinned against the direct integral in the tests.
    """
    x, y = exact_knots(frac_max, fracs, curve)
    abs_auc = audc(x, y)
    clean = float(y[0])
    if not clean > 0:
        raise ValueError(f"{who}: clean mIoU is {clean}, so retention (curve/curve[0]) is undefined "
                         f"— a model that scores 0 with nothing missing has not trained.")
    return abs_auc, clean, 100.0 * abs_auc / clean


def _atomic_write_csv(path, fieldnames, rows):
    """Write via a temp file + os.replace: a killed run cannot leave a half-written deliverable that
    still parses as CSV. (Same discipline as phase8 / phase8F_multi.)"""
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def require_class_set_contract():
    """Refuse to run unless phase2_degradation scores EVERY method on the fixed macro class set.

    `miou` averages over whichever classes are present in y_true, and each seed shifts the
    checkerboard, so without a fixed set each seed reports a different estimand and their mean
    averages different quantities — the defect `common_class_set` exists to fix. This script did not
    pass `class_set` at all, so its mIoU was not even the same metric as the Phase 2 deliverable it
    is meant to corroborate.

    The check is a CONTRACT, not a courtesy, because of the asymmetry it guards: `eval_mlp` routes
    through `miou_over(..., class_set)` while `eval_proposed` takes a `class_set=` argument and does
    not read it. Passing class_set into a build where that is still true is WORSE than not passing
    it: b1/b2/b3 would then be scored on the fixed 14-class set and b4/b6/proposed on a drifting
    one, so the within-G paired comparison — the only comparison this script is entitled to make —
    would stop being like-for-like. Not passing it at all is at least uniformly wrong.
    """
    missing = []
    if not hasattr(P2, "common_class_set") or not hasattr(P2, "SPLIT_BLOCK"):
        missing.append("phase2_degradation.common_class_set / SPLIT_BLOCK do not exist")
    if "class_set" not in inspect.signature(P2.run_seed).parameters:
        missing.append("phase2_degradation.run_seed does not accept class_set=")
    try:
        src = inspect.getsource(P2.eval_proposed)
    except OSError:                                    # pragma: no cover - source always available
        src = ""
    # Comment lines are stripped before the search. The fix that satisfies this check also carries a
    # comment explaining itself, and that comment naturally mentions `miou_over` — so a revert that
    # left the prose behind would keep passing a naive substring test while the call underneath went
    # back to the bare metric.
    src = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    if "miou_over" not in src:
        missing.append(
            "phase2_degradation.eval_proposed accepts class_set= but never reads it (its body "
            "calls the bare miou()) — so b1/b2/b3 would be scored on the fixed macro class set "
            "and b4/b6/proposed on whichever classes each seed's split happens to contain. One "
            "line: `return miou_over(yte, pred, ..., class_set)`, as eval_mlp already does")
    if missing:
        raise RuntimeError(
            "phase2_degradation does not yet honour a fixed macro class set for every method:\n  - "
            + "\n  - ".join(missing)
            + "\nThe group sweep needs the SAME estimand as the Phase 2 deliverable it corroborates, "
              "and the same one for every method it compares. Fix phase2_degradation first.")


def selfcheck_fraction_axis():
    """Numeric guards for this file. No training, no data — safe to run anywhere.

    The previous version tested ONE curve shape, mIoU(frac) = 100*(1-frac). A straight line is
    integrated exactly by the trapezoid rule on ANY axis, so that test was structurally incapable of
    detecting the grid/knot misalignment that made the cross-G numbers wrong — it passed throughout.
    The convex cases below are the regression test for it.
    """
    def _fail(msg):
        # NOT `assert`: python -O strips assert statements, and a guard that evaporates under an
        # interpreter flag reports "[selfcheck] OK" for a broken implementation.
        raise RuntimeError(f"[selfcheck] FAILED: {msg}")

    # ---- 1. the missing-FRACTION axis is what makes G comparable (raw counts are not) ----
    def curve(G, max_missing):
        fr = np.arange(0, max_missing + 1) / G
        return fr, 100.0 * (1.0 - fr)                        # identical physics, different sampling
    fr5, y5 = curve(5, 4)                                    # G=5  reaches 0.8 missing fraction
    fr20, y20 = curve(20, 10)                                # G=20 reaches 0.5 missing fraction
    raw5, raw20 = audc(np.arange(len(y5)), y5), audc(np.arange(len(y20)), y20)
    auc5 = frac_auc_exact(0.5, fr5, y5)
    auc20 = frac_auc_exact(0.5, fr20, y20)
    flat = frac_auc_exact(0.5, fr20, np.full_like(y20, 42.0))
    print(f"[selfcheck] raw-count AUDC:  G5={raw5:.2f}  G20={raw20:.2f}  (differ; ranges 0..0.8 vs 0..0.5)")
    print(f"[selfcheck] fraction AUC:    G5={auc5:.2f}  G20={auc20:.2f}  (match on the common 0..0.5 range)")
    print(f"[selfcheck] flat mIoU=42 curve -> {flat:.4f}  (must be 42: a MEAN, not an area)")
    if abs(auc5 - auc20) > 1e-9:
        _fail(f"same physics must give the same fraction-AUC: {auc5} vs {auc20}")
    if abs(raw5 - raw20) <= 1.0:
        _fail(f"raw-count AUDC should conflate the ranges: {raw5} vs {raw20}")
    if abs(flat - 42.0) > 1e-9:
        _fail(f"AUC must be a per-point MEAN in [0,100], got {flat} for mIoU=42")

    # ---- 2. exact integration vs the retired grid, on a CONVEX curve (the old blind spot) ----
    f = lambda x: 70.0 * np.exp(-2.5 * np.asarray(x))                                  # noqa: E731

    def reference_mean(frac_max, fracs, y):
        """Independent reference: walk the segments and apply the analytic trapezoid to each.
        Deliberately a different SHAPE of code from exact_knots (a Python loop over segments, not a
        vectorised knot union), so agreeing with it is evidence rather than a tautology."""
        fracs = np.asarray(fracs, float); total = 0.0; a = float(fracs[0])
        while a < frac_max - 1e-15:
            i = int(np.searchsorted(fracs, a, side="right")) - 1
            b = min(float(fracs[i + 1]), frac_max)
            total += 0.5 * (np.interp(a, fracs, y) + np.interp(b, fracs, y)) * (b - a)
            a = b
        return total / (frac_max - float(fracs[0]))

    # The DEFAULT sweep: its linspace contains every knot, so grid == exact. This is the guard that
    # says the integrator change moved no previously published default number.
    d5, d10, d20 = np.arange(0, 5) / 5, np.arange(0, 7) / 10, np.arange(0, 11) / 20
    grid = np.linspace(0.0, 0.5, 21)
    for G, fr in ((5, d5), (10, d10), (20, d20)):
        g = frac_auc(grid, fr, f(fr))
        e = frac_auc_exact(0.5, fr, f(fr))
        r = reference_mean(0.5, fr, f(fr))
        if abs(g - e) > 1e-12 or abs(e - r) > 1e-12:
            _fail(f"default sweep G={G}: grid={g!r} exact={e!r} reference={r!r} must all agree")
    print("[selfcheck] default {5,10,20}: grid == exact == reference to 1e-12 (no published number moves)")

    # --groups 6 7: max_knots is 6 and the OLD guard passed at --grid 6, yet that linspace misses
    # every G=7 knot. The gap between the two G even changes SIGN.
    s6, s7 = np.arange(0, 6) / 6, np.arange(0, 7) / 7
    fmax = 5.0 / 6.0
    bad = np.linspace(0.0, fmax, 6)
    g6, g7 = frac_auc(bad, s6, f(s6)), frac_auc(bad, s7, f(s7))
    e6, e7 = frac_auc_exact(fmax, s6, f(s6)), frac_auc_exact(fmax, s7, f(s7))
    if (abs(e6 - reference_mean(fmax, s6, f(s6))) > 1e-12
            or abs(e7 - reference_mean(fmax, s7, f(s7))) > 1e-12):
        _fail("exact integration must match the independent segment-walk reference at G=6,7")
    print(f"[selfcheck] --groups 6 7 @ --grid 6 (which the OLD guard accepted): "
          f"G7 grid={g7:.5f} vs exact={e7:.5f} ({g7 - e7:+.5f})")
    print(f"[selfcheck]   G6-G7 gap: grid={g6 - g7:+.5f} vs exact={e6 - e7:+.5f}  <- SIGN FLIP")
    if abs(g7 - e7) < 0.3:
        _fail(f"the 6/7 misalignment must still be demonstrated: grid={g7} exact={e7}")
    if np.sign(g6 - g7) == np.sign(e6 - e7):
        _fail(f"the 6/7 case must flip the cross-G sign: grid gap {g6 - g7}, exact gap {e6 - e7}")

    # ---- 3. retention identity: the ratio IS the integral of the retention curve ----
    x, y = exact_knots(0.5, d20, f(d20))
    direct = audc(x, y / y[0]) * 100.0
    ratio = 100.0 * frac_auc_exact(0.5, d20, f(d20)) / f(d20)[0]
    if abs(direct - ratio) > 1e-9:
        _fail(f"retention AUC identity broken: integral {direct} vs ratio {ratio}")
    print(f"[selfcheck] retention AUC: integral={direct:.6f} == 100*absolute/clean={ratio:.6f}")

    # ---- 4. the guards that stop a silently-wrong summary ----
    for who, fn in (
        ("zero-width range", lambda: frac_auc_exact(0.0, d20, f(d20))),
        ("extrapolation past the measured range", lambda: frac_auc_exact(0.9, d20, f(d20))),
        ("non-monotonic measured axis", lambda: frac_auc_exact(0.4, np.array([0.0, .4, .2, .5]),
                                                               np.array([70., 55., 45., 40.]))),
        ("curve/axis length mismatch", lambda: frac_auc_exact(0.4, d20, f(d20)[:-1])),
        ("non-finite curve", lambda: frac_auc_exact(0.4, d20, np.where(d20 == 0.2, np.nan, f(d20)))),
    ):
        try:
            fn()
        except ValueError:
            continue
        _fail(f"{who} must raise, not return a plausible number")
    print("[selfcheck] OK: fraction-axis AUC is cross-G comparable, exact, and a mean in [0,100].")
    return (raw5, raw20), (auc5, auc20)


def _validate(args):
    """Every pure-argument error, rejected BEFORE hw.setup / load_data / any training.

    `--trials 0` used to be caught by degradation_curve — correctly, but only after every model of
    the first seed had been trained. A configuration typo should cost seconds, not a GPU-hour."""
    # argparse's nargs="+" already forbids an empty list from the CLI, but _validate is the
    # contract for the whole module: with no seeds every mean is a NaN over an empty array (a
    # RuntimeWarning, not an error) and with no groups there is no row 0 to take CSV headers from.
    if len(args.seeds) < 1:
        raise ValueError("--seeds needs at least one seed.")
    if len(args.groups) < 1:
        raise ValueError("--groups needs at least one group count.")
    if len(set(args.seeds)) != len(args.seeds):
        dup = sorted({s for s in args.seeds if list(args.seeds).count(s) > 1})
        raise ValueError(f"--seeds must be unique, got duplicates {dup}: a repeated seed retrains "
                         f"to the identical result and is then counted twice in every mean, "
                         f"doubling its weight while n_seeds and the paired-win denominator both "
                         f"claim {len(args.seeds)} independent runs.")
    if len(set(args.groups)) != len(args.groups):
        dup = sorted({g for g in args.groups if list(args.groups).count(g) > 1})
        raise ValueError(f"--groups must be unique, got duplicates {dup}: the repeat costs a full "
                         f"retrain of every seed and then overwrites the first result, leaving "
                         f"duplicate rows in the CSV for one group count.")
    for G in args.groups:
        # G=1 would make max_missing 0 -> a single-point "curve" at fraction 0, which drags the
        # common maximum (a min over G) to 0 and collapses the range to zero width for EVERY G, not
        # just this one. That used to divide by zero and write `inf` into the CSV for all methods at
        # all group counts. G=1 is also vacuous (no group left to drop) and rejected by run_seed.
        if G < 2:
            raise ValueError(f"--groups needs every G >= 2, got {G}: with one group nothing can go "
                             f"missing, so its reachable missing fraction is 0 and the shared "
                             f"cross-G range collapses to zero width for every other G too.")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}: at 0 the MLP baselines and the "
                         f"supervised fine-tune never run, while SGMAE still pretrains for "
                         f"max(1, epochs//2) = 1 epoch — an untrained-vs-pretrained comparison "
                         f"presented as a like-for-like one.")
    if args.trials < 1:
        raise ValueError(f"--trials must be >= 1, got {args.trials}: each m>0 point averages "
                         f"`trials` random drop-set draws wherever the space is too large to "
                         f"enumerate.")
    if args.jobs is not None and args.jobs < 1:
        raise ValueError(f"--jobs must be >= 1 if given, got {args.jobs}.")
    if args.grid is not None:
        raise ValueError(
            "--grid was removed: the summary is now the EXACT integral of the measured piecewise-"
            "linear curve (see exact_knots), so there is no resampling grid to size. The flag is "
            "rejected rather than ignored because a run that silently drops a flag you passed is "
            "the failure mode this file exists to document.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--groups", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--grid", type=int, default=None,
                    help="REMOVED (exact integration needs no grid); passing it is an error")
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent seed workers (default: adaptive; also BANDSIM_WORKERS)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                    help="device for the workers (default: auto; also BANDSIM_DEVICE)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="disable deterministic algorithms for a small speedup")
    ap.add_argument("--smoke", action="store_true", help="1 seed / 12 epochs, quick sanity")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the missing-fraction-axis numeric guards and exit (no training)")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck_fraction_axis(); return 0

    # A smoke run writes *_smoke.* so it cannot replace a real sweep. This script had no --smoke
    # flag and therefore no suffix protection, yet experiments/integrity_check.py ran it as
    # `--seeds 0 --groups 5 10 --epochs 12` pointed at the CANONICAL filename — so the integrity
    # harness itself overwrote the deliverable with a 1-seed 12-epoch 2-group run. The evidence was
    # sitting in paper/: n_seeds=1, every *_std 0.0, no G=20 row at all and no provenance sidecar,
    # under a STATUS_REPORT line quoting three G values and three numbers the file does not contain.
    out_tag = ""
    if args.smoke:
        args.seeds = [0]; args.epochs = 12; args.trials = 4; args.groups = [5, 10]
        out_tag = "_smoke"
        print("SMOKE RUN: 1 seed / 12 epochs / G in {5,10} — writing *_smoke.csv. These numbers are "
              "a sanity check, not results.")
    _validate(args)
    # The sweep is a SET: sorting makes the output name, the row order and the sidecar independent
    # of the order the group counts were typed in.
    args.groups = sorted(args.groups)
    # Named after the sweep. `--groups 5 10 20` and `--groups 5 10` integrate over DIFFERENT common
    # ranges (0..0.5 vs 0..0.6) — different estimands — and writing both to one filename silently
    # replaced one with the other, leaving nothing in the file to say which sweep produced it.
    tag = "G" + "-".join(str(g) for g in args.groups)
    require_class_set_contract()

    hw.setup(deterministic=not args.nondeterministic, prefer=args.device); print("HW:", hw.info())
    cube, gt = P2.load_data()

    # The macro class set, fixed ONCE across every seed and shared by every method — the same set
    # phase2_degradation.main() computes, so this sweep reports the SAME estimand as the deliverable
    # it corroborates. Offsets s and s+block give byte-identical splits, so duplicates collapse.
    uniq_off = sorted({int(sd) % P2.SPLIT_BLOCK for sd in args.seeds})
    class_set, present = P2.common_class_set(gt, P2.SPLIT_BLOCK, uniq_off)
    print(f"MACRO CLASS SET: {len(class_set)}/{P2.NUM_CLASSES} classes present in every split "
          f"({len(uniq_off)} distinct offsets); mIoU averages over these and ONLY these.")
    excluded = [c for c in range(P2.NUM_CLASSES) if c not in class_set]
    if excluded:
        print("  EXCLUDED (absent from at least one split): "
              + ", ".join(f"class {c} ({present[c]}/{len(uniq_off)})" for c in excluded)
              + " — this CHANGES the metric definition versus a full-16 mIoU; say so where the "
                "numbers appear.")

    # ---- pass 1: run every G; keep per-seed curves on the MISSING-FRACTION axis ----
    per_G = {}
    frac_max_common = 1.0
    for G in args.groups:
        max_missing = min(6, G - 1) if G <= 10 else G // 2   # per-G physical sweep (unchanged)
        fracs = np.arange(0, max_missing + 1) / G            # missing FRACTION in [0,1]
        frac_max_common = min(frac_max_common, float(fracs.max()))  # reachable by EVERY G
        results = parallel.run_jobs(
            P2.run_seed, args.seeds,
            shared=dict(cube=cube, gt=gt, n_groups=G, max_missing=max_missing,
                        trials=args.trials, epochs=args.epochs, class_set=class_set),
            prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
            label=f"groupabl/G{G}/seed")
        curves = {k: [] for k in EXPECTED_METHODS}
        params = None
        # P2.run_seed returns (curves, params, raw_rows) since c993b7e added the per-drop-set raw
        # record; this sweep aggregates curves and params and does not re-emit phase2's raw rows
        # (its own raw artefact is per-G AUDC, written below). The starred slot keeps this call
        # site honest about arity instead of silently truncating a future fourth element.
        for sd, (cv, pr, *_raw) in zip(args.seeds, results):
            # The method list is a CONTRACT with run_seed, checked rather than assumed. Hard-coding
            # a subset is how B4 and B6 came to be trained at every G and every seed and then
            # excluded from every comparison, table and conclusion this script produced.
            if set(cv) != set(EXPECTED_METHODS):
                raise RuntimeError(
                    f"run_seed's method set changed: missing={sorted(set(EXPECTED_METHODS) - set(cv))}, "
                    f"unexpected={sorted(set(cv) - set(EXPECTED_METHODS))}. Update EXPECTED_METHODS "
                    f"deliberately — a silent subset drops trained baselines out of the comparison.")
            for k in EXPECTED_METHODS:
                curves[k].append(_check_curve(cv[k], fracs, f"G={G} seed={sd} method={k}"))
            if params is not None and pr != params:
                raise RuntimeError(f"G={G}: parameter counts differ between seeds ({params} vs "
                                   f"{pr}) — the architecture must depend on G alone, not on seed.")
            params = pr
            print(f"G={G} seed {sd} (max missing fraction {fracs.max():.2f}): "
                  + " ".join(f"{k}[clean]={cv[k][0]:.1f}" for k in EXPECTED_METHODS))
        per_G[G] = dict(fracs=fracs, curves=curves, params=params, max_missing=max_missing)

    if frac_max_common <= 0.0:
        raise ValueError(f"no G in {args.groups} can reach a positive missing fraction "
                         f"(frac_max_common={frac_max_common}) — nothing to integrate.")
    print(f"\ncommon missing-fraction range: 0..{frac_max_common:.4f} (exact piecewise-linear "
          f"integration on each curve's own knots — no resampling grid)")
    caps = {G: per_G[G]["params"] for G in args.groups}
    if len({tuple(sorted(p.items())) for p in caps.values()}) > 1:
        print("NOTE: parameter counts differ across G (the attention model embeds with "
              "nn.Linear(S, d_model) and S is the largest group size), so a cross-G LEVEL "
              "difference is confounded with capacity: "
              + "; ".join(f"G={G}: {caps[G]}" for G in args.groups))

    # ---- pass 2: summarise each G over the common range ----
    rows, raw_rows = [], []
    baselines = [k for k in EXPECTED_METHODS if k != "proposed"]
    for G in args.groups:
        fracs = per_G[G]["fracs"]; curves = per_G[G]["curves"]
        row = {"groups": G, "frac_min": 0.0, "frac_max": round(float(frac_max_common), 6),
               "n_seeds": len(args.seeds), "epochs": args.epochs, "trials": args.trials,
               "max_missing": per_G[G]["max_missing"], "seed_sd_ddof": 1,
               "params_mlp": per_G[G]["params"].get("MLP"),
               "params_proposed": per_G[G]["params"].get("Proposed"),
               "params_b4": per_G[G]["params"].get("B4"), "params_b6": per_G[G]["params"].get("B6")}
        aucs, cleans, rets = {}, {}, {}
        for k in EXPECTED_METHODS:
            a, c, r = [], [], []
            for sd, cv in zip(args.seeds, curves[k]):
                abs_auc, clean, ret = seed_summary(frac_max_common, fracs, cv,
                                                   who=f"G={G} seed={sd} {k}")
                a.append(abs_auc); c.append(clean); r.append(ret)
                for m, fr in enumerate(fracs):
                    raw_rows.append({"groups": G, "seed": sd, "method": k, "missing_groups": m,
                                     "missing_fraction": f"{fr:.6f}", "miou": f"{cv[m]:.6f}",
                                     "seed_clean_miou": f"{clean:.6f}",
                                     "seed_abs_auc": f"{abs_auc:.6f}",
                                     "seed_retention_auc": f"{ret:.6f}",
                                     "common_frac_max": f"{frac_max_common:.6f}"})
            aucs[k] = np.array(a); cleans[k] = np.array(c); rets[k] = np.array(r)
            row[f"{k}_auc_mean"] = _fmt(aucs[k].mean())
            row[f"{k}_auc_sd_ddof1"] = _fmt(_sd1(aucs[k]))
            row[f"{k}_clean_mean"] = _fmt(cleans[k].mean())
            row[f"{k}_retention_auc_mean"] = _fmt(rets[k].mean())      # drop AUC = clean - auc

        # Paired against EVERY baseline. Same seed = same split, same init, same drop-set list, so
        # each difference is paired and its spread is what decides whether "proposed wins" is a
        # finding or a coin flip. The old code picked ONE rival by highest mean AUC and reported
        # only that margin — which does not imply beating the others on any seed (see docstring).
        for k in baselines:
            d = aucs["proposed"] - aucs[k]
            row[f"margin_vs_{k}_mean"] = _fmt(d.mean())
            row[f"margin_vs_{k}_sd_ddof1"] = _fmt(_sd1(d))
            row[f"wins_vs_{k}"] = int((d > 0).sum())
        worst, beats = paired_verdict(aucs, baselines)
        row["worst_margin"] = _fmt(worst)
        row["beats_every_baseline_every_seed"] = beats
        row["best_rival"] = max(baselines, key=lambda k: aucs[k].mean())   # descriptive only
        rows.append(row)

        print(f"  -> G={G} mean mIoU over missing fraction [0,{frac_max_common:.2f}]: "
              + "  ".join(f"{k}={aucs[k].mean():.1f}" for k in EXPECTED_METHODS))
        print("     retention AUC (% of clean): "
              + "  ".join(f"{k}={rets[k].mean():.1f}" for k in EXPECTED_METHODS))
        print("     paired margin vs each baseline: "
              + "  ".join(f"{k}={(aucs['proposed'] - aucs[k]).mean():+.2f}"
                          f"({int((aucs['proposed'] - aucs[k] > 0).sum())}/{len(args.seeds)})"
                          for k in baselines))
        print(f"     worst margin over ALL baselines and ALL seeds: {worst:+.2f} mIoU "
              f"-> beats every baseline on every seed: {worst > 0}")

    summary = P(f"results_phase2_group_ablation_{tag}{out_tag}.csv")
    raw = P(f"results_phase2_group_ablation_{tag}{out_tag}_raw.csv")
    _atomic_write_csv(summary, list(rows[0].keys()), rows)
    _atomic_write_csv(raw, list(raw_rows[0].keys()), raw_rows)
    print(f"\nwrote {summary}")
    print(f"      {raw}  ({len(raw_rows)} rows: every measured point, so every column above is "
          f"recomputable without re-running)")
    print(f"*_auc_mean is the MEAN mIoU over the COMMON missing-fraction range "
          f"[0,{frac_max_common:.2f}] (a mean in [0,100]); *_retention_auc_mean is the same "
          f"integral as % of that seed's clean mIoU. Cross-G LEVEL differences carry a "
          f"coarse-sampling bias (~0.8 mIoU between G=5 and G=20 on an identical synthetic "
          f"response), a capacity difference and a different drop-set sampling regime — compare "
          f"methods WITHIN each G. See the module docstring.")

    all_win = all(r["beats_every_baseline_every_seed"] for r in rows)
    tally = "; ".join("G={}: worst margin {}".format(r["groups"], r["worst_margin"]) for r in rows)
    print(f"Robust only if 'proposed' beats EVERY baseline on EVERY seed at EVERY G: {all_win} "
          f"({tally})")
    n = len(args.seeds)
    print(f"  n={n} seeds: a clean sweep against ONE baseline is p={0.5 ** n:.3f} by an exact "
          f"one-sided sign test. The {len(baselines)} baselines share seeds, splits and drop sets, "
          f"so that is NOT a joint p-value — quote it per baseline, or use the per-seed AUCs in the "
          f"raw CSV for a paired test of your choosing.")

    # The per-G physical sweep is decided INSIDE main() (max_missing = min(6, G-1) if G <= 10 else
    # G//2), so args alone cannot say how far each G was pushed — and that is what makes the
    # *_auc_mean columns comparable or not. Read it back off the measured axis rather than
    # re-deriving the rule, so the record cannot drift from what was run.
    ok = stamp(P(f"results_phase2_group_ablation_{tag}{out_tag}.csv"), args,
               extra={"groups_swept": list(args.groups), "methods": list(EXPECTED_METHODS),
                      "max_missing_by_group": {int(G): int(per_G[G]["max_missing"])
                                               for G in args.groups},
                      "params_by_group": {int(G): per_G[G]["params"] for G in args.groups},
                      "common_missing_fraction_max": float(frac_max_common),
                      "macro_class_set": [int(c) for c in class_set],
                      "n_classes_total": int(P2.NUM_CLASSES),
                      "split_block": int(P2.SPLIT_BLOCK),
                      # getattr, not P2.ENUMERATION_CAP: this line runs AFTER every model has
                      # trained, so an AttributeError here would throw away the whole sweep to
                      # record a field about it. A null in the sidecar says "not recorded"; a
                      # traceback at this point says nothing and costs the run.
                      "enumeration_cap": getattr(P2, "ENUMERATION_CAP", None),
                      "integration": "exact piecewise-linear on measured knots (no grid)",
                      "seed_sd_ddof": 1, "raw_evidence_csv": os.path.basename(raw)})
    # Stamp the RAW file too, not only the summary. `raw_evidence_csv` above is enough to FIND the
    # long form but not to ATTRIBUTE it: anyone who re-analyses the per-seed rows -- which is the
    # entire reason the long form ships -- otherwise holds a CSV with no commit, no args and no
    # class set of its own. That is not hypothetical; the per-level crossover analysis was run off
    # a *_raw.csv, and doctor's stamping check scores deliverables one file at a time.
    ok_raw = stamp(raw, args, extra={"summary_csv": os.path.basename(summary),
                                     "n_rows": len(raw_rows),
                                     "groups_swept": list(args.groups),
                                     "methods": list(EXPECTED_METHODS),
                                     "macro_class_set": [int(c) for c in class_set],
                                     "max_missing_by_group": {int(G): int(per_G[G]["max_missing"])
                                                              for G in args.groups},
                                     "seed_sd_ddof": 1})
    if ok is None or ok_raw is None:
        which = ", ".join(n for n, o in (("summary", ok), ("raw", ok_raw)) if o is None)
        print(f"PROVENANCE FAILED ({which}): the CSVs above are on disk but UNATTRIBUTED — "
              f"do not cite them.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
