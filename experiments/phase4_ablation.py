#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 (optional ablation) — Design C (thin cirrus) & Design D (instrument noise / striping).

⚠️ NEITHER CORRUPTION MODEL IS CALIBRATED PHYSICS, and both source modules say so in their own
headers. The scoping is repeated HERE because a reader arriving from the paper sees "Design C —
thin cirrus" and "Hyperion-like SNR" on this page, and the caveats are two imports away:
  * bandsim/cirrus.py — "ILLUSTRATIVE ONLY, NOT PHYSICALLY VALIDATED": extinction_shape() is a
    hand-crafted Gaussian near 1.375 um, not an ice-cloud (Mie / T-matrix / Baum-Yang) optical
    model, and it does NOT reproduce the operational Sentinel-2 L1C B10 cirrus-band response.
  * bandsim/noise.py — "SCHEMATIC / STANDARD-FORM, NOT CALIBRATED TO ANY REAL SENSOR":
    hyperion_like_snr() is a hand-drawn VNIR~500 -> SWIR~50 falloff, not a measured NEdL curve, and
    "Hyperion-like" names the SHAPE, not the instrument.
Both modules instruct that these designs be EXCLUDED from any physics-claiming result; the D5
reliability paper rests on real 6S transmittance and real pyspectral SRF instead. This file is a
robustness stress test across corruption SHAPES. Nothing in it is a radiometric claim, and "tau"
here is a knob, not a retrieved optical depth.

Tests whether the proposed model's robustness carries ACROSS corruption physics, not just missing
bands. On every measurement so far IT DOES NOT — this panel is a NEGATIVE result and must be
reported as one:
  * Design C (cirrus): the proposed model is decisively WORSE than the B2 dropout baseline at every
    non-trivial optical depth, on every seed.
  * Design D2 (dead columns): the two are indistinguishable — a paired margin far smaller than the
    per-seed spread.
  * Design D1 (stripe gain): split into D1a/D1b below; only D1a can move at all.
Do not cite this file as cross-physics evidence FOR the method. Missing bands (Phase 2) is where the
method wins; cirrus is where it loses, and saying so is what makes the Phase 2 result credible.

⚠️ THE MAGNITUDES ARE SUPERSEDED — RE-RUN BEFORE CITING. This docstring used to quote specific
figures (a paired margin of -5.28+-1.74 mIoU at tau=0.3, retention 2.4% vs 12.0% at tau=1.0, a D2
margin of +0.01 to +0.08). They were produced BEFORE commit 61a2fb0 (the AVIRIS wavelength axis and
the mean-mIoU estimand) and before the D1 ordering fix and the D2 exact-severity fix in this file,
so no run behind them is reproducible from the current tree. The DIRECTION above has survived every
change so far and is far larger than the seed spread; the numbers are deliberately not restated,
because a figure left in a docstring is quoted long after the run that produced it is gone.

Corruption axes, with striping split into its two INDEPENDENT sub-effects so the ablation attributes
each cleanly (instead of conflating them on one "striping strength" axis):
  Design D1 — multiplicative stripe gain  g_c ~ N(1, eps^2)   (NO dead columns), SNR noise on
  Design D2 — dead detector columns       fraction f zeroed    (NO stripe gain),  SNR noise on

DESIGN D1 IS SPLIT IN TWO, AND THE CONTRAST BETWEEN THEM IS THE FINDING.
  D1a  y = g_c * x + n     gain on the SIGNAL, noise added after  -> SURVIVES the norm (plotted)
  D1b  y = g_c * (x + n)   gain multiplies the noise as well      -> DEGENERATE (recorded, not plotted)
"Survives" is a claim about the INPUT: what the model is fed genuinely changes, and run_seed records
by how much. Whether either model RESPONDS is a separate question, and on the runs so far the
answer is nearly no: the metric barely moves and D1a's own paired verdicts come out
indistinguishable. That is a weak robustness observation about the models; it is not what makes the
axis valid, and the two must never be reported as one finding.
Both use the same per-column scalar gain g_c ~ N(1, eps^2) and the same noise realisation. Under the
per-spectrum mean normalisation this script applies, D1b cancels exactly -- (g*(x+n))/mean(g*(x+n))
== (x+n)/mean(x+n) -- so its retention is pinned at 100% by algebra and says nothing about either
model. D1a does not cancel, because the additive term does not carry the gain. Measured by
selfcheck_d1_identity() at eps in {0.05, 0.1, 0.2}:
    D1b  max|meannorm(corrupt) - meannorm(noise-only)| = 6.7e-16   (float rounding)
    D1a  MIN over eps of the same quantity            = 3.3e-03   (13 orders of magnitude larger)

AN EARLIER VERSION OF THIS FILE GOT THIS WRONG and it cost a whole corruption axis. It stated that
"to make D1 measure anything, the gain would have to be per-band-and-column (spectrally varying), or
the normalization would have to not be scale-invariant". Neither is necessary: moving the additive
noise AFTER the gain is enough, and that ordering is the one bandsim.noise itself documents --
detector responsivity multiplies the incoming signal, while read noise is a property of the
electronics and does not scale with it. The old order also had a second problem: add_band_noise sizes
sigma from the per-band mean of whatever it is handed, so gaining first and then calling it would
have given a high-gain column proportionally more noise. D1a therefore extracts the noise realisation
from the CLEAN cube once and adds it after the gain, which keeps n independent of g and makes eps=0
reduce EXACTLY to the noise-only baseline (verified: difference 0.0), so the retention denominator is
the same condition on both axes.

D1b is kept rather than deleted because the pair is the result: the SAME gain magnitude is invisible
or measurable depending only on where the noise enters. It is flagged in the CSV's `degenerate`
column and never plotted -- a line pinned at 100% by algebra is read as the most robust result on a
retention figure no matter what the caption says. main() re-checks BOTH at runtime: D1b must be flat
and D1a must not be, and either failing means the corruption chain no longer implements the model its
axis is named for.

DESIGN D2 REALISES ITS FRACTION EXACTLY (dead_col_mode="exact"), for the same reason in a different
guise: a sweep axis must apply the severity its label claims. add_striping's DEFAULT kills each
column by an independent Bernoulli(f) draw -- the right model when the draw is the nuisance, the
wrong one when f is what the x-axis means. Measured on this 145-column cube over the 5 default
seeds, as the fraction of TEST pixels actually zeroed:
    nominal 1%  ->  0.00% - 2.38%   (seed 0 realised ZERO dead columns: an uncorrupted point sitting
                                     at x=0.01, retention 100% by construction, not by robustness)
    nominal 3%  ->  2.55% - 6.71%
    nominal 5%  ->  3.32% - 8.34%   (OVERLAPS the 3% range -- so the axis did not order its own
                                     conditions, and seed 0's 3% and 5% points destroyed the
                                     identical 6.71% and returned the same mIoU)
Exact mode zeroes floor(f*145+0.5) = 1/4/7 columns on every seed, nested along the sweep so a level
only ADDS dead columns. What remains seed-dependent is WHICH columns are hit and therefore how many
test pixels die, so that is measured from the corrupted cube and written to the CSV
(`realised_dead_cols`, `dead_test_px_pct_mean/std`) instead of assumed from the request. It matters
for reading the panel: a zeroed spectrum carries no information, so that fraction is the share of the
evaluation on which NEITHER model can do better than whatever single class it maps the constant
vector to. A drop of that order is therefore UNAVOIDABLE, and D2's absolute retention level measures
how much of the test set was destroyed rather than how robust either model is. The PAIRED margin is
not affected the same way -- both models face the identical destroyed pixels -- which is why D2's
null result is meaningful while D2's retention LEVEL is not a robustness score.

Reports retention = mIoU(corrupt) / mIoU(that design's OWN corruption-off baseline). The four axes
have TWO distinct zero points, not one shared condition: Design C's tau=0 is CLEAN (no cirrus AND no
instrument noise), whereas D1a/D1b/D2's eps=0 / f=0 are all SNR-noise-ON but striping-OFF. Those
three are the SAME condition and must stay so for their retentions to be comparable, so run_seed
checks it numerically rather than trusting three seed literals to stay in sync. A zero baseline
makes retention undefined -> reported as NaN (not a misleading +inf).

Corruptions are applied on the FULL 2-D cube (H,W,B) so striping runs along real image columns;
test pixels are then extracted with the disjoint test mask. Test spectra are MEAN-normalized (each
spectrum divided by its band-mean) and standardized with CLEAN-TRAIN stats (realistic fixed-
normalization deployment). Mean normalization makes the classifier brightness/scale-robust, so the
C/D ablations isolate spectral-SHAPE distortion -- the physical point of cirrus and striping -- from
global radiance scaling.

The parenthetical that used to justify that choice was FALSE, and it is worth recording because it
is the same mistake as the D1 one above: reasoning about an invariance without measuring it. It
claimed "per-band standardization would be invariant to cirrus's per-band multiplicative scaling and
would hide Design C entirely". That holds only when the per-band statistics are recomputed on the
CORRUPTED image, which is the opposite of what this pipeline does. Measured on the test split,
max|z(corrupt) - z(clean)| at tau = 0.1 / 0.3 / 1.0:
    per-band standardization, statistics from the TEST image    4e-10 / 1e-09 / 7e-09   invariant
    per-band standardization, statistics from CLEAN TRAIN         11.4 / 31.1  / 80.2   NOT
    mean-brightness normalization (what this file applies)       0.039 / 0.112 / 0.339   NOT
So the fixed clean-train standardization used here would not have hidden Design C; it would have
shown it enormously. And in the case where the invariance does hold it is STRONGER than the claim,
absorbing the additive path term too, because per-band standardization is invariant to any per-band
AFFINE map and apply_cirrus is affine: rho*exp(-tau*k(lam)) + path(lam).

What mean normalization actually costs is D1b, stated as a design property rather than an accident:
it divides out any BAND-INDEPENDENT scalar exactly (measured: 1.3e-15 for a uniform x1.37 gain),
which is precisely why a per-column stripe gain has to enter before the noise to be visible at all.
The scale robustness and the D1b degeneracy are one fact seen twice.

Reuses Phase 2 training. Requires bandsim.cirrus / bandsim.noise (already implemented).

Outputs (../paper/):
  figs/fig_ablation_cd.pdf         - retention vs tau (cirrus), vs stripe gain (D1a) and vs
                                     dead-col fraction (D2). THREE panels: the degenerate D1b axis
                                     is measured but withheld (see above).
  results_phase4_ablation.csv      - ALL FOUR designs, including the flagged D1b rows and D2's
                                     realised dead-column / destroyed-test-pixel severity
  results_phase4_raw.csv           - one row per (design, level, seed). Every aggregate above is
                                     recomputable from it, which is the only way a dispersion can
                                     be checked: ddof is invisible in a rounded summary.

Usage:
  python experiments/phase4_ablation.py --seeds 0 1 2 3 4
"""
import os
import sys
import csv
import argparse
import numpy as np
from scipy import stats            # already a hard dependency: bandsim.io's splitter uses SciPy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch  # threads/device configured adaptively by bandsim.hw / bandsim.parallel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from phase2_degradation import (load_data, train_mlp, pretrain_sgmae, finetune_proposed,
                                group_present_mask, NUM_CLASSES)
from bandsim.io import disjoint_block_split, AVIRIS_WL_NM
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.cirrus import apply_cirrus
from bandsim.noise import hyperion_like_snr, add_band_noise, add_striping
from bandsim.model import GroupedCrossBandAttention
from bandsim import hw, parallel
from bandsim.metrics import miou
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)

TAUS = [0.0, 0.1, 0.3, 0.6, 1.0]           # Design C cirrus optical depth
STRIPE_EPS = [0.0, 0.05, 0.1, 0.2]         # Design D1 multiplicative stripe-gain strength (no dead cols)
DEAD_FRACS = [0.0, 0.01, 0.03, 0.05]       # Design D2 dead detector-column fraction (no stripe gain)

# A REPORTING threshold, not a scientific one -- stated as a constant so it can be argued with
# rather than buried in a comparison. Below it, both methods have collapsed and their paired margin
# is a ratio between near-zero retentions; the cirrus axis spends most of its range there, which is
# a property of the grid (TAUS) and not of either model.
FLOOR_RETENTION_PCT = 25.0

# Every corruption axis that is MEASURED and written to the CSV. `degenerate` is non-empty for an
# axis whose curve is fixed by algebra rather than by the model, and the string lands in the CSV's
# `degenerate` column so a downstream table cannot quote those rows as a result.
CSV_DESIGNS = [
    {"design": "C_cirrus", "key": "C", "grid": TAUS, "degenerate": ""},
    {"design": "D1a_gain_then_noise", "key": "D1a", "grid": STRIPE_EPS, "degenerate": ""},
    {"design": "D1b_noise_then_gain", "key": "D1b", "grid": STRIPE_EPS,
     "degenerate": "gain_multiplies_noise_too_so_mean_norm_cancels_it"},
    {"design": "D2_dead_cols", "key": "D_dead", "grid": DEAD_FRACS, "degenerate": ""},
]

# Every axis that is PLOTTED — a strict SUBSET of CSV_DESIGNS, because a curve that cannot move
# does not belong in a figure whose whole point is comparing how flat the curves are. See the D1
# section of the module docstring for the full argument and for where the finding lives instead.
FIGURE_PANELS = [
    {"design": "C_cirrus", "xlabel": r"Cirrus optical depth $\tau$",
     "title": "Design C — thin cirrus"},
    {"design": "D1a_gain_then_noise", "xlabel": r"Stripe-gain strength $\epsilon$  ($y=g_cx+n$)",
     "title": "Design D1a — column gain, noise after"},
    {"design": "D2_dead_cols", "xlabel": r"Dead-column fraction $f$ (no stripe gain)",
     "title": "Design D2 — SNR + dead columns"},
]


def omitted_designs():
    """Designs that are MEASURED but not PLOTTED, in CSV order.

    A separate function so the figure's "measured but not plotted" caption is derived from the two
    specs rather than hand-written next to them: a hand-written caption is exactly the thing that
    goes stale when someone re-adds a panel, and it would then tell the reader an axis is missing
    while the axis is sitting right there. Empty list => no caption is drawn at all.
    """
    plotted = {p["design"] for p in FIGURE_PANELS}
    return [d["design"] for d in CSV_DESIGNS if d["design"] not in plotted]


def mean_brightness_norm(X, eps=1e-8):
    """Per-pixel MEAN brightness normalization: divide each spectrum by its band-mean.

        x'(lam) = x(lam) / mean_lam[ x(lam) ]

    This is MEAN normalization (dividing by the per-spectrum band-mean), NOT an L1 norm
    (x / sum_lam |x|): the two differ by the constant band count and by the absolute value on any
    negatives, so a mean-normalized spectrum has band-mean 1 whereas an L1-normalized one sums to 1.
    It simulates the per-acquisition radiometric normalization a real HSI pipeline applies, making
    the classifier brightness/scale-robust so the C/D ablations isolate spectral-SHAPE distortion
    from global radiance scaling. Applied to BOTH train and test here for consistency.

    ROBUSTNESS: a spectrum whose band-mean is ~0 relative to its own magnitude would EXPLODE under
    plain `x / (mean + eps)` (e.g. [-1, 0, 1] -> mean 0 -> ~[-1e8, 0, 1e8]). Such rows FALL BACK to
    L2 normalization instead (denominator = ||x||_2 >= |x|, so the output stays bounded by 1). The
    trigger is |mean| < 1e-3 * ||x||_2. Guard example: mean_brightness_norm([[-1., 0., 1.]]) ->
    ~[-0.707, 0, 0.707] (bounded), not [-1e8, 0, 1e8].

    WHEN IT FIRES (an earlier version of this docstring got both parts of this wrong):
      * For a STRICTLY POSITIVE spectrum the worst case is one-hot, giving mean/||x||_2 = 1/n_bands
        = 1/200 = 0.005 — a 5x margin over the 1e-3 trigger, NOT the "~0.07" (= 1/sqrt(n_bands))
        previously claimed. Measured on this cube the minimum ratio is 0.056, so the fallback indeed
        never fires on clean or cirrus-corrupted reflectance and those results are unchanged.
      * But it DOES fire, and the old claim that it "NEVER fires" was false: Design D2 zeroes whole
        detector columns, so those pixels have an ALL-ZERO spectrum (mean == 0 AND ||x||_2 == 0).
        There the fallback denominator is eps and the row maps to all-zeros — a defensible
        convention for a spectrum that carries no information, and strictly better than the 0/0 NaN
        plain mean-division would produce, but those rows are constant vectors, not spectra.
        Measured under D2's exact dead-column sweep (1 / 4 / 7 of the 145 columns):
              f=0.01 ->  145/21025 px of the image (0.69%),  1.14 +- 0.07% of the TEST pixels
              f=0.03 ->  580/21025 px (2.76%),               3.40 +- 0.86% of the test pixels
              f=0.05 -> 1015/21025 px (4.83%),               5.77 +- 1.18% of the test pixels
        The test-set column is the one that matters: it is the fraction of EVALUATED pixels on
        which no model can do better than whatever class it maps the constant vector to, so it
        bounds how much of D2's absolute retention drop is attributable to any model. (The figures
        that stood here before, 1305 and 1450 pixels, were seed 0's Bernoulli realisation — 9 and
        10 columns for a nominal 3% and 5%; see run_seed for why that draw was replaced.)
    This is a boundedness guard, not a finiteness guard: a NaN or +inf already present in the input
    still propagates (NaN in -> NaN out). Nothing in the current corruption chain produces one
    (add_band_noise / add_striping both validate their parameters).
    """
    X = np.asarray(X, float)
    mean = X.mean(axis=-1, keepdims=True)
    l2 = np.sqrt((X * X).sum(axis=-1, keepdims=True))
    # "near-zero mean" = band-mean below 0.1% of the spectrum's L2 magnitude -> mean-division would
    # blow up; use the (always >= |mean|) L2 norm as a bounded fallback denominator for those rows.
    near_zero = np.abs(mean) < 1e-3 * (l2 + eps)
    denom = np.where(near_zero, l2 + eps, mean)
    return X / denom


def selfcheck_d1_identity(n_cols=11, n_bands=200, seed=0):
    """Prove BOTH halves of the D1 story numerically: g*(x+n) cancels, g*x+n does not.

    The earlier version proved only the first and concluded that "to make D1 measure anything, the
    gain would have to be per-band-and-column, or the normalisation would have to not be
    scale-invariant". That conclusion is wrong, and this check is what shows it: moving the additive
    noise AFTER the gain makes the same per-column scalar survive the per-spectrum mean, because the
    noise term does not carry the gain. An entire corruption axis was discarded on that reasoning.

    Explicit raises rather than assert: `python -O` strips assert, and this function's whole purpose
    is to stop a degenerate curve being read as robustness."""
    from bandsim.io import AVIRIS_WL_NM as _WL
    rng = np.random.default_rng(seed)
    cube = rng.uniform(1000.0, 8000.0, (7, n_cols, n_bands))          # positive reflectance-like
    snr = hyperion_like_snr(_WL[:n_bands])
    noised = add_band_noise(cube, snr, np.random.default_rng(900))
    noise_only = noised - cube
    base = mean_brightness_norm(noised.reshape(-1, n_bands))

    worst_b, least_a = 0.0, np.inf
    for eps in STRIPE_EPS:
        if eps == 0.0:
            continue
        gained = add_striping(cube, np.random.default_rng(901), stripe_eps=eps,
                              dead_col_frac=0.0, col_axis=1)
        a = mean_brightness_norm((gained + noise_only).reshape(-1, n_bands))       # g*x + n
        b = mean_brightness_norm(add_striping(noised, np.random.default_rng(901), stripe_eps=eps,
                                              dead_col_frac=0.0, col_axis=1
                                              ).reshape(-1, n_bands))              # g*(x+n)
        least_a = min(least_a, float(np.abs(a - base).max()))
        worst_b = max(worst_b, float(np.abs(b - base).max()))
    print(f"[selfcheck D1] over eps={[e for e in STRIPE_EPS if e > 0]}:")
    print(f"  D1b  g*(x+n): max|meannorm(corrupt) - meannorm(noise-only)| = {worst_b:.3e}  (cancels)")
    print(f"  D1a  g*x + n: MIN over eps of the same quantity            = {least_a:.3e}  (survives)")
    if not np.isfinite(worst_b) or worst_b >= 1e-12:
        raise RuntimeError(f"D1b should cancel exactly under mean normalisation; got {worst_b}")
    if not np.isfinite(least_a) or least_a <= 1e-6:
        raise RuntimeError(f"D1a should NOT cancel -- the noise must be added AFTER the gain; "
                           f"smallest deviation over eps was {least_a}")
    print("[selfcheck D1] CONFIRMED: the SAME per-column gain is invisible when it multiplies the "
          "noise too (D1b) and measurable when it does not (D1a). D1b is not a robustness result; "
          "D1a is the axis to read.")
    return worst_b, least_a


def sample_std(x):
    """SAMPLE standard deviation (ddof=1); NaN for n < 2.

    np.std defaults to ddof=0, the POPULATION formula, which underestimates the spread of a sample
    by sqrt((n-1)/n) -- 10.6% at the 5 seeds this file runs by default. Every "+-" here is estimated
    from a handful of seeds and every one of them was too small. n<2 returns NaN rather than 0.0: a
    single seed carries no dispersion estimate, and 0.0 would advertise a perfectly precise one.
    """
    a = np.asarray(x, float)
    return float(a.std(ddof=1)) if a.size >= 2 else float("nan")


def collapsed_levels(axes, floor_pct):
    """Levels where BOTH methods retain less than `floor_pct` against their own zero point.

    Module level rather than a closure inside main(), for the reason the phase 3 session recorded
    in shared_layer.md: claim-bearing logic inside a closure forces a test to reimplement what it
    checks, which is a tautology.

    A paired margin between two COLLAPSED methods is a ratio between near-zero scores, not an
    operational difference, and the cirrus axis spends most of its range there -- a property of the
    TAUS grid, not of either model. Flagged rather than dropped: the levels are real measurements,
    but "loses" there does not mean what it means where both methods still work.

    `floor_pct` is a REPORTING policy, not a testable claim -- there is no ground truth for it, so
    it is a named constant and it is written into the provenance stamp to be argued with. A level
    whose retention is undefined (a zero baseline gives NaN) is NOT flagged: undefined is not low.
    """
    out = []
    for name, acc, grid in axes:
        for lvl in grid:
            if lvl == 0.0:
                continue
            pr = paired_retention(acc[lvl]["proposed"], acc[0.0]["proposed"])[0]
            br = paired_retention(acc[lvl]["b2"], acc[0.0]["b2"])[0]
            if max(pr, br) < floor_pct:          # NaN compares False, which is what we want
                out.append((name, lvl))
    return out


def paired_retention(values, base):
    """PAIRED per-seed retention: mean and std of (value_s / base_s) over seeds s, in %.

    The previous statistic was a ratio of seed-MEANS, mean(values)/mean(base). That is not a paired
    comparison: it discards the seed pairing (each seed has its own data split, init and corruption
    realisation, so value_s and base_s are strongly correlated) and, crucially, it has NO dispersion
    at all — one number per cell, no way to tell a real effect from seed noise. On this experiment
    the two point estimates agree to <= 0.07 percentage points, so switching does not move the
    conclusions; what changes is that a spread now comes with them, and it is large: up to 6.98 pp
    (Design C, tau=0.1, proposed), which is comparable to the proposed-vs-B2 gap being plotted.
    Returns (mean_pct, std_pct); a zero baseline for any seed gives NaN (retention undefined).
    """
    v = np.asarray(values, float); b = np.asarray(base, float)
    if v.shape != b.shape:
        raise ValueError(f"paired_retention needs one value per seed: {v.shape} vs {b.shape}")
    if v.size == 0:
        raise ValueError("paired_retention needs at least one seed")
    if np.any(b == 0):
        return float("nan"), float("nan")
    r = v / b * 100.0
    return float(r.mean()), sample_std(r)


@torch.no_grad()
def eval_both(m_prop, m_b2, Xstd, yte, groups):
    # .eval() here rather than relying on the trainers. TODAY THIS CHANGES NOTHING and no number in
    # this file is affected -- verified twice over: phase2's train_mlp/finetune_proposed both end
    # with model.eval() so the models arrive in eval mode, and GroupedCrossBandAttention's six
    # nn.Dropout layers are constructed at p=0.0, so even train mode would be a no-op. But both of
    # those are somebody else's invariant, either can change without phase 4 being touched, and the
    # failure mode is silent rather than loud: active dropout returns a different mIoU on every call
    # of the same input, which is indistinguishable from seed noise in a table of means and stds.
    m_prop.eval(); m_b2.eval()
    dev = next(m_prop.parameters()).device
    Xt = torch.from_numpy(Xstd).to(dev)
    pm = group_present_mask(Xstd.shape[0], groups, [])   # all bands present (perturbation, not loss)
    pp = m_prop(Xt, torch.from_numpy(pm).to(dev)).argmax(1).cpu().numpy()
    pb = m_b2(Xt).argmax(1).cpu().numpy()
    return miou(yte, pp, NUM_CLASSES), miou(yte, pb, NUM_CLASSES)


def run_seed(seed, cube, gt, n_groups, epochs, block=10):
    tr_mask, te_mask = disjoint_block_split(gt, block=block, guard=1, offset=seed)
    # per-pixel MEAN brightness normalization (train+test) -> scale-robust pipeline for C/D
    Xtr_n = mean_brightness_norm(cube[tr_mask]); ytr = gt[tr_mask].astype(int) - 1
    yte = gt[te_mask].astype(int) - 1
    mu = Xtr_n.mean(0); sd = Xtr_n.std(0) + 1e-8
    Xtr = ((Xtr_n - mu) / sd).astype(np.float32)
    groups = contiguous_groups(cube.shape[-1], n_groups)
    cwl = group_center_wavelengths(AVIRIS_WL_NM, groups)

    m_b2 = train_mlp(Xtr, ytr, groups, seed, group_dropout=True, epochs=epochs)
    m_prop = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES)
    pretrain_sgmae(m_prop, Xtr, groups, seed, epochs=max(1, epochs // 2))
    finetune_proposed(m_prop, Xtr, ytr, groups, seed, epochs=epochs)

    def std_test(cube_corrupt):
        Xn = mean_brightness_norm(cube_corrupt[te_mask])
        return ((Xn - mu) / sd).astype(np.float32)

    out = {"C": {}, "D1a": {}, "D1b": {}, "D_dead": {}, "D2_severity": {},
           "D1a_input_delta": {}}
    # Design C — thin cirrus tau sweep (applied on the full 2-D cube)
    for tau in TAUS:
        cube_c = apply_cirrus(cube, AVIRIS_WL_NM, tau)          # (H,W,B)
        out["C"][tau] = eval_both(m_prop, m_b2, std_test(cube_c), yte, groups)

    # Design D — per-band SNR noise ALWAYS on; the SAME noise realization (seed+900) is reused in
    # both sweeps so each isolates ONE striping effect against an identical noise baseline.
    snr = hyperion_like_snr(AVIRIS_WL_NM)
    # The noise REALIZATION, extracted once from the clean cube so it can be added AFTER the gain
    # without the gain feeding back into its magnitude. add_band_noise sizes sigma from the per-band
    # mean of whatever it is handed, so gaining first and then calling it would give a high-gain
    # column proportionally more noise -- which is not the detector model: read noise is a property
    # of the electronics, not of the column's responsivity.
    noised = add_band_noise(cube, snr, np.random.default_rng(seed + 900))        # x + n
    noise_only = noised - cube                                                   # n, independent of g

    # D1a -- gain applied to the SIGNAL, noise added after:  y = g_c * x + n
    # This is the model bandsim.noise documents, and it is the one that MEASURES something: the
    # per-column scalar no longer factors out of the per-spectrum mean, because the additive noise
    # does not carry it. Measured on a synthetic cube at eps=0.2:
    #   g*(x+n) -> max|meannorm(corrupt) - meannorm(noise-only)| = 6.7e-16   (cancels)
    #   g*x + n ->                                                 1.1e-02   (does not)
    # At eps=0 the gain is the identity, so this reduces EXACTLY to the noise-only baseline
    # (verified: difference 0.0), which is what makes the retention denominator correct.
    x0a = std_test(noised)                       # the eps=0 condition, by definition
    for eps in STRIPE_EPS:
        if eps > 0:
            gained = add_striping(cube, np.random.default_rng(seed + 901),
                                  stripe_eps=eps, dead_col_frac=0.0, col_axis=1)
            Xa = std_test(gained + noise_only)
        else:
            Xa = x0a
        out["D1a"][eps] = eval_both(m_prop, m_b2, Xa, yte, groups)
        # What reaches the MODEL, recorded apart from what the model DOES with it. A flat D1a mIoU
        # is not evidence that the gain cancelled -- it is equally consistent with a model that is
        # insensitive to it -- and only the input separates the two. Measured against x0a rather
        # than against whatever the loop happened to see first, so it does not depend on the order
        # of STRIPE_EPS.
        out["D1a_input_delta"][eps] = float(np.abs(Xa - x0a).max())

    # D1b -- the ORIGINAL order, kept as a CONTROL rather than deleted:  y = g_c * (x + n)
    # Here the gain multiplies signal and noise alike, so it is exactly cancelled by the per-spectrum
    # mean normalisation and the curve is pinned at 100% by algebra. It is retained because the
    # contrast with D1a is the actual finding: the SAME gain magnitude is informative or invisible
    # depending only on where the noise enters. It is flagged degenerate and never plotted.
    for eps in STRIPE_EPS:
        cube_b = noised
        if eps > 0:
            cube_b = add_striping(noised, np.random.default_rng(seed + 901),
                                  stripe_eps=eps, dead_col_frac=0.0, col_axis=1)
        out["D1b"][eps] = eval_both(m_prop, m_b2, std_test(cube_b), yte, groups)
    # D2 — dead detector columns (fraction zeroed), NO stripe gain (stripe_eps=0)
    #
    # dead_col_mode="exact" because `dead` is the SWEEP AXIS here, not a nuisance parameter: it must
    # be realised as a definite number of columns rather than as a Bernoulli expectation. Under the
    # default Bernoulli draw the realised severity varied so much across seeds that the axis did not
    # even ORDER the conditions. Measured on this cube over the 5 default seeds, as a fraction of
    # TEST pixels destroyed:
    #     nominal 1%  ->  0.00% - 2.38%      (seed 0 realised ZERO dead columns: an uncorrupted
    #                                         point plotted at x=0.01, i.e. retention 100% by
    #                                         construction rather than by robustness)
    #     nominal 3%  ->  2.55% - 6.71%
    #     nominal 5%  ->  3.32% - 8.34%      (overlaps 3% -- a nominally worse point was strictly
    #                                         less corrupted on 2 of 5 seeds)
    # and seed 0's 3% and 5% points destroyed the IDENTICAL 6.71% of test pixels, which is why they
    # returned the same mIoU. In exact mode the count is floor(f*145+0.5) = 1/4/7 columns, identical
    # across seeds, and the sweep is nested so a level only ADDS dead columns.
    #
    # Reuses `noised` instead of recomputing add_band_noise(seed+900): it is the same call with the
    # same seed, so this is bit-identical, and sharing the object makes "D1 and D2 have the same
    # zero point" structural rather than a coincidence of two seed literals staying in sync.
    for dead in DEAD_FRACS:
        cube_n, info = noised, None
        if dead > 0:
            cube_n, info = add_striping(noised, np.random.default_rng(seed + 902),
                                        stripe_eps=0.0, dead_col_frac=dead, col_axis=1,
                                        return_info=True, dead_col_mode="exact")   # W columns
        out["D_dead"][dead] = eval_both(m_prop, m_b2, std_test(cube_n), yte, groups)
        # Realised severity, measured on the OUTCOME rather than on the request: a test spectrum
        # that is identically zero carries no information, and mean_brightness_norm maps every one
        # of them to the same constant vector (see its docstring). It is the share of the
        # evaluation on which NEITHER model can beat its own constant prediction, so a drop of
        # that order is UNAVOIDABLE and the absolute retention level measures destruction rather
        # than robustness. The paired margin is not affected the same way -- both models face the
        # identical destroyed pixels -- so proposed-minus-B2 stays interpretable.
        out["D2_severity"][dead] = (0 if info is None else info["dead_col_count"],
                                    float((np.abs(cube_n[te_mask]).max(axis=-1) == 0).mean() * 100))

    # All three D axes' zero points are meant to be the SAME condition (SNR noise on, striping off),
    # which is what makes their retention denominators comparable and what the printed table claims.
    # Check it rather than asserting it in a comment: a stray edit to one of the seed literals would
    # otherwise give D1 and D2 different baselines while both were still labelled "SNR-only".
    z = [out["D1a"][0.0], out["D1b"][0.0], out["D_dead"][0.0]]
    if not all(np.allclose(z[0], zi, rtol=0, atol=1e-9) for zi in z[1:]):
        raise RuntimeError(f"D1a/D1b/D2 zero points must be the identical SNR-only condition, but "
                           f"D1a={z[0]}, D1b={z[1]}, D2={z[2]} -- a corruption seed or the order of "
                           f"operations changed, so the retention denominators are no longer the "
                           f"same experiment.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent seed workers (default: adaptive; also BANDSIM_WORKERS)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                    help="device for the workers (default: auto; also BANDSIM_DEVICE)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="disable deterministic algorithms for a small speedup")
    ap.add_argument("--selfcheck", action="store_true",
                    help="prove the Design D1 stripe-gain degeneracy numerically and exit (no training)")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck_d1_identity(); return
    # --smoke shares the deliverable paths with a full run unless it is tagged (see the same
    # footgun in phase2_degradation): a 1-seed sanity check would otherwise overwrite the 5-seed
    # results_phase4_ablation.csv and fig_ablation_cd.pdf with numbers nothing marks as smoke.
    tag = ""
    if args.smoke:
        args.seeds = [0]; args.epochs = 12
        tag = "_smoke"
        print("SMOKE RUN: 1 seed / 12 epochs — writing to *_smoke.* so the real deliverables are "
              "NOT overwritten. These numbers are a sanity check, not results.")
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device); print('HW:', hw.info())

    cube, gt = load_data()
    accC = {t: {"proposed": [], "b2": []} for t in TAUS}
    accD1a = {e: {"proposed": [], "b2": []} for e in STRIPE_EPS}
    accD1b = {e: {"proposed": [], "b2": []} for e in STRIPE_EPS}
    accDd = {d: {"proposed": [], "b2": []} for d in DEAD_FRACS}
    # Realised D2 severity per level: (dead columns, % of TEST pixels zeroed) per seed. Reported
    # rather than assumed -- the nominal fraction is now exact in columns, but WHICH columns are hit
    # still decides how many test pixels die, and that is the quantity the retention drop is against.
    sevDd = {d: {"cols": [], "px": []} for d in DEAD_FRACS}
    import time
    t0 = time.time()
    results = parallel.run_jobs(
        run_seed, args.seeds,
        shared=dict(cube=cube, gt=gt, n_groups=args.groups, epochs=args.epochs),
        prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
        label="phase4/seed")
    for sd, out in zip(args.seeds, results):
        for t in TAUS:
            accC[t]["proposed"].append(out["C"][t][0]); accC[t]["b2"].append(out["C"][t][1])
        for e in STRIPE_EPS:
            accD1a[e]["proposed"].append(out["D1a"][e][0]); accD1a[e]["b2"].append(out["D1a"][e][1])
            accD1b[e]["proposed"].append(out["D1b"][e][0]); accD1b[e]["b2"].append(out["D1b"][e][1])
        for d in DEAD_FRACS:
            accDd[d]["proposed"].append(out["D_dead"][d][0]); accDd[d]["b2"].append(out["D_dead"][d][1])
            sevDd[d]["cols"].append(out["D2_severity"][d][0])
            sevDd[d]["px"].append(out["D2_severity"][d][1])
        print(f"seed {sd}: cirrus tau1.0 prop={out['C'][1.0][0]:.1f} "
              f"| D1a eps0.2 prop={out['D1a'][0.2][0]:.1f} | D1b eps0.2 prop={out['D1b'][0.2][0]:.1f}"
              f" | dead0.05 prop={out['D_dead'][0.05][0]:.1f}")
    print(f"(all {len(args.seeds)} seeds in {time.time()-t0:.1f}s)")

    def ret(d, key0, keys, method):
        # PAIRED retention relative to THIS design's OWN corruption-off baseline d[key0] (C:
        # clean/no-noise; D1/D2: SNR-only) -- the three baselines differ, so each curve is
        # self-relative. Paired per seed (see paired_retention) and returned WITH its spread, so the
        # figure can carry error bars instead of implying a precision the 5 seeds do not support.
        return zip(*[paired_retention(d[k][method], d[key0][method]) for k in keys])

    def paired_margin(d, level, alpha=0.05):
        """proposed - b2 per seed at one level: mean, sample std, wins, and a paired-t interval.

        Paired because both methods see the identical split and corruption realisation for a given
        seed, so the per-seed difference removes the (large) between-seed variance.

        The INTERVAL is what makes the verdict honest. Counting a level as a LOSS whenever the mean
        margin is negative treats -0.22 mIoU and -6.72 mIoU as the same finding, on 5 seeds whose
        per-seed spread reaches ~3.8 -- which is exactly the error this file already refuses to make
        in the other direction when it declines to call +0.01 a win. A level is now named only when
        the interval excludes zero; otherwise it is INDISTINGUISHABLE, which is a result and not a
        missing one. Per-level and uncorrected for multiplicity: with 5 seeds the interval is wide
        enough that this is the generous direction, and the count of levels tested is printed so a
        reader can adjust -- and does not have to, because main() also reports the
        Bonferroni-corrected tally beside the uncorrected one. On every measurement so far the
        cirrus levels have survived that correction; the intervals are deliberately not restated
        here for the same reason the magnitudes at the top of this file are not.

        The nominal 95% assumes the per-seed differences are roughly normal, which 5 seeds cannot
        verify. It is used regardless because the distribution-free alternative has no power at
        this size: a two-sided sign test on n=5 has a MINIMUM attainable p of 2*(1/2)^5 = 0.0625,
        so it could never reject at 0.05 however consistent the sign. Reporting the interval and
        naming the assumption is the honest version of that trade; more seeds is the real fix.
        """
        m = np.asarray(d[level]["proposed"], float) - np.asarray(d[level]["b2"], float)
        mean, sd, n = float(m.mean()), sample_std(m), m.size
        wins = int((m > 0).sum())
        if not np.isfinite(sd):
            return mean, sd, wins, float("nan"), float("nan"), "n<2"
        half = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1)) * sd / np.sqrt(n)
        lo, hi = mean - half, mean + half
        return mean, sd, wins, lo, hi, ("loses" if hi < 0 else
                                        "wins" if lo > 0 else "indistinguishable")

    acc_by_key = {"C": accC, "D1a": accD1a, "D1b": accD1b, "D_dead": accDd}
    # The family for the multiplicity correction: the nonzero levels of the three MEASURING axes.
    # D1b is excluded because its margin is the eps=0 margin repeated by algebra -- including its
    # four levels would both inflate the correction and enter one piece of evidence four times.
    MEASURING = (("C", accC, TAUS), ("D1a", accD1a, STRIPE_EPS), ("D2", accDd, DEAD_FRACS))
    n_lvl = sum(1 for _, _, g in MEASURING for lv in g if lv != 0.0)
    alpha_bonf = 0.05 / n_lvl

    # Design D1 is cancelled analytically by mean_brightness_norm (see the module docstring); detect
    # it in the ACTUAL results too, so a future change that accidentally revives the degeneracy is
    # caught here rather than published as a flat robustness curve.
    # Two runtime checks, and the CONTRAST between them is the finding. D1b must be flat (a gain
    # that multiplies signal and noise alike cancels under per-spectrum mean normalisation); D1a
    # must NOT be (the same gain applied before the noise does not factor out). Either one failing
    # means the corruption chain no longer implements the model the axis is named for.
    def _flat(acc):
        return all(np.allclose(acc[e]["proposed"], acc[0.0]["proposed"]) and
                   np.allclose(acc[e]["b2"], acc[0.0]["b2"]) for e in STRIPE_EPS)
    d1b_flat, d1a_flat = _flat(accD1b), _flat(accD1a)
    if d1b_flat:
        print("\n*** D1b DEGENERATE (as designed): y = g_c*(x+n) is cancelled exactly by "
              "mean_brightness_norm, so mIoU is identical at every stripe_eps. Measured, written to "
              "the CSV with a `degenerate` flag, and NOT plotted -- a line pinned at 100% by algebra "
              "reads as perfect robustness in a retention figure.")
    else:
        print("\n*** D1b IS NO LONGER FLAT. Either the corruption order changed, the gain became "
              "spectrally varying, an offset was introduced, or the normalisation stopped being "
              "scale-invariant. Inspect the transformation, not just this metric: identical "
              "predictions can hide a changed input, and a changed metric can come from model "
              "nondeterminism rather than from the transform.")
    # The D1a check is on the INPUT, not on the metric. An earlier version tested mIoU and called
    # a flat curve a corruption-chain defect -- but a model that is simply insensitive to the gain
    # produces exactly the same flat curve -- and on this experiment the metric barely moves, which
    # the paired verdicts below duly report as indistinguishable. That is squarely the range where
    # the output cannot tell "the corruption cancelled" from "the model did not care".
    d1a_delta = min(min(o["D1a_input_delta"][e] for e in STRIPE_EPS if e > 0) for o in results)
    if not (d1a_delta > 0.0):
        raise RuntimeError(
            f"D1a must change what the model is FED: y = g_c*x + n does not factor out of the "
            f"per-spectrum mean. The smallest change over seeds and eps is {d1a_delta}, so either "
            f"the noise is no longer added after the gain or eps is not reaching add_striping. "
            f"Nothing on that axis can be read as robustness.")
    print(f"*** D1a REACHES THE MODEL: the standardised test matrix moves by at least "
          f"{d1a_delta:.3g} at every nonzero eps, so the column gain survives the normalisation. "
          f"This is the axis the earlier version of this file discarded on the mistaken grounds "
          f"that only a per-band gain or a non-scale-invariant normalisation could make D1 "
          f"informative -- moving the noise after the gain does it.")
    print(f"    Whether either MODEL responds is a SEPARATE question: mIoU is "
          f"{'flat to 1e-5' if d1a_flat else 'not flat'} across eps, and the paired verdicts below "
          f"say how much of that is signal. A near-null response is a weak robustness observation "
          f"about the models -- it is not what makes the axis valid.")

    # ---- raw per-seed rows ----
    # Every aggregate below is recomputable from this file, which is why it exists: a mutation
    # campaign on the aggregate CSV alone showed that ddof could be switched back to the population
    # formula at any of three call sites without a single test noticing. A dispersion is invisible
    # in the output -- 1.74 looks exactly as plausible as 1.95 -- so the only way to guard it is to
    # publish the numbers it was computed from.
    with open(P(f"results_phase4_raw{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "param", "seed", "proposed_miou", "b2_miou",
                    "realised_dead_cols", "dead_test_px_pct"])
        for spec in CSV_DESIGNS:
            acc = acc_by_key[spec["key"]]
            for lvl in spec["grid"]:
                for i, sd_ in enumerate(args.seeds):
                    sev = ((sevDd[lvl]["cols"][i], f"{sevDd[lvl]['px'][i]:.4f}")
                           if spec["design"] == "D2_dead_cols" else ("", ""))
                    w.writerow([spec["design"], lvl, sd_, f"{acc[lvl]['proposed'][i]:.6f}",
                                f"{acc[lvl]['b2'][i]:.6f}", *sev])

    # ---- csv ----
    with open(P(f"results_phase4_ablation{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # Every reported quantity now carries its across-seed spread, and the head-to-head is the
        # PAIRED per-seed difference (+ how many seeds it actually won) rather than a gap between
        # two independent means -- the old header had no dispersion column at all, so a 0.01-mIoU
        # "win" and a 6-mIoU loss were presented with identical apparent authority.
        w.writerow(["design", "param", "n_seeds",
                    "proposed_miou_mean", "proposed_miou_std",
                    "proposed_retention_mean", "proposed_retention_std",
                    "b2_miou_mean", "b2_miou_std", "b2_retention_mean", "b2_retention_std",
                    # ..._std columns are SAMPLE std (ddof=1). The interval is a 95% paired-t on
                    # the per-seed margin: `paired_verdict` names a level only when it excludes 0,
                    # so a -0.22 mIoU mean margin is no longer filed as a loss beside a -6.72 one.
                    "paired_margin_mean", "paired_margin_std", "paired_wins",
                    "paired_ci_lo", "paired_ci_hi", "paired_verdict",
                    # Same interval at alpha/n_levels over the family named by MEASURING, with
                    # its BOUNDS and not only its label: a label alone cannot be checked against
                    # the alpha it claims to use. Blank on D1b, which is not in that family.
                    "paired_ci_lo_bonf", "paired_ci_hi_bonf", "paired_verdict_bonferroni",
                    # REALISED severity, D2 only: the nominal fraction is a request, and until this
                    # commit it was a Bernoulli expectation whose realisation did not even order the
                    # sweep. `realised_dead_cols` is now deterministic in (frac, n_cols) and prints
                    # as a RANGE if it ever stops being, which is the signal that exactness broke.
                    "realised_dead_cols", "dead_test_px_pct_mean", "dead_test_px_pct_std",
                    "degenerate"])

        def severity_cells(design, lvl):
            """The three severity columns, filled ONLY for D2 -- the one axis whose realised
            severity is distinct from its nominal parameter. Blank elsewhere rather than 0, because
            a number here would claim a measurement that was never taken on those axes."""
            if design != "D2_dead_cols":
                return ["", "", ""]
            c, px = sevDd[lvl]["cols"], sevDd[lvl]["px"]
            cols = str(c[0]) if len(set(c)) == 1 else f"{min(c)}-{max(c)}"
            return [cols, f"{np.mean(px):.2f}", f"{np.std(px):.2f}"]
        # Each design's retention is vs its OWN corruption-off baseline (C: tau=0 clean/no-noise;
        # D1: eps=0 SNR-only; D2: f=0 SNR-only) -- these are DIFFERENT baselines, not one shared one.
        # EVERY design is written here, including the D1 rows that the figure omits: dropping the
        # panel must not drop the result. The `degenerate` flag comes from CSV_DESIGNS but is
        # cleared if the runtime check above found D1 moving after all, so the column always
        # describes THIS run rather than a standing expectation.
        ns = len(args.seeds)
        for spec in CSV_DESIGNS:
            design, acc, grid = spec["design"], acc_by_key[spec["key"]], spec["grid"]
            # The only axis whose expected degeneracy is checked at runtime is D1; if this run found
            # it moving, the flag is dropped so the column never contradicts the numbers beside it.
            degen = "" if (design == "D1b_noise_then_gain" and not d1b_flat) else spec["degenerate"]
            for lvl in grid:
                pr_m, pr_s = paired_retention(acc[lvl]["proposed"], acc[0.0]["proposed"])
                br_m, br_s = paired_retention(acc[lvl]["b2"], acc[0.0]["b2"])
                mm, ms, wins, lo, hi, verd = paired_margin(acc, lvl)
                w.writerow([design, lvl, ns,
                            f"{np.mean(acc[lvl]['proposed']):.2f}", f"{sample_std(acc[lvl]['proposed']):.2f}",
                            f"{pr_m:.1f}", f"{pr_s:.1f}",
                            f"{np.mean(acc[lvl]['b2']):.2f}", f"{sample_std(acc[lvl]['b2']):.2f}",
                            f"{br_m:.1f}", f"{br_s:.1f}",
                            f"{mm:+.2f}", f"{ms:.2f}", f"{wins}/{ns}",
                            f"{lo:+.2f}", f"{hi:+.2f}", verd,
                            *(("", "", "") if design == "D1b_noise_then_gain"
                              else tuple(f"{v:+.2f}" if isinstance(v, float) else v
                                         for v in paired_margin(acc, lvl, alpha_bonf)[3:6])),
                            *severity_cells(design, lvl), degen])

    # ---- figure: the MEASURING axes only (C, D2). D1 is recorded in the CSV but not drawn ----
    # See FIGURE_PANELS for why: its curve is flat by algebra, and a flat line in a retention figure
    # is read as robustness no matter what the caption says.
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    spec_by_design = {d["design"]: d for d in CSV_DESIGNS}
    fig, axes = plt.subplots(1, len(FIGURE_PANELS), figsize=(3.1 * len(FIGURE_PANELS), 2.6),
                             squeeze=False)
    # errorbar, not plot: the across-seed spread reaches ~7 percentage points here, which is the
    # same order as the proposed-vs-B2 gap. Drawing bare lines implied a precision 5 seeds cannot
    # support and let a within-noise difference read as a separation.
    # These bars are the UNPAIRED across-seed sample spread of each method. The head-to-head is
    # PAIRED, and the two answer different questions: overlapping bars do NOT mean the difference is
    # null, because the between-seed variance they show is exactly what the pairing removes. The
    # paired interval and its verdict are per level in the CSV, and the caption says so.
    for ax, pan in zip(axes[0], FIGURE_PANELS):
        grid = spec_by_design[pan["design"]]["grid"]
        acc = acc_by_key[spec_by_design[pan["design"]]["key"]]
        for method, style, color, label in (("proposed", "-^", "#1f6f3a", "Proposed"),
                                            ("b2", "-s", "#e67e22", "B2 dropout")):
            m, s = ret(acc, 0.0, grid, method)
            ax.errorbar(grid, m, yerr=s, fmt=style, color=color, lw=1.7, ms=4,
                        capsize=2.5, elinewidth=0.9, label=label)
        ax.set_xlabel(pan["xlabel"]); ax.set_ylabel("mIoU retention (%)")
        ax.set_title(pan["title"], fontsize=8.5)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
    # Say in the figure itself that an axis was measured and withheld, so the missing panel reads as
    # a decision with a reason rather than as an axis nobody ran.
    # Two things a reader cannot get from the axes themselves, so they are drawn on the figure:
    # what was measured and withheld, and what the error bars are NOT. Schematic-physics scoping
    # goes here too -- "Design C, thin cirrus" reads as physics unless the panel says otherwise.
    notes = ["Schematic corruption models (bandsim/cirrus.py, bandsim/noise.py): a robustness "
             "stress test, not calibrated radiometry.",
             f"Bars are the across-seed sample spread of each method, NOT the paired comparison — "
             f"overlapping bars do not imply a null difference. Paired 95% intervals are per level "
             f"in results_phase4_ablation{tag}.csv."]
    omitted = omitted_designs()
    if omitted:
        notes.append(f"{', '.join(omitted)} measured but not plotted: a per-column scalar gain "
                     f"cancels exactly under mean-brightness normalization, so its retention is "
                     f"100% by algebraic identity. Values in the same CSV.")
    # fig.text does NOT wrap, and a caption running off both edges is worse than no caption: the
    # first draft of the scoping note measured 1118 px in a 930 px figure. wrap=True makes
    # matplotlib do it against the real figure width. The fix before this one estimated
    # characters-per-inch instead, and that constant was 20% off between two of these very notes
    # purely from character mix -- it fitted the 3-panel figure and overflowed a 1-panel one, which
    # is exactly how a magic constant fails: silently, and only in the case nobody rendered.
    # Each note's height is then MEASURED to place the next and to size the margin tight_layout
    # leaves, so adding a note cannot quietly overwrite the panels.
    y = 0.006
    for note in reversed(notes):
        t = fig.text(0.5, y, note, ha="center", va="bottom", fontsize=5.4, color="#b03030",
                     wrap=True)
        fig.canvas.draw()                      # wrapping and the extent both need a renderer pass
        y += (t.get_window_extent(renderer=fig.canvas.get_renderer()).height
              / (fig.get_size_inches()[1] * fig.dpi)) + 0.008
    fig.tight_layout(rect=(0, y + 0.01, 1, 1))
    fig.savefig(P(f"figs/fig_ablation_cd{tag}.pdf")); plt.close(fig)

    print("\n===== Phase 4 C/D ablation (mean +/- std over {} seeds) =====".format(len(args.seeds)))
    print("(retention % is PAIRED per seed vs EACH design's own zero baseline: C tau=0 is "
          "clean/no-noise; D1/D2 eps=0/f=0 are SNR-only)")
    for design, acc, grid, name in (
            ("C", accC, TAUS, "Design C (cirrus)"),
            ("D1a", accD1a, STRIPE_EPS, "Design D1a (g*x + n -- gain then noise)"),
            ("D1b", accD1b, STRIPE_EPS, "Design D1b (g*(x+n) -- noise then gain, DEGENERATE)"),
            ("D2", accDd, DEAD_FRACS, "Design D2 (SNR + dead columns, no stripe gain)")):
        print(f"{name}:  level -> mIoU and PAIRED retention% (base = this design's own zero point)")
        for lvl in grid:
            pr_m, pr_s = paired_retention(acc[lvl]["proposed"], acc[0.0]["proposed"])
            mm, ms, wins, lo, hi, verd = paired_margin(acc, lvl)
            # D2 carries its REALISED severity, because its nominal x is a request: the retention
            # drop is against however many test spectra were actually zeroed, and a zeroed spectrum
            # carries no information for either model. Printing it next to the retention is what
            # keeps the absolute level from being read as a robustness score.
            sev = ""
            if design == "D2":
                sev = (f"   [{sevDd[lvl]['cols'][0]} dead cols, "
                       f"{np.mean(sevDd[lvl]['px']):.2f}+-{sample_std(sevDd[lvl]['px']):.2f}% of test px zeroed]")
            print(f"  {lvl:<5} prop={np.mean(acc[lvl]['proposed']):5.1f}+-{sample_std(acc[lvl]['proposed']):4.1f} "
                  f"({pr_m:5.1f}+-{pr_s:4.1f}%)   b2={np.mean(acc[lvl]['b2']):5.1f}+-{sample_std(acc[lvl]['b2']):4.1f}"
                  f"   paired(prop-b2)={mm:+5.2f} [{lo:+5.2f},{hi:+5.2f}] {verd:<17}"
                  f" wins {wins}/{len(args.seeds)}{sev}")
    # State the direction of the result explicitly. This panel has historically been described as
    # showing the proposed model is robust across corruption physics; on the numbers above it is
    # not, and an unlabelled table invites the reader to assume the intended direction.
    # Three ways, not two. The previous line counted a level as a LOSS whenever the mean paired
    # margin was negative, which on 5 seeds with a per-seed spread up to ~3.8 mIoU files noise as a
    # finding -- and did it while the same file was carefully declining to call +0.01 a win.
    # D1a is included now that it measures. D1b is NOT: its margin is the eps=0 margin repeated by
    # algebra, so counting its four levels would enter one piece of evidence four times.
    verdicts, verdicts_bonf = {}, {}
    for name, acc, grid in MEASURING:
        for lvl in grid:
            if lvl == 0.0:
                continue
            verdicts.setdefault(paired_margin(acc, lvl)[5], []).append((name, lvl))
            verdicts_bonf.setdefault(paired_margin(acc, lvl, alpha_bonf)[5], []).append((name, lvl))
    if sum(len(v) for v in verdicts.values()) != n_lvl:      # raise: `python -O` drops assert
        raise RuntimeError(f"the verdict tally covers {sum(len(v) for v in verdicts.values())} "
                           f"levels but the Bonferroni family was sized at {n_lvl}")
    print(f"\nVERDICT over the {n_lvl} non-trivial levels of the three MEASURING axes (C, D1a, D2),"
          f"\nby 95% paired-t interval on proposed - B2, per level, uncorrected for multiplicity:")
    for v in ("loses", "wins", "indistinguishable", "n<2"):
        if verdicts.get(v):
            print(f"  {v:<18} {len(verdicts[v]):>2}: {verdicts[v]}")
    # Answer the multiplicity objection in the artefact instead of flagging it and moving on: at
    # 10 levels the family-wise error rate of an uncorrected sweep reaches ~40%, and a reviewer
    # will say so. Bonferroni is conservative, which is the right direction for a claim we want
    # to survive rather than one we want to make.
    # Not every "loses" means the same thing. Where BOTH methods have collapsed, the margin
    # compares two near-zero scores: a real ratio, but not an operational difference, and the table
    # should not read as though tau=1.0 and tau=0.1 were the same kind of evidence. The levels stay
    # in the tally -- they are real measurements -- and are flagged instead of dropped.
    floored = collapsed_levels(MEASURING, FLOOR_RETENTION_PCT)
    if floored:
        print(f"  NOTE: at {len(floored)} of the {n_lvl} levels BOTH methods retain "
              f"<{FLOOR_RETENTION_PCT:.0f}%, so the margin there compares two collapsed models "
              f"rather than two working ones: {floored}")
    print(f"same, Bonferroni-corrected (alpha={alpha_bonf:.4f} = 0.05/{n_lvl}):")
    for v in ("loses", "wins", "indistinguishable", "n<2"):
        if verdicts_bonf.get(v):
            print(f"  {v:<18} {len(verdicts_bonf[v]):>2}: {verdicts_bonf[v]}")
    # Derived from THIS run, not written in advance: the previous line asserted "D1 measures
    # nothing (degenerate)" unconditionally, which stopped being true the moment D1a was added.
    n_lose, n_win = len(verdicts.get("loses", [])), len(verdicts.get("wins", []))
    n_null, n_untested = len(verdicts.get("indistinguishable", [])), len(verdicts.get("n<2", []))
    # UNTESTED is not INDISTINGUISHABLE, and the first draft of this block conflated them: with one
    # seed no interval exists at all, and it still printed "no level separates the two methods",
    # which is a claim about a comparison that never happened.
    if n_untested:
        print(f"  ({n_untested} of {n_lvl} levels have fewer than 2 seeds, so no interval exists "
              f"for them. That is UNTESTED, not indistinguishable.)")
    if n_untested == n_lvl:
        print("NOTHING WAS COMPARED at 95%: every level is below 2 seeds. Do not read a direction "
              "into the margins above -- run more seeds.")
    elif n_lose and not n_win:
        print("This is a NEGATIVE result for cross-physics robustness and must be reported as one.")
    elif n_lose and n_win:
        print(f"MIXED: proposed separates from B2 downward at {n_lose} level(s) and upward at "
              f"{n_win}. Report both; neither direction alone describes this panel.")
    elif n_win:
        print(f"Proposed separates upward at {n_win} level(s) and never downward -- but check the "
              f"level count above before reading that as cross-physics robustness.")
    else:
        print(f"No level separates the two methods at 95% ({n_null} tested, none separating). "
              f"Report as INDISTINGUISHABLE, which is a result; it is not evidence of robustness, "
              f"and at these seed counts it is not evidence of equivalence either.")
    # Two clauses, deliberately not one: the axis being valid is a fact about the INPUT and the
    # models' response is a fact about the MODELS. This line used to say "D1a measures (mIoU moves
    # with eps)", which asserted the first on the evidence of the second.
    print(f"D1a's corruption REACHES the model -- the standardised test matrix moves by at least "
          f"{d1a_delta:.3g} at every nonzero eps, so the axis is valid. Whether the models respond "
          f"is separate: mIoU {'moves' if not d1a_flat else 'does NOT move'} with eps. "
          f"D1b {'is degenerate as designed' if d1b_flat else 'is NOT flat, which it should be'} "
          f"-- see the module docstring for why the pair is the finding.")
    # Same `tag` as the write, so a smoke sidecar cannot describe the 5-seed deliverable. The three
    # corruption grids are module constants, not arguments: a later edit to TAUS/STRIPE_EPS/DEAD_FRACS
    # would otherwise leave an old CSV's `param` column unattributable to the sweep that produced it.
    # d1_degenerate_this_run records the RUNTIME finding, which is what the `degenerate` column was
    # set from -- a standing expectation and a measured one are not the same claim.
    stamp(P(f"results_phase4_ablation{tag}.csv"), args,
          extra={"designs": [s["design"] for s in CSV_DESIGNS],
                 "cirrus_taus": TAUS, "stripe_eps": STRIPE_EPS, "dead_fracs": DEAD_FRACS,
                 "plotted_designs": [p["design"] for p in FIGURE_PANELS],
                 "d1b_degenerate_this_run": bool(d1b_flat), "d1a_flat_this_run": bool(d1a_flat),
                 # How much the standardised TEST MATRIX moves at the smallest nonzero eps, over
                 # all seeds. This -- not the mIoU -- is what says the D1a corruption survives the
                 # normalisation; d1a_flat_this_run beside it is a fact about the MODELS.
                 "d1a_min_input_delta": d1a_delta,
                 # The D2 axis is only interpretable together with how its fraction was realised:
                 # the same DEAD_FRACS under "bernoulli" is a different experiment (see run_seed).
                 "dead_col_mode": "exact",
                 # So a downstream table cannot quote these rows as physics: both corruption
                 # models are schematic and their own modules exclude them from physics claims.
                 "physics_scope": "schematic_stress_test_not_calibrated_radiometry",
                 "dispersion": "sample_std_ddof1", "interval": "paired_t_95pct_per_level",
                 # The closing summary is a print; this is the machine-readable version of it, so a
                 # reader (and a test) gets the tally without parsing stdout. "n<2" is a separate
                 # bucket ON PURPOSE: a level with no interval is untested, not indistinguishable.
                 "verdict_counts": {k: len(v) for k, v in sorted(verdicts.items())},
                 "verdict_counts_bonferroni": {k: len(v) for k, v in
                                              sorted(verdicts_bonf.items())},
                 "bonferroni_family_size": n_lvl,
                 "floor_retention_pct": FLOOR_RETENTION_PCT,
                 "levels_where_both_methods_collapsed": floored,
                 "raw_rows": f"results_phase4_raw{tag}.csv",
                 "d2_realised_dead_cols": {str(d): sevDd[d]["cols"][0] for d in DEAD_FRACS},
                 "d2_dead_test_px_pct": {str(d): round(float(np.mean(sevDd[d]["px"])), 3)
                                         for d in DEAD_FRACS}})
    stamp(P(f"results_phase4_raw{tag}.csv"), args,
          extra={"aggregates": f"results_phase4_ablation{tag}.csv", "seeds": list(args.seeds)})
    print(f"\nwrote: {P(f'figs/fig_ablation_cd{tag}.pdf')}  {P(f'results_phase4_ablation{tag}.csv')}"
          f"  {P(f'results_phase4_raw{tag}.csv')}")


if __name__ == "__main__":
    main()
