#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S5 (A+B) — a CLEAN-trained classifier evaluated under a 6S-derived gaseous-transmittance
(column-water-vapour, CWV) shift, with NO re-training and NO re-fitting (source->shift transfer).

WHAT THIS MEASURES, AND WHAT IT DOES NOT  (read before quoting any number)
---------------------------------------------------------------------------
SPECTRAL-ONLY, COMMON-GRID. Swapping the target sensor here swaps ONE thing: the measured bandpass
set (Design A, pyspectral RSR). Every band is read off the SAME Indian Pines pixel grid, so spatial
resolution (S2 is 10/20/60 m, OLI 30 m), PSF/MTF, resampling, co-registration, SNR, quantization and
the two product chains are NOT modelled. This is a bandpass-set comparison on a common spatial grid,
NOT a Sentinel-2-versus-Landsat-OLI sensor-system comparison; do not describe it as one.

GASEOUS-ONLY. Design B applies 6S `transmittance_global_gas` (water vapour + O2/O3/CO2). Aerosol
scattering, path radiance and adjacency are NOT modelled, so this is a CWV stress-test, not a full
radiative-transfer shift. The strong absorption cores are hard-masked in EVERY condition.

TWO ARMS, AND THE SECOND ONE IS WHY THIS FILE EXISTS
-----------------------------------------------------
Phase 3 already measured what happens when rho*T is fed to a clean-normalised model on the HSI
axis: mIoU 54.4 -> 3.2, and it collapses even with NO bands dropped, because a band with a small
sd leaves the training distribution under even a modest multiplicative factor (median shift
-0.25 sigma, WORST BAND -27 sigma). Phase 3 was therefore forced to split its result in two and to
call the rho*T path "uncorrected gas-only radiometry proxy" -- never "the atmosphere result".

This script used to report ONLY that arm, on sensor bands, and call it the flagship. It now reports
both, so the reader can see which part of the degradation is recoverable radiometry and which part
is information that no correction can return:

  A1  UNCORRECTED (gas-only radiometry proxy).  rho*T then SRF, evaluated with the clean scaler.
      This is the historical curve. It is an UNCORRECTED-RADIOMETRY SENSITIVITY, and it mixes
      genuine band suppression with scaler out-of-distribution.

  A2  ORACLE PER-BAND GAIN CORRECTION (the control).  The same cube divided by the per-band gain
      g_b (see oracle_gain), i.e. what a PERFECT radiometric correction of the smooth multiplicative
      factor would recover.

      A2 ~ clean            => the A1 drop is recoverable gain/normalisation, NOT information loss.
      A2 still well below   => that part is a CANDIDATE for genuine loss -- and only a candidate.
                               A2 removes the per-band gain, not every route to scaler OOD: a
                               low-variance band can still leave the training distribution on the
                               residual alone. Read A2's drop TOGETHER with its zshift_max column,
                               and call it information loss only when that residual shift is small.

  The band-loss arm (atmosphere decides WHICH bands are unusable, survivors keep clean values) is
  phase 3's PRIMARY and is not duplicated here; the per-band transmittances this script writes to
  the *_bands.csv let a reader apply any threshold themselves.

WHAT THE PRE-FIX RESULT ACTUALLY SAID (read before assuming the hypothesis holds). The committed
results_phase5_ab_flagship.csv (no provenance sidecar, so its seed count and config are unknowable)
reports S2 66.73 -> 63.26 / 63.73 / 63.90 and OLI 63.60 -> 61.38 / 61.41 / 61.40 across
CWV 0.5 / 2.0 / 4.0. Two things follow, and neither is what the hypothesis expects:
  * the CWV response is FLAT. An 8x increase in column water vapour moves OLI by 0.03 mIoU and
    moves S2 UP. Essentially all of the effect is a one-off clean -> any-atmosphere step, which is
    the signature of a fixed gain/normalisation offset rather than a water-vapour gradient.
  * OLI retains MORE than S2 (0.965 vs 0.958), i.e. the OPPOSITE ordering to the hypothesis.
Those numbers were produced by arm A1 alone, compared by eye between two unpaired error bars. That
is exactly why this file now runs A2 and reports a PAIRED difference-in-differences: the ordering
question has to be settled by the paired test, not by two curves that happen not to touch.

WHY OLI RETAINS MORE, MEASURED (no training needed -- pure numpy on the LUT, the measured RSR and
the seed-0 split; the run reprints all of it). Per-band mean standardized shift over the test
pixels, in units of the band's own clean-train sd:

Both shift columns are measured FROM CLEAN and carry their SIGN, matching the emitted *_bands.csv
(shift_sigma_cwv*) so table and artefact cross-read directly -- every band attenuates, so every
shift is negative, and dropping the sign hid the one bit that says so. The "next largest" rows are
selected by |clean->4.0|, the column the table shows: an earlier draft rebased the COLUMNS onto
clean->X but left the row SELECTION on the 0.5->4.0 increment, which ranks OLI differently and
named the wrong band (B9/pan at -0.20 over B3 at -0.24).

                              gain@0.5  gain@4.0   clean->0.5  clean->4.0
    S2  B09   943 nm            0.6150    0.2177     -3.12 s     -6.17 s   <- the water-vapour band
    S2  B10  1374 nm (0.02%)    0.2967    0.0401    -10.36 s    -14.16 s   <- DEGENERATE, artefact
    S2  B12  2202 nm            0.9625    0.9229     -0.40 s     -0.84 s   <- next largest, S2
    OLI B8  2201 nm             0.9611    0.9223     -0.41 s     -0.84 s   <- OLI's LARGEST
    OLI B3   560 nm             0.9604    0.9570     -0.22 s     -0.24 s   <- next largest, OLI

Three things follow, and they reframe the whole experiment:

  1. THE CWV AXIS IS ACTIVE, BUT OVERWHELMINGLY CONCENTRATED IN ONE CHANNEL. On the HSI axis 93 of
     192 unmasked bands move more than 0.05 in transmittance between CWV 0.5 and 4.0 (max 0.42, at
     947 nm). After SRF integration two MEASURING S2 bands still clear 0.05 -- B09 at 0.397 and B08
     at 0.065, a factor of 6 apart, -6.17 s against -0.57 s in sd units from clean. The scope word is
     load-bearing: the DEGENERATE B10 also clears it, at 0.257, so counting nominal bands gives three
     and a top-two ratio of 1.55 rather than 6. B09 is the NARROW
     943 nm water-vapour band, the one channel designed to sense exactly this quantity; every other
     multispectral bandpass averages the feature away over its width.

  2. A MECHANISM THAT PREDICTS THE BACKWARDS ORDERING. S2's worst real band lands 6.17 sigma out of
     distribution; OLI's worst reaches 0.84 -- S2 is ~7x more perturbed, and the difference is one channel OLI
     does not have. If that channel matters, the band giving S2 its clean-condition advantage is
     precisely the band water vapour destroys: an asset in-distribution, a liability under the shift
     it was built to sense. That makes the pre-fix ordering (OLI 0.965 vs S2 0.958) the ordering this
     mechanism PREDICTS rather than an anomaly -- but predicted is not proven. What is measured here
     is feature-space displacement; whether it moves mIoU depends on how much weight the classifier
     puts on B09, which this probe cannot see.
     THE DECISIVE TEST is an S2-minus-B09 ablation: if S2's retention rises to OLI's when its
     water-vapour band is removed, the causal claim holds. Until that is run, state the mechanism and
     the prediction, not the conclusion.

  3. THE FLAT CURVE IS CONSISTENT WITH SATURATION -- and the axis is definitely NOT inert, which is
     the part that is measured. B09 is already 3.1 sigma out of distribution at CWV 0.5 and only
     reaches ~6 sigma by CWV 4.0; a feature that far out plausibly has the classifier's response
     already saturated, so extra water vapour buys little extra error. That is a hypothesis about the
     model, not a measurement of it. Arm A2 bears on it: if removing the per-band gain recovers most
     of the clean->CWV step, the step was gain rather than lost information.

CONSEQUENCE FOR THE HEADLINE NUMBER. S2 B10 retains 0.02% of its SRF mass and yet carries the
LARGEST excursion in the table, 14.16 sigma at CWV 4.0 -- 2.3x the real signal band -- because its
residual comes from wings sitting on the absorption-core edge. Under the default
`--degenerate-bands keep` that channel is fed to the model. Whether it actually moves mIoU cannot be settled by the shift alone (a
near-zero channel may carry a near-zero learned weight), so the `keep` and `drop` runs are BOTH
required and the paper number is the one they agree on.

THE BAND SET IS THE CANONICAL PRODUCT CONTRACT, UNCONDITIONALLY. pyspectral returns the RSR
STORE's band list, not the sensor's surface-reflectance product -- for OLI that is 9 bands whose
numbering is not USGS's (its B8 is SWIR-2 at 2201 nm, its B6 the 1375.7 nm cirrus band, its B9 the
590-ish nm PANCHROMATIC channel spanning B2/B3/B4 at an unmodelled 15 m), and for S2 it is 13
including B10 cirrus. Earlier this file compared DIFFERENT band sets per --srf source, then grew a
`--band-set reflective` opt-in that fixed it in place. Both are gone: every SRF now passes through
bandsim.srf.select_canonical_bandset, which matches by CENTRE WAVELENGTH (never by name -- the
store and USGS both spell an OLI band "B6" and mean 1373 vs 1609 nm) onto this repo's *_CENTERS_NM
product definitions (S2 13->12, OLI 9->7), records which source band each canonical band came from
and what was dropped, and is the same contract phase 2 cross-sensor pins. There is no flag to turn
it off, because its off state was a wrong experiment, not an option.

PROTOCOL (unchanged, and now verified at runtime)
--------------------------------------------------
For each sensor: (1) fit ONE scaler on the CLEAN source TRAIN pixels, (2) train ONE MLP on those
normalized clean pixels, (3) FREEZE model + scaler and evaluate on EVERY condition's held-out TEST
pixels with no re-training and no re-fitting. clean = in-distribution reference. A per-condition
re-trained ORACLE would measure per-condition separability, not transfer, and is deliberately not
what this reports.

WHAT THE ERROR BARS ARE (and are not)
--------------------------------------
The `--seeds` are checkerboard OFFSETS of ONE scene, and one seed drives the split, the shuffle and
the init together. The checkerboard period is `block` = 10 px, not 2*block (offsets 10 and 20
reproduce offset 0 exactly), so offsets 0..4 are five distinct but heavily overlapping splits; the run prints the measured pairwise test-mask Jaccard overlap.
Spread here is split+init variability on a single scene -- NOT scene-level or sensor-level
generalisation, and a CI computed from it must never be read as one. A second scene is
phase6_second_dataset.py's job.

DATA SCOPE. Indian_pines_corrected.mat: "corrected" is the BAND SUBSET, not atmospheric correction.
The 200-band axis is the nominal 220-band AVIRIS axis minus 20 water/noise bands (bandsim.io), so
(a) wavelengths are NOMINAL, not per-scene calibrated -- a few nm of error matters at a narrow
absorption edge -- and (b) the saturated 1400/1900 nm cores are already absent. The CWV-diagnostic
940/1130 nm water features DO survive and are what the ladder acts on.

Requires the 6S table (experiments/precompute_6s_table.py). Uses the S3-hardened split.

Outputs (../paper/):
  figs/fig_ab_flagship{tag}.pdf                      - uncorrected | gain-corrected, per sensor
  results_phase5_ab_flagship{tag}.csv                - aggregate per (sensor, arm, condition)
  results_phase5_ab_flagship_perseed{tag}.csv        - RAW per-seed rows (recompute any statistic)
  results_phase5_ab_flagship_bands{tag}.csv          - per-band SRF mass / transmittance / gain
  results_phase5_ab_flagship_paired{tag}.csv         - paired OLI-minus-S2 difference-in-differences

Usage:
  python experiments/phase5_ab_flagship.py --seeds 0 1 2 3 4
  python experiments/phase5_ab_flagship.py --smoke                    # 2 seeds, 12 epochs (_smoke)
  python experiments/phase5_ab_flagship.py --degenerate-bands drop --tag _nodead   # sensitivity
"""
import os, sys, csv, json, argparse, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
from bandsim import hw, parallel
from bandsim.io import load_mat_cube, disjoint_block_split, axis_sha256, AVIRIS_WL_NM
from bandsim.config_runner import allowed_claim_scopes, derived_validation_status
from bandsim.pipeline import simulate
from bandsim.atmosphere import hard_mask_absorption_cores, load_cached_transmittance
from bandsim.srf import (pyspectral_srf, gaussian_srf, build_resampling_matrix,
                         select_canonical_bandset,
                         SENTINEL2_MSI_CENTERS_NM, LANDSAT8_OLI_CENTERS_NM)
from bandsim.provenance import stamp, file_sha256
from phase2_cross_sensor import train_mlp, eval_mlp   # split train / eval -> train once, eval under shift

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
def P(rel):
    return os.path.join(PAPER_DIR, rel)

DATA = os.path.join(os.path.dirname(_HERE), "data", "indian_pines")
TABLE = os.path.join(os.path.dirname(_HERE), "data", "srf_cache", "T_6s_grid.npz")
CWVS = [0.5, 2.0, 4.0]
_PLATFORM = {"sentinel2": ("Sentinel-2A", "msi"), "landsat_oli": ("Landsat-8", "oli")}
_CENTERS = {"sentinel2": SENTINEL2_MSI_CENTERS_NM, "landsat_oli": LANDSAT8_OLI_CENTERS_NM}
NUM_CLASSES = 16

# A sensor band that retains less than this fraction of its SRF mass after the absorption-core mask
# is not a measurement of that band, it is the mask's leakage. Measured (bandsim.atmosphere): S2 B10
# retains 0.02% and OLI's 1373 nm cirrus band 0%. See --degenerate-bands.
MIN_BAND_THROUGHPUT = 0.01
# Purely NUMERICAL guard on the scaler: a column that is constant on the clean train split has no
# scale to divide by. `sd + 1e-8` only avoids the ZeroDivisionError; it still divides a residual by
# 1e-8. Bands above the floor keep `sd + 1e-8` EXACTLY, so this changes no historical number unless
# a genuinely constant band exists. The substantive guard is MIN_BAND_THROUGHPUT, not this.
SD_FLOOR_REL = 1e-12
# Fixed so the interval is reproducible: a CI that moves between runs of the same data is not a
# provenance-able number.
BOOTSTRAP_SEED = 20260719
N_BOOT = 10000


# --------------------------------------------------------------------------------------- synthesis

def sensor_srf(wl, sensor, srf_source):
    """Build the sensor's CANONICAL SRF dict ONCE (it does not depend on the atmosphere).

    Returns (srf, detectors, report). The raw source dict -- pyspectral's store list or the
    gaussian synthesis -- is reduced through bandsim.srf.select_canonical_bandset, matched by
    CENTRE WAVELENGTH onto this repo's *_CENTERS_NM product definitions (S2 12, OLI 7). That is the
    same contract phase 2 cross-sensor pins, and it replaces this file's own reflective_band_mask
    (a second implementation of the same match, 20 nm tolerance against the shared one's 25 nm,
    which is exactly the kind of drift a single definition exists to prevent).

    `detectors` is keyed by CANONICAL name via the report's source mapping, so provenance records
    WHICH physical response was integrated -- pyspectral's RSR store is versioned and ESA has
    revised S2A's B01/B02 responses, so 'srf=pyspectral' alone does not identify the operator."""
    if srf_source == "pyspectral":
        plat, inst = _PLATFORM[sensor]
        raw, detectors_src = pyspectral_srf(wl, plat, inst, return_detectors=True)
    else:
        # names= is REQUIRED: auto-numbering renames S2's B8A->B9 and B9->B10 (see gaussian_srf).
        raw = gaussian_srf(wl, list(_CENTERS[sensor].values()), fwhm_nm=30.0,
                           names=list(_CENTERS[sensor].keys()))
        detectors_src = {k: "gaussian_fwhm30" for k in raw}
    # B1 EXCLUDED BY DECISION, shared with phase2_cross_sensor (see its synth_sensor): the corrected
    # AVIRIS axis cannot sample B1's full response, and select_canonical_bandset alone would keep a
    # TRUNCATED B1 whose per-band mass passes MIN_BAND_THROUGHPUT -- the dishonest case the
    # cross-sensor refusal guard exists for. Recorded in the report under "excluded".
    srf, report = select_canonical_bandset(wl, raw, _CENTERS[sensor], exclude=("B1",),
                                           exclude_reason=("the AVIRIS-based synthesis axis starts at 437.7 nm; S2/OLI B1 (443 nm) has 16-24% of its response below the first sampled wavelength, and synthesizing a truncated, renormalised B1 would fabricate signal the sensor never measured on this cube"))
    detectors = {name: detectors_src[m["source_band"]] for name, m in report["matched"].items()}
    return srf, detectors, report


def sensor_cube(cube, wl, srf, cwv, srf_source):
    """Flagship chain: atmosphere(CWV) on the HSI axis, THEN SRF -> sensor bands. Returns (out, info).

    The strong absorption cores (6S is unreliable there) are hard-masked in EVERY condition,
    including clean. The clean reference is atmosphere-OFF (gaseous transmittance T=1) but carries
    the SAME core mask, so the ONLY thing that varies clean->CWV is the CWV gaseous-transmittance
    gradient -- NOT a one-off absorption-core removal confounded with it.

    `info` is RETURNED, not discarded. It carries the honesty labels and the keep-mask Design B
    actually applied, and main() asserts that mask IS the clean path's. Note precisely how much that
    check is worth, because an earlier comment here overstated it: on the CLEAN branch this function
    puts its own `keep` into `info`, so main compares an object with itself and the assertion cannot
    fail. The content is entirely on the CWV branch, where `simulate` recomputes the mask from the
    axis independently -- a dropped `hard_mask_cores` yields None, and np.array_equal(None, keep) is
    False, so the case that would silently mix a band removal into the CWV shift IS caught."""
    # The labels below are what pipeline.simulate() propagates into `info` so the qualifier travels
    # with the numbers. They are DERIVED from the execution path and cross-checked against
    # config_runner's vocabulary, so this hand-built cfg cannot over-claim either.
    #
    # NOTE ON config_runner: build_cfg() is deliberately NOT used. Its `_SRF_SOURCES` whitelist is
    # {"gaussian"} -- routing this script through it would silently replace the measured pyspectral
    # RSR with a single 30 nm Gaussian (2.3x too wide on the red-edge, 5.8x too narrow on S2 B12),
    # i.e. it would destroy the one thing this comparison rests on. An earlier docstring claimed the
    # simulation ran "through config_runner"; it never did, and it should not.
    designs = {"A": {"enable": True}, "B": {"enable": cwv is not None}}
    scope = "gaseous_absorption" if cwv is not None else "band_set_geometry"
    if scope not in allowed_claim_scopes(designs):
        raise RuntimeError(f"claim_scope {scope!r} overstates designs {designs}")
    vs_a = derived_validation_status("A", {"srf_source": srf_source})
    cfg = {"seed": 0, "claim_scope": scope,
           "A": {"enable": True, "srf": srf, "validation_status": vs_a}}
    if cwv is not None:
        cfg["B"] = {"enable": True, "hard_mask_cores": True,
                    # 6S table keyed by CWV only (AOD inert, dropped); float() matches config_runner
                    "cache_npz": TABLE, "cache_key": f"cwv{float(cwv)}",
                    "validation_status": derived_validation_status("B", {})}
        return simulate(cube, wl, cfg)
    # clean = atmosphere OFF (T=1) but the SAME hard core-mask as the CWV conditions, applied on the
    # HSI axis BEFORE SRF exactly as Design B does (T=1 * keep == keep-mask only), then SRF.
    keep = hard_mask_absorption_cores(wl)                  # True = usable band; cores -> 0
    out, info = simulate(np.asarray(cube, float) * keep.astype(float), wl, cfg)
    info["atmos_keep_mask"] = keep                         # so main() can compare the two paths
    return out, info


def band_throughput(W, keep):
    """Fraction of each sensor band's SRF mass that SURVIVES the absorption-core mask.

    The mask zeroes SOURCE samples on the HSI axis; build_resampling_matrix's rows stay normalised
    over the FULL grid, so a masked band is not deleted -- it is attenuated by exactly this fraction
    and becomes a near-zero channel while `cube.shape[-1]` still counts it. That is why the figure
    legend used to say "13 bands" for a sensor with 12 informative ones."""
    return (W * keep[None, :].astype(float)).sum(1)        # rows sum to 1 => this IS the fraction


def oracle_gain(W, keep, T, min_mass=1e-6):
    """Per-band gain g_b that a PERFECT correction of the smooth multiplicative factor would invert.

        clean band   c_b = sum_i rho_i keep_i W_bi
        shifted band u_b = sum_i rho_i T_i keep_i W_bi
        g_b            = sum_i T_i keep_i W_bi / sum_i keep_i W_bi        (keep-weighted mean T)

    u_b / g_b == c_b exactly when rho is flat across the band's surviving support; the residual is
    the in-band covariance between rho and T, i.e. spectral distortion INSIDE the bandpass that no
    per-band scalar can undo. Dividing by g_b therefore gives an UPPER BOUND on radiometric
    correction, which is what makes it the right control for "is the drop recoverable gain?".

    Deliberately NOT sensor_band_transmittance()'s W @ T: that averages T over the FULL bandpass
    including the masked core, so for a cirrus band it would return a deep-absorption value and
    divide a near-zero residual by it.

    `min_mass` is a NUMERICAL guard only -- below it the ratio is 0/0 and g_b = 1 leaves the band
    alone. It is deliberately NOT MIN_BAND_THROUGHPUT, and that distinction was briefly got wrong in
    a way worth recording, because the wrong version was argued for confidently and was checkable
    all along:

      the claim was that at 2e-4 of retained mass, dividing S2 B10 by its gain (0.0401 at CWV 4.0)
      would MULTIPLY BY 25 an already 10.4-sigma artefact and could push arm A2 below arm A1.

    Measured on the seed-0 split, B10 at CWV 4.0: clean mean 0.3921, uncorrected mean 0.0163 -- a
    factor 0.0416, i.e. almost exactly g_b. The standardized shift is -14.16 sigma uncorrected and
    +0.49 sigma after dividing by g_b; over all 13 bands z_abs_max goes 14.31 -> 4.25 against a clean
    baseline of 4.22 (an earlier
    draft said 4.15 -- centred on the TEST mean rather than the clean TRAIN mean the scaler
    actually uses, in the very paragraph arguing this revert). Division REMOVES the shift; it does not amplify it. The dominant term is the
    multiplicative attenuation itself, which is exactly what g_b cancels -- the amplified-noise
    intuition was about a term that is not dominant here.

    So the sub-threshold band is corrected like any other. Arm A2 is defined as an UPPER BOUND on
    what a perfect per-band gain correction recovers; an oracle that declines to correct a channel
    it knows the gain of is not that bound. Whether such a channel should be USED at all is a
    separate question, and it has a separate control: `--degenerate-bands drop`."""
    m = (W * keep[None, :].astype(float)).sum(1)
    num = (W * (keep.astype(float) * np.asarray(T, float))[None, :]).sum(1)
    return np.where(m > min_mass, num / np.maximum(m, 1e-300), 1.0), m


# ---------------------------------------------------------------------------------- fingerprinting

def _state_signature(model):
    """SHA-256 over the FULL model state, not just `parameters()`.

    Order-sensitive over state_dict() name/dtype/shape/bytes PLUS the training-mode flag. Hashing
    only parameters would miss registered buffers, non-parameter state and a `.train()` flip; this
    model happens to have none of those today (MLPBaseline is Linear/ReLU, dropout=0), but the check
    is what licenses the claim "the model is frozen", so it should not depend on that staying true.
    A scalar SUM would additionally be fooled by a compensating +d/-d swap between two weights."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode("utf-8"))
        h.update(str(v.dtype).encode("utf-8"))
        h.update(str(tuple(v.shape)).encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    h.update(b"training=1" if model.training else b"training=0")
    return h.hexdigest()


def _scaler_signature(mu, sdv):
    """SHA-256 over the RAW BYTES of the (mu, sdv) scaler arrays. Unlike id(mu)/id(sdv) object
    identity, a byte hash also detects an IN-PLACE content change (`mu[:] = ...`) that keeps the same
    object -> it verifies the scaler fit ONCE on the clean source is reused UNCHANGED."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mu, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(sdv, dtype=np.float64).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------- one seed

def run_seed(seed, cubes, gains, keep_cols, measuring, gt, sensors, cwvs, epochs):
    """One seed, HONEST source->shift protocol; returns RAW per-condition rows (never aggregates).

    Everything an aggregate could be recomputed from is returned: the mIoU of every arm, the
    standardized-shift diagnostics that say WHY it moved, the train-split mIoU that says whether the
    model converged at all, and the split's identity/coverage. main() writes these verbatim to the
    per-seed CSV so a reader can recompute any interval without re-running the experiment."""
    tr, te = disjoint_block_split(gt, block=10, guard=1, offset=seed)
    # Split integrity. A guard band that silently emptied a split, or overlapping masks, would still
    # produce a plausible mIoU -- the failure mode this repo keeps meeting is wrong-but-not-crashing.
    if not tr.any() or not te.any():
        raise RuntimeError(f"seed {seed}: empty split (train={int(tr.sum())}, test={int(te.sum())})")
    if bool((tr & te).any()):
        raise RuntimeError(f"seed {seed}: train and test masks overlap in {int((tr & te).sum())} px")
    ytr = gt[tr].astype(int) - 1
    yte = gt[te].astype(int) - 1
    # miou() averages over classes PRESENT in y_true, so a seed whose test split lost a class is
    # averaging over a DIFFERENT class set than its neighbours. Record it rather than assume.
    classes_te = sorted(int(c) for c in np.unique(yte))
    split_sig = hashlib.sha256(np.ascontiguousarray(tr).tobytes()
                               + np.ascontiguousarray(te).tobytes()).hexdigest()[:16]

    rows = []
    for s in sensors:
        cols = keep_cols[s]
        # positions WITHIN the sliced block whose band is above the throughput floor
        meas = [k for k, i in enumerate(cols) if i in measuring[s]]
        # --- source = CLEAN condition: fit scaler + train model ONCE, then freeze ---
        Xtr = cubes[(s, None)][tr][:, cols]
        Xte_clean = cubes[(s, None)][te][:, cols]
        mu = Xtr.mean(0)
        raw_sd = Xtr.std(0)
        sdv = raw_sd + 1e-8
        degenerate = raw_sd <= SD_FLOOR_REL * np.maximum(np.abs(mu), 1.0)
        sdv = np.where(degenerate, 1.0, sdv)               # constant column -> centre only, no /1e-8
        model = train_mlp((Xtr - mu) / sdv, ytr, seed, epochs=epochs)
        frozen_sig = _state_signature(model)               # full-state fingerprint after training
        scaler_sig = _scaler_signature(mu, sdv)
        if model.training:
            raise RuntimeError("model returned from train_mlp is in TRAIN mode, not eval")
        train_miou = eval_mlp(model, (Xtr - mu) / sdv, ytr)["mIoU"]   # convergence witness

        def record(arm, cwv, Xte):
            """Evaluate the FROZEN model + FROZEN scaler on one condition and record why it moved."""
            # Runtime contract (not `assert`: must survive `python -O`). Checked BEFORE **and**
            # AFTER: a before-only check can never catch the last evaluation mutating the model.
            if _state_signature(model) != frozen_sig:
                raise RuntimeError(f"{s}/{arm}/{cwv}: model state changed since freeze (re-trained?)")
            if _scaler_signature(mu, sdv) != scaler_sig:
                raise RuntimeError(f"{s}/{arm}/{cwv}: scaler bytes changed since freeze (re-fit?)")
            if not np.isfinite(Xte).all():
                raise RuntimeError(f"{s}/{arm}/{cwv}: non-finite features reached the "
                                   f"model (a zero oracle gain would do this, and NaN "
                                   f"logits still argmax to a plausible mIoU)")
            Z = (Xte - mu) / sdv
            r = eval_mlp(model, Z, yte)
            if _state_signature(model) != frozen_sig:
                raise RuntimeError(f"{s}/{arm}/{cwv}: model state changed DURING evaluation")
            if _scaler_signature(mu, sdv) != scaler_sig:               # the comment above says
                raise RuntimeError(f"{s}/{arm}/{cwv}: scaler changed DURING evaluation")   # BOTH
            # The mechanism phase 3 measured on the HSI axis, measured here on sensor bands: how far
            # did the clean scaler push each band out of the training distribution?
            shift = (Z - (Xte_clean - mu) / sdv).mean(0)   # (B,) mean standardized shift per band
            a = np.abs(shift)
            rows.append({
                "seed": seed, "sensor": s, "arm": arm,
                "condition": "clean" if cwv is None else f"cwv{cwv:g}",
                "cwv": "" if cwv is None else cwv,
                "miou": r["mIoU"], "OA": r["OA"], "AA": r["AA"], "kappa": r["kappa"],
                "train_miou": train_miou,
                "zshift_median": float(np.median(a)), "zshift_max": float(a.max()),
                # ..._measuring excludes bands below the throughput floor. Without it the column and
                # the --dry-run headline shared a NAME and not a scope: 14.16 (all fed bands, driven
                # by S2 B10) against 6.17 (measuring only). The docstring tells the reader to judge
                # arm A2 by this column, so which bands it covers cannot be left to inference.
                "zshift_median_measuring": float(np.median(a[meas])) if meas else float("nan"),
                "zshift_max_measuring": float(a[meas].max()) if meas else float("nan"),
                # Mapped back through `cols` to an index into the sensor's FULL band set, which is
                # what the *_bands.csv is indexed by. np.argmax(a) alone is a position in the SLICED
                # block, so under --degenerate-bands drop it would name a different band than the
                # one it points at -- silently, and only in the sensitivity run.
                "zshift_max_band": int(cols[int(np.argmax(a))]),
                "z_abs_max": float(np.abs(Z).max()),
                "n_bands_used": len(cols), "n_degenerate_scaler_cols": int(degenerate.sum()),
                "n_classes_test": len(classes_te), "n_test_px": int(te.sum()),
                "split_sha16": split_sig,
            })

        record("clean", None, Xte_clean)
        for c in cwvs:
            Xte_u = cubes[(s, c)][te][:, cols]
            record("uncorrected", c, Xte_u)                       # A1: rho*T, clean scaler
            # gains are indexed over the sensor's FULL band set; slice them the same way the cube was
            # sliced, or --degenerate-bands drop divides an (N, k) block by an (nbands,) vector.
            record("gain_corrected", c, Xte_u / gains[(s, c)][cols])   # A2: oracle gain removed
    return rows


# ------------------------------------------------------------------------------------- statistics

def paired_bootstrap_ci(d, n_boot=N_BOOT, alpha=0.05, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap CI for the mean of the per-seed PAIRED differences `d`.

    Paired by construction: every seed contributes one difference computed from ONE split, so the
    split variance that dominates the marginal spreads cancels. Percentile, not BCa: at n=5 there
    are only 5**5 distinct resamples and the acceleration estimate is not worth trusting.

    THIS IS NOT A SCENE-LEVEL INTERVAL. The resampling units are checkerboard offsets of a single
    Indian Pines scene that share most of their pixels (the run prints the overlap), so the interval
    describes split+init variability and nothing wider."""
    d = np.asarray(d, float)
    # Below 3 units a percentile bootstrap is not merely wide, it is DEGENERATE: at n=1 every
    # resample is the same value, the interval collapses to zero width, and `lo > 0` then reports
    # a single unreplicated measurement as SIGNIFICANT. integrity_check.py runs this script with
    # `--seeds 0`, so that path is reached in normal use. NaN propagates to `significant_at_95`
    # as False, and the sign tally (`oli_worse_wins`) remains the honest small-n statistic.
    if d.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _col(rows, sensor, arm, cond, field="miou"):
    """Values of `field` for one (sensor, arm, condition) cell, ordered by seed.

    Ordered by SEED, not by row order: every paired statistic below subtracts two of these
    element-wise, so a differently-ordered pair would silently difference mismatched splits."""
    sel = [r for r in rows if r["sensor"] == sensor and r["arm"] == arm and r["condition"] == cond]
    return np.array([r[field] for r in sorted(sel, key=lambda r: r["seed"])], float)


def _sd1(a):
    """Sample SD (ddof=1). numpy's default ddof=0 treats these seeds as the whole population."""
    return float(np.asarray(a, float).std(ddof=1)) if np.asarray(a).size > 1 else float("nan")


def _fmt(x, prec=2, plus=False):
    """Format a statistic, or an EMPTY cell when it is undefined (too few seeds).

    Deliberately not the string "nan", and deliberately not a fabricated 0.00.
    integrity_check.csv_finite_and_sane() does float(cell) and REJECTS the whole file on a NaN,
    while `except ValueError: continue` silently skips any cell that does not parse -- so "nan"
    fails the harness at --seeds 0 and an empty cell is read as "not defined" by the harness, by
    pandas and by a human alike. Writing 0.00 (what this file did before the ddof fix) is worse
    than either: it asserts ZERO dispersion from a single measurement, in the sd column, which is
    exactly the false-precision artefact the rest of this script exists to prevent."""
    x = float(x)
    return "" if x != x else (f"{x:+.{prec}f}" if plus else f"{x:.{prec}f}")


def _warn_if_replacing(args, out_paths):
    """Say so, loudly, when this run is about to replace a file it is not a re-run of.

    `out_paths` is EVERY file this run will write, aggregate first. Both halves need that:

      * the seed-count check reads `n_seeds`, which only the aggregate carries;
      * the CONFIG CHANGE check runs against the first sidecar that exists among ANY of them.

    Passing only the aggregate was a demonstrated hole, not a theoretical one. Under --dry-run the
    bands table is the ONLY file written, so a dry-run-only workflow never creates the aggregate:
    os.path.exists(aggregate) is False, which kills the downgrade branch AND the "unknowable"
    branch, and the missing sidecar skips the config branch. Reproduced: two dry-runs at the same
    tag replaced a 13+9-row band table with a 12+7-row one, with zero warning -- verbatim the
    failure this function's relocation was written to close.

    Called BEFORE the first write. It cannot refuse: integrity_check.py reads a non-zero exit as
    CRASH and an un-refreshed output as STALE, so refusing or redirecting would break a harness this
    file must not edit. THE ACTUAL FIX belongs there -- this script now has --smoke."""
    out_path = out_paths[0]
    # experiments/integrity_check.py runs this script as `--seeds 0 --epochs 12` against the
    # UNSUFFIXED deliverable, so a 1-seed smoke-scale run replaces a 5-seed result -- and its sd
    # column collapses to an empty cell while the figure LOOKS more precise. That is the class of
    # defect that once inverted phase 3's conclusion. It is deliberately NOT fatal here:
    # integrity_check treats a non-zero exit as CRASH and an un-refreshed output as STALE, so both
    # refusing and redirecting would break a harness this file must not edit.
    # THE ACTUAL FIX belongs in integrity_check.py: this script now HAS --smoke, so that entry
    # should read `phase5_ab_flagship.py --smoke` / `results_phase5_ab_flagship_smoke.csv`.
    #
    # Read the seed count from the CSV's OWN n_seeds column, sidecar only as a fallback. That
    # ordering matters: integrity_check backs up and byte-restores paper/*.csv and *.tex, but NOT
    # *.provenance.json -- so a 1-seed harness run leaves a sidecar sitting next to a restored
    # 5-seed CSV, describing a downgrade that did not survive. Trusting the sidecar first would let
    # that fabricated record disarm this guard on the next real run. (The same gap leaves the
    # _perseed/_bands/_paired companions behind after a harness run, since they are not in its
    # pre-run glob; nothing here can fix that from inside this file.)
    _out = out_path
    old_n = None
    if os.path.exists(_out):
        try:
            with open(_out, newline="") as f:
                old_n = max(int(r["n_seeds"]) for r in csv.DictReader(f) if r.get("n_seeds"))
        except Exception:
            old_n = None
    corroborated = old_n is not None
    if old_n is None and os.path.exists(_out + ".provenance.json"):
        try:
            with open(_out + ".provenance.json") as f:
                old_n = len((json.load(f).get("args") or {}).get("seeds") or []) or None
        except Exception:
            old_n = None
    if old_n and old_n > len(args.seeds):
        print(f"  !! DOWNGRADE: replacing a {old_n}-seed result with {len(args.seeds)} seeds. "
              f"Use --tag or --smoke to keep both.")
    elif os.path.exists(_out) and not corroborated:
        # Reaching here with old_n set means the count came from a sidecar the CSV cannot confirm.
        # That is not a safe silence: integrity_check restores paper/*.csv but NOT *.provenance.json,
        # so a 1-seed harness run leaves a sidecar beside a byte-restored 5-seed CSV -- and trusting
        # it made `1 > 5` false and skipped this branch too. Say the count is unproven either way.
        print(f"  !! the existing deliverable carries no n_seeds column, so its seed count is "
              f"{'UNCORROBORATED (sidecar says ' + str(old_n) + ')' if old_n else 'unknowable'} -- "
              f"this run cannot be compared to what it replaces.")
    # Seed count is not the only way one run silently replaces a different one. --srf, --band-set,
    # --degenerate-bands and --seeds each change WHICH NUMBERS these files hold while leaving the
    # filename and often the row count identical. A same-size run under a different configuration is
    # not a re-run, it is a different experiment.
    #
    # Iterates EVERY path, and that is the whole point: under --dry-run the bands table is the only
    # file written, so checking the aggregate alone found nothing to compare against and printed
    # nothing. Reproduced twice -- once by review, once again after a "fix" that changed this
    # function's signature and its docstring but not this loop.
    for path in out_paths:
        side = path + ".provenance.json"
        if not os.path.exists(side):
            continue
        try:
            with open(side) as f:
                old_args = (json.load(f).get("args") or {})
        except Exception:
            continue
        differs = {k: (old_args.get(k), getattr(args, k))
                   for k in ("srf", "band_set", "degenerate_bands", "epochs", "seeds")
                   if k in old_args and old_args.get(k) != getattr(args, k, None)}
        if differs:
            print(f"  !! CONFIG CHANGE: {os.path.basename(path)} was produced with "
                  + ", ".join(f"{k}={o!r} (now {n!r})" for k, (o, n) in differs.items())
                  + ". Same path, different experiment -- use --tag to keep both.")
            break



def _provenance(args, wl, nbands, keep_cols, n_measuring, srf_meta, sensors, overlap):
    """The record every artefact of this run is stamped with.

    A function rather than an inline dict because --dry-run stamps it too: the go/no-go table has to
    be as traceable as the full result, and a second hand-maintained copy would drift from this one.

    The 6S LUT and the SRF store are the two inputs that decide every number here, and neither is
    under version control as a VALUE: precompute_6s_table.py can regenerate the LUT on a different
    axis, and pyspectral's RSR store is versioned (ESA has revised S2A's B01/B02 responses). Either
    can change a result while the CSV looks identical, so both are hashed/pinned."""
    try:
        from importlib.metadata import version as _ver
        pyspectral_version = _ver("pyspectral")
    except Exception:
        pyspectral_version = None
    return {
        "lut": TABLE, "lut_sha256": file_sha256(TABLE), "cwvs": CWVS, "sensors": sensors,
        "bands_by_sensor": nbands, "bands_used_by_sensor": {s: len(keep_cols[s]) for s in sensors},
        "bands_measuring_by_sensor": n_measuring,
        "srf_source": args.srf, "band_set": "canonical_contract",
        "pyspectral_version": pyspectral_version, "srf": srf_meta,
        "wavelength_axis_sha256": axis_sha256(wl), "n_hsi_bands": int(wl.size),
        "cube_sha256": file_sha256(os.path.join(DATA, "Indian_pines_corrected.mat")),
        "gt_sha256": file_sha256(os.path.join(DATA, "Indian_pines_gt.mat")),
        "hw": hw.info(), "deterministic": not args.nondeterministic,
        "dry_run": bool(getattr(args, "dry_run", False)),
        "claim_scope": "gaseous_absorption",
        "validation_status": {"A": derived_validation_status("A", {"srf_source": args.srf}),
                              "B": derived_validation_status("B", {})},
        "claim_limits": ("spectral-only bandpass-set comparison on a COMMON spatial grid (no PSF/"
                         "MTF/resolution/SNR/quantization); gaseous transmittance only (no aerosol/"
                         "path radiance); seeds are checkerboard offsets of ONE scene, so intervals "
                         "are split+init variability, not scene-level"),
        "seed_split_jaccard_overlap": overlap,
        "bootstrap": {"method": "percentile", "n_boot": N_BOOT, "seed": BOOTSTRAP_SEED,
                      "undefined_below_n": 3},
        "degenerate_band_policy": args.degenerate_bands,
        "min_band_throughput": MIN_BAND_THROUGHPUT,
        "protocol": "scaler+MLP fit ONCE on clean source; frozen (full state_dict fingerprint "
                    "checked before AND after every evaluation) and evaluated under each CWV with "
                    "no re-training and no re-fit",
        "absorption_cores": "hard-masked in EVERY condition, clean included; Design B's mask "
                            "asserted identical to the clean path's at runtime",
        "arms": {"uncorrected": "rho*T then SRF, clean scaler -- uncorrected gas-only radiometry "
                                "proxy (phase 3's SUPPLEMENTARY arm; NOT 'L1C')",
                 "gain_corrected": "same cube divided by the keep-weighted per-band mean "
                                   "transmittance -- an UPPER BOUND on radiometric correction; the "
                                   "residual gap to clean is in-band spectral distortion"},
    }


# ------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--srf", default="pyspectral", choices=["pyspectral", "gaussian"])
    ap.add_argument("--degenerate-bands", default="keep", choices=["keep", "drop"],
                    help="sensor bands retaining <1%% of their SRF mass after the core mask: "
                         "'keep' (default, historical) feeds the near-zero channel to the model; "
                         "'drop' removes the column from every condition (sensitivity run)")
    ap.add_argument("--tag", default="", help="suffix for ALL outputs, so a variant cannot overwrite "
                                              "the deliverable produced by a different config")
    ap.add_argument("--smoke", action="store_true", help="2 seeds / 12 epochs, *_smoke outputs only")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every band/CWV diagnostic and write the *_bands table, then exit "
                         "BEFORE any training. No GPU, no model, seconds. This is the go/no-go: it "
                         "says whether the CWV axis moves this band set at all, and by how many "
                         "sigma, which decides whether the full run is worth its cost")
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent seed workers (default: adaptive; also BANDSIM_WORKERS)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                    help="device for the workers (default: auto; also BANDSIM_DEVICE)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="disable deterministic algorithms for a small speedup")
    args = ap.parse_args()

    # ---- argument contracts. `--epochs 0` trained nothing and still wrote a deliverable; duplicate
    # seeds silently double-weight one split inside every mean and every bootstrap resample.
    if args.epochs < 1:
        sys.exit(f"ERROR: --epochs must be >= 1, got {args.epochs} (0 would write an untrained result)")
    if args.jobs is not None and args.jobs < 1:
        sys.exit(f"ERROR: --jobs must be >= 1, got {args.jobs}")
    if len(set(args.seeds)) != len(args.seeds):
        sys.exit(f"ERROR: --seeds must be unique, got {args.seeds} (a repeat re-weights that split)")
    if args.tag and not args.tag.startswith("_"):
        args.tag = "_" + args.tag
    if args.smoke:
        # Report the ACTUAL count: `--smoke --seeds 0` leaves one seed, and a hard-coded "2 seeds"
        # would describe a run that then writes empty sd cells for a reason the log denied.
        args.seeds, args.epochs = args.seeds[:2], 12
        print(f"[smoke] {len(args.seeds)} seed(s) / {args.epochs} epochs — writing *_smoke "
              f"artefacts, NOT the real deliverables")
    sfx = args.tag + ("_smoke" if args.smoke else "")

    os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)   # in main(), not at import time
    # --dry-run does no model work, so it must not touch a GPU. prefer="cpu" is NOT enough and an
    # earlier comment here wrongly claimed it was: hw.setup calls torch.cuda.manual_seed_all under
    # `if n_gpus() > 0`, which is gated on VISIBLE devices, not on `prefer` -- so a context was still
    # created on every card (a few hundred MB each), possibly mid-campaign and against doctor.py's
    # own VRAM gates, while the --dry-run help said "No GPU". Hide the devices instead. Set before
    # hw.setup and before anything else touches CUDA, which nothing at import time does.
    if args.dry_run and args.device != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    hw.setup(deterministic=not args.nondeterministic,
             prefer=args.device or ("cpu" if args.dry_run else None))
    print("HW:", hw.info(), "| SRF:", args.srf, "| degenerate-bands:", args.degenerate_bands)
    if not os.path.exists(TABLE):
        print(f"ERROR: missing 6S table {TABLE}"); sys.exit(1)

    # ---- inputs + contracts ----------------------------------------------------------------
    wl = AVIRIS_WL_NM
    cube = load_mat_cube(os.path.join(DATA, "Indian_pines_corrected.mat"),
                         key="indian_pines_corrected").astype(np.float64)
    gt = load_mat_cube(os.path.join(DATA, "Indian_pines_gt.mat"), key="indian_pines_gt").astype(int)
    # Checked here rather than left to fail deep inside a matmul or a CUDA label assert, where the
    # message names a tensor shape instead of the data problem.
    if cube.ndim != 3 or gt.ndim != 2 or cube.shape[:2] != gt.shape:
        sys.exit(f"ERROR: cube {cube.shape} / gt {gt.shape} are not a matching (H,W,C)/(H,W) pair")
    if cube.shape[-1] != wl.size:
        sys.exit(f"ERROR: cube has {cube.shape[-1]} bands but the axis has {wl.size}")
    if not np.isfinite(cube).all():
        sys.exit(f"ERROR: cube has {int((~np.isfinite(cube)).sum())} non-finite values")
    if not (np.diff(wl) > 0).all():
        sys.exit("ERROR: wavelength axis is not strictly increasing")
    if gt.min() < 0 or gt.max() > NUM_CLASSES:
        sys.exit(f"ERROR: gt labels outside [0, {NUM_CLASSES}] (range [{gt.min()}, {gt.max()}])")

    sensors = ["sentinel2", "landsat_oli"]
    keep = hard_mask_absorption_cores(wl)

    # ---- the 6S table, checked as PHYSICS before it is used --------------------------------
    # The LUT decides every CWV number here. Its axis is already guarded (a wrong-axis table was a
    # real, silent defect); this adds the monotonicity that an axis check cannot see: more column
    # water vapour can never RAISE gaseous transmittance. A mis-keyed or mislabelled column would
    # otherwise produce a perfectly plausible curve.
    # require_generation_mode: a table RESAMPLED onto this axis has the right length, the right
    # wl_nm and the right axis hash, so it passes every other check -- while its values near narrow
    # absorption edges are not what 6S returns there. This experiment is ABOUT those edges.
    Ts = {c: load_cached_transmittance(TABLE, f"cwv{float(c)}", n_bands=wl.size,
                                       expected_wavelengths_nm=wl,
                                       require_generation_mode="direct_6s") for c in CWVS}
    for lo, hi in zip(CWVS, CWVS[1:]):
        if not (Ts[hi] <= Ts[lo] + 1e-9).all():
            bad = int((Ts[hi] > Ts[lo] + 1e-9).sum())
            sys.exit(f"ERROR: 6S table not monotone in CWV: T(cwv{hi}) > T(cwv{lo}) in {bad} bands. "
                     f"More water vapour cannot increase gaseous transmittance -- the table's keys "
                     f"or columns are wrong.")

    # The FIRST seed's split, used only to express each band's shift in sd units below. That number
    # needs no model -- it is (shifted - clean)/clean_train_sd in feature space -- which is the whole
    # reason --dry-run can promise it. It is also the quantity that actually ranks bands: gain span
    # and sigma shift order them DIFFERENTLY (S2 B12 has a smaller span than B08 and a larger shift;
    # a fully-masked band can have a span of 0 and the largest shift of all), so a go/no-go read off
    # the span alone would not be the decision it appears to be.
    tr0, te0 = disjoint_block_split(gt, block=10, guard=1, offset=args.seeds[0])

    # ---- synthesis: SRF built ONCE per sensor (it does not depend on the atmosphere) --------
    cubes, gains, srf_meta, band_rows, keep_cols, nbands, n_measuring, measuring = \
        {}, {}, {}, [], {}, {}, {}, {}
    for s in sensors:
        # Selection happens INSIDE sensor_srf via the canonical contract, before anything
        # downstream is built, so the cubes, W, the gains, the band count and the figure legend are
        # all consistent by construction rather than by a column mask applied in four places.
        srf, detectors, sel_report = sensor_srf(wl, s, args.srf)
        dropped = sel_report["dropped"]
        if dropped and args.srf == "gaussian":
            # gaussian_srf is BUILT from _CENTERS, so the canonical match must be a no-op here. A
            # drop means a band's SRF-WEIGHTED effective centre has moved further than the
            # tolerance from the centre it was constructed at -- which happens when the wavelength
            # axis has a gap under that band -- and it would silently shrink the band set the
            # gaussian arm compares. Fail rather than accept it.
            sys.exit(f"ERROR: the canonical contract dropped {dropped} from the GAUSSIAN {s} SRF, "
                     f"which is built from those very centres. The axis must have a gap under "
                     f"them; investigate rather than accept a shrunken band set.")
        if dropped:
            print(f"  {s}: canonical band contract drops {dropped} (source bands with no product "
                  f"centre within tolerance: pan / cirrus / out-of-product channels)")
        if not srf:
            sys.exit(f"ERROR: the canonical contract left {s} with no bands")
        W, names = build_resampling_matrix(wl, srf)
        centers = W @ wl
        mass = band_throughput(W, keep)
        # Where the SURVIVING response actually sits. For a band whose bandpass lies inside an
        # absorption core these differ by a lot -- S2 B10 is nominally a 1375 nm cirrus band, but
        # after the mask its residual signal comes from the wings, so `center_nm` names a wavelength
        # the channel no longer measures. Reporting both is what makes "degenerate" concrete.
        Wk = W * keep[None, :].astype(float)
        centers_kept = np.where(mass > 1e-12, (Wk @ wl) / np.maximum(mass, 1e-300), np.nan)
        srf_meta[s] = {
            "band_names": list(names), "detectors": [detectors[n] for n in names],
            "band_centers_nm": [round(float(x), 3) for x in centers],
            "kept_srf_fraction": [round(float(x), 6) for x in mass],
            "srf_matrix_sha256": hashlib.sha256(np.ascontiguousarray(W, "<f8").tobytes()).hexdigest(),
            # which source band each canonical band came from, mismatch in nm, and what was dropped
            "canonical_selection": sel_report,
        }
        for c in [None] + CWVS:
            out, info = sensor_cube(cube, wl, srf, c, args.srf)
            # The docstring's central claim is that clean and every CWV condition differ by T ALONE.
            # That is only true if Design B masked EXACTLY the bands the clean path masked, and that
            # was a comment, not a check. simulate() reports the mask it used, so compare them.
            if not np.array_equal(info.get("atmos_keep_mask"), keep):
                sys.exit(f"ERROR: {s}/cwv={c}: Design B's core mask differs from the clean path's -- "
                         f"clean->CWV would then mix a band removal with the CWV shift")
            if list(info.get("band_names", names)) != list(names):
                sys.exit(f"ERROR: {s}/cwv={c}: band names/order changed between conditions")
            cubes[(s, c)] = out
            if c is not None:
                gains[(s, c)], _ = oracle_gain(W, keep, Ts[c])
        nbands[s] = cubes[(s, None)].shape[-1]
        # Per-band standardized shift on the first seed's split -- the same quantity run_seed's
        # `record` computes per condition, but in feature space only, so it is available here with
        # no model. sd is the CLEAN TRAIN sd, i.e. the scaler the frozen model will actually use.
        sd0 = cubes[(s, None)][tr0].std(0) + 1e-8
        cl0 = cubes[(s, None)][te0]
        shift_sigma = {c: (cubes[(s, c)][te0] - cl0).mean(0) / sd0 for c in CWVS}
        dead = [i for i, m in enumerate(mass) if m < MIN_BAND_THROUGHPUT]
        keep_cols[s] = ([i for i in range(nbands[s]) if i not in set(dead)]
                        if args.degenerate_bands == "drop" else list(range(nbands[s])))
        for i, n in enumerate(names):
            band_rows.append({
                "sensor": s, "band": n, "detector": detectors[n],
                "center_nm": f"{centers[i]:.2f}", "center_nm_after_mask": _fmt(centers_kept[i]),
                "kept_srf_fraction": f"{mass[i]:.6f}",
                "status": ("degenerate" if mass[i] < MIN_BAND_THROUGHPUT else
                           "truncated" if mass[i] < 0.999 else "intact"),
                "used_by_model": i in set(keep_cols[s]),
                # The keep-weighted mean transmittance of this band == the oracle gain applied in
                # arm A2. Named for the GAIN, not for T, because a band below the NUMERICAL floor is
                # forced to 1.0 and 1.0 is emphatically not its transmittance. On this axis that is
                # OLI's cirrus band alone (mass exactly 0); S2 B10 sits at 1.7e-4, ABOVE the floor,
                # so it carries real gains -- an earlier version of this comment said "a degenerate
                # band" and was written while the floor was MIN_BAND_THROUGHPUT, which was reverted.
                **{f"oracle_gain_cwv{c:g}": f"{gains[(s, c)][i]:.6f}" for c in CWVS},
                # How far the clean scaler pushes this band out of the training distribution, in its
                # own sd. Model-free, so --dry-run can emit it, and it is the number that decides
                # whether a gain change matters rather than merely existing. THIS IS ARM A1
                # (UNCORRECTED) -- it sits beside the A2 gain factor and would otherwise invite the
                # post-correction reading; joining to the per-seed CSV matches arm=="uncorrected".
                **{f"shift_sigma_cwv{c:g}": f"{shift_sigma[c][i]:+.2f}" for c in CWVS},
            })
        # IS THE CWV AXIS ACTUALLY DOING ANYTHING TO THIS SENSOR? This repo has already shipped one
        # INERT axis: an earlier CWV x AOD grid varied AOD over a quantity that does not depend on
        # it, so every AOD column was identical and the "AOD ablation" measured nothing. A flat
        # degradation curve is that defect's exact symptom, and the committed pre-fix result IS flat
        # (OLI moves 0.03 mIoU across an 8x change in water vapour). The monotonicity check above
        # cannot see it -- identical columns are trivially monotone. So measure the range: after SRF
        # integration a wide multispectral bandpass averages a narrow absorption feature away, and if
        # the per-band gain barely moves between the lowest and highest CWV then the axis is inert
        # ON THIS BAND SET and a flat curve is the correct physics, not a finding about sensors.
        #
        # Evaluated over keep_cols, NOT over every nominal band. The verdict is about the channels
        # the model actually sees, and the widest-moving band can easily be one it does not: S2 B10
        # sits inside the 1350-1450 nm core and keeps ~2e-4 of its mass from wings lying right on
        # the water-vapour edge, so it plausibly has the LARGEST gain span of any S2 band -- and it
        # is the band --degenerate-bands drop removes, and a near-zero channel either way. Scoring
        # inertness on it would let a channel that carries no class information rescue the check.
        span_all = np.abs(gains[(s, CWVS[0])] - gains[(s, CWVS[-1])])
        # Scored over bands that are BOTH fed to the model AND actual measurements. keep_cols alone
        # is not enough: under the default --degenerate-bands keep it still contains the near-zero
        # channels, so a degenerate band would be back inside the verdict. It is harmless right now,
        # but NOT for the reason an earlier comment gave: that one said oracle_gain pins degenerate
        # bands to 1.0 so their span is exactly 0, which the threshold revert made false -- S2 B10's
        # span is 0.2565, the SECOND LARGEST of any S2 band. It is harmless only because 0.2565 does
        # not beat B09's 0.3973. That is luck, not a guarantee, which is why the scope is stated
        # here rather than inherited from another function's behaviour.
        scored = [i for i in keep_cols[s] if mass[i] >= MIN_BAND_THROUGHPUT]
        if not scored:
            sys.exit(f"ERROR: {s} has no band above the {MIN_BAND_THROUGHPUT:.0%} throughput floor")
        n_measuring[s] = len(scored); measuring[s] = set(scored)
        span = span_all[scored]
        arg = scored[int(np.argmax(span))]
        srf_meta[s]["cwv_gain_span"] = {"min_used": float(span.min()), "max_used": float(span.max()),
                                        "max_any_band": float(span_all.max()),
                                        "argmax_used_band": names[arg], "inert_below": 0.01}
        print(f"  {s}: per-band oracle gain moves {span.min():.4f}..{span.max():.4f} across "
              f"CWV {CWVS[0]:g}->{CWVS[-1]:g} over the {len(scored)} MEASURING bands the model uses "
              f"(max on {names[arg]}; {span_all.max():.4f} over all {len(names)} nominal bands)")
        # The gain span is physics; THIS is what decides whether it reaches the classifier, and the
        # two rank bands differently -- so both are printed and the CSV carries both per band.
        sig = np.abs(shift_sigma[CWVS[-1]])
        sig_used = sig[scored]
        a_used = scored[int(np.argmax(sig_used))]
        a_all = int(np.argmax(sig))
        print(f"  {s}: |standardized shift| at CWV {CWVS[-1]:g} (seed {args.seeds[0]} split): "
              f"median {np.median(sig_used):.2f} s, max {sig_used.max():.2f} s on {names[a_used]} "
              f"over measuring bands"
              + (f"  -- but {sig.max():.2f} s on {names[a_all]}, which is BELOW the throughput floor"
                 if a_all != a_used else ""))
        if span.max() < 0.01:
            print(f"  !! {s}: the CWV axis is NEARLY INERT on this band set -- no band's gain moves "
                  f"more than {span.max():.4f} across the whole CWV range. A flat mIoU curve here "
                  f"measures the SRF averaging out a narrow absorption feature, NOT sensor "
                  f"robustness. Do not report it as a degradation result.")
        if dead:
            print(f"  ! {s}: bands {[names[i] for i in dead]} retain <{MIN_BAND_THROUGHPUT:.0%} of "
                  f"their SRF mass after the core mask "
                  f"({', '.join(f'{mass[i]:.2%}' for i in dead)}) -- "
                  f"{'DROPPED' if args.degenerate_bands == 'drop' else 'kept (near-zero channel)'}")
        print(f"  {s}: {nbands[s]} bands nominal, {len(keep_cols[s])} fed to the model, "
              f"{sum(1 for m in mass if m >= MIN_BAND_THROUGHPUT)} above the throughput floor")

    # ---- how independent are the "seeds", really? -------------------------------------------
    # Offsets 0..4 shift the checkerboard by 0..4 px, so consecutive seeds keep most of their
    # assignment. The period is `block` = 10 px, NOT 2*block: raising both block indices by one
    # leaves (bi+bj) parity unchanged, so offset s and offset s+block are the same split byte for
    # byte (verified: offsets 10 and 20 reproduce offset 0). Sweeping more than `block` offsets
    # counts splits twice; the default 0..4 stays inside one period. Measure it, because the spread these seeds produce is quoted as an
    # error bar and a reader is entitled to know what it is a spread OVER. Expect ~0.448 mean (0.632
    # for ADJACENT seeds) against 0.333 for two genuinely independent half-splits -- these are
    # POSITIVELY CORRELATED replicates, so every sd and CI below is an UNDERESTIMATE, not a
    # conservative bound.
    masks = [disjoint_block_split(gt, block=10, guard=1, offset=sd)[1] for sd in args.seeds]
    jac = [float((a & b).sum()) / float(max((a | b).sum(), 1))
           for i, a in enumerate(masks) for b in masks[i + 1:]]
    # None, not NaN: json.dump writes a bare `NaN` token, which is not valid JSON (RFC 8259). Python
    # reads it back, `jq` and every JS reader do not -- and this dict goes into a stamped sidecar.
    overlap = {"mean": float(np.mean(jac)) if jac else None,
               "max": float(np.max(jac)) if jac else None, "n_pairs": len(jac)}
    print("test-split Jaccard overlap between seeds: "
          + (f"mean {overlap['mean']:.2f}, max {overlap['max']:.2f}" if jac else "n/a (one seed)")
          + " — these are offsets of ONE scene, not independent scenes")

    # ---- per-band csv (depends on NOTHING downstream, so it is written before any training) ----
    _out = P(f"results_phase5_ab_flagship{sfx}.csv")
    _warn_if_replacing(args, [_out] + [P(f"results_phase5_ab_flagship_{n}{sfx}.csv")
                                       for n in ("bands", "perseed", "paired")])
    prov = _provenance(args, wl, nbands, keep_cols, n_measuring, srf_meta, sensors, overlap)
    with open(P(f"results_phase5_ab_flagship_bands{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(band_rows[0].keys()))
        w.writeheader(); w.writerows(band_rows)

    if args.dry_run:
        # Everything above is physics and bookkeeping: the LUT guards, the per-band throughput, the
        # oracle gains, the CWV-inertness verdict, the split overlap. None of it needs a model, so
        # none of it should cost a GPU to find out. Stop here.
        #
        # The bands table is NOT thin-vs-full -- a dry run and a full run at the SAME configuration
        # produce it identically, because nothing in it depends on a model. It IS configuration-
        # dependent, and an earlier comment here wrongly called it "a pure function of the LUT, the
        # SRF and the core mask": `used_by_model` follows --degenerate-bands, and the ROW SET follows
        # --band-set and --srf (13+9 rows against 12+7). So a `--dry-run --band-set reflective` can
        # leave a 19-row table beside a full run's aggregate CSV that still says 13/9 bands -- which
        # is exactly why _warn_if_replacing now runs BEFORE this write rather than after it.
        stamp(P(f"results_phase5_ab_flagship_bands{sfx}.csv"), args, extra=prov)
        print(f"\n[dry-run] wrote {P(f'results_phase5_ab_flagship_bands{sfx}.csv')} and stopped "
              f"before training.\n"
              f"          Read the inertness verdict and the degenerate-band lines above: they "
              f"decide whether\n"
              f"          the full run is worth its cost. Re-run without --dry-run to train.")
        return

    # ---- run --------------------------------------------------------------------------------
    results = parallel.run_jobs(
        run_seed, args.seeds,
        shared=dict(cubes=cubes, gains=gains, keep_cols=keep_cols, measuring=measuring,
                    gt=gt, sensors=sensors,
                    cwvs=CWVS, epochs=args.epochs),
        prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
        label="phase5/seed")
    if len(results) != len(args.seeds):
        sys.exit(f"ERROR: {len(results)} results for {len(args.seeds)} seeds")
    rows = [r for res in results for r in res]
    cov = {r["n_classes_test"] for r in rows}
    if len(cov) > 1:
        print(f"  ! WARNING: test splits cover DIFFERENT class counts across seeds ({sorted(cov)}); "
              f"miou averages over present classes, so those seed means are not over the same set")
    for sd in args.seeds:
        g = lambda s, a, c: next(r["miou"] for r in rows if r["seed"] == sd and r["sensor"] == s
                                 and r["arm"] == a and r["condition"] == c)
        print(f"seed {sd}: S2 clean={g('sentinel2','clean','clean'):.1f} "
              f"unc4={g('sentinel2','uncorrected','cwv4'):.1f} "
              f"gain4={g('sentinel2','gain_corrected','cwv4'):.1f} | "
              f"OLI clean={g('landsat_oli','clean','clean'):.1f} "
              f"unc4={g('landsat_oli','uncorrected','cwv4'):.1f} "
              f"gain4={g('landsat_oli','gain_corrected','cwv4'):.1f}")

    # ---- per-seed RAW csv (everything else is recomputable from this) ------------------------
    fields = list(rows[0].keys())
    with open(P(f"results_phase5_ab_flagship_perseed{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["sensor"], r["arm"], str(r["cwv"]), r["seed"])):
            w.writerow(r)

    # ---- aggregate csv -----------------------------------------------------------------------
    arms = [("clean", [None]), ("uncorrected", CWVS), ("gain_corrected", CWVS)]
    # P(...) inline, not the `_out` name: tests/test_smoke_isolation.py records a write site
    # only when the path argument is literally a P(...) call, so hoisting this to a variable
    # made the HEADLINE deliverable the one write its lock could not see.
    with open(P(f"results_phase5_ab_flagship{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # ddof=1: these seeds are a SAMPLE of possible splits, not the population. numpy's default
        # ddof=0 is ~89% of the sample SD at n=5, i.e. an error bar 11% too small, quoted as if it
        # were the estimand. Both retention estimands are written because they are not equal: the
        # ratio of means discards the pairing, the mean of per-seed ratios keeps it and has a spread.
        w.writerow(["sensor", "bands_nominal", "bands_used", "arm", "condition", "cwv",
                    "eval_protocol", "miou_mean", "miou_sd_ddof1", "miou_sem",
                    "drop_vs_clean_mean", "drop_vs_clean_sd_ddof1",
                    "retention_ratio_of_means", "retention_paired_mean", "retention_paired_sd_ddof1",
                    "zshift_median_mean", "zshift_max_mean",
                    "zshift_max_measuring_mean", "n_seeds"])
        for s in sensors:
            clean_v = _col(rows, s, "clean", "clean")
            for arm, conds in arms:
                for c in conds:
                    cond = "clean" if c is None else f"cwv{c:g}"
                    v = _col(rows, s, arm, cond)
                    drop = clean_v - v
                    ret = v / clean_v if np.all(clean_v > 0) else np.full_like(v, np.nan)
                    w.writerow([
                        s, nbands[s], len(keep_cols[s]), arm, cond, "" if c is None else c,
                        "in_distribution" if c is None else
                        ("under_shift_no_refit" if arm == "uncorrected" else
                         "under_shift_oracle_gain_corrected_no_refit"),
                        _fmt(v.mean()), _fmt(_sd1(v)), _fmt(_sd1(v) / np.sqrt(v.size)),
                        _fmt(drop.mean()), _fmt(_sd1(drop)),
                        _fmt(v.mean() / clean_v.mean(), 3) if clean_v.mean() > 0 else "",
                        _fmt(ret.mean(), 3), _fmt(_sd1(ret), 3),
                        _fmt(_col(rows, s, arm, cond, 'zshift_median').mean()),
                        _fmt(_col(rows, s, arm, cond, 'zshift_max').mean()),
                        _fmt(_col(rows, s, arm, cond, 'zshift_max_measuring').mean()), v.size])

    # ---- the headline question, as a PAIRED test --------------------------------------------
    # "Does OLI degrade more than S2?" is a difference of differences, and both differences share a
    # seed (same split, same test pixels), so it must be evaluated paired. Two independent error
    # bars that happen not to touch is not that test, and it was the only evidence offered before.
    #
    # TWO ESTIMANDS, because they are not the same question and can disagree. The absolute DiD
    # differences mIoU POINTS; the retention DiD differences each sensor's fraction of its OWN clean
    # score. With unequal clean baselines (66.73 vs 63.60 pre-fix -- a 4.7% gap) the sensor that
    # loses more points need not be the one that retains less: drops of 3.00 (S2) and 2.95 (OLI)
    # make S2 look worse in points while OLI retains LESS (0.9536 vs 0.9550). This file's own
    # summary of the pre-fix result is stated in RETENTION terms, so retention gets its own
    # interval instead of being read off the absolute one.
    # SIGN CONVENTION for both: > 0 means OLI degrades MORE.
    paired = []
    for arm in ("uncorrected", "gain_corrected"):
        for c in CWVS:
            cond = f"cwv{c:g}"
            cl_s2 = _col(rows, "sentinel2", "clean", "clean")
            cl_ol = _col(rows, "landsat_oli", "clean", "clean")
            v_s2, v_ol = _col(rows, "sentinel2", arm, cond), _col(rows, "landsat_oli", arm, cond)
            d_s2, d_ol = cl_s2 - v_s2, cl_ol - v_ol
            row = {"arm": arm, "cwv": c, "n_seeds": int(d_s2.size),
                   "drop_s2_mean": _fmt(d_s2.mean()), "drop_oli_mean": _fmt(d_ol.mean())}
            finite = bool(np.all(cl_s2 > 0) and np.all(cl_ol > 0))
            for key, d in (("abs", d_ol - d_s2),
                           ("ret", (v_s2 / cl_s2) - (v_ol / cl_ol) if finite
                                   else np.full(d_s2.shape, np.nan))):
                lo, hi = paired_bootstrap_ci(d)
                # BOTH tails. `lo > 0` alone calls a significantly NEGATIVE DiD -- S2 degrading more,
                # which is the direction the pre-fix numbers point at -- "not significant", printed
                # beside an interval that excludes zero. The direction is NAMED, never implied by a
                # bare boolean whose column name only covers one of the two answers.
                direction = "oli_worse" if lo > 0 else "s2_worse" if hi < 0 else "inconclusive"
                row.update({
                    f"did_{key}_mean": _fmt(d.mean(), 3, plus=True),
                    f"did_{key}_sd_ddof1": _fmt(_sd1(d), 3),
                    f"did_{key}_ci95_lo": _fmt(lo, 3, plus=True),
                    f"did_{key}_ci95_hi": _fmt(hi, 3, plus=True),
                    f"did_{key}_direction": direction,
                    f"did_{key}_significant_at_95": direction != "inconclusive",
                    f"did_{key}_oli_worse_wins": f"{int((d > 0).sum())}/{d.size}",
                })
            row["ci_units"] = "checkerboard offsets of ONE Indian Pines scene (NOT scenes/sensors)"
            paired.append(row)
    with open(P(f"results_phase5_ab_flagship_paired{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader(); w.writerows(paired)

    # ---- figure: the two arms side by side ---------------------------------------------------
    # x is CATEGORICAL: clean is its OWN tick, not a fabricated CWV=0 coordinate (clean is
    # atmosphere-off, not CWV=0). Per-seed points are drawn under the mean so the reader sees n
    # and its dispersion instead of an unlabelled bar.
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
    colors = {"sentinel2": "#2980b9", "landsat_oli": "#c0392b"}
    # The INFORMATIVE count leads. band_throughput's docstring says, past tense, that the legend
    # "used to say 13 bands for a sensor with 12 informative ones" -- and under the default
    # --degenerate-bands keep it still rendered "13 of 13", because keep_cols counts what is fed to
    # the model, not what measures anything. The count that reached the console and the bands CSV
    # never reached the figure, which is the artefact a reader actually sees.
    labels = {s: f"{'Sentinel-2' if s == 'sentinel2' else 'Landsat OLI'} "
                 f"({n_measuring[s]} informative, {len(keep_cols[s])} fed)" for s in sensors}
    titles = {"uncorrected": "A1  uncorrected gas-only radiometry\n(clean scaler, no correction)",
              "gain_corrected": "A2  oracle per-band gain removed\n(upper bound on correction)"}
    xpos = np.arange(1 + len(CWVS))
    for ax, arm in zip(axes, ("uncorrected", "gain_corrected")):
        for s in sensors:
            conds = ["clean"] + [f"{c:g}" for c in CWVS]
            vals = [_col(rows, s, "clean", "clean")] + \
                   [_col(rows, s, arm, f"cwv{c:g}") for c in CWVS]
            for x, v in zip(xpos, vals):
                ax.plot(np.full(v.size, x), v, ".", ms=2.6, alpha=0.45, color=colors[s], zorder=1)
            # yerr=None below n=2, NOT 0.0: a zero-height bar is a drawn assertion of zero
            # dispersion from a single measurement -- the same false-precision artefact the CSV
            # avoids by writing an empty cell. Routed through _sd1 so the two cannot drift apart.
            sds = [_sd1(v) for v in vals]
            ax.errorbar(xpos, [v.mean() for v in vals],
                        yerr=(sds if all(x == x for x in sds) else None),
                        fmt="-o", color=colors[s], lw=1.7, ms=4, capsize=2, zorder=2,
                        label=labels[s] if arm == "uncorrected" else None)
            ax.set_xticks(xpos); ax.set_xticklabels(conds)
        ax.set_xlim(-0.35, len(xpos) - 0.65)
        ax.set_title(titles[arm], fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mIoU (%)")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    # fig.text, not fig.supxlabel: supxlabel needs matplotlib >= 3.4 and a figure that fails to
    # render is a worse outcome than a caption placed by hand.
    fig.text(0.5, 0.02,
             "Test-time atmosphere:  clean (in-dist)  |  column water vapour (g/cm$^2$).  "
             "points = per-seed, bars = mean $\\pm$ sample SD (ddof=1) over "
             f"{len(args.seeds)} splits of one scene", ha="center", fontsize=6.5)
    fig.suptitle("Clean-trained model under 6S gaseous-transmittance (CWV) shift — spectral-only, "
                 "common grid", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    fig.savefig(P(f"figs/fig_ab_flagship{sfx}.pdf")); plt.close(fig)

    # ---- console summary ---------------------------------------------------------------------
    print(f"\n===== A+B: clean(source)-trained model under 6S gaseous-transmittance shift, no "
          f"re-fit (mean over {len(args.seeds)} seeds) =====")
    for s in sensors:
        clean = _col(rows, s, "clean", "clean").mean()
        for arm in ("uncorrected", "gain_corrected"):
            row = " ".join(f"cwv{c:g}={_col(rows, s, arm, f'cwv{c:g}').mean():.1f}"
                           f"({_col(rows, s, arm, f'cwv{c:g}').mean() / clean * 100:.0f}%)"
                           for c in CWVS)
            print(f"{s:14s}({len(keep_cols[s])}b) {arm:15s}: clean={clean:.1f} | {row}")
    print("\nPAIRED difference-in-differences (>0 = OLI degrades MORE), both estimands:")
    for p in paired:
        for k, lbl in (("abs", "mIoU pts"), ("ret", "retention")):
            print(f"  {p['arm']:15s} cwv{p['cwv']:g} {lbl:9s}: {p[f'did_{k}_mean'] or '(undef)':>7s} "
                  f"[{p[f'did_{k}_ci95_lo'] or '?':>7s}, {p[f'did_{k}_ci95_hi'] or '?':>7s}]  "
                  f"wins {p[f'did_{k}_oli_worse_wins']:>5s}  -> {p[f'did_{k}_direction']}")
    print("  ('inconclusive' means the interval spans zero; 's2_worse' is a REAL finding, not a "
          "null one. Interval is over checkerboard offsets of ONE scene -- not a scene-level claim.)")
    z = max(rows, key=lambda r: r["zshift_max"])
    print(f"\nWorst standardized band shift: {z['zshift_max']:.1f} sigma "
          f"({z['sensor']}/{z['arm']}/{z['condition']}, band index {z['zshift_max_band']}). "
          f"Compare the two arms before calling any of this information loss.")

    # ---- stamp everything ---------------------------------------------------------------------
    # scripts/doctor.py requires a sidecar for EVERY paper/results_*.csv, and the three companions
    # are deliverables in their own right (the per-seed file is what any re-analysis reads). Stamp
    # all four rather than leaving three unprovenanced tables in paper/.
    for _name in ("", "_perseed", "_bands", "_paired"):
        stamp(P(f"results_phase5_ab_flagship{_name}{sfx}.csv"), args, extra=prov)
    print(f"\nwrote: {P(f'figs/fig_ab_flagship{sfx}.pdf')}\n"
          f"       {P(f'results_phase5_ab_flagship{sfx}.csv')} (+ _perseed / _bands / _paired, "
          f"each with a .provenance.json)")


if __name__ == "__main__":
    main()
