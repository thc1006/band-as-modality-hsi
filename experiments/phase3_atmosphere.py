#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 (M&S signboard) — Design B: robustness vs 6S GASEOUS-transmittance water-vapour band loss.

Models train on CLEAN train-region pixels, then are evaluated on test pixels under a 6S gaseous
transmittance T_gas(lam; CWV). Higher column water vapour suppresses the water-vapour bands
(~940/1130 nm) harder, pushing more of them below the usability threshold -> a physics-driven
BAND-LOSS gradient. Two products are reported, and keeping them apart is essential:

  PRIMARY (corrected-product proxy). Atmospheric correction inverts the smooth multiplicative
  T_gas, but a band whose transmittance is too low to invert has no usable signal left. So the
  atmosphere's role here is ONLY to decide WHICH BANDS ARE MISSING (T below TAU_MISSING, plus the
  strong absorption cores 6S cannot be trusted in); the surviving bands keep their UNCHANGED values.
  Note precisely what that means numerically, because the wording used to overstate it: in PRIMARY
  the T VALUES never reach a feature. T enters ONLY through the threshold, so two atmospheres with
  different T but the same thresholded mask produce bit-identical model inputs (asserted in
  tests/test_phase3_atmosphere_guards.py). The CWV axis in the primary result is therefore a
  DISCRETE band-loss ladder (10 -> 19 -> 32 -> 40 bands missing at TAU=0.5 on the corrected
  2026-07-20 axis), not a continuous
  radiometric degradation. IDEALISATION: correction is assumed exact on the surviving bands; a real
  Sen2Cor-style correction leaves residual error we do not model.

  SUPPLEMENTARY (uncorrected GAS-ONLY radiometry proxy). rho*T is fed to a model trained on
  corrected reflectance. This is deliberately NOT called "L1C" anywhere any more. Sentinel-2 L1C is
  TOA reflectance, and the surface->TOA forward model needs path radiance, direct/diffuse
  transmittance, spherical albedo, Rayleigh + aerosol scattering and viewing/solar geometry; rho*T
  is ONE multiplicative gaseous factor out of that set. Calling it L1C claims a product we do not
  simulate, so the name is "uncorrected gas-only radiometry proxy" in the code, the CSV and the
  paper.
  Measured, it is not a mild extra stress but a wipe-out that swamps the band-loss signal: at
  CWV=0.5 proposed scores 39.3 under band loss alone (5 seeds, 60 epochs, corrected 2026-07-20
  axis; the numbers in results_phase3_atmosphere.csv and its .provenance.json) and collapses once
  rho*T is applied. Phase 5's two-arm design showed WHY that collapse is not information loss: a
  clean-fit scaler evaluated on rho*T measures scaler OOD, so the rho*T figure is a normalisation
  artefact and belongs with a per-band gain-corrected control arm, never quoted alone. An earlier
  revision claimed "every measured figure in this file is quoted HERE and only here"; that was
  false in its own file (the fixed-core costs, the tau grid and the sd range are all quoted further
  down) and the aspiration is dropped rather than restated. What replaces it is a TEST:
  tests/test_phase3_atmosphere_guards.py reconciles the quoted figures against the shipped CSV, so
  drift fails a check instead of relying on a discipline the file did not keep. It collapses even
  with NO bands dropped -- which shows only that the collapse is not a BAND-LOSS effect. It does
  NOT, on its own, show the cause is normalisation: zero bands dropped plus multiplicative
  suppression is still a radiometric change. The cause is normalisation rather
  than physics -- surviving bands are
  standardised with clean-train mu/sd, and while the median shift is -0.25 sigma the worst band moves
  -27 sigma, because a band with a small sd leaves the training distribution under even a modest
  multiplicative factor. The mechanism is measurable in the cube itself: the lowest-variance bands
  are indices 199, 103, 144, 198, 143, 102 with sd 7.1-15.7 against a mean near 1010-1090. Do NOT
  describe these as "the residual shoulders of the truncated water windows" -- that was checked and
  it is false for half of them: 199 and 198 sit at 2490.4 and 2480.8 nm (the SWIR terminus, where
  AVIRIS SNR is simply poor) and 144 at 1963.0 nm; only 103, 143 and 102 are inside the hard-masked
  cores. A modest T on ANY low-sd band moves it by tens of sd, wherever that band happens to be.
  AND THE SHIFT IS NOT ONE OUTLIER: measured on the seed-0 split, 14-15% of SURVIVING bands move
  more than 3 sigma at every CWV. That is systemic domain shift, which makes "artefact" the wrong
  word for it -- what this arm really measures is a model standardised on one radiometric domain
  being applied to another WITHOUT re-standardising, which is a real deployment failure mode rather
  than a bug in the experiment.
  DO NOT attach a mechanism story to the supplementary NUMBERS, in either direction. They sit at or
  below chance and they are not even monotone in CWV: b2_uncorrected runs 1.37 -> 6.28 -> 4.43 and
  proposed runs 3.21 -> 1.38 -> 0.94, so PROPOSED WINS ONLY AT CWV=0.5 and B2 "wins" at 2.0 and
  4.0. Quoting the CWV=0.5 pair alone -- which an earlier revision did -- reports the one level
  where the method looks good out of three that are all noise. Report all three or state plainly
  that the arm is at chance and no ordering is meaningful. An earlier version of this script applied
  rho*T unconditionally, which is why it appeared to show graceful degradation only while the 6S
  table's wavelength axis was ALSO wrong.

DATA PROVENANCE (this bounds what the experiment may claim). The cube is
Indian_pines_corrected.mat. "corrected" in that filename refers to the BAND SUBSET, not to
atmospheric correction: the 200-band axis is the nominal 220-band AVIRIS axis with the 20
water-absorption/noise bands removed (1-based 104-108, 150-163, 220; see bandsim.io.AVIRIS_WL_NM,
which literally np.delete()s them). Measured, the stored values are integer-valued in [955, 9604]
with mean 2652. Do not read a physical quantity off that range in EITHER direction: it is equally
consistent with scaled DN/radiance and with reflectance x 1e4 (0.0955-0.9604, mean 0.265 -- an
ordinary vegetation-scene reflectance range, and the convention Sentinel-2 L2A uses). An earlier
revision of THIS paragraph called it "calibrated-radiance-like", which is the same category of
unsupported assertion it was written to remove. The defensible statement is that the scaling
convention is UNDOCUMENTED in this repository, so the cube is not a quantity we can PROVE is a BOA
surface-reflectance product. An earlier version of this docstring asserted "Indian
Pines is itself already a surface-reflectance product" and used that to call PRIMARY an "L2A-style"
result; nothing in the repo establishes it, so both the assertion and the L2A label are gone. Treat
PRIMARY as a corrected-product PROXY on a benchmark feature cube. Two consequences that must reach
the paper:
  * the strongest water-vapour channels are ALREADY ABSENT before this script runs, so the
    "clean_full" reference is a benchmark200 ceiling, not a raw-AVIRIS full spectrum -- the writer
    below emits it as `benchmark200_full` for exactly that reason. NOTE the shipped
    paper/results_phase3_atmosphere.csv PREDATES this and still says `clean_full`, with the older
    column names and no per-seed file: it was written at commit d3866b9 from a dirty tree, and
    bandsim/{atmosphere,model,io,metrics,grouping}.py have all changed since. The VALUES quoted in
    this docstring reconcile against it, but it must be RE-RUN AND RE-STAMPED before submission --
    a reviewer reproducing it today gets a different schema;
  * the CWV ladder acts on the SHOULDERS of the water windows, not their cores (only 8 of 200 bands
    fall inside the hard-masked cores at all: 1358.9-1445.2 and 1800.0-1819.2 nm);
  * THE WAVELENGTH AXIS IS NOMINAL, NOT MEASURED. bandsim.io builds it as
    linspace(400, 2500, 220) minus the 20 removed indices, and io.py's own comment says to load a
    per-scene wavelength vector for exact calibration. Every threshold decision in this file is
    taken against that reconstructed axis. Note the tension, because a reviewer will: this script
    REFUSES a transmittance table that was interpolated onto the axis, on the grounds that
    interpolated values "are not what 6S returns at absorption edges" -- while thresholding those
    values against an axis that is itself nominal. At a water-window edge T falls from ~0.9 to ~0.1
    over some tens of nm, so a few-nm axis error moves bands across TAU. The stamped
    `wavelength_axis_sha256` records the axis's IDENTITY, which is not its CORRECTNESS.
Lead the paper's atmosphere story with the real Sen2Cor L1C->L2A transition (phase5/phase8R), which
uses a genuine L2A product, and use this as the controlled gaseous-absorption axis.

A benchmark200_full reference (nothing further removed) is reported next to clean (absorption cores
removed) so the fixed-core cost is separable from the CWV effect: measured, that fixed removal alone
costs B2 30.8 mIoU but proposed only 23.4, i.e. much of the margin predates any CWV shift.

TEST-TIME INPUT PARITY (a narrower claim than it first appears -- read the limits). The proposed
model takes an explicit group-present mask and B2 does not, so it is fair to ask whether the margin
is just an oracle telling proposed which zeros mean "missing". Measured on the shipped LUT at
TAU_MISSING=0.5, GROUP_ABSENT_FRAC=0.5 and the default 10 groups, the number of ABSENT GROUPS is 0
at clean, 0 at CWV=0.5 and 0 at CWV=2.0, and 1 at CWV=4.0. In every condition but one the
present-mask is all-True, so AT TEST TIME proposed receives exactly what B2 receives (a
zero-imputed spectrum, no indicator). This is mechanical, not merely arithmetic: with an all-present
mask the learnable mask_token is the ONLY channel through which presence can enter the forward pass,
and perturbing it to 1e6 changes the output by exactly 0.0 (asserted in the tests). At CWV=4.0 the
single absent group has its surviving bands REPLACED by the mask token, so proposed there sees
strictly LESS band content than B2 -- `proposed_nomask` in the CSV measures that arm directly.

WHAT THIS DOES **NOT** ESTABLISH, and the file must not be read as establishing it:
  * It does NOT make the margin non-oracular. The oracle is consumed during TRAINING, not at test
    time: finetune_proposed is told exactly which groups it zeroed, train_mlp is told nothing. Equal
    test-time inputs relocate that advantage, they do not remove it. Do not cite the +11.1 clean
    margin (corrected 2026-07-20 axis: proposed 47.3 against B2 36.2) as evidence that the mask
    MECHANISM works -- and note the mask is now provably inert in ALL FOUR conditions on this axis,
    since no group crosses the absence threshold at any CWV.
  * The mask is not the only asymmetry, and listing it alone would be worse than listing none.
    Proposed also gets SGMAE pretraining -- `epochs//2` = 30 extra passes, so 90 gradient epochs to
    B2's 60 -- plus a wavelength PE, plus a transformer against an MLP. (It is the SMALLER model:
    70,692 params vs 121,360.) The honest summary is that this compares two METHODS, not one
    mechanism.
  * phase2's B4 does NOT isolate the mask, and an earlier revision of this paragraph claimed it
    did, backwards. `train_hcs` passes `pm` to the model, so B4 HAS the present-mask; proposed-vs-B4
    holds the mask constant and varies PE type, SGMAE and the training mask distribution at once.
    A baseline that isolates the mask (B2 + an explicit missingness indicator) DOES NOT EXIST in
    this repository. Say so rather than pointing at a baseline that answers a different question.

THE MECHANISM IS INERT WHERE THE MARGIN IS LARGEST -- state this, do not bury it. `absent == 0` in
3 of the 4 headline conditions, and across the whole lower half of the tau grid (empty for every
CWV at tau<=0.4). So in exactly those conditions the group-mask mechanism that "band-as-modality"
names is provably doing nothing, and the margin there is evidence for "transformer + SGMAE +
wavelength PE beats an MLP under band corruption" -- a real and publishable claim, but NOT evidence
for the mask contribution. The parity result and the mechanism's inertness are the SAME FACT seen
from two sides; quoting the first as a defence while omitting the second would be advocacy.

CORRUPTION-SHIFT CONFOUND (affects every condition, and no arm here controls for it). Both models
are trained on WHOLE-group corruption -- train_mlp zeroes whole groups, finetune_proposed masks
whole groups. But no phase-3 condition ever removes a whole group: the per-group missing fractions
top out at 0.50 (CWV=4.0: [0, 0, .25, .20, .15, .50, 0, .45, 0, .30]). Every test point is a
PARTIAL, SCATTERED corruption pattern that neither model was trained on, so part of what is being
measured is which architecture tolerates an out-of-distribution corruption shape. This is
orthogonal to the parity question and survives it.

KNIFE EDGE, disclosed: that CWV=4.0 group sits at a missing fraction of EXACTLY 0.500, so `>=`
versus `>` flips it. `--group-absent-frac` is a CLI knob, it is stamped in the provenance, and any
group landing within GROUP_FRAC_EPS of the threshold is warned about at run time.

Threshold sensitivity: TAU_MISSING=0.5 is a modelling choice, NOT a physical law -- a band at
T=0.49 is not qualitatively different from one at T=0.51, and a defensible criterion would be
noise-aware (post-correction SNR / retrieval uncertainty), which this script does not model. What
the code CAN do is show the ranking is not an artefact of the knob, so `--tau-sweep` runs the
TAU x CWV grid and writes results_phase3_tau_sweep{sfx}.csv. An earlier version of this docstring
claimed the ranking "survived a sweep of TAU in {0.3, 0.5, 0.7}" -- there is NO artefact in the repo
backing that claim, so it is withdrawn until `--tau-sweep` has been run and stamped. What IS
verifiable without training anything is how the knob moves the masks: the missing-band count at
CWV=4.0 runs 23/26/37/47/57 for TAU 0.3/0.4/0.5/0.6/0.7, and the absent-group set is empty for
TAU<=0.4 everywhere, [5] at TAU=0.5, and [5,7,9] by TAU=0.7 -- so TAU is also, silently, the knob
that decides whether the group-mask mechanism engages at all.

SCOPE (honest): this is a 6S GASEOUS-transmittance / COLUMN-WATER-VAPOUR (CWV) stress-test, NOT a
full radiative-transfer ablation. 6S gaseous transmittance is aerosol-independent, so the single
degradation axis is CWV (there is no aerosol axis). Aerosol scattering, path radiance and adjacency
are NOT modelled — lead the paper's atmosphere story with the real Sen2Cor L1C->L2A transition and
use this as a controlled supplementary gaseous-absorption axis. See EXP_PHASE3_PLAN.md.

Requires the precomputed 6S table (run experiments/precompute_6s_table.py in the sixs env):
  data/srf_cache/T_6s_grid.npz   (one gaseous-transmittance column per CWV)

Reuses Phase 2 training (import from phase2_degradation).

Outputs (../paper/):
  figs/fig_robust_vs_cwv.pdf         - mIoU vs CWV (Proposed vs B2 dropout)
  results_phase3_atmosphere.csv      - aggregated table (mean/std over seeds)
  results_phase3_perseed.csv         - the RAW per-seed rows the table above is computed from, plus
                                       per-seed class support, so a reader can recompute the summary,
                                       run a PAIRED test, and see whether mIoU was averaged over the
                                       same class set in every seed (guard=1 can cost a small class)
  results_phase3_tau_sweep.csv       - only with --tau-sweep
  results_phase3_random_control.csv  - only with --random-control
--smoke writes the same artefacts under a `_smoke` suffix; it must never touch the paths above.

Usage:
  python experiments/phase3_atmosphere.py --seeds 0 1 2 3 4
  python experiments/phase3_atmosphere.py --smoke                 # 1 seed, quick (_smoke outputs)
  python experiments/phase3_atmosphere.py --tau-sweep             # TAU x CWV grid (see above)
  python experiments/phase3_atmosphere.py --random-control        # count-matched random band loss
"""
import os
import sys
import csv
import argparse
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch  # threads/device configured adaptively by bandsim.hw / bandsim.parallel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from phase2_degradation import (load_data, train_mlp, pretrain_sgmae, finetune_proposed,
                                group_present_mask, common_class_set, miou_over, NUM_CLASSES)
from bandsim.io import disjoint_block_split, AVIRIS_WL_NM, axis_sha256
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.atmosphere import (hard_mask_absorption_cores, apply_atmosphere,
                                load_cached_transmittance)
from bandsim.model import GroupedCrossBandAttention
from bandsim import hw, parallel
# bandsim.metrics.miou is no longer imported: every call goes through phase2_degradation.miou_over
# with the FIXED class set, so a stray miou() here would silently reintroduce the drifting estimand.
from bandsim.provenance import stamp, file_sha256

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)

TABLE = os.path.join(os.path.dirname(_HERE), "data", "srf_cache", "T_6s_grid.npz")
CWVS = [0.5, 2.0, 4.0]
# A band whose 6S GASEOUS transmittance falls below TAU_MISSING has lost >50% of its radiance to
# molecular absorption -> treated as effectively MISSING (physics-informed band-present mask).
# What happens to the SURVIVING bands depends on the product being modelled, and this comment used
# to state the wrong one as if it were the only one: in PRIMARY (correct_radiometry=True, the result
# the paper leads with) the survivors are left at their UNCHANGED values, T_eff=1, so T touches
# nothing except the threshold. Only in the SUPPLEMENTARY arm are they suppressed by rho'=rho*T.
# A THRESHOLD, NOT A LAW: usability really depends on SNR, calibration and retrieval uncertainty,
# so T=0.49 vs T=0.51 is not a physical discontinuity -- this is a modelling cutoff, swept with
# --tau-sweep, and it is NOT a "full 6S" claim.
TAU_MISSING = 0.5
GROUP_ABSENT_FRAC = 0.5  # a group with >=50% of its bands missing is marked absent (proposed)
# A group whose missing fraction lands this close to GROUP_ABSENT_FRAC is reported: on the shipped
# LUT the CWV=4.0 group 5 sits at EXACTLY 0.500, so `>=` vs `>` decides whether the group-mask
# mechanism engages at all in the headline worst-case condition. That is a knife edge, not a
# setting, and it must be visible in the log rather than discovered by a reviewer.
GROUP_FRAC_EPS = 0.05


def resolve_cwv_key(table, cwv):
    """The ONE key in `table` holding this CWV's transmittance column, or an error naming why not.

    6S gaseous transmittance is aerosol-independent, so the table holds ONE physical column per
    CWV; we match on the CWV prefix so this is robust to the exact key layout (a legacy table
    tagged each column with an inert atmosphere suffix). Those legacy duplicates are *supposed* to
    be identical, and this used to `return` on the FIRST match in archive order without checking:
    if they ever disagreed, the run silently picked whichever numpy happened to list first and the
    provenance recorded nothing about which one it was. Ambiguity is now resolved only if the
    candidates are bitwise-equal, and the chosen key is returned so the caller can stamp it.

    Opened with a context manager: an NpzFile holds a file descriptor until closed and this is
    called once per CWV per seed-worker."""
    pref = f"cwv{cwv}"
    with np.load(table, allow_pickle=False) as data:
        cands = [k for k in data.files if k == pref or k.startswith(pref + "_")]
        if not cands:
            raise KeyError(f"no CWV={cwv} transmittance column in {table} "
                           f"(keys={sorted(data.files)})")
        if len(cands) > 1:
            ref = np.asarray(data[cands[0]], float)
            bad = [k for k in cands[1:] if not np.array_equal(np.asarray(data[k], float), ref)]
            if bad:
                raise ValueError(
                    f"{table} has {len(cands)} columns matching CWV={cwv} ({sorted(cands)}) and "
                    f"they are NOT identical (differ: {sorted(bad)}). Refusing to pick one by "
                    f"archive order -- that choice would change every number this script reports "
                    f"and appear nowhere in the output. Regenerate the table with one column per "
                    f"CWV (experiments/precompute_6s_table.py).")
    return sorted(cands)[0]


def cwv_transmittance(table, cwv, wl=AVIRIS_WL_NM):
    """Gaseous transmittance column T_gas(lam) in [0,1] for a column-water-vapour value.

    Resolving the key via resolve_cwv_key but LOADING through
    bandsim.atmosphere.load_cached_transmittance keeps this script on the same wavelength-identity
    guard as the simulator: a bare np.load() here used to accept any 200-long column, including one
    computed on a gapless linspace axis rather than the gapped AVIRIS axis this script hard-masks
    and plots against. `require_generation_mode="direct_6s"` additionally refuses a table that was
    INTERPOLATED onto this axis -- such a table reproduces the axis and its hash exactly and is
    still not what 6S returns at the narrow absorption edges this experiment thresholds on, which
    is where the entire CWV ladder comes from."""
    return load_cached_transmittance(table, resolve_cwv_key(table, cwv), n_bands=len(wl),
                                     expected_wavelengths_nm=wl,
                                     require_generation_mode="direct_6s")


def physics_missing_bands(T, keep, *, tau_missing):
    """Bands effectively MISSING under the atmosphere: 6S gaseous transmittance below tau_missing
    (deep molecular absorption -> negligible surviving radiance) OR inside a hard-masked strong
    absorption core (6S unreliable there -> also unusable). Returns boolean missing-mask (C,).

    `tau_missing` is KEYWORD-ONLY AND REQUIRED. It used to default to the module global, which
    Python binds ONCE at def time: under --tau-sweep, which rebinds the sweep value per condition,
    every caller that omitted the argument would have kept silently thresholding at the original
    0.5 while the CSV reported the swept tau. A sweep that does not actually sweep is worse than no
    sweep, so the argument cannot be omitted.

    Modelling scope (documented, honest): this returns ONLY the band-present mask. Whether the
    surviving bands are additionally suppressed by rho'=rho*T is the CALLER's choice of product
    (see eval_condition's `correct_radiometry`); an earlier version of this docstring described the
    multiplicative suppression as part of this axis unconditionally, which is not what the primary
    result does. It is first-order (no aerosol scattering / path radiance / adjacency); see
    EXP_PHASE3_PLAN.md.
    """
    T = np.asarray(T, float)
    keep = np.asarray(keep, bool)
    # This function is importable and is called directly by the tests and by --tau-sweep, so it
    # validates its own preconditions rather than trusting the LUT loader upstream of the normal path.
    if T.ndim != 1 or keep.ndim != 1 or T.shape != keep.shape:
        raise ValueError(f"T and keep must be 1-D with the same length: got {T.shape} and "
                         f"{keep.shape} (a mismatched pair broadcasts instead of raising)")
    if not np.isfinite(T).all() or T.min() < -1e-6 or T.max() > 1.0 + 1e-6:
        raise ValueError(f"T must be a finite transmittance in [0,1], got range "
                         f"[{T.min():.3g}, {T.max():.3g}]")
    if not 0.0 <= float(tau_missing) <= 1.0:
        raise ValueError(f"tau_missing must lie in [0,1] (it is compared against a transmittance), "
                         f"got {tau_missing}")
    return (T < float(tau_missing)) | (~keep)


def control_rng(seed, cwv):
    """RNG for the count-matched random control, derived ARITHMETICALLY from (seed, condition).

    Module-level for the same reason as product_multiplier: the derivation used to be written
    inline inside run_seed, so the guard for it had to REIMPLEMENT the expression -- and a mutation
    dropping `cwv` from the seed vector (making all three CWV cells draw the IDENTICAL band set, so
    the control silently became one draw reported three times) passed the whole suite. A test that
    re-derives what it is checking is checking itself.

    NOT hash(): Python salts str/tuple hashing per process unless PYTHONHASHSEED is pinned, and
    bandsim.parallel.run_jobs SPAWNS its workers -- a hash-derived seed would give a different
    "reproducible" control on every run, and a different one per worker.
    """
    return np.random.default_rng([int(seed), int(round(float(cwv) * 10)), 3])


def product_multiplier(T, correct_radiometry):
    """The per-band multiplier T_eff applied to the surviving bands, for the chosen product.

    MODULE-LEVEL AND TINY ON PURPOSE. This one branch IS the difference between the primary and
    supplementary arms -- the whole "the atmosphere decides WHICH bands are missing, it does not
    rescale the survivors" claim reduces to whether this returns ones. It used to live inside
    `eval_condition`, a closure nested in `run_seed`, which no test can reach: the guard for the
    file's flagship claim had to REIMPLEMENT the arm inline, which made it tautological and let a
    mutation reintroducing suppression into PRIMARY pass the whole suite. Behaviour that carries a
    claim has to be callable from a test.

    correct_radiometry=True  -> ones: the survivors keep their values, T enters only via the mask.
    correct_radiometry=False -> T:    rho' = rho*T, the gas-only uncorrected proxy.
    """
    T = np.asarray(T, float)
    return np.ones_like(T) if correct_radiometry else T


def build_eval_input(Xte_raw, T_eff, missing, mu, sd):
    """Model input for one condition: suppress, mask, standardise, then impute missing as 0.

    Also module-level for testability (see product_multiplier). ORDER MATTERS and is asserted in
    the tests: a missing band must end up at standardized 0 -- the train-mean value both models
    were TRAINED to see for a missing band -- not at the raw-0 -> -mu/sd sentinel that
    (raw-mu)/sd leaves behind, which is an out-of-distribution value neither model ever saw.
    Getting that backwards mixes the suppression effect with a spurious sentinel shift.
    """
    Xte_raw_corr = apply_atmosphere(Xte_raw, T_eff, keep_mask=~missing)
    X = ((Xte_raw_corr - mu) / sd).astype(np.float32)
    X[:, missing] = 0.0
    return X


def absent_groups_for(missing, groups, group_absent_frac):
    """Groups the proposed model is told are ABSENT, plus the per-group missing fractions.

    Returned together on purpose: the fractions are what reveal a group sitting on the decision
    boundary, and the boundary is live -- at CWV=4.0 one group is at EXACTLY group_absent_frac.
    """
    fracs = np.array([float(np.asarray(missing)[np.asarray(idx)].mean()) for idx in groups])
    absent = [g for g in range(len(groups)) if fracs[g] >= group_absent_frac]
    return absent, fracs


@contextlib.contextmanager
def eval_mode(model):
    """Put a model in EVALUATION mode for the block, then restore what it was.

    `@torch.no_grad()` is NOT this. no-grad stops autograd recording; eval mode switches Dropout
    off and BatchNorm onto running statistics -- they are orthogonal, and only the second one
    changes the FUNCTION the model computes.

    This is a no-op on today's numbers because both training entry points in phase2_degradation end
    with `model.eval()`, so models ARRIVE here already in eval mode. Be careful with the weaker
    reason it is tempting to give instead -- "every module is built with dropout=0.0 and there is no
    BatchNorm, so train and eval compute the same thing". That is FALSE as stated, and measurably
    so: on the proposed model at dropout=0.0 under no_grad, train mode and eval mode differ by
    max|delta| ~3e-07, because nn.MultiheadAttention dispatches to a different fused kernel in eval.
    The difference is numerically negligible here, but the invariant that actually protects the
    result is "the model is in eval mode", not "the model has no dropout".
    Which is precisely why this is worth pinning: that guarantee is currently held by the last line
    of two functions in another module, neither of which is this function's to keep, and a model
    arriving in train mode would silently shift the reported mIoU rather than fail. Restoring the
    previous mode keeps it safe to call on a model the caller is still training.
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


@torch.no_grad()
def eval_proposed(model, Xte_std, yte, groups, absent_groups, class_set=None):
    dev = next(model.parameters()).device
    pm = group_present_mask(Xte_std.shape[0], groups, absent_groups)
    with eval_mode(model):
        pred = model(torch.from_numpy(Xte_std.astype(np.float32)).to(dev),
                     torch.from_numpy(pm).to(dev)).argmax(1).cpu().numpy()
    return miou_over(yte, pred, NUM_CLASSES, class_set)


@torch.no_grad()
def eval_mlp(model, Xte_std, yte, class_set=None):
    dev = next(model.parameters()).device
    with eval_mode(model):
        pred = model(torch.from_numpy(Xte_std.astype(np.float32)).to(dev)).argmax(1).cpu().numpy()
    return miou_over(yte, pred, NUM_CLASSES, class_set)


def run_seed(seed, cube, gt, n_groups, epochs, block=10, tau_missing=TAU_MISSING,
             group_absent_frac=GROUP_ABSENT_FRAC, tau_sweep=(), random_control=False,
             class_set=None):
    """One seed: train on clean train-region pixels, evaluate every atmosphere condition.

    `tau_missing` / `group_absent_frac` are PARAMETERS, not reads of the module globals. They used
    to be globals read inside the nested eval_condition, which is fine in-process and wrong across
    `bandsim.parallel.run_jobs`: its parallel path spawns workers that re-import this module, so a
    sweep value assigned in the parent's namespace would never reach them and every worker would
    silently threshold at the default while the CSV reported the swept value. Passing them through
    `shared=` is the only form that survives the spawn boundary.
    """
    tr_mask, te_mask = disjoint_block_split(gt, block=block, guard=1, offset=seed)
    Xtr_raw = cube[tr_mask]; ytr = gt[tr_mask].astype(int) - 1
    Xte_raw = cube[te_mask]; yte = gt[te_mask].astype(int) - 1
    mu = Xtr_raw.mean(0); sd = Xtr_raw.std(0) + 1e-8
    Xtr = ((Xtr_raw - mu) / sd).astype(np.float32)
    groups = contiguous_groups(cube.shape[-1], n_groups)
    cwl = group_center_wavelengths(AVIRIS_WL_NM, groups)
    keep = hard_mask_absorption_cores(AVIRIS_WL_NM)   # True = usable band; strong cores -> missing

    # train on clean
    m_b2 = train_mlp(Xtr, ytr, groups, seed, group_dropout=True, epochs=epochs)
    # b1 = the SAME MLP with band-group dropout OFF, so (b2 - b1) is dropout's main effect under
    # PHYSICAL (6S) degradation -- the share measured synthetically in phase2X but never under the
    # physically-grounded band loss this file applies (task #47). Same architecture, only the training
    # augmentation differs, so the contrast isolates the dropout regularizer, not the model.
    m_b1 = train_mlp(Xtr, ytr, groups, seed, group_dropout=False, epochs=epochs)
    m_prop = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES)
    pretrain_sgmae(m_prop, Xtr, groups, seed, epochs=max(1, epochs // 2))
    finetune_proposed(m_prop, Xtr, ytr, groups, seed, epochs=epochs)

    def eval_missing(missing, T_eff):
        """Score both models given an explicit band-missing mask and an explicit multiplier.

        Split out of eval_condition so the count-matched RANDOM control goes through the SAME
        imputation, standardisation and group-masking code as the physics arm. A control that
        reached the models by a different path would be comparing two code paths, not two band-loss
        POLICIES, which is the entire question it exists to answer."""
        Xte_corr = build_eval_input(Xte_raw, T_eff, missing, mu, sd)
        absent, fracs = absent_groups_for(missing, groups, group_absent_frac)
        prop = eval_proposed(m_prop, Xte_corr, yte, groups, absent, class_set)
        # THE SAME model on THE SAME input with the group-mask mechanism switched OFF (nothing
        # declared absent). One extra forward pass, no retraining. This is the ablation that
        # actually measures the mask: where `absent` is empty it is identical to `proposed` by
        # construction (and asserted so in the tests), and where it is non-empty -- only CWV=4.0 on
        # the shipped LUT -- the difference IS the mask mechanism's contribution, isolated. Without
        # it the file argues about the mask in prose while never measuring it.
        prop_nomask = prop if not absent else eval_proposed(m_prop, Xte_corr, yte, groups, [],
                                                            class_set)
        return {
            "proposed": prop,
            "proposed_nomask": prop_nomask,
            "b2": eval_mlp(m_b2, Xte_corr, yte, class_set),
            "b1": eval_mlp(m_b1, Xte_corr, yte, class_set),
            "n_absent": len(absent),
            "n_missing_bands": int(missing.sum()),
            # The DISTANCE of the closest group to the absent/present decision boundary. Reported
            # because on the shipped LUT a group sits at exactly the threshold at CWV=4.0, so this
            # number -- not the setting -- is what says whether the result is on a knife edge.
            "min_frac_margin": float(np.min(np.abs(fracs - group_absent_frac))) if len(fracs) else 1.0,
            "group_missing_fracs": [round(float(f), 4) for f in fracs],
        }

    def eval_condition(T, correct_radiometry=True, apply_core_mask=True):
        """Evaluate ONE atmosphere condition given its gaseous-transmittance column T (T=1 -> clean).

        The SAME physics-informed missing mask (deep-absorption bands < TAU_MISSING, PLUS the hard-
        masked strong-absorption cores ~keep) and the SAME train-mean (standardized-0) imputation are
        used for EVERY condition. Crucially the CLEAN reference (T=1) therefore also carries the
        core mask, so clean->CWV isolates the CWV gradient. Previously clean kept ALL bands present
        while the CWV conditions removed the cores, so the clean->CWV drop confounded a FIXED core
        removal (~keep) with the CWV shift.

        `correct_radiometry` chooses WHICH product we are modelling, and it matters enormously:

        True  (PRIMARY) -- a CORRECTED-product proxy. Atmospheric correction inverts the smooth
              multiplicative gaseous transmittance, but a band whose transmittance is too low to
              invert has no usable signal left, so the atmosphere's effect is to decide WHICH BANDS
              ARE MISSING. This is the band-loss mechanism the paper is about. Be precise about the
              consequence, which the previous wording blurred: T_eff is ONES here, so the T VALUES
              never enter a feature and the CWV axis is a discrete band-loss ladder rather than a
              radiometric gradient. This arm is NOT labelled "L2A-style" any more -- the cube is a
              benchmark feature cube whose values (integer, 955-9604) this repository cannot prove
              are BOA surface reflectance; see DATA PROVENANCE in the module docstring.
        False (SUPPLEMENTARY) -- an uncorrected GAS-ONLY radiometry proxy: rho*T is fed to a model
              trained on corrected reflectance. NOT "L1C": L1C is TOA reflectance and needs path
              radiance, spherical albedo, Rayleigh/aerosol scattering and geometry, none of which
              are here. Measured, this is not a mild extra stress but a total wipe-out that swamps
              the band-loss signal entirely -- see the module docstring for the measured figures,
              quoted there once so the two places cannot drift apart. The collapse persists with NO
              bands dropped at all, which is what identifies it as a normalisation artefact rather
              than a physical effect. The cause is normalisation, not physics -- surviving bands are
              standardised with clean-train mu/sd, and while the median shift is only -0.25 sigma the
              worst band moves -27 sigma, because bands with a small sd are thrown far out of
              distribution by even a modest multiplicative factor. Report it as an uncorrected-
              radiometry sensitivity, never as the atmosphere->band-loss result.
        """
        # apply_core_mask=False drops ONLY the T-driven bands, giving the untouched benchmark200
        # ceiling used for the benchmark200_full reference below.
        core = keep if apply_core_mask else np.ones(keep.shape, bool)
        missing = physics_missing_bands(T, core, tau_missing=tau_missing)
        # Deep-absorption / core bands are hard-masked to 0 (missing) in both modes; the surviving
        # bands are left at their corrected values (T_eff=1) or suppressed by rho*T (T_eff=T).
        return eval_missing(missing, product_multiplier(T, correct_radiometry))

    def eval_random_control(n_missing, rng):
        """Count-matched RANDOM band loss: drop `n_missing` bands drawn uniformly, nothing else.

        The control for the question the physics arm cannot answer on its own -- is the margin
        specific to WHICH bands the atmosphere takes, or is it generic missing-band robustness at
        the same COUNT? The physics mask is not a random set: it is contiguous, concentrated on the
        water-window shoulders, and it removes low-variance bands. If proposed beats B2 by the same
        margin under a uniform mask of equal size, then "atmospheric robustness" is the wrong name
        for the finding and "missing-band robustness" is the right one. Radiometry is left
        untouched (T_eff=1) so this is matched to the PRIMARY arm specifically.

        READ THE DIRECTION CAREFULLY -- this control is NOT neutral, and mistaking its sign for a
        result would be worse than not running it. The uniform arm matches the physics arm on COUNT
        ONLY, and the counts include the 8 fixed absorption-core bands the physics arm removes in
        EVERY condition. Those 8 are not a random sample: their variance sits low in the cube
        (sd-ranks 1, 4, 5, 7, 14 of 200 for five of them) though NOT uniformly so (ranks 32, 34 and
        60 for the other three). NO a-priori direction is asserted from that, for two reasons. First,
        variance is not information: this file's own headline says removing exactly those 8 bands
        costs B2 30.8 mIoU, which is direct evidence they matter, and an earlier revision of this
        paragraph called them "among the least informative in the cube" while the same file reported
        that 19.4 -- a flat self-contradiction. Second, a prediction that reads "lower is as
        expected, equal means generic robustness, higher is unexplained" is unfalsifiable and would
        launder any outcome into a confirmation.
        The interpretable quantity is the PAIRED GAP (proposed - b2) under each policy, not either
        policy's absolute mIoU: if the gap is the same under both, the identity of the lost bands is
        not what the method exploits. A defensible alternative -- holding the 8 cores masked in both arms and
        randomising only the CWV-driven increment -- would answer the narrower question "does the
        identity of the CWV increment matter"; this one answers "does the identity of the whole
        missing set matter". Neither is more correct, but they are different questions and the CSV
        must not be read as if it answered the other one.

        CONFOUND AT CWV=4.0, DISCLOSED -- do not read that cell as a clean policy comparison. A
        uniform draw spreads its bands evenly (~n/G per group), so it essentially never reaches the
        0.5 absent threshold: simulated over 200 draws at the matched counts, the uniform arm gets
        0 absent groups at every ladder step on the corrected axis (fractions top out at 0.40, so
        the group-mask mechanism never engages at the default threshold). Historical, stretched-axis
        note: the physics arm had 1 absent
        group at CWV=4.0. So in the ONE condition where the group-mask mechanism engages at all, the
        two arms differ in band identity AND in whether the mask is active -- the paired gap there
        confounds "which bands" with "mask on vs mask off". `proposed_nomask` is written alongside
        precisely so that cell can be read with the mechanism held off in both arms; use it, or
        restrict the policy claim to CWV=0.5 and 2.0 where both arms have 0 absent groups.
        """
        C = Xte_raw.shape[1]
        missing = np.zeros(C, bool)
        missing[rng.choice(C, size=int(n_missing), replace=False)] = True
        return eval_missing(missing, np.ones(C, float))

    # clean reference = atmosphere OFF (T=1 everywhere) but the SAME hard core-mask as CWV (see
    # eval_condition): T=1 -> only ~keep bands are "missing", so no CWV suppression, cores masked.
    clean = eval_condition(np.ones_like(keep, dtype=float))
    clean["proposed_uncorrected"] = clean["proposed"]     # T=1: correction is a no-op, so identical
    clean["b2_uncorrected"] = clean["b2"]
    clean["b1_uncorrected"] = clean["b1"]
    # benchmark200_full = the untouched ceiling, nothing FURTHER removed. Reported so a reader can
    # separate the cost of the FIXED absorption-core removal from the CWV-driven band loss. Measured
    # over 5 seeds that fixed removal alone costs B2 30.8 mIoU (67.0 -> 36.2) but proposed only 23.4
    # (71.4 -> 64.9), so most of the method's margin is already established BEFORE any CWV shift.
    # Presenting the whole clean->CWV gap as an atmosphere effect would overstate what CWV does.
    # NOT a raw-AVIRIS full spectrum: the benchmark already lacks 20 water/noise bands (see DATA
    # PROVENANCE), which is why this condition is no longer called "clean_full".
    full = eval_condition(np.ones_like(keep, dtype=float), apply_core_mask=False)
    clean["proposed_full"] = full["proposed"]
    clean["b2_full"] = full["b2"]
    clean["b1_full"] = full["b1"]
    res = {"clean": clean}
    for cwv in CWVS:
        T = cwv_transmittance(TABLE, cwv)
        r = eval_condition(T, correct_radiometry=True)    # PRIMARY: atmosphere -> band loss
        # Carry the uncorrected-radiometry variant alongside rather than in a separate run, so the
        # two mechanisms are reported together and can never silently be conflated again.
        u = eval_condition(T, correct_radiometry=False)
        r["proposed_uncorrected"] = u["proposed"]
        r["b2_uncorrected"] = u["b2"]
        r["b1_uncorrected"] = u["b1"]
        res[cwv] = r

    # Per-seed class support. guard=1 can leave a small class with no TEST pixels in some seeds, and
    # `miou` averages over classes PRESENT IN y_true -- so two seeds can be averaging over different
    # class sets, and the std across seeds would silently mix "harder split" with "fewer classes".
    # Recorded per seed so that is auditable instead of assumed.
    res["_meta"] = {
        "seed": int(seed),
        "n_test": int(yte.size),
        "n_train": int(ytr.size),
        "classes_in_test": int(np.unique(yte).size),
        "class_support": {int(c): int(n) for c, n in zip(*np.unique(yte, return_counts=True))},
        "tau_missing": float(tau_missing),
        "group_absent_frac": float(group_absent_frac),
    }

    # --tau-sweep: the SAME trained models re-evaluated at other thresholds. Re-using the models is
    # correct AND cheap -- tau is a TEST-TIME masking choice that never touches training, so
    # retraining per tau would only add seed noise to a comparison that is meant to isolate tau.
    if tau_sweep:
        sweep = []
        for tau in tau_sweep:
            for cwv in CWVS:
                T = cwv_transmittance(TABLE, cwv)
                m = physics_missing_bands(T, keep, tau_missing=tau)
                r = eval_missing(m, np.ones_like(T, dtype=float))
                sweep.append({"tau": float(tau), "cwv": float(cwv), **r})
        res["_tau_sweep"] = sweep

    # --random-control: uniform band loss matched to each CWV's band COUNT (see eval_random_control).
    if random_control:
        ctrl = []
        for cwv in CWVS:
            n_missing = res[cwv]["n_missing_bands"]
            # Seeded from BOTH the seed and the condition (see control_rng) so each cell is an
            # independent draw and the whole control is reproducible from the provenance alone.
            r = eval_random_control(n_missing, control_rng(seed, cwv))
            ctrl.append({"cwv": float(cwv), "n_missing_bands": int(n_missing), **r})
        res["_random_control"] = ctrl
    return res


def band_loss_geometry_report(tau_missing, group_absent_frac, n_groups, table=TABLE):
    """Lines describing the band-loss geometry, INCLUDING any knife-edge warning.

    Returns the lines instead of printing them so a test can read what the run would tell a reader.
    Deleting the knife-edge branch used to leave the suite green: a warning nothing asserts on is a
    warning that can silently stop firing, and this one carries the disclosure that the CWV=4.0
    result flips on `>=` versus `>`.

    Computed before training because it depends only on the LUT and the thresholds -- it is what
    decides whether the proposed model's group-mask mechanism engages at all.
    """
    keep = hard_mask_absorption_cores(AVIRIS_WL_NM)
    groups = contiguous_groups(len(AVIRIS_WL_NM), n_groups)
    out = []
    for c in CWVS:
        m = physics_missing_bands(cwv_transmittance(table, c), keep, tau_missing=tau_missing)
        absent, fr = absent_groups_for(m, groups, group_absent_frac)
        margin = float(np.min(np.abs(fr - group_absent_frac)))
        out.append(f"  CWV={c}: {int(m.sum())}/{len(keep)} bands missing, absent groups {absent}")
        # `<=` with an explicit tolerance, NOT a bare `<`. At the shipped defaults CWV=2.0's closest
        # group sits at 0.45, and abs(0.45-0.5) evaluates to 0.049999999999999996 -- so a bare
        # `< 0.05` fires there by floating-point accident rather than by decision. Firing IS the
        # intent (0.45 is genuinely within one eps of the threshold); making it deterministic means
        # the warning does not appear or vanish on a rounding change upstream.
        if margin <= GROUP_FRAC_EPS + 1e-12:
            out.append(
                f"    [knife edge] a group sits {margin:.3f} from the absent threshold "
                f"(fracs={[round(f, 3) for f in fr]}). '>=' vs '>' flips whether the proposed "
                f"model is told a group is absent, so this condition's result is NOT robust to "
                f"an arbitrarily small change in --group-absent-frac. Report it as such.")
    return out


def seed_sd(values):
    """Dispersion ACROSS SEEDS: the sample sd (ddof=1), or NaN below two seeds.

    `np.std` defaults to ddof=0, the POPULATION formula, which treats the seeds as the entire
    population rather than a sample of the split x initialisation space we are generalising over.
    It is too small by sqrt((n-1)/n) -- 18% at the 3 seeds this phase ships, 10.6% at 5. An earlier
    revision named the column `_std_pop` and documented the choice, which is honest but still the
    wrong estimator for the quantity: every `+/-` a reader interprets as run-to-run variability is
    understated.

    NaN rather than 0.0 for a single seed. ddof=1 with n=1 is a 0/0, and reporting a spread of
    exactly zero for one run reads as perfect agreement rather than as no information -- which is
    what a --smoke run (1 seed) would otherwise print.
    """
    v = np.asarray(values, float)
    if v.size < 2:
        return float("nan")
    return float(v.std(ddof=1))


def summary_rows(agg, absent_info, missing_info, conds, n_seeds):
    """Header + data rows of the summary CSV, as a list, so the ROWS can be asserted on.

    Returning rows rather than writing them lets a test check the labels and arithmetic that
    actually reach the artefact. That matters here because the previous guard for the
    `clean_full` -> `benchmark200_full` rename searched the SOURCE for the old label -- which
    `"clean" + "_full"` evades, and which says nothing about what the writer emits anyway.

    The *_uncorrected columns are the SUPPLEMENTARY no-atmospheric-correction variant. They are
    written next to the primary numbers on purpose: reported alone they look like an atmosphere
    result, when in fact they are dominated by a normalisation artefact (see eval_condition).
    *_std_sample is the SAMPLE sd over seeds (ddof=1; see seed_sd) -- it estimates the run-to-run
    variability we generalise over, so the population formula would understate it by 18% at 3 seeds.
    It is still a spread, not a significance statement: the seeds vary the split offset AND the
    model init together, so it mixes both sources, and n is small. `paired_diff` is the statistic
    for "proposed > B2".
    """
    rows = [["condition", "cwv", "n_absent_groups", "n_missing_bands",
             "proposed_miou_mean", "proposed_miou_std_sample", "b2_miou_mean", "b2_miou_std_sample",
             "proposed_retention", "b2_retention",
             "proposed_gasonly_uncorrected_mean", "b2_gasonly_uncorrected_mean",
             # The mask-mechanism ablation: the SAME proposed model on the SAME input with nothing
             # declared absent. Equal to proposed_miou_mean wherever n_absent_groups==0 (i.e.
             # everywhere but CWV=4.0 on the shipped LUT), which is itself the finding: the
             # mechanism is inert in most conditions.
             "proposed_nomask_mean", "paired_diff_mean", "n_seeds",
             # task #47: dropout's share of the plain-MLP->proposed gap UNDER PHYSICAL degradation.
             # b1 = MLP no group-dropout; dropout_main_effect = b2 - b1; share = (b2-b1)/(proposed-b1).
             "b1_miou_mean", "b1_miou_std_sample", "dropout_main_effect_mean",
             "proposed_minus_b1_mean", "dropout_share_pct"]]

    def _b1cols(prop_list, b2_list, b1_list):
        pm, b2m, b1m = np.mean(prop_list), np.mean(b2_list), np.mean(b1_list)
        de = b2m - b1m; tot = pm - b1m
        share = de / tot * 100 if abs(tot) > 1e-6 else float("nan")
        return [f"{b1m:.2f}", f"{seed_sd(b1_list):.2f}", f"{de:.2f}", f"{tot:.2f}",
                (f"{share:.1f}" if np.isfinite(share) else "-")]

    pclean = np.mean(agg["clean"]["proposed"]); bclean = np.mean(agg["clean"]["b2"])
    # Written FIRST so the ceiling is the reader's first reference point: the clean row below
    # already carries the fixed absorption-core removal, and that removal is not free.
    # Named benchmark200_full, NOT clean_full: the benchmark cube has already had 20 water/noise
    # bands removed upstream, so "0 removed" is relative to the 200-band benchmark, not to a raw
    # AVIRIS spectrum (see DATA PROVENANCE in the module docstring).
    pf = agg["clean"]["proposed_full"]; bf = agg["clean"]["b2_full"]; b1f = agg["clean"]["b1_full"]
    # "-" for the uncorrected columns: T=1 makes correction a no-op here, so there is no
    # uncorrected variant to report. Writing the CORRECTED value under an *_uncorrected header
    # (which this row used to do) silently hands a corrected number to anyone aggregating it.
    rows.append(["benchmark200_full", "-", 0, 0, f"{np.mean(pf):.2f}", f"{seed_sd(pf):.2f}",
                 f"{np.mean(bf):.2f}", f"{seed_sd(bf):.2f}", "-", "-",
                 "-", "-", f"{np.mean(pf):.2f}",
                 f"{np.mean(np.asarray(pf) - np.asarray(bf)):.2f}", n_seeds] + _b1cols(pf, bf, b1f))
    for cond in conds:
        p = agg[cond]["proposed"]; b = agg[cond]["b2"]
        pu = np.mean(agg[cond]["proposed_uncorrected"]); bu = np.mean(agg[cond]["b2_uncorrected"])
        pnm = np.mean(agg[cond]["proposed_nomask"])
        pd = np.mean(np.asarray(p) - np.asarray(b))
        if cond == "clean":
            rows.append(["clean", "-", absent_info.get("clean", 0), missing_info.get("clean", 0),
                         f"{np.mean(p):.2f}", f"{seed_sd(p):.2f}",
                         f"{np.mean(b):.2f}", f"{seed_sd(b):.2f}", "1.00", "1.00",
                         f"{pu:.2f}", f"{bu:.2f}", f"{pnm:.2f}", f"{pd:.2f}", n_seeds]
                        + _b1cols(p, b, agg[cond]["b1"]))
        else:
            rows.append([f"cwv{cond}", cond, absent_info.get(cond, 0), missing_info.get(cond, 0),
                         f"{np.mean(p):.2f}", f"{seed_sd(p):.2f}",
                         f"{np.mean(b):.2f}", f"{seed_sd(b):.2f}",
                         f"{np.mean(p)/pclean:.2f}", f"{np.mean(b)/bclean:.2f}",
                         f"{pu:.2f}", f"{bu:.2f}", f"{pnm:.2f}", f"{pd:.2f}", n_seeds]
                        + _b1cols(p, b, agg[cond]["b1"]))
    return rows


def check_results_complete(results, seeds):
    """Refuse a results list that cannot be zipped safely against `seeds`.

    `bandsim.parallel.run_jobs` documents one result per item in item order, and its parallel path
    pre-allocates by index and re-raises worker exceptions, so a short list should be impossible.
    The guard exists because the CONSUMER is a zip(), and zip() TRUNCATES SILENTLY: if that
    contract ever weakened, the failure mode is a table averaged over fewer seeds than its own
    provenance claims, which no reader could detect. Module-level so the guard itself is testable
    -- deleting it used to leave the suite green.
    """
    if len(results) != len(seeds):
        raise RuntimeError(f"run_jobs returned {len(results)} results for {len(seeds)} seeds; "
                           f"refusing to zip() them into a silently short-averaged table")
    if any(r is None for r in results):
        raise RuntimeError(f"run_jobs returned a None result for seed(s) "
                           f"{[s for s, r in zip(seeds, results) if r is None]}")
    return results


def check_mask_invariant(cond, seed, res_cond, absent_info, missing_info):
    """The missing mask is a function of (LUT, tau, keep, groups) ONLY -- never of the seed.

    So `n_absent` and `n_missing_bands` are INVARIANTS across seeds, not per-seed observations.
    They used to be overwritten by each seed in turn, which reports the last seed's value as
    everyone's. Assert the invariant rather than assuming it; module-level so the assertion is
    itself reachable from a test.
    """
    if cond in absent_info and absent_info[cond] != res_cond["n_absent"]:
        raise RuntimeError(
            f"condition {cond}: seed {seed} sees {res_cond['n_absent']} absent groups but an "
            f"earlier seed saw {absent_info[cond]}. The missing mask is a function of the LUT and "
            f"tau ONLY, so this means something seed-dependent has leaked into it.")
    if cond in missing_info and missing_info[cond] != res_cond["n_missing_bands"]:
        raise RuntimeError(
            f"condition {cond}: seed {seed} sees {res_cond['n_missing_bands']} missing bands but "
            f"an earlier seed saw {missing_info[cond]} (see above -- the mask must not depend on "
            f"the seed)")


def provenance_extra(args, cube, absent_info, missing_info, class_set=None, class_present=None):
    """The `extra` payload stamped beside the results CSV.

    A FUNCTION, not a dict literal inside main(), so a test can assert on the PAYLOAD rather than on
    the source text that builds it. That distinction is not pedantic: the earlier guard checked
    `"cube_physical_quantity" in <source>`, which a mutation satisfied by deleting the entire stamp
    and leaving the words in a comment, and a second mutation satisfied while inverting the caveat
    to claim the cube IS verified reflectance. A magic-word search over source is not a check on
    what gets written.

    The INPUT CUBE is hashed here. Its identity was previously assumed from a filename, and the
    filename is exactly what misled this script into calling the cube a surface-reflectance product
    -- "corrected" there means the 200-band subset. A digest plus the measured value range lets a
    reader check which artefact was actually read, and check the claim independently.
    """
    dd = os.path.join(os.path.dirname(_HERE), "data", "indian_pines")
    cube_path = os.path.join(dd, "Indian_pines_corrected.mat")
    gt_path = os.path.join(dd, "Indian_pines_gt.mat")
    return {
        "lut": TABLE, "lut_sha256": file_sha256(TABLE),
        # WHICH column was used, not merely which file. resolve_cwv_key refuses ambiguous matches,
        # and recording the resolution makes that check auditable after the fact.
        "lut_keys_used": {str(c): resolve_cwv_key(TABLE, c) for c in CWVS},
        "lut_generation_mode": "direct_6s (required; a resampled table is refused)",
        "wavelength_axis_sha256": axis_sha256(AVIRIS_WL_NM),
        "wavelength_axis_provenance": (
            "NOMINAL, NOT MEASURED: linspace(400,2500,220) minus the 20 removed indices "
            "(bandsim.io.AVIRIS_WL_NM). Every TAU threshold decision is taken against this "
            "reconstructed axis. The sha256 above records its IDENTITY, not its CORRECTNESS."),
        "n_bands": int(len(AVIRIS_WL_NM)),
        "cube_path": cube_path, "cube_sha256": file_sha256(cube_path),
        "gt_path": gt_path, "gt_sha256": file_sha256(gt_path),
        "cube_value_range": [float(np.min(cube)), float(np.max(cube))],
        # MACHINE-READABLE, not just prose. A mutation that rewrote the sentence below to claim the
        # cube IS verified reflectance -- while keeping the word "UNVERIFIED" in it -- passed a
        # guard that searched the prose for that word. A downstream reader should key on a boolean,
        # and a test should assert on one; inverting this flag has to be deliberate.
        "cube_physical_quantity_verified": False,
        "cube_physical_quantity": (
            "UNVERIFIED. Benchmark feature cube; integer-valued, range recorded above. That range "
            "is consistent with scaled DN/radiance AND with reflectance x 1e4, so it discriminates "
            "neither. This repository does NOT establish it as a BOA surface-reflectance product, "
            "so the primary arm is a corrected-product PROXY and is not labelled L2A."),
        "benchmark_bands_removed_upstream": (
            "20 of the nominal 220 AVIRIS bands (1-based 104-108, 150-163, 220) are already absent "
            "from this cube, so benchmark200_full is a 200-band ceiling, not a raw full spectrum. "
            "See bandsim.io.AVIRIS_WL_NM."),
        "tau_missing": args.tau_missing,
        "group_absent_frac": args.group_absent_frac,
        "tau_sweep": args.tau_sweep or None,
        "random_control": bool(args.random_control),
        "n_absent_groups_per_condition": {str(k): int(v) for k, v in absent_info.items()},
        "n_missing_bands_per_condition": {str(k): int(v) for k, v in missing_info.items()},
        # The answer to "is the proposed model's present-mask an oracle B2 lacks?" -- recorded as
        # data rather than argued in prose. Where this is 0 the mask is all-True.
        "information_parity_note": (
            "n_absent_groups==0 means the proposed model's group-present mask is all-True, i.e. AT "
            "TEST TIME it receives exactly what B2 receives (zero-imputed spectrum, no missingness "
            "indicator). Where it is >0 the absent group's surviving bands are replaced by a mask "
            "token, so proposed sees strictly LESS band content than B2, not more. This does NOT "
            "make the margin non-oracular: the mask is consumed during TRAINING, where proposed is "
            "told which groups it dropped and B2 is not. See proposed_nomask for the ablation."),
        # The macro estimand, recorded so a reader knows WHICH classes the mIoU averages over.
        # Without it, "mIoU" names two different quantities across phases and nothing in the
        # artefact says so.
        "macro_class_set_0based": list(class_set) if class_set is not None else None,
        "macro_n_classes": len(class_set) if class_set is not None else None,
        "macro_class_set_note": (
            "mIoU is averaged over the classes present in EVERY seed's test split "
            "(phase2_degradation.common_class_set), matching phases 1 and 2. Averaging over "
            "whatever each split happens to contain makes the mean over seeds a mean of different "
            "estimands."),
        "classes_per_split_count": (list(map(int, class_present))
                                    if class_present is not None else None),
        "conditions": ["benchmark200_full", "clean(core-masked)"] + [f"cwv{c}" for c in CWVS],
    }


def validate_args(args, n_bands=None):
    """Reject configurations that would produce a plausible WRONG number instead of crashing.

    Raises ValueError; main() converts that to `ap.error`. Split out of main() so it is reachable
    from a test: a mutation deleting every one of these checks used to leave the whole suite green,
    because main() is a monolith no test can enter. Also normalises `--tau-sweep` (fills the default
    grid, dedups, sorts) IN PLACE, so the normalisation is testable for the same reason.

    n_bands defaults to this script's own axis; passed explicitly by the tests.
    """
    n_bands = len(AVIRIS_WL_NM) if n_bands is None else n_bands
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"--seeds must be distinct (got {args.seeds}); a repeated seed trains the "
                         f"identical model twice and would be counted as two independent "
                         f"replicates in the std")
    if args.groups < 2:
        raise ValueError("--groups must be >= 2: with a single group, group dropout and SGMAE group "
                         "masking are both no-ops, so 'proposed' would silently stop being the "
                         "method under test")
    if args.groups > n_bands:
        # Query the axis, do not hardcode 200: the constant and the axis are two things that can
        # disagree, and the axis is the one contiguous_groups() actually partitions.
        raise ValueError(f"--groups must not exceed the band count ({n_bands})")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if getattr(args, "jobs", None) is not None and args.jobs <= 0:
        raise ValueError("--jobs must be > 0")
    if not 0.0 <= args.tau_missing <= 1.0:
        raise ValueError("--tau-missing is compared against a transmittance and must lie in [0,1]")
    if not 0.0 <= args.group_absent_frac <= 1.0:
        raise ValueError("--group-absent-frac is a fraction of a group's bands and must lie in [0,1]")
    if args.tau_sweep is not None:
        if not args.tau_sweep:
            args.tau_sweep = [0.3, 0.4, 0.5, 0.6, 0.7]
        if any(not 0.0 <= t <= 1.0 for t in args.tau_sweep):
            raise ValueError("--tau-sweep values must all lie in [0,1]")
        args.tau_sweep = sorted(set(args.tau_sweep))
    return args


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
    ap.add_argument("--tau-missing", type=float, default=TAU_MISSING,
                    help=f"transmittance below which a band counts as missing (default {TAU_MISSING})")
    ap.add_argument("--group-absent-frac", type=float, default=GROUP_ABSENT_FRAC,
                    help=f"missing-band fraction at which a GROUP is declared absent to the proposed "
                         f"model (default {GROUP_ABSENT_FRAC}; on the shipped LUT one group sits at "
                         f"exactly this value at CWV=4.0)")
    ap.add_argument("--tau-sweep", type=float, nargs="*", default=None,
                    metavar="TAU",
                    help="also evaluate these thresholds (default grid if given with no values) and "
                         "write results_phase3_tau_sweep.csv")
    ap.add_argument("--random-control", action="store_true",
                    help="also evaluate count-matched UNIFORM band loss, to separate "
                         "atmosphere-specific from generic missing-band robustness")
    args = ap.parse_args()
    try:
        validate_args(args)
    except ValueError as e:
        ap.error(str(e))
    # A --smoke run shared its output paths with a full run, so a 1-seed sanity check silently
    # replaced the 5-seed CWV table the paper quotes — and nothing in the CSV recorded the swap.
    # Route smoke output to its own names; the deliverables below stay whatever produced them.
    sfx = ""
    if args.smoke:
        args.seeds = [0]; args.epochs = 12
        sfx = "_smoke"
        print("[smoke] 1 seed / 12 epochs — writing *_smoke artefacts, NOT the real deliverables")
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device); print('HW:', hw.info())

    if not os.path.exists(TABLE):
        print(f"ERROR: missing 6S table {TABLE}. Run precompute_6s_table.py in the sixs env.")
        sys.exit(1)

    # Report the band-loss geometry BEFORE training anything: it depends only on the LUT and the
    # thresholds, it is what decides whether the proposed model's group-mask mechanism engages at
    # all, and finding it in the log beats deriving it from the results afterwards.
    print(f"band-loss geometry (tau={args.tau_missing}, group_absent_frac={args.group_absent_frac}):")
    for line in band_loss_geometry_report(args.tau_missing, args.group_absent_frac, args.groups):
        print(line)

    cube, gt = load_data()
    # FIXED MACRO CLASS SET, shared with phases 1/2 (phase2_degradation.common_class_set).
    # `miou` averages over whatever classes the split happens to contain, and each seed shifts the
    # checkerboard, so the mean over seeds was averaging DIFFERENT ESTIMANDS. Measured on this
    # config: seeds 0/1/2 each see 15 of 16 classes but not the same 15, while seeds 3/4 see all 16.
    # The common set is 14 of 16 for both the 3-seed and 5-seed lists, dropping Grass-pasture-mowed
    # and Oats -- which have 2-8 test pixels, so their per-class IoU is uninterpretable anyway and
    # was contributing pure noise to the mean. Phases 1 and 2 already score this way; until phase 3
    # did too, its mIoU was not comparable with theirs in the same table.
    class_set, class_present = common_class_set(gt, 10, args.seeds, guard=1, num_classes=NUM_CLASSES)
    print(f"macro class set: {len(class_set)}/{NUM_CLASSES} classes present in EVERY seed's test "
          f"split (dropped: {[c + 1 for c in range(NUM_CLASSES) if class_present[c] != len(args.seeds)]}"
          f", 1-based) — every seed now reports the same estimand")
    conds = ["clean"] + list(CWVS)
    agg = {cond: {"proposed": [], "proposed_nomask": [], "b2": [], "b1": [],
                  "proposed_uncorrected": [], "b2_uncorrected": [], "b1_uncorrected": []}
           for cond in conds}
    absent_info = {}
    missing_info = {}
    import time
    t0 = time.time()
    results = parallel.run_jobs(
        run_seed, args.seeds,
        shared=dict(cube=cube, gt=gt, n_groups=args.groups, epochs=args.epochs,
                    tau_missing=args.tau_missing, group_absent_frac=args.group_absent_frac,
                    tau_sweep=tuple(args.tau_sweep or ()), random_control=args.random_control,
                    class_set=class_set),
        prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
        label="phase3/seed")
    check_results_complete(results, args.seeds)
    per_seed_rows = []
    for sd, res in zip(args.seeds, results):
        for cond in conds:
            agg[cond]["proposed"].append(res[cond]["proposed"])
            agg[cond]["b2"].append(res[cond]["b2"])
            agg[cond]["proposed_nomask"].append(res[cond]["proposed_nomask"])
            agg[cond]["b1"].append(res[cond]["b1"])
            agg[cond]["proposed_uncorrected"].append(res[cond]["proposed_uncorrected"])
            agg[cond]["b2_uncorrected"].append(res[cond]["b2_uncorrected"])
            agg[cond]["b1_uncorrected"].append(res[cond]["b1_uncorrected"])
            # clean now carries the SAME hard core-mask (see run_seed), so report ITS missing count
            # too rather than hardcoding 0 (the cores are masked in every condition, incl. clean).
            # These counts are seed-INVARIANTS, not per-seed observations (see check_mask_invariant).
            check_mask_invariant(cond, sd, res[cond], absent_info, missing_info)
            absent_info[cond] = res[cond]["n_absent"]
            missing_info[cond] = res[cond]["n_missing_bands"]
            meta = res["_meta"]
            per_seed_rows.append({
                "seed": sd, "condition": "clean" if cond == "clean" else f"cwv{cond}",
                "cwv": "-" if cond == "clean" else cond,
                "proposed_miou": res[cond]["proposed"], "b2_miou": res[cond]["b2"],
                "b1_miou": res[cond]["b1"],
                # dropout's PHYSICAL-degradation main effect for this seed/condition (task #47)
                "dropout_main_effect": res[cond]["b2"] - res[cond]["b1"],
                # The PAIRED difference, per seed. This -- not two independent means with SD bars --
                # is the statistic a claim of "proposed > B2" rests on, because both models share the
                # split, the standardisation and the mask within a seed.
                "paired_diff": res[cond]["proposed"] - res[cond]["b2"],
                "proposed_nomask": res[cond]["proposed_nomask"],
                "proposed_uncorrected": res[cond]["proposed_uncorrected"],
                "b2_uncorrected": res[cond]["b2_uncorrected"],
                "n_missing_bands": res[cond]["n_missing_bands"],
                "n_absent_groups": res[cond]["n_absent"],
                "min_group_frac_margin": res[cond]["min_frac_margin"],
                "n_test_px": meta["n_test"], "classes_in_test": meta["classes_in_test"],
            })
        # benchmark200_full lives only on the clean condition (the no-removal ceiling, not a CWV level)
        agg["clean"].setdefault("proposed_full", []).append(res["clean"]["proposed_full"])
        agg["clean"].setdefault("b2_full", []).append(res["clean"]["b2_full"])
        agg["clean"].setdefault("b1_full", []).append(res["clean"]["b1_full"])
        per_seed_rows.append({
            "seed": sd, "condition": "benchmark200_full", "cwv": "-",
            "proposed_miou": res["clean"]["proposed_full"], "b2_miou": res["clean"]["b2_full"],
            "b1_miou": res["clean"]["b1_full"],
            "dropout_main_effect": res["clean"]["b2_full"] - res["clean"]["b1_full"],
            "paired_diff": res["clean"]["proposed_full"] - res["clean"]["b2_full"],
            # No band is missing here, so the mask mechanism cannot engage: nomask == proposed.
            "proposed_nomask": res["clean"]["proposed_full"],
            # "-", NOT the corrected value. This row has no uncorrected variant (T=1 makes the two
            # identical), and writing the corrected number under an *_uncorrected header hands a
            # corrected figure to anyone who aggregates that column.
            "proposed_uncorrected": "-", "b2_uncorrected": "-", "b1_uncorrected": "-",
            "n_missing_bands": 0, "n_absent_groups": 0, "min_group_frac_margin": "-",
            "n_test_px": res["_meta"]["n_test"],
            "classes_in_test": res["_meta"]["classes_in_test"],
        })
        print(f"seed {sd}: clean prop={res['clean']['proposed']:.1f} "
              f"| cwv4 prop={res[4.0]['proposed']:.1f} b2={res[4.0]['b2']:.1f}")
    print(f"(all {len(args.seeds)} seeds in {time.time()-t0:.1f}s)")

    # ---- csv ----
    with open(P(f"results_phase3_atmosphere{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for row in summary_rows(agg, absent_info, missing_info, conds, len(args.seeds)):
            w.writerow(row)

    # ---- per-seed raw rows: the table above must be RECOMPUTABLE from these ----
    with open(P(f"results_phase3_perseed{sfx}.csv"), "w", newline="") as f:
        cols = ["seed", "condition", "cwv", "proposed_miou", "proposed_nomask", "b2_miou", "b1_miou",
                "paired_diff", "dropout_main_effect", "proposed_uncorrected", "b2_uncorrected",
                "b1_uncorrected", "n_missing_bands", "n_absent_groups", "min_group_frac_margin",
                "n_test_px", "classes_in_test"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in per_seed_rows:
            w.writerow(row)

    # ---- optional: tau sweep (one row per seed x tau x cwv) ----
    if args.tau_sweep:
        with open(P(f"results_phase3_tau_sweep{sfx}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "tau", "cwv", "n_missing_bands", "n_absent_groups",
                        "min_group_frac_margin", "proposed_miou", "b2_miou", "paired_diff"])
            for sd, res in zip(args.seeds, results):
                for r in res.get("_tau_sweep", []):
                    w.writerow([sd, r["tau"], r["cwv"], r["n_missing_bands"], r["n_absent"],
                                f"{r['min_frac_margin']:.4f}", f"{r['proposed']:.2f}",
                                f"{r['b2']:.2f}", f"{r['proposed'] - r['b2']:.2f}"])
        print(f"wrote: {P(f'results_phase3_tau_sweep{sfx}.csv')} "
              f"(tau in {args.tau_sweep}; the docstring's withdrawn sweep claim can be re-made "
              f"from THIS artefact, and only from it)")

    # ---- optional: count-matched random band-loss control ----
    if args.random_control:
        with open(P(f"results_phase3_random_control{sfx}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "cwv", "n_missing_bands", "policy", "proposed_miou", "b2_miou",
                        "paired_diff", "n_absent_groups"])
            for sd, res in zip(args.seeds, results):
                for r in res.get("_random_control", []):
                    cwv = r["cwv"]
                    phys = res[cwv]
                    w.writerow([sd, cwv, r["n_missing_bands"], "physics_6s",
                                f"{phys['proposed']:.2f}", f"{phys['b2']:.2f}",
                                f"{phys['proposed'] - phys['b2']:.2f}", phys["n_absent"]])
                    w.writerow([sd, cwv, r["n_missing_bands"], "uniform_random",
                                f"{r['proposed']:.2f}", f"{r['b2']:.2f}",
                                f"{r['proposed'] - r['b2']:.2f}", r["n_absent"]])
        print(f"wrote: {P(f'results_phase3_random_control{sfx}.csv')} — compare the PAIRED_DIFF of "
              f"`physics_6s` against `uniform_random` at the SAME band count. If they match, the "
              f"finding is generic missing-band robustness, not atmosphere-specific robustness, "
              f"and the paper must say so. Compare the paired gaps, NOT the absolute mIoU: the "
              f"uniform arm spends the same band budget on a different, non-matched part of the "
              f"spectrum, so its ABSOLUTE level is not comparable in either direction (see "
              f"eval_random_control -- no a-priori sign is claimed).")

    # ---- figure: mIoU per condition, proposed vs b2. x is CATEGORICAL: clean is its OWN tick, not a
    # fabricated CWV=0.15 coordinate (clean is atmosphere-off, not CWV=0.15); CWV ticks show the true
    # g/cm^2 values ----
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    xpos = np.arange(1 + len(CWVS))                     # 0 = clean tick, 1.. = CWV conditions
    for method, color, lab in [("proposed", "#1f6f3a", "Proposed"), ("b2", "#e67e22", "B2 dropout")]:
        ys = [np.mean(agg["clean"][method])] + [np.mean(agg[c][method]) for c in CWVS]
        es = [seed_sd(agg["clean"][method])] + [seed_sd(agg[c][method]) for c in CWVS]
        ax.errorbar(xpos, ys, yerr=es, fmt="-o", color=color, lw=1.6, ms=3.5, capsize=2, label=lab)
    # PUT THE BAND COUNT ON THE AXIS. In PRIMARY the model input is a pure function of WHICH bands
    # are missing -- T's values never reach a feature -- so a bare "CWV (g/cm^2)" axis shows a
    # four-point band-loss ladder wearing the label of a continuous physical dose-response, and a
    # journal reader never sees the module docstring that explains the difference. The count is the
    # quantity that actually varies, so it is on the tick.
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"clean\n({missing_info.get('clean', 0)} lost)"]
                       + [f"{c:g}\n({missing_info.get(c, 0)} lost)" for c in CWVS])
    ax.set_xlim(-0.35, len(CWVS) + 0.35)
    ax.set_xlabel("Test-time atmosphere: CWV (g/cm$^2$) and resulting bands lost of "
                  f"{len(AVIRIS_WL_NM)}", fontsize=7.5)
    ax.set_ylabel("mIoU (%)")
    # Name the MECHANISM, not just the driver: what varies across these ticks is which bands 6S
    # renders unusable, on a corrected-product proxy -- not a radiometric rescaling.
    ax.set_title("Robustness to 6S-driven band loss (corrected-product proxy)", fontsize=8.5)
    ax.grid(alpha=0.3)
    # SAY what the error bars are. An unlabelled bar is read as a confidence interval by default,
    # and these are neither a CI nor an SE: they are the population sd over seeds, and the seeds
    # move the split and the init together.
    ax.legend(fontsize=6.5, frameon=False, loc="lower left",
              title=f"bars: $\\pm$1 sample sd over {len(args.seeds)} seeds (not a CI)",
              title_fontsize=5.5)
    fig.tight_layout(); fig.savefig(P(f"figs/fig_robust_vs_cwv{sfx}.pdf")); plt.close(fig)

    print("\n===== Phase 3 atmosphere robustness (mean over {} seeds) =====".format(len(args.seeds)))
    print("PRIMARY: corrected-product proxy -- the atmosphere decides WHICH BANDS ARE MISSING.")
    pf = np.mean(agg["clean"]["proposed_full"]); bf = np.mean(agg["clean"]["b2_full"])
    pc = np.mean(agg["clean"]["proposed"]); bc = np.mean(agg["clean"]["b2"])
    print(f"benchmark200_full (0 FURTHER bands removed): proposed={pf:.1f}  b2={bf:.1f}"
          f"   <- ceiling ON THE 200-BAND BENCHMARK (20 water/noise bands already absent upstream)")
    print(f"clean ({missing_info.get('clean',0)} core bands removed): proposed={pc:.1f}  b2={bc:.1f}"
          f"   <- fixed-core cost: proposed -{pf-pc:.1f}, b2 -{bf-bc:.1f} mIoU")
    for cwv in CWVS:
        p = np.mean(agg[cwv]["proposed"]); b = np.mean(agg[cwv]["b2"])
        print(f"CWV={cwv} (absent {absent_info.get(cwv,0)} grp, {missing_info.get(cwv,0)} missing bands): "
              f"proposed={p:5.1f}  b2={b:5.1f}")
    print("\nSUPPLEMENTARY: uncorrected GAS-ONLY radiometry (rho*T fed to a model trained on")
    print("corrected data). NOT 'L1C' -- no path radiance / spherical albedo / scattering / geometry.")
    print("This is NOT the atmosphere->band-loss result: it is dominated by a normalisation artefact")
    print("(surviving bands with a small sd move tens of sigma), so it collapses even with 0 bands dropped.")
    for cwv in CWVS:
        print(f"CWV={cwv}: proposed={np.mean(agg[cwv]['proposed_uncorrected']):5.1f}  "
              f"b2={np.mean(agg[cwv]['b2_uncorrected']):5.1f}")
    print("\nRead the ABSOLUTE mIoU, not retention: retention is each method's score divided by its")
    print("OWN clean score, so the method that starts lower has less to lose and can post a higher")
    print("retention while being worse everywhere. Whether that happens in THIS run is computed")
    print("below rather than quoted: a hardcoded '0.85 vs 0.84' used to print here regardless of")
    print("what the run produced, which is the drift bug this file spends a docstring warning about.")
    for cwv in CWVS:
        pr = np.mean(agg[cwv]["proposed"]) / pc; br = np.mean(agg[cwv]["b2"]) / bc
        if br > pr:
            print(f"  CWV={cwv}: B2 posts the higher RETENTION ({br:.2f} vs {pr:.2f}) while scoring "
                  f"{np.mean(agg[cwv]['b2']):.1f} vs {np.mean(agg[cwv]['proposed']):.1f} mIoU.")
    # Stamp HOW this table was produced next to the table. Without it a reader cannot tell a
    # 5-seed run from a --smoke, and the 6S table's identity is the one input that silently
    # rewrote every number here once already.
    #
    # The INPUT CUBE is hashed too. Its identity was previously assumed from a filename, and the
    # filename is exactly what misled this script into calling the cube a surface-reflectance
    # product -- "corrected" there means the 200-band subset. A digest plus the measured value range
    # lets a reader check which artefact was actually read, and check the claim independently.
    stamp(P(f"results_phase3_atmosphere{sfx}.csv"), args,
          extra=provenance_extra(args, cube, absent_info, missing_info, class_set, class_present))
    stamp(P(f"results_phase3_perseed{sfx}.csv"), args,
          extra={"note": "raw per-seed rows; results_phase3_atmosphere.csv must be recomputable "
                         "from this file", "lut_sha256": file_sha256(TABLE),
                 "tau_missing": args.tau_missing, "group_absent_frac": args.group_absent_frac})
    if args.tau_sweep:
        stamp(P(f"results_phase3_tau_sweep{sfx}.csv"), args,
              extra={"note": "TAU x CWV sweep on the SAME trained models (tau is a test-time masking "
                             "choice and never touches training)", "tau_grid": args.tau_sweep,
                     "lut_sha256": file_sha256(TABLE)})
    if args.random_control:
        stamp(P(f"results_phase3_random_control{sfx}.csv"), args,
              extra={"note": "count-matched uniform band loss vs the 6S mask, same imputation and "
                             "group-masking path; separates atmosphere-specific from generic "
                             "missing-band robustness", "lut_sha256": file_sha256(TABLE),
                     "tau_missing": args.tau_missing})
    print(f"\nwrote: {P(f'figs/fig_robust_vs_cwv{sfx}.pdf')}  {P(f'results_phase3_atmosphere{sfx}.csv')}"
          f"  {P(f'results_phase3_perseed{sfx}.csv')}")


if __name__ == "__main__":
    main()
