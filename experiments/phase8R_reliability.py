#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8R (★ FLAGSHIP) — certified reliability / abstention over the PHYSICAL degradation space
on real Sentinel-2 (CloudSEN12).

Everyone TOLERATES missing bands; nobody CERTIFIES when to trust the prediction under
PHYSICALLY-grounded band loss. Per operational degradation state
    clean  ->  dropB10 (the L1C->L2A cirrus removal)  ->  dropB1B9B10 (atmospheric loss)
           ->  L2A_real (the REAL Sen2Cor surface-reflectance product, B10 genuinely absent)
we quantify three things:

  (1) RELIABILITY DEGRADES: misclassification-detection AUROC drops and selective risk (AURC)
      rises across the degradation states. NOTE the attribution limit: `dropB10` and `dropB1B9B10`
      are isolated missing-band controls, but `L2A_real` is an OPERATIONAL L1C->L2A DOMAIN SHIFT --
      Sen2Cor output is BOA surface reflectance, so it changes atmospheric correction, quantification
      and processing at the same time as B10 goes absent. The clean->L2A_real difference must not be
      attributed to band loss alone; `dropB10` is the arm that isolates that.
  (2) A CLEAN-CALIBRATED THRESHOLD TRACKS A HIGHER HELD-OUT LOSS UNDER SHIFT. Two different numbers
      are reported because the gap between them is the observation:
        * crc_calibration_selection_stat — (n/(n+1))*Rhat_calib + B/(n+1) at the selected threshold,
          the statistic CRC minimised over lambda. <= alpha by construction whenever a threshold was
          feasible, so on its own it can NEVER show a failure — a plot of it alone is flat at alpha.
        * crc_heldout_empirical_joint_loss — the JOINT confidently-wrong mass E[kept AND wrong] the
          held-out set incurred at that same threshold.
      WHAT THIS IS NOT. The left number is not a per-split upper bound on the right one. CRC
      (Angelopoulos et al. 2022) Theorem 2.1 controls the MARGINAL expectation E[L_{n+1}] over the
      JOINT draw of calibration and test under exchangeability. A single split whose empirical loss
      exceeds alpha does not contradict it, and this file no longer calls that a voided certificate.
      The theorem-level quantity is the MEAN over independent (split_seed, model_seed) draws with an
      interval, which is why the raw per-run rows are written and the split itself is re-drawn.
  (3) RECALIBRATING ON THE OPERATIONAL STATE (Mondrian) TRACKS A LOWER HELD-OUT LOSS. Calibration and
      evaluation then come from the same state, which is the setting standard CRC is stated for. The
      operational lesson is that one clean calibration does not transfer across regimes.

Grounding: naive-CP-fails-under-shift and the mask-/regime-conditional fix are established
theory — weighted conformal under covariate shift (Tibshirani et al. 2019) and mask-conditional
coverage for general missing-data mechanisms (Fan et al., arXiv:2512.14221, 2025). Our
contribution is APPLYING + CERTIFYING it over PHYSICAL spectral degradation for real RS
segmentation, anchored by the real L1C->L2A transition. Discrete operational states let us use
the simplest valid instance — per-state (Mondrian) recalibration.

Methodology guards honored:
  - Guard 1: the atmospheric anchor is the REAL L2A product, not a 6S-on-L1C stress test.
  - Guard 2: calibration/evaluation are split by ROI (location) within the TEST ROIs (which are
    roi-disjoint from train) — NOT at pixel level, and NOT merely by patch index — so conformal
    exchangeability is not broken by spatial autocorrelation. The SAME unit is used for the CRC
    grouping: CloudSEN12's 975 test patches come from only 195 roi_ids (exactly 5 patches per
    location, differing by acquisition DATE), so grouping CRC by patch would count those 5 as 5
    independent calibration units, inflate n ~5x and shrink the B/(n+1) correction to ~1/5 of what
    the data entitles it to. The exchangeable unit is the ROI on BOTH the split and the bound.

Honesty: conformal_at_risk is an RCPS-style target-risk OPERATING POINT (report achieved risk
descriptively), and per-pixel spectral mIoU is deliberately NOT a spatial-SOTA number — this is
a reliability study. The point is the GAP between naive and degradation-aware risk control.
Two further honesty guards, because both are ways this result could be oversold:
  - COVERAGE IS PART OF THE RESULT. The certified quantity is a JOINT mass, which any method can
    drive to 0 by abstaining, so Mondrian "restoring control" while accepting far fewer pixels is
    a weaker claim than it looks. Every risk number is reported and plotted with its coverage.
  - A FAIRNESS CONTROL is run. "Naive" makes the threshold AND the temperature stale at once, so a
    third arm (clean-calibrated threshold, state-specific temperature) attributes the breach to the
    threshold rather than to temperature miscalibration.

Outputs (../paper/):
  results_phase8R_reliability.csv          per (method,state): acc/AURC/AUGRC/AUROC, plug-in
                                           CONDITIONAL risk, and for each of the three CRC arms the
                                           calibration BOUND, the REALIZED eval joint loss, the
                                           coverage and the feasibility rate, plus the count of
                                           exchangeable ROI units behind the bound
  figs/fig_phase8R_conformal_shift.pdf     CRC bound vs realized JOINT risk, and coverage, per state
--smoke writes both under a `_smoke` suffix. It calibrates on a few dozen ROIs, few enough that the
CRC B/(n+1) floor alone can decide the naive-vs-Mondrian comparison this experiment exists to make,
so it must never touch the paths above. Its 80-patch cap also samples patch INDICES at random out of
975, which almost never draws two patches of the same location, so a smoke run does NOT exhibit the
~5x patch->ROI unit collapse that the full run does (~487 calib patches -> ~97 calib ROIs); read the
smoke's unit counts as a mechanism check, not as the full run's numbers.

Usage:
  python experiments/phase8R_reliability.py --smoke          # 2 splits x 1 model seed, quick (_smoke outputs)
  python experiments/phase8R_reliability.py --split-seeds 0 1 2 3 4 --model-seeds 0 1 2 --jobs 5
  # the paper's 10x10 flagship (results_phase8R_*_10seed.csv, 2x V100) -- re-run it under ANOTHER tag,
  # or the committed files are overwritten:
  python experiments/phase8R_reliability.py --split-seeds $(seq 0 9) --model-seeds $(seq 0 9) \\
      --out-tag _10seed_repro
"""
import os, sys, csv, argparse, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
from bandsim.grouping import group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention
from bandsim.reliability import (confidence_msp, aurc, augrc, selective_auroc,
                                 fit_temperature, conformal_at_risk, conformal_risk_control)
from bandsim import hw, parallel
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)

METHODS = ["proposed", "b2"]
TARGET_RISK = 0.10
NUM_CLASSES = P8.NUM_CLASSES


@torch.no_grad()
def logits_at(kind, model, X, groups, drop):
    """Logits (N,C) for `kind` on inputs X with the given groups dropped (band-exact)."""
    dev = next(model.parameters()).device
    if kind in ("proposed", "b4", "b6"):
        pm = P2.group_present_mask(X.shape[0], groups, drop)
        return model(torch.from_numpy(X).to(dev), torch.from_numpy(pm).to(dev)).cpu().numpy()
    Xc = P2.zero_missing(X, groups, drop)
    return model(torch.from_numpy(Xc).to(dev)).cpu().numpy()


def test_roi_ids(split="test"):
    """The `roi_id` of every test patch, in the order load_split enumerates patches."""
    import pandas as pd
    meta = pd.read_csv(os.path.join(P8.DATA, split, "metadata.csv"))
    if "roi_id" not in meta.columns:
        raise KeyError(f"{split}/metadata.csv has no roi_id column (have {list(meta.columns)[:8]}...) "
                       f"-- the calib/eval split cannot be made ROI-disjoint without it")
    return meta["roi_id"].to_numpy()


def scene_component_ids(split="test"):
    """The scene-connected-component id of every patch (the P0-2 exchangeable unit: ROIs sharing any
    s2_id unioned, 195 -> 184 on test). Thin alias delegating to the SINGLE source of truth
    P8.scene_component_ids, so this module's call sites and tests keep one implementation."""
    return P8.scene_component_ids(split)


def test_patch_split(n_test_patches, calib_frac=0.5, max_patches=None, seed=0, roi_ids=None):
    """Disjoint calib/eval TEST-patch id sets, split BY ROI.

    Splitting on the patch INDEX is not enough, and calling that "patch-level rigour" was wrong.
    CloudSEN12's 975 test patches come from only 195 distinct `roi_id`s -- about five patches per
    location, differing by acquisition DATE (963 distinct s2_id) rather than by place. A random
    patch-index split therefore puts the same ground, imaged on different days, on both sides of the
    calibration/evaluation boundary, so the units are spatially dependent and the conformal
    guarantee is unearned. This is the same defect that inflated Phase 4R's bound before its split
    was moved to whole spatial blocks.

    Splitting on `roi_id` makes calibration and evaluation locations disjoint. The honest cost is
    sample size: n drops from ~975 patches to ~195 ROIs, so CRC's B/(n+1) correction grows -- which
    is the bound being computed on units that are actually independent rather than on a count that
    was never real.

    BOTH sides must be non-empty. n_test_patches=1 used to return calib={0}, eval={} -- metrics then
    average over nothing and CRC "calibrates" on a single unit (its B/(n+1) floor is 0.5, so nothing
    is certifiable below alpha=0.5 anyway). Degenerate requests now raise instead of being returned.
    The clamp also stops calib_frac>=1 from emptying eval; ordinary calls are unaffected."""
    rng = np.random.default_rng(70000 + seed)
    ids = np.arange(n_test_patches)
    if max_patches is not None and max_patches < n_test_patches:
        ids = rng.choice(n_test_patches, size=max_patches, replace=False)
    if len(ids) < 2:
        raise ValueError(f"need >= 2 test patches for a disjoint calib/eval split, got {len(ids)}")

    if roi_ids is None:
        # No ROI information: fall back to the patch-index split, but say plainly that the result is
        # NOT ROI-disjoint so nobody reads it as the rigorous arm.
        print("[warn] test_patch_split: no roi_ids given -- falling back to a patch-INDEX split. "
              "Calibration and evaluation may share locations; this is NOT an ROI-disjoint split.")
        rng.shuffle(ids)
        k = min(max(1, int(len(ids) * calib_frac)), len(ids) - 1)
        return np.sort(ids[:k]), np.sort(ids[k:])

    r = np.asarray(roi_ids)[ids]
    uniq = np.unique(r)
    if len(uniq) < 2:
        raise ValueError(f"need >= 2 distinct ROIs for an ROI-disjoint split, got {len(uniq)} "
                         f"from {len(ids)} patches")
    perm = rng.permutation(uniq)
    kr = min(max(1, int(len(perm) * calib_frac)), len(perm) - 1)
    cal_rois, ev_rois = perm[:kr], perm[kr:]
    cal = np.sort(ids[np.isin(r, cal_rois)])
    ev = np.sort(ids[np.isin(r, ev_rois)])
    if len(cal) == 0 or len(ev) == 0:
        raise ValueError(f"ROI split emptied a side: {len(cal)} calib / {len(ev)} eval patches")
    shared = set(np.asarray(roi_ids)[cal].tolist()) & set(np.asarray(roi_ids)[ev].tolist())
    if shared:
        raise ValueError(f"calib and eval share {len(shared)} ROI(s) -- the split is not disjoint")
    print(f"  ROI-disjoint split: {len(cal_rois)} calib ROIs ({len(cal)} patches) / "
          f"{len(ev_rois)} eval ROIs ({len(ev)} patches)")
    return cal, ev


def reliability_over_states(kind, model, tmp, cal, ev, groups, states, unit_cal=None,
                            unit_ev=None, *, target):
    """For one trained model: per degradation state, descriptive reliability + a clean-calibrated
    versus a state-calibrated conformal arm, plus a temperature control.

    tmp / cal / ev are dicts {"l1c": (X,y), "l2a": (X,y)} of TEMPERATURE-fitting / CRC-calibration /
    evaluation pixels, drawn from THREE DISJOINT ROI sets.

    WHY TEMPERATURE GETS ITS OWN SPLIT. It used to be fitted on the same pixels CRC then calibrated
    on. CRC needs the loss L_i(lambda) to be exchangeable across calibration and test units, and a
    score function fitted on the calibration set makes calibration scores in-sample while test scores
    stay out-of-sample -- they are not exchangeable, and the theorem does not apply. The usual defence
    ("temperature is a monotone rescale, so it cannot matter") is false for multi-class MSP: with
    logits A=[-1.407,-2.531,-1.247,0.083] and B=[-4.650,-0.438,-2.492,-1.465], MSP(A)>MSP(B) at
    T=0.5 and MSP(B)>MSP(A) at T=5.0. Temperature reorders which samples clear a threshold, so
    fitting it on the calibration set is not harmless.

    unit_cal / unit_ev are the per-pixel EXCHANGEABLE-UNIT ids handed to CRC as calib/eval groups.
    They must be ROI ids, not patch ids: CloudSEN12 gives ~5 patches per location (same ground,
    different acquisition dates), so patch grouping counts 5 correlated units as 5 independent ones,
    inflating n ~5x and shrinking the B/(n+1) term to ~1/5 of what the data entitles it to.

    Returns a list of LONG-FORM rows, one per (state, arm), each carrying the selected threshold and
    the temperature that produced it -- without those a result cannot be audited or re-derived."""
    rows = []
    # --- temperatures, both fitted on the TEMPERATURE ROIs (disjoint from calib and eval) ---
    Xt_cl, yt_cl = tmp["l1c"]
    T_clean = fit_temperature(logits_at(kind, model, Xt_cl, groups, []), yt_cl)

    # CLEAN calibration reference (a naive deployment calibrates ONCE, on clean)
    Xcl_c, ycl = cal["l1c"]
    lg_cal_clean = logits_at(kind, model, Xcl_c, groups, [])
    conf_cal_clean = confidence_msp(lg_cal_clean / T_clean)
    corr_cal_clean = (lg_cal_clean.argmax(1) == ycl).astype(int)

    for name, src, drop in states:
        Xc, yc = cal[src]
        Xe, ye = ev[src]
        Xt, yt = tmp[src]
        lg_cal = logits_at(kind, model, Xc, groups, drop)
        lg_ev = logits_at(kind, model, Xe, groups, drop)
        corr_cal = (lg_cal.argmax(1) == yc).astype(int)
        corr_ev = (lg_ev.argmax(1) == ye).astype(int)

        # this state's temperature, also from the TEMPERATURE ROIs
        T = fit_temperature(logits_at(kind, model, Xt, groups, drop), yt)
        conf_ev = confidence_msp(lg_ev / T)
        conf_cal = confidence_msp(lg_cal / T)
        conf_ev_clean = confidence_msp(lg_ev / T_clean)
        conf_ev_raw = confidence_msp(lg_ev)          # H-3: T=1, the un-scaled deployment scoring rule

        # THE PLUG-IN MARGIN IS SIZED FROM THE SAME UNIT CRC USES. It was not: `unit_cal` was
        # handed to conformal_risk_control on the very next lines while conformal_at_risk was
        # called without it, so its finite-sample margin counted 194,000 correlated PIXELS as
        # independent evidence instead of the 97 ROIs the split is actually built from. That is a
        # ~45x understatement of the margin (0.00068 against 0.0305 at the raw counts), which makes
        # `conservative=True` a no-op: the threshold comes out too aggressive and the achieved risk
        # runs above target with nothing to announce it. The margin now uses a design-effect n,
        # which lands between the two extremes at the measured intra-ROI correlation.
        mond = conformal_at_risk(corr_cal, conf_cal, corr_ev, conf_ev, target_risk=target,
                                 calib_group=unit_cal)
        naive = conformal_at_risk(corr_cal_clean, conf_cal_clean, corr_ev, conf_ev_clean,
                                  target_risk=target, calib_group=unit_cal)
        # The ROW-SIZED operating point is kept alongside, not discarded. The difference between
        # these two is the entire point this experiment is making -- that counting correlated units
        # as independent buys coverage that was never earned -- and quoting only the corrected
        # number would hide the size of the effect. Coverage only; the risks are in the pair above.
        # H-1: pass a SINGLETON group per row (np.arange) rather than calib_group=None. With None the
        # refinement pass is skipped entirely, so the row-sized arm differed from the grouped arm in
        # the ALGORITHM (refined vs unrefined) as well as in n -- confounding the very comparison. A
        # per-row group makes n_eff = the row count with rho=0, so BOTH arms run the same refinement
        # and the only thing that varies is the exchangeable-unit count.
        mond_rows = conformal_at_risk(corr_cal, conf_cal, corr_ev, conf_ev, target_risk=target,
                                      calib_group=np.arange(corr_cal.size))
        naive_rows = conformal_at_risk(corr_cal_clean, conf_cal_clean, corr_ev, conf_ev_clean,
                                       target_risk=target, calib_group=np.arange(corr_cal_clean.size))
        arms = {
            "mondrian": (conformal_risk_control(corr_cal, conf_cal, corr_ev, conf_ev, alpha=target,
                                                calib_group=unit_cal, eval_group=unit_ev), T),
            "naive": (conformal_risk_control(corr_cal_clean, conf_cal_clean, corr_ev, conf_ev_clean,
                                             alpha=target, calib_group=unit_cal, eval_group=unit_ev),
                      T_clean),
            # FAIRNESS CONTROL: "naive" makes the threshold AND the temperature stale at once. This
            # arm keeps the state's own temperature and only the clean-calibrated threshold, so the
            # gap can be attributed. Close to naive => it tracks the stale THRESHOLD; close to
            # Mondrian => it was mostly stale TEMPERATURE and the conformal framing is overstated.
            "naiveThr_freshT": (conformal_risk_control(
                corr_cal_clean, confidence_msp(lg_cal_clean / T), corr_ev, conf_ev, alpha=target,
                calib_group=unit_cal, eval_group=unit_ev), T),
        }
        base = {
            "state": name,
            "acc": float(corr_ev.mean()) * 100,
            "aurc": aurc(corr_ev, conf_ev) * 100,
            "augrc": augrc(corr_ev, conf_ev) * 100,
            "auroc": selective_auroc(corr_ev, conf_ev) * 100,
            # H-3: AURC/AUROC under a FIXED deployment scoring rule (T=1 raw, and the clean-fitted T),
            # NOT just the per-state re-fitted T above. Temperature reorders multi-class MSP (see the
            # docstring), so re-fitting T per state hides part of the degradation that a deployed,
            # once-calibrated detector would actually suffer. These are the columns for "how does a
            # FIXED rule fail under shift"; the state-T columns answer "how much can re-calibration buy".
            "aurc_rawT": aurc(corr_ev, conf_ev_raw) * 100,
            "auroc_rawT": selective_auroc(corr_ev, conf_ev_raw) * 100,
            "aurc_cleanT": aurc(corr_ev, conf_ev_clean) * 100,
            "auroc_cleanT": selective_auroc(corr_ev, conf_ev_clean) * 100,
            "mondrian_plugin_cond_risk": mond["risk"] * 100,
            "mondrian_plugin_cov": mond["coverage"] * 100,
            "naive_plugin_cond_risk": naive["risk"] * 100,
            "naive_plugin_cov": naive["coverage"] * 100,
            # what the margin was actually sized from, so the correction is auditable rather than
            # asserted: n_eff sits between the ROI count and the pixel count at the estimated icc.
            "plugin_margin_neff": float(mond["n_calib_units"]),
            "plugin_margin_icc": float(mond["icc"]),
            "plugin_degenerate": float(mond["degenerate"]),
            # the discarded alternative, for the contrast (see the call site)
            "mondrian_plugin_cov_rowsized": mond_rows["coverage"] * 100,
            "naive_plugin_cov_rowsized": naive_rows["coverage"] * 100,
            "mondrian_plugin_cond_risk_rowsized": mond_rows["risk"] * 100,
            "naive_plugin_cond_risk_rowsized": naive_rows["risk"] * 100,
        }
        for arm, (crc, temp_used) in arms.items():
            rows.append(dict(base, arm=arm,
                             temperature=float(temp_used),
                             crc_threshold=float(crc["threshold"]),
                             crc_calibration_selection_stat=crc["calib_crc_bound"] * 100,
                             crc_heldout_empirical_joint_loss=crc["eval_group_joint_risk"] * 100,
                             crc_heldout_coverage=crc["eval_group_coverage"] * 100,
                             crc_heldout_selective_risk=crc["eval_group_selective_risk"] * 100,
                             crc_feasible=int(crc["feasible"]),
                             n_calib_units=int(crc["n_calib_units"]),
                             n_eval_units=int(crc["n_eval_units"])))
    return rows


# ROI budget for the three-way split of the TEST ROIs. Temperature gets the smallest share: it fits
# ONE parameter, while the CRC bound's B/(n+1) term is driven directly by the calibration ROI count.
TEMP_FRAC, CALIB_FRAC = 0.25, 0.375


def split_test_rois(rid_te, split_seed):
    """Three DISJOINT ROI sets -> boolean pixel masks (temperature, CRC calibration, evaluation).

    The split is re-drawn per `split_seed`. It used to be hardcoded to seed=0, so every model seed
    shared ONE calibration draw and the mean over seeds could not estimate the marginal expectation
    CRC Theorem 2.1 is about -- the seeds varied only model init and the training subsample."""
    uro = np.unique(rid_te)
    if len(uro) < 3:
        raise ValueError(f"need >= 3 distinct test ROIs for a three-way split, got {len(uro)}")
    perm = np.random.default_rng(70000 + split_seed).permutation(uro)
    n_t = max(1, int(round(TEMP_FRAC * len(perm))))
    n_c = max(1, int(round(CALIB_FRAC * len(perm))))
    if n_t + n_c >= len(perm):
        raise ValueError(f"split leaves no evaluation ROIs ({len(perm)} total)")
    ro_t, ro_c, ro_e = perm[:n_t], perm[n_t:n_t + n_c], perm[n_t + n_c:]
    for a, b, na, nb in ((ro_t, ro_c, "temp", "calib"), (ro_c, ro_e, "calib", "eval"),
                         (ro_t, ro_e, "temp", "eval")):
        if set(a.tolist()) & set(b.tolist()):
            raise ValueError(f"{na}/{nb} ROI sets overlap -- the split is not disjoint")
    return (np.isin(rid_te, ro_t), np.isin(rid_te, ro_c), np.isin(rid_te, ro_e))


def train_models(model_seed, Xtr, ytr, groups, cwl, subsample_frac, epochs):
    """The TRAINING half of a (split_seed, model_seed) draw. It reads neither `split_seed` nor any RNG
    state left behind by earlier work: the pixel subsample is drawn from default_rng(model_seed), every
    constructor is preceded by hw.seed_model(model_seed), and the train/pretrain/finetune helpers seed
    their own explicit streams. So wherever training is deterministic the trained models are a
    function of model_seed alone -- the property H-10 (train ONCE per model seed, evaluate it on every
    split; run_two_stage) rests on, where the per-cell protocol retrained the identical model once per
    split seed. Observed on the 4x P100 box (2026-10-13, torch 2.6.0+cu124, CUDA, deterministic=True)
    -- an observation, not a guarantee: the flagship's model seed 0 gave identical b2 and proposed
    weight hashes on each of the four GPUs at 1 and at 4 intra-op threads and when trained twice in one
    process, and the per-cell and two-stage protocols gave identical rows at full precision for all ten
    model seeds (experiments/H10_TRAIN_ONCE_VERIFICATION.md). On the CPU the weights depend on the
    intra-op thread count, which is why run_two_stage gives both of its stages ONE thread budget; and repeated
    two-thread CPU campaigns occasionally disagreed in the last bits of some evaluation rows, at a
    locus not isolated, while at one thread per worker they never did in 120 repetitions of a small
    synthetic campaign (bandsim.parallel, "Determinism").

    Training once also realises EXACTLY the design two_way_se assumes ("a given model_seed recurs across
    all splits (same trained model, same subsample)"), including where training is not deterministic
    (e.g. --nondeterministic), where per-cell retraining only approximated it.

    The training subsample is over PIXELS, not ROIs -- the docstring used to say "train ROIs", which
    it never was. Pixel subsampling over-represents whichever patches contribute more pixels and its
    variance is pixel-resampling variance, not spatial-unit variance; it is kept only because the
    train loader does not return patch ids, and the label now says so."""
    # No P2.NUM_CLASSES = ... here. That module-global rewrite is the non-reentrant pattern the
    # merge campaign removed from phase8 (it is what made PR #11's b2m test order-dependent: any
    # test that ran this file first left P2 configured for 4 classes). num_classes is passed
    # explicitly at the one call that needs it; 8R's own logits_at scores the models directly, so
    # nothing else reads the global.
    rs = np.random.default_rng(model_seed)
    k = max(1, int(round(subsample_frac * Xtr.shape[0])))
    sub = rs.choice(Xtr.shape[0], size=k, replace=False)
    Xs, ys = Xtr[sub], ytr[sub]

    # bs=auto_bs: same rationale and same caveat as phase8 -- P2's 256 default is sized for 21k
    # pixels, the flagships train on hundreds of thousands (phase8R: 0.8 x 900k = 720k px; phase8R2,
    # which shares this function: 0.8 x its source-domain pixels, 552k on the surface axis), and the
    # models are launch-bound at 256 (measured ~25 min/run that this brings to ~2-3). A
    # hyperparameter change, recorded in the sidecar.
    bs = P2.auto_bs(Xs.shape[0])
    hw.seed_model(model_seed)                                       # P0-2: seed before the B2 constructor
    m_b2 = P2.train_mlp(Xs, ys, groups, model_seed, group_dropout=True, epochs=epochs,
                        num_classes=NUM_CLASSES, bs=bs)
    hw.seed_model(model_seed)                                       # P0-2: and before the proposed model --
    m_prop = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES)    # init reproducible, not order-dependent
    P2.pretrain_sgmae(m_prop, Xs, groups, model_seed, epochs=max(1, epochs // 2), bs=bs)
    P2.finetune_proposed(m_prop, Xs, ys, groups, model_seed, epochs=epochs, bs=bs)
    return {"proposed": m_prop, "b2": m_b2}


def evaluate_cell(split_seed, model_seed, models, Xte, yte, rid_te, groups, *, target):
    """The EVALUATION half of a (split_seed, model_seed) draw: re-split the test ROIs for `split_seed`
    and score both trained models on every state. Reads no process-global RNG (split_test_rois seeds
    its own generator from split_seed; bandsim.reliability draws no random numbers), so wherever the
    computation is deterministic its rows depend only on its arguments -- not on where or when the
    models were trained. (On the CPU the logits moved in the last bits with the intra-op thread count at
    the larger two of the four input sizes measured -- up to 2.4e-7 in absolute logit at 145,650 and
    388,400 rows, not at 2,400 or 60,000 -- which run_two_stage holds fixed; repeated two-thread CPU
    campaigns occasionally disagreed in some evaluation rows, at a locus not isolated, and never in 120
    repetitions at one thread per worker -- bandsim.parallel, "Determinism". On a P100 the rows of a
    flagship cell evaluated after train_job/eval_job's GPU->CPU->bytes->GPU round trip matched
    run_seed's at full precision. Both measured 2026-10-13 with scripts/verify_h10.py;
    experiments/H10_TRAIN_ONCE_VERIFICATION.md.)

    `target` is the CRC risk level (alpha) and is REQUIRED here and on every hop from main() down to
    reliability_over_states -- none of them has a default any more. A previous version left run_seed
    without the parameter, so reliability_over_states silently used its default 0.10 while --alpha was
    written to the plot, console and provenance, mislabelling any run with --alpha != 0.10. A required
    argument cannot be dropped silently, and tests/test_phase8R_p0_fixes.py checks that every CRC call
    receives the value passed in."""
    mt, mc, me = split_test_rois(rid_te, split_seed)

    def part(mask):
        return {k: (Xte[k][mask], yte[mask]) for k in ("l1c", "l2a")}
    tmp, cal, ev = part(mt), part(mc), part(me)

    g_b1 = P8._assert_singleton(groups, P8.B1_IDX, "B1")
    g_b9 = P8._assert_singleton(groups, P8.B9_IDX, "B9")
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    states = [("clean", "l1c", []), ("dropB10", "l1c", [g_b10]),
              ("dropB1B9B10", "l1c", [g_b1, g_b9, g_b10]), ("L2A_real", "l2a", [g_b10])]

    out = []
    for meth in METHODS:
        for r in reliability_over_states(meth, models[meth], tmp, cal, ev, groups, states,
                                         unit_cal=rid_te[mc], unit_ev=rid_te[me], target=target):
            out.append(dict(r, method=meth, split_seed=split_seed, model_seed=model_seed,
                            n_temp_roi=int(np.unique(rid_te[mt]).size),
                            n_calib_roi=int(np.unique(rid_te[mc]).size),
                            n_eval_roi=int(np.unique(rid_te[me]).size)))
    return out


def run_seed(job, Xtr, ytr, Xte, yte, rid_te, groups, cwl, subsample_frac, epochs, *, target):
    """One (split_seed, model_seed) draw in ONE call: train, then evaluate -- the per-cell protocol
    main() used before H-10, kept as the single-cell reference path. main() runs the same two halves
    through run_two_stage: train_job once per model seed, then eval_job once per cell."""
    split_seed, model_seed = job
    models = train_models(model_seed, Xtr, ytr, groups, cwl, subsample_frac, epochs)
    return evaluate_cell(split_seed, model_seed, models, Xte, yte, rid_te, groups, target=target)


def train_job(model_seed, Xtr, ytr, groups, cwl, subsample_frac, epochs):
    """H-10 stage 1, one job per MODEL seed: train once and return the models as PICKLED BYTES of
    their CPU copies, which the parent hands to every evaluation job of that seed.

    Bytes, not module objects, on purpose. A module that crosses the process pool (returned here, or
    shared with the evaluation jobs) is pickled by torch's multiprocessing reducer, which moves every
    tensor into shared memory and passes one file descriptor per tensor -- 40 per seed, so 400 for each
    evaluation job of a 10-seed campaign. That crashed the pool with "Too many open files" under the
    common `ulimit -n 1024` (measured 2026-10-13; this box's limit of 1048576 hid it). Plain pickle
    writes each tensor's storage BY VALUE (torch.save), so a seed travels as ~0.6 MB of immutable
    bytes: no descriptor, every weight round-trips exactly, and no evaluation job can alter the model
    another one receives. .cpu() first so the bytes load on any worker, GPU or CPU; the eval() mode the
    trainers leave the models in is part of the pickled state."""
    models = train_models(model_seed, Xtr, ytr, groups, cwl, subsample_frac, epochs)
    return pickle.dumps({k: m.cpu() for k, m in models.items()}, protocol=pickle.HIGHEST_PROTOCOL)


def eval_job(job, trained, Xte, yte, rid_te, groups, *, target):
    """H-10 stage 2, one job per (split_seed, model_seed) CELL: evaluate the cell against the model
    trained once for its seed. `trained` maps model_seed -> train_job's bytes; unpickling gives this
    job a private copy, moved to this worker's device."""
    split_seed, model_seed = job
    dev = hw.device()
    models = {k: m.to(dev) for k, m in pickle.loads(trained[model_seed]).items()}
    return evaluate_cell(split_seed, model_seed, models, Xte, yte, rid_te, groups, target=target)


def run_two_stage(eval_fn, jobs, *, train_shared, eval_shared, prefer=None, n_jobs=None, threads=None,
                  deterministic=True, label="two-stage", log=print, eval_per_worker_gb=None):
    """H-10's campaign driver, shared by phase8R and phase8R2: train every model seed that occurs in
    `jobs` ONCE (train_job, one parallel job per seed), then evaluate every (split_seed, model_seed)
    cell against its seed's model (`eval_fn`, one parallel job per cell; each job receives `trained` =
    {model_seed: train_job's bytes} for ALL seeds, plus `eval_shared`, and uses its own seed's entry).
    Returns (results, compute): the results in `jobs` order -- the order, and wherever training and
    evaluation are bitwise deterministic (observed: CUDA on the 4x P100 box; the CPU at one thread per
    worker, on small synthetic campaigns) the values, that the per-cell protocol (run_seed per cell)
    produced -- and both stages' plans for the provenance sidecar.

    ONE THREAD BUDGET FOR BOTH STAGES. Unless `threads` is given (None or 0 mean not given; a negative
    count raises), both stages get the per-worker intra-op thread count the per-cell protocol had for
    this campaign (plan_jobs over the CELLS, BANDSIM_THREADS included). Left to plan_jobs, the training
    stage -- one job per model seed, so fewer jobs -- can get MORE threads per worker (whenever there
    are fewer seeds than cell workers), and CPU weights depend on the thread count: the two-stage rows
    then differed from the per-cell ones on the CPU (found in review, 2026-10-13). On the P100s the
    count did not change CUDA-trained weights, so there holding it changed no number.

    `eval_per_worker_gb` is the VRAM budget (decimal GB) of ONE evaluation job for the size-aware
    workers-per-GPU default (bandsim.parallel.run_jobs); None keeps the default, and a NaN or
    non-positive one raises before anything is trained. A caller whose
    evaluation is heavier than the default budget passes it (phase8R2 does). Here it changes only how
    many evaluation jobs share a GPU: the thread budget is pinned above, so no number moves with it.

    `compute` records, per stage, the plan the run_jobs call actually ran (its on_plan callback, not a
    re-plan): device, worker count and whether it was forced, workers per GPU, the VRAM budget that took
    effect ("inf" for an infinite one, which JSON cannot hold), threads per worker -- plus the requested
    budget and the deterministic flag, which is what a reader needs to judge whether the per-cell
    equality can be expected.

    Stage 1 uses at most one worker per model seed; a single seed (e.g. --smoke) trains serially in
    this process, after which run_jobs releases this process's CUDA cache before the evaluation workers
    start."""
    model_seeds = list(dict.fromkeys(ms for _, ms in jobs))
    if threads is not None and threads < 0:
        raise ValueError(f"threads must be >= 0 (None or 0: the per-cell plan's), got {threads}")
    if eval_per_worker_gb is not None:               # refuse a bad budget BEFORE anything is trained
        eval_per_worker_gb = float(eval_per_worker_gb)
        hw.per_worker_budget_gb(eval_per_worker_gb)  # raises on NaN / non-positive
    if not threads:
        threads = parallel.plan_jobs(len(jobs), prefer=prefer, jobs=n_jobs).threads_per_worker
    plans = {}
    blobs = parallel.run_jobs(train_job, model_seeds, shared=train_shared, prefer=prefer, jobs=n_jobs,
                              threads=threads, deterministic=deterministic, log=log,
                              label=f"{label}/train", on_plan=lambda p: plans.__setitem__("train", p))
    trained = dict(zip(model_seeds, blobs))
    results = parallel.run_jobs(eval_fn, jobs, shared=dict(eval_shared, trained=trained), prefer=prefer,
                                jobs=n_jobs, threads=threads, deterministic=deterministic, log=log,
                                label=f"{label}/eval", per_worker_gb=eval_per_worker_gb,
                                on_plan=lambda p: plans.__setitem__("eval", p))
    def gb(x):                       # json.dump would write an inf budget as the non-JSON token Infinity
        return x if x is None or np.isfinite(float(x)) else str(float(x))

    def stage(p):
        return {"plan": p.describe(), "device": p.device_mode, "workers": p.workers,
                "workers_forced": p.workers_forced, "max_workers_per_gpu": p.per_gpu,
                "gpu_oversub": p.oversub, "gpu_oversub_forced": p.oversub_forced,
                "vram_budget_gb": gb(p.budget_gb), "threads_per_worker": p.threads_per_worker}
    compute = {"train_stage": stage(plans["train"]), "eval_stage": stage(plans["eval"]),
               "device": plans["eval"].device_mode, "threads_per_worker": plans["eval"].threads_per_worker,
               "deterministic_flag": bool(deterministic),
               "eval_vram_budget_gb_requested": gb(eval_per_worker_gb), "parent_process": hw.info()}
    return results, compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)),
                    help="ROI re-splits. CRC's guarantee is a MARGINAL expectation over the joint "
                         "draw of calibration AND test, so a single split cannot estimate it.")
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--subsample-frac", type=float, default=0.8)
    ap.add_argument("--patches-train", type=int, default=3000)
    ap.add_argument("--px-train", type=int, default=300)
    ap.add_argument("--patches-test", type=int, default=None, help="cap on #test patches")
    ap.add_argument("--px-test", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=TARGET_RISK)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    ap.add_argument("--nondeterministic", action="store_true")
    ap.add_argument("--test-pixel-seed", type=int, default=54321,
                    help="seed of the per-patch test-pixel draw (54321 = the paper's). The reported SE "
                         "leaves this draw out (H-8); varying it measures that uncertainty. Give a "
                         "non-default seed an --out-tag, or the run writes over the canonical CSVs.")
    ap.add_argument("--out-tag", default="", help="filename suffix for ALL outputs (e.g. _10seed) so "
                    "a re-run with more seeds does not clobber the canonical (untagged) results; must "
                    "be filename-safe. --smoke forces _smoke and ignores this.")
    args = ap.parse_args()
    for name, v, lo, hi in (("subsample_frac", args.subsample_frac, 0.0, 1.0),
                            ("alpha", args.alpha, 0.0, 1.0)):
        if not (lo < v <= hi):
            raise ValueError(f"--{name.replace('_','-')} must be in ({lo}, {hi}], got {v}")
    if args.epochs < 1 or args.px_test < 1 or args.px_train < 1:
        raise ValueError("epochs / px-train / px-test must all be >= 1")
    for nm, sq in (("split-seeds", args.split_seeds), ("model-seeds", args.model_seeds)):
        if len(set(sq)) != len(sq):
            raise ValueError(f"--{nm} has duplicates ({sq}); a repeated seed is the SAME draw and "
                             f"would be double-counted as an independent run in the two-way SE")
    _safe = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    if args.out_tag and (set(args.out_tag) - _safe):
        raise ValueError(f"--out-tag must be filename-safe [A-Za-z0-9_.-], got {args.out_tag!r}")

    sfx = args.out_tag
    if args.smoke:
        args.split_seeds, args.model_seeds, args.epochs = [0, 1], [0], 10
        args.patches_train = 80; args.px_train = 200; args.patches_test = 80; args.px_test = 200
        sfx = "_smoke"
        print("[smoke] 2 splits x 1 model seed / 10 epochs / 80 test patches — *_smoke artefacts only")
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device); print("HW:", hw.info())

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)

    print("loading CloudSEN12 train + test ...")
    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=args.px_train,
                             n_patches=args.patches_train, seed=12345)

    # Load the TEST pixels ONCE with their roi_id; the three-way ROI partition is then a cheap index
    # operation per split_seed, instead of re-reading the patches for every re-split.
    import pandas as pd
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    n_test = len(meta)
    roi_of_patch = test_roi_ids("test")
    comp_of_patch = scene_component_ids("test")   # P0-2: scene-connected components = the true unit
    ids = np.arange(n_test)
    # SCENE-LEVEL train/test separation (C6 leak-guard, added 2026-07-22). CloudSEN12's shipped
    # splits are NOT Sentinel-2-product-disjoint: 4 products appear in BOTH train and test, so a
    # scene the model trained on can reappear in the conformal calibration/eval set. The reliability
    # GAP (naive vs Mondrian, same trained model) is invariant to this, but the absolute accuracy and
    # risks are not, so we drop every test patch whose s2_id also occurs in TRAIN. (val is not used
    # by this experiment, and val n test = 0 in any case.)
    train_products = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leak = meta["s2_id"].isin(train_products).to_numpy()
    if leak.any():
        print(f"[leak-guard] dropping {int(leak.sum())} of {n_test} test patches whose Sentinel-2 "
              f"product also appears in TRAIN (scene-level train/test separation)")
        ids = ids[~leak[ids]]
    if args.patches_test is not None and args.patches_test < ids.size:
        ids = np.sort(np.random.default_rng(70000).choice(ids, size=args.patches_test,
                                                          replace=False))
    def load_test(product, want_pid=False):
        return P8.load_split("test", product, pixels_per_patch=args.px_test, patch_ids=ids,
                             seed=args.test_pixel_seed, return_patch_id=want_pid)
    X_l1c, y_te, pid_te = load_test("L1C", want_pid=True)
    X_l2a, y_l2a = load_test("L2A")
    if not np.array_equal(y_te, y_l2a):
        raise ValueError("L1C/L2A pixel misalignment: the two products must enumerate the SAME "
                         "pixels in the same order, or every state comparison is between "
                         "different ground")
    rid_te = roi_of_patch[pid_te]                 # ROI id per pixel (reported, and the P0-2 baseline)
    cid_te = comp_of_patch[pid_te]                # scene-component id per pixel = the EXCHANGEABLE UNIT
    if np.unique(cid_te).size < 3:
        raise ValueError(f"only {np.unique(cid_te).size} distinct scene-components -- need >= 3")

    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr = norm(Xtr)
    Xte = {"l1c": norm(X_l1c), "l2a": norm(X_l2a)}
    jobs = [(ss, ms) for ss in args.split_seeds for ms in args.model_seeds]
    print(f"train {Xtr.shape[0]} px | test {Xte['l1c'].shape[0]} px over "
          f"{np.unique(rid_te).size} ROIs -> {np.unique(cid_te).size} scene-components (the unit) | "
          f"alpha {args.alpha:.0%} | "
          f"{len(args.split_seeds)} splits x {len(args.model_seeds)} model seeds = {len(jobs)} runs")

    # H-10 (run_two_stage): train each model seed ONCE, then evaluate every (split_seed, model_seed)
    # cell against it. Wherever training is deterministic it depends on model_seed alone
    # (train_models), so the old per-cell loop retrained the identical model once per split seed --
    # 90% of the campaign's training on a 10x10 design. `jobs` keeps its split-major order, so the
    # rows, the CSV and every aggregate come out in exactly the order run_seed produced them.
    results, compute = run_two_stage(
        eval_job, jobs,
        train_shared=dict(Xtr=Xtr, ytr=ytr, groups=groups, cwl=cwl,
                          subsample_frac=args.subsample_frac, epochs=args.epochs),
        # rid_te is the EXCHANGEABLE UNIT handed to the split and CRC grouping: pass the scene-component
        # ids (P0-2), not the raw roi_ids, so two ROIs sharing a Sentinel-2 product are one unit.
        eval_shared=dict(Xte=Xte, yte=y_te, rid_te=cid_te, groups=groups, target=args.alpha),
        prefer=args.device, n_jobs=args.jobs, deterministic=not args.nondeterministic,
        label="phase8R")
    rows = [r for sub in results for r in sub]

    # ---- RAW long-form rows: one per (split_seed, model_seed, method, state, arm) ----------------
    # Without these a result cannot be audited: no paired bootstrap, no way to see which run was
    # infeasible, no threshold or temperature to re-derive the number from. The aggregate below is
    # computed FROM this file, so the two can never disagree.
    raw_fields = ["split_seed", "model_seed", "method", "state", "arm", "acc", "aurc", "augrc",
                  "auroc", "aurc_rawT", "auroc_rawT", "aurc_cleanT", "auroc_cleanT",   # H-3
                  "temperature", "crc_threshold", "crc_calibration_selection_stat",
                  "crc_heldout_empirical_joint_loss", "crc_heldout_coverage",
                  "crc_heldout_selective_risk", "crc_feasible", "n_calib_units", "n_eval_units",
                  "n_temp_roi", "n_calib_roi", "n_eval_roi", "mondrian_plugin_cond_risk",
                  "mondrian_plugin_cov", "naive_plugin_cond_risk", "naive_plugin_cov",
                  # H-2: audit columns that were computed then dropped by extrasaction="ignore".
                  # The margin's design-effect n, its icc, the degeneracy flag, and the ROW-SIZED
                  # (pixel-unit) plug-in coverage/risk are the evidence that the correlated-unit
                  # correction is real -- without them the CSV cannot be re-derived or challenged.
                  "plugin_margin_neff", "plugin_margin_icc", "plugin_degenerate",
                  "mondrian_plugin_cov_rowsized", "naive_plugin_cov_rowsized",
                  "mondrian_plugin_cond_risk_rowsized", "naive_plugin_cond_risk_rowsized"]
    with open(P(f"results_phase8R_raw{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    # ---- aggregate: the estimand is the MEAN over independent runs, with a paired contrast --------
    states = ["clean", "dropB10", "dropB1B9B10", "L2A_real"]
    ARMS = ["naive", "naiveThr_freshT", "mondrian"]

    def sel(meth, st, arm, feasible_only=True):
        v = [r for r in rows if r["method"] == meth and r["state"] == st and r["arm"] == arm]
        return [r for r in v if r["crc_feasible"]] if feasible_only else v

    def two_way_se(a, splits, models):
        """Cameron-Gelbach-Miller two-way cluster-robust SE of the mean (P0-3). The cells are a CROSSED
        model-seed x split-seed design (e.g. 10 x 10 = 100), not iid draws: a given model_seed recurs
        across all splits (same trained model, same subsample) and a given split_seed across all models
        (same ROI partition), so an iid std/sqrt(n) understates the SE. SE^2 = V_split + V_model - V_iid,
        each V the cluster-robust variance of the mean treating that factor as the cluster. Degrades
        to the iid SE when neither factor clusters."""
        N = a.size
        e = a - a.mean()
        def V(labels):
            lab = np.asarray(labels, dtype=object)
            return sum(float(e[lab == g].sum()) ** 2 for g in set(lab.tolist())) / N ** 2
        viid = float((e ** 2).sum()) / N ** 2
        core = V(splits) + V(models) - viid
        return float(np.sqrt(core if core > 0 else viid))   # iid floor: never a misleading 0 on non-constant data

    def mstat(vals, splits=None, models=None):
        a = np.asarray(vals, float)
        if a.size == 0:
            return float("nan"), float("nan"), 0
        if a.size == 1:
            return float(a[0]), 0.0, 1
        # The estimand is the MEAN; report its SE. With the (split_seed, model_seed) labels use the
        # two-way cluster-robust SE (the design is crossed, so iid std/sqrt(n) is optimistic).
        if splits is not None and models is not None:
            se = two_way_se(a, splits, models)
        else:
            se = float(a.std(ddof=1) / np.sqrt(a.size))
        return float(a.mean()), se, a.size

    agg_fields = ["method", "state", "arm", "n_runs", "n_feasible", "feasible_rate",
                  "mean_heldout_joint_loss_pct", "se_heldout_joint_loss_pct",
                  "mean_coverage_pct", "se_coverage_pct", "mean_selection_stat_pct",
                  "mean_selective_risk_pct", "mean_acc_pct", "mean_calib_units",
                  "paired_minus_mondrian_pct", "se_paired_minus_mondrian_pct"]
    with open(P(f"results_phase8R_reliability{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields); w.writeheader()
        for meth in METHODS:
            for st in states:
                mond = {(r["split_seed"], r["model_seed"]): r
                        for r in sel(meth, st, "mondrian")}
                for arm in ARMS:
                    fe = sel(meth, st, arm)
                    allr = sel(meth, st, arm, feasible_only=False)
                    fss = [r["split_seed"] for r in fe]; fms = [r["model_seed"] for r in fe]
                    m_loss, se_loss, k = mstat([r["crc_heldout_empirical_joint_loss"] for r in fe],
                                               fss, fms)
                    m_cov, se_cov, _ = mstat([r["crc_heldout_coverage"] for r in fe], fss, fms)
                    # PAIRED contrast on the runs where BOTH arms certified -- comparing two means
                    # taken over different feasible subsets is not a comparison of the arms.
                    dp = [(r["split_seed"], r["model_seed"],
                           r["crc_heldout_empirical_joint_loss"]
                           - mond[(r["split_seed"], r["model_seed"])]["crc_heldout_empirical_joint_loss"])
                          for r in fe if (r["split_seed"], r["model_seed"]) in mond]
                    m_d, se_d, _ = mstat([x[2] for x in dp], [x[0] for x in dp], [x[1] for x in dp])
                    w.writerow({
                        "method": meth, "state": st, "arm": arm,
                        "n_runs": len(allr), "n_feasible": k,
                        "feasible_rate": f"{(k / len(allr) if allr else float('nan')):.3f}",
                        "mean_heldout_joint_loss_pct": f"{m_loss:.3f}",
                        "se_heldout_joint_loss_pct": f"{se_loss:.3f}",
                        "mean_coverage_pct": f"{m_cov:.3f}", "se_coverage_pct": f"{se_cov:.3f}",
                        "mean_selection_stat_pct":
                            f"{mstat([r['crc_calibration_selection_stat'] for r in fe])[0]:.3f}",
                        "mean_selective_risk_pct":
                            f"{mstat([r['crc_heldout_selective_risk'] for r in fe])[0]:.3f}",
                        "mean_acc_pct": f"{mstat([r['acc'] for r in allr])[0]:.3f}",
                        "mean_calib_units": f"{mstat([r['n_calib_units'] for r in fe])[0]:.1f}",
                        "paired_minus_mondrian_pct": f"{m_d:.3f}",
                        "se_paired_minus_mondrian_pct": f"{se_d:.3f}"})

    # ---- figure ----------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    xs = np.arange(len(states))
    style = [("naive", "Naive (clean-calibrated)", "#c0392b"),
             ("naiveThr_freshT", "Naive thr + state temp (control)", "#8e6a00"),
             ("mondrian", "Degradation-aware (Mondrian)", "#1f6f3a")]
    for key, lab, col in style:
        m = [mstat([r["crc_heldout_empirical_joint_loss"] for r in sel("proposed", st, key)])
             for st in states]
        ax.errorbar(xs, [x[0] for x in m], yerr=[x[1] for x in m], marker="o", lw=1.8, ms=4,
                    color=col, label=lab, capsize=2)
        ax.plot(xs, [mstat([r["crc_calibration_selection_stat"] for r in sel("proposed", st, key)])[0]
                     for st in states], ls="--", lw=1.1, color=col, marker="o", ms=4,
                mfc="white", mew=1.1, alpha=0.9)
        c = [mstat([r["crc_heldout_coverage"] for r in sel("proposed", st, key)]) for st in states]
        axc.errorbar(xs, [x[0] for x in c], yerr=[x[1] for x in c], marker="o", lw=1.8, ms=4,
                     color=col, capsize=2)
    ax.axhline(args.alpha * 100, ls=":", color="k", lw=1, label=f"α={args.alpha:.0%}")
    ax.plot([], [], color="0.35", lw=1.8, marker="o", ms=4,
            label="mean held-out  E[kept ∧ wrong]  ± SE")
    ax.plot([], [], color="0.35", lw=1.1, ls="--", marker="o", ms=4, mfc="white",
            label="CRC calibration selection statistic")
    for a in (ax, axc):
        a.set_xticks(xs); a.set_xticklabels(states, rotation=20, ha="right", fontsize=7)
        a.grid(alpha=0.3)
    ax.set_ylabel("JOINT risk  E[accepted ∧ wrong]  (%)")
    ax.set_title(f"Mean over {len(jobs)} runs (Proposed)", fontsize=8.5)
    ax.legend(fontsize=5.5, frameon=False)
    axc.set_ylabel("Coverage at the CRC threshold (%)")
    axc.set_title("...and the coverage it was bought with", fontsize=8.5)
    fig.tight_layout(); fig.savefig(P(f"figs/fig_phase8R_conformal_shift{sfx}.pdf")); plt.close(fig)

    # ---- console ---------------------------------------------------------------------------------
    print(f"\n===== Phase 8R (real S2; mean over {len(jobs)} runs = "
          f"{len(args.split_seeds)} ROI splits x {len(args.model_seeds)} model seeds; alpha "
          f"{args.alpha:.0%}) =====")
    for meth in METHODS:
        print(f"\n[{meth}]  state         arm                 mean held-out loss ± SE   "
              f"coverage ± SE      sel.stat   feas   paired vs Mondrian ± SE")
        for st in states:
            for arm in ARMS:
                fe = sel(meth, st, arm); allr = sel(meth, st, arm, feasible_only=False)
                ml, sl, k = mstat([r["crc_heldout_empirical_joint_loss"] for r in fe])
                mc, sc, _ = mstat([r["crc_heldout_coverage"] for r in fe])
                ss_ = mstat([r["crc_calibration_selection_stat"] for r in fe])[0]
                mond = {(r["split_seed"], r["model_seed"]): r for r in sel(meth, st, "mondrian")}
                dd = [r["crc_heldout_empirical_joint_loss"]
                      - mond[(r["split_seed"], r["model_seed"])]["crc_heldout_empirical_joint_loss"]
                      for r in fe if (r["split_seed"], r["model_seed"]) in mond]
                md, sd_, _ = mstat(dd)
                print(f"      {st:<12}  {arm:<18}  {ml:6.2f} ± {sl:4.2f}          "
                      f"{mc:6.2f} ± {sc:4.2f}   {ss_:6.2f}  {k:2d}/{len(allr):<2d}  "
                      f"{md:+6.2f} ± {sd_:4.2f}")
    print("\nKEY: the estimand is the MEAN held-out joint loss over INDEPENDENT (ROI split, model")
    print("     seed) draws -- that is the quantity CRC Theorem 2.1 bounds. A single run's loss")
    print("     above alpha refutes nothing, and no cell here is labelled a voided certificate.")
    print("     'sel.stat' is the calibration selection statistic, <= alpha by construction whenever")
    print("     a threshold was feasible, so it can never show a failure on its own.")
    print("     READ EVERY LOSS WITH ITS COVERAGE: a joint mass goes to 0 by abstaining.")
    print("     'paired vs Mondrian' is computed only on runs where BOTH arms certified.")
    print("     Temperature is fitted on ROIs DISJOINT from the CRC calibration ROIs.")
    print("     dropB10 / dropB1B9B10 isolate band loss; L2A_real is an OPERATIONAL L1C->L2A domain")
    print("     shift (BOA reflectance, atmospheric correction, B10 absent) -- do not attribute the")
    print("     clean->L2A_real difference to band loss alone.")

    prov = {"target_risk": args.alpha,
            "estimand": "MEAN held-out group joint loss over (split_seed, model_seed) draws, on the "
                        "FIXED sampled pixels (H-8: px-test pixels/patch, one pixel-sampling seed "
                        f"{args.test_pixel_seed} shared across all runs, so the reported SE is over the ROI split and "
                        "model init ONLY -- it does NOT include pixel-sampling uncertainty, and the "
                        "controlled quantity is the loss on this ROI-composite pixel sample, not the "
                        "full-image or full-patch pixel risk); NOT a per-run certificate check",
            "se_method": f"two-way cluster-robust over split_seed x model_seed (P0-3; the {len(jobs)} "
                         f"cells are a crossed {len(args.model_seeds)}-model x {len(args.split_seeds)}-"
                         f"split design, not {len(jobs)} iid draws)",
            "method_comparison_caveat": "H-9: proposed (SGMAE pretrain + finetune) and b2 (supervised "
                                        "MLP) are NOT compute/step/exposure-matched; this file's finding "
                                        "is the WITHIN-method naive-vs-Mondrian coverage gap, not a "
                                        "proposed-beats-b2 claim",
            "exchangeable_unit": "scene-connected component (P0-2: ROIs sharing any s2_id unioned; "
                                 "195 test ROIs -> 184 components), used on BOTH the split and the CRC "
                                 "grouping",
            "roi_split": f"three-way disjoint: temp {TEMP_FRAC:.0%} / calib {CALIB_FRAC:.0%} / eval "
                         f"{1 - TEMP_FRAC - CALIB_FRAC:.0%}, re-drawn per split_seed",
            "temperature_design": "fitted on TEMPERATURE ROIs, disjoint from the CRC calibration "
                                  "ROIs (temperature reorders MSP across samples, so fitting it on "
                                  "the calibration set breaks exchangeability)",
            "n_runs": len(jobs), "n_split_seeds": len(args.split_seeds),
            "train_protocol": "H-10 two-stage (run_two_stage): each model_seed trained ONCE (train_job), "
                              "then every (split_seed, model_seed) cell evaluated against that model "
                              "(eval_job) -- one model per seed across all splits, the crossed design "
                              "two_way_se assumes. It reproduces per-cell retraining (run_seed) if "
                              "training and evaluation are bitwise deterministic in this run's "
                              "configuration (compute: device, threads_per_worker, deterministic_flag) "
                              "and identical across the GPUs and processes used -- observed on 4x P100 "
                              "(torch 2.6.0+cu124, CUDA, deterministic=True) and, on small synthetic "
                              "campaigns, on the CPU at one thread per worker; not guaranteed elsewhere",
            "compute": compute,
            # Recomputed with train_models' own expression (k = round(frac * n)), not stored from a
            # worker: every job draws the same-SIZED subsample, so the value is identical across
            # runs and deterministic from args. A hyperparameter -- see P2.auto_bs.
            "train_bs": int(P2.auto_bs(max(1, int(round(args.subsample_frac * Xtr.shape[0]))))),
            "n_model_seeds": len(args.model_seeds),
            "n_test_rois": int(np.unique(rid_te).size),
            "training_subsample": "PIXEL subsample, not ROI subsample",
            "states": states, "arms": ARMS, "methods": METHODS}
    for nm in ("reliability", "raw"):
        stamp(P(f"results_phase8R_{nm}{sfx}.csv"), args, extra=prov)
    print(f"\nwrote: {P(f'results_phase8R_reliability{sfx}.csv')}")
    print(f"       {P(f'results_phase8R_raw{sfx}.csv')}  (long-form, one row per run x state x arm)")
    print(f"       {P(f'figs/fig_phase8R_conformal_shift{sfx}.pdf')}")


if __name__ == "__main__":
    main()
