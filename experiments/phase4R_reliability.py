#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4R (★honest ace) — reliability / abstention under missing bands.

Everyone TOLERATES missing bands; to our knowledge — after the literature search documented in
docs/review/ (SEnSeI, HyperspectralMAE, DOFA, Panopticon all target tolerance, not abstention) — no
prior work quantifies WHEN NOT TO TRUST the prediction under missing bands. "To our knowledge" and a
citable search, not a bare "the first": novelty is a claim about the literature and this file cannot
evidence one. What it produces, for HSI PER-PIXEL classification (each pixel's spectrum is classified
independently — no spatial convolution or neighbourhood context, so "segmentation" would oversell it):
  - risk-coverage curves at increasing #missing groups (abstention recovers low risk)
  - AURC / AUGRC and misclassification-detection AUROC vs #missing groups
  - per-m conformal risk control: an APPROXIMATE finite-sample bound (CRC, Angelopoulos+ 2022) on
    the JOINT confidently-wrong mass P(accepted AND wrong) <= 10%, with a feasibility flag —
    reported ALONGSIDE an APPROXIMATE plug-in operating point targeting the CONDITIONAL selective
    risk P(wrong | accepted). The two estimands are kept clearly distinct, and neither is a
    guarantee HERE: see reliability_at_m for why this scene only approximates exchangeability.
    The rigorous certificate is phase8R (ROI-level split on real Sentinel-2).
using temperature-scaled max-softmax confidence. See docs/guide/EXP_PHASE4R_PLAN.md.

Compares the Proposed model against the strongest baseline B2 (band-group dropout MLP).
Reuses Phase 2 training (import from phase2_degradation).

TWO CONFORMAL ARMS, TWO EXCHANGEABLE UNITS, TWO DIFFERENT QUESTIONS. Which one you want depends
entirely on what is RANDOM when the system is deployed:
  crc_*     unit = spatial BLOCK. Answers "a new REGION of this scene, under the SAME band-loss
            patterns calibration saw". n ~ 27 blocks.
  crcmask_* unit = the DROP MASK. Answers "a genuinely NEW instrument failure". n = --trials.
            This is the operational question an instrument-failure argument is really making, and
            it is far harder: CRC needs B/(n+1) <= alpha, so n >= 1/alpha - 1 masks. At alpha=10%
            that is >= 9 DISTINCT FAILURE PATTERNS before the bound can even be stated, and the
            original --trials 8 could not state it at any threshold. Reported infeasible, loudly,
            rather than quietly borrowing the block arm's larger n.

Outputs (../paper/):
  figs/fig_risk_coverage.pdf         - Proposed risk-coverage at m=0/3/6 (KEY FIGURE)
  figs/fig_reliability_vs_missing.pdf- AURC & selective-AUROC vs #missing (Proposed vs B2)
  results_phase4R_reliability.csv    - per-m aggregate: mean, std and n_valid for every metric
  results_phase4R_raw.csv            - LONG FORM, one row per (seed, method, m, metric); the
                                       aggregate is lossy and the re-run costs GPU-hours
  results_phase4R_paired.csv         - per-m paired Proposed-minus-B2 differences
  results_phase4R_summary.tex

OUTPUT PROTECTION IS BY CONFIG, NOT BY --smoke. The deliverables are written ONLY when the run
matches CANONICAL exactly; anything else is redirected to a suffixed path naming what differed.
--smoke was previously the only guarded case, so `--epochs 3` or `--max-missing 4` still overwrote
the paper's numbers in place. That is the same defect wearing different flags: guard the CLASS.

Usage:
  python experiments/phase4R_reliability.py --smoke          # 1 seed, quick (_smoke outputs)
  python experiments/phase4R_reliability.py --seeds 0 1 2 3 4
"""
import os
import sys
import csv
import hashlib
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch  # threads/device configured adaptively by bandsim.hw / bandsim.parallel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # for phase2_degradation
sys.path.insert(0, os.path.dirname(_HERE))      # for bandsim
from phase2_degradation import (load_data, prep, train_mlp, pretrain_sgmae,
                                finetune_proposed, group_present_mask, zero_missing, NUM_CLASSES,
                                _vec_group_subset)
from bandsim.grouping import build_group_matrix
from bandsim.model import MLPBaseline
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention
from bandsim.io import AVIRIS_WL_NM, disjoint_block_split
from bandsim import hw, parallel
from bandsim.provenance import stamp
from bandsim.reliability import (confidence_msp, risk_coverage_curve, aurc, augrc,
                                 selective_auroc, fit_temperature, conformal_at_risk,
                                 conformal_risk_control)

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)


class _MaskAwareMLP(torch.nn.Module):
    """B2's MLP with the group-presence mask CONCATENATED to the zero-filled spectrum.

    The ablation this exists for. B2 sees a spectrum with the missing groups set to 0, which in
    standardized space is the training MEAN -- so the information "this group is absent" is present
    but only as an exact-zero coincidence a linear first layer cannot cleanly detect. The proposed
    model is handed the presence mask explicitly. Any gap between them therefore confounds TWO
    things: the architecture (cross-band attention, wavelength PE, SGMAE pretraining) and the mere
    fact of being told what is missing. This baseline holds the second constant so the first can be
    read on its own -- it is B2's exact network with G extra input units.
    """

    def __init__(self, n_bands, n_groups, num_classes, hidden=256):
        super().__init__()
        self.net = MLPBaseline(n_bands + n_groups, num_classes, hidden=hidden)

    def forward(self, x, present):
        return self.net(torch.cat([x, present], dim=1))


def train_mlp_maskaware(Xtr, ytr, groups, seed, epochs=60, hidden=256, lr=1e-3, bs=256,
                        num_classes=None):
    """B2M: train _MaskAwareMLP on a BUDGET IDENTICAL to train_mlp(group_dropout=True).

    Same optimiser, learning rate, batch size, epoch count, hidden width, and -- critically -- the
    same two RNG streams at the same offsets, so B2 and B2M see the IDENTICAL sequence of dropout
    masks and the identical minibatch ordering. The only difference between the two runs is whether
    the network is told which groups were dropped. Anything else would put the comparison back in
    the same confound it exists to remove.

    Deliberately local to this file rather than added to phase2_degradation: that module is edited
    by many concurrent branches, and this baseline is only needed for phase4R's question. It reuses
    phase2's own primitives (`_vec_group_subset`, `build_group_matrix`, `MLPBaseline`) so it cannot
    drift from B2's dropout distribution. Promote it if another phase ever needs it.
    """
    dev = hw.device()
    torch.manual_seed(seed)
    shuffle_rng = np.random.default_rng(seed)            # identical to train_mlp
    mask_rng = np.random.default_rng(seed + 1009)        # identical to train_mlp
    G = len(groups); C = Xtr.shape[1]
    model = _MaskAwareMLP(C, G, NUM_CLASSES if num_classes is None else num_classes,
                          hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(ytr).long().to(dev)
    n = Xt.shape[0]
    Mgb = build_group_matrix(C, groups).astype(np.float32)
    model.train()
    for _ in range(epochs):
        perm = torch.from_numpy(shuffle_rng.permutation(n)).to(dev)
        for s in range(0, n, bs):
            bi = perm[s:s + bs]
            xb = Xt[bi]; yb = yt[bi]
            # same call, same stream position, same distribution as train_mlp's dropout arm
            drop = _vec_group_subset(mask_rng, xb.shape[0], G, 0, 4, leave_one=True).astype(np.float32)
            bmask = (drop @ Mgb == 0.0).astype(np.float32)
            xb = xb * torch.from_numpy(bmask).to(dev)
            present = torch.from_numpy(1.0 - drop).to(dev)          # (b,G), 1 = group present
            opt.zero_grad()
            loss = lossf(model(xb, present), yb)
            loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def logits_maskaware(model, X, groups, drop):
    model.eval()                                    # see logits_proposed
    dev = next(model.parameters()).device
    pm = group_present_mask(X.shape[0], groups, drop).astype(np.float32)
    xz = zero_missing(X, groups, drop)
    return model(torch.from_numpy(xz).to(dev), torch.from_numpy(pm).to(dev)).cpu().numpy()


@torch.no_grad()
def logits_proposed(model, X, groups, drop):
    # eval() as well as no_grad(): they are NOT the same switch. no_grad() stops the graph being
    # built; eval() is what puts Dropout/BatchNorm into inference behaviour. Today every trainer in
    # phase2_degradation returns an already-eval() model and both nets default to dropout=0.0, so
    # this changes no number -- it removes the IMPLICIT CONTRACT that a future dropout>0, or a
    # trainer that forgets its final eval(), would break silently inside the confidence scores this
    # entire file's conclusions are computed from.
    model.eval()
    dev = next(model.parameters()).device
    pm = group_present_mask(X.shape[0], groups, drop)
    return model(torch.from_numpy(X).to(dev), torch.from_numpy(pm).to(dev)).cpu().numpy()


@torch.no_grad()
def logits_mlp(model, X, groups, drop):
    model.eval()                                    # see logits_proposed
    dev = next(model.parameters()).device
    return model(torch.from_numpy(zero_missing(X, groups, drop)).to(dev)).cpu().numpy()


def pooled_logits(kind, model, X, y, gid, groups, m, trials, rng):
    """Pool logits+labels over `trials` random drops of m groups. Returns (logits, labels, unit_id).

    Every pixel reappears once per trial, so the pool holds len(X)*trials ROWS but only len(X)
    distinct pixels. `gid` (one exchangeable-unit id per row of X) is tiled alongside so the caller
    can hand conformal_risk_control the SPATIAL BLOCK each row came from -- counting the repeats as
    independent calibration units would shrink CRC's B/(n+1) term by a factor of `trials`.

    Row i of the pool belongs to trial i // len(X), so the caller can derive a MASK id without this
    function returning one -- which is how the mask-level conformal arm gets its exchangeable unit."""
    G = len(groups)
    # Validate at the ENTRY, not by whatever numpy raises several frames down. Each of these was
    # reachable and none announced itself:
    #   * `kind` fell through `if kind == "proposed" ... else` to the MLP path, so a typo
    #     ("propoesd") silently evaluated B2 twice and reported it as the proposed-vs-B2 comparison;
    #   * m < 0 failed `m > 0` and ran as the CLEAN condition, mislabelled as a missing-band level;
    #   * m > G surfaced as "Cannot take a larger sample than population" from inside rng.choice;
    #   * trials < 1 produced an empty list and died in np.concatenate AFTER the models had trained.
    if kind not in METHODS:
        raise ValueError(f"kind must be one of {METHODS}, got {kind!r} -- an unrecognised kind "
                         f"used to fall through to the B2 path and be reported as the other method")
    if not 0 <= int(m) <= G:
        raise ValueError(f"m must lie in [0, {G}] (number of band groups), got {m}")
    if int(trials) < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    L, Y, U = [], [], []
    for _ in range(trials if m > 0 else 1):
        drop = rng.choice(G, size=m, replace=False).tolist() if m > 0 else []
        lg = globals()[_LOGITS_FN[kind]](model, X, groups, drop)
        L.append(lg); Y.append(y); U.append(gid)
    return np.concatenate(L), np.concatenate(Y), np.concatenate(U)


def _per_mask_stats(corr, conf, msk, thr):
    """(accuracy, coverage, selective risk) for EACH drop mask separately, at threshold `thr`.

    The pooled metrics everywhere else average over the mask mixture, which answers "how bad is a
    RANDOM band loss". An instrument-failure argument is usually about a PARTICULAR one -- the
    detector that actually died -- so the mixture mean can look survivable while one mask inside it
    is catastrophic. These are the per-mask numbers the mean is hiding.
    """
    acc, cov, sel = [], [], []
    kept_all = conf >= thr
    for u in np.unique(msk):
        s = msk == u
        acc.append(float(corr[s].mean()))
        k = s & kept_all
        cov.append(float(k.sum()) / max(1, int(s.sum())))
        sel.append(float(1.0 - corr[k].mean()) if k.any() else float("nan"))
    return np.array(acc), np.array(cov), np.array(sel)


def _per_class_stats(corr, conf, labels, thr, n_classes):
    """(coverage, selective risk) per CLASS at threshold `thr`, plus the number of classes present.

    Indian Pines is severely imbalanced, and a selective classifier can buy a flattering GLOBAL
    coverage by abstaining almost entirely on the small classes -- the abstention lands where the
    pixels are few, so it barely shows in the pooled number. Per-class coverage is what exposes
    that: a class with 0% coverage has been silently dropped from the map, and no global statistic
    in this file would say so.
    """
    kept = conf >= thr
    cov, sel = [], []
    for c in range(int(n_classes)):
        s = labels == c
        if not s.any():
            continue                       # class absent from this eval split; not a zero-coverage
        k = s & kept
        cov.append(float(k.sum()) / int(s.sum()))
        sel.append(float(1.0 - corr[k].mean()) if k.any() else float("nan"))
    return np.array(cov), np.array(sel)


def reliability_at_m(logits_t, labels_t, logits_c, labels_c, blk_c, logits_e, labels_e, blk_e,
                     *, target_risk, msk_c, msk_e, logits_ei, labels_ei, msk_ei):
    """Fit temperature on a SEPARATE temp set, calibrate conformal on a DISJOINT calib set, and
    evaluate on a THIRD disjoint eval set. temp/calib/eval are disjoint test PIXELS, so the score
    function (temperature) is fixed BEFORE conformal calibration -- a shared temp+calib set would
    let the temperature adapt to the very pixels CRC then calibrates on.

    Two conformal quantities are returned, with DIFFERENT ESTIMANDS and DIFFERENT status:
      * crc_*  : conformal_risk_control (Angelopoulos+ 2022) on the JOINT mass
        P(accepted AND wrong) <= target_risk. NOT the conditional error rate among accepted pixels,
        and meaningless without crc_coverage next to it (abstaining drives the joint mass to 0).
        `blk_c`/`blk_e` are SPATIAL-BLOCK ids: the exchangeable unit is the checkerboard block, so
        the `trials` repeated rows of one pixel -- and the mutually correlated pixels of one block --
        collapse to ONE calibration unit instead of inflating n (crc_units reports the surviving n).
      * crcmask_* : the SAME CRC machinery with the DROP MASK as the exchangeable unit instead of
        the block, moving toward the estimand an instrument-failure claim needs -- "a NEW failure
        pattern" rather than "a new region under the failures we already saw". n is `trials`, so the
        B/(n+1) term alone needs n >= B/alpha - 1 (9 at alpha=10%) before any bound exists, and it
        requires the OPPOSITE mask arrangement from the block arm: calib and eval draw INDEPENDENT
        masks, because the test unit must be a mask calibration never saw. Hence `logits_ei`.
        Undefined at m=0 (one "no bands missing" pattern, nothing to average over) -> NaN, including
        its unit count and its floor.

        READ THIS BEFORE QUOTING IT AS "CERTIFIED FOR A NEW FAILURE". IT IS NOT, AND CANNOT BE HERE.
        The exchangeable unit is really the PAIR (mask, pixel-set), and the pixel-set dimension has
        n = 1: every calibration loss is computed on the one calibration region, every evaluation
        loss on the one evaluation region. Both readings of exchangeability fail. Conditional on the
        two regions the calibration losses are i.i.d. but the test loss has a DIFFERENT MARGINAL --
        different pixels, different class composition. Treating the regions as random instead makes
        all n calibration units mutually correlated through the shared calibration region while the
        test unit is not -- which is precisely the dependence asymmetry the block arm's shared mask
        stream was introduced to remove, transposed into the region dimension.

        Measured, one fixed calib region against one fixed eval region with only masks resampled,
        `gap` being an eval overconfidence shift in logits:

            gap   trials   bound%   realized%   P(realized > alpha)
            0.0       32     9.93        7.19                  0.5%
            0.0      128     9.95        9.30                 13.5%
            0.5       32     9.93       10.83                 80.0%
            0.5      128     9.95       12.51                100.0%
            1.0       32     9.93       14.36                100.0%

        The bound never moves; only the realization does. And RAISING --trials makes it WORSE: the
        only thing protecting the arm is the B/(n+1) conservatism slack, which shrinks from 2.74 to
        0.65 points as n goes 32 -> 128. So read crcmask_* as a bound CONDITIONAL ON THIS
        CALIBRATION REGION, exposed to any calibration-vs-evaluation shift -- and a shift is a bias,
        which no finite-sample term shrinks. Making the region part of the draw needs MANY regions,
        i.e. many scenes; that is phase8R, where a whole location is the unit.
      * conf_* : conformal_at_risk — an APPROXIMATE plug-in operating point targeting the
        CONDITIONAL selective risk P(wrong | accepted). NOT a guarantee: the achieved eval risk only
        fluctuates around the target, so report it descriptively. `blk_c` goes to it too: its
        finite-sample margin must be sized from an EFFECTIVE sample size, and the pooled rows are
        neither independent nor as few as the blocks.

        DO NOT read the margin as the explanation of the committed table. The old run had
        plugin_cond_risk above its 10% target in all 14 cells (10.65-16.68). That FACT is real, but
        the row-count margin is not a sufficient cause: the old effective target was 0.0974 and the
        residual above THAT is +3.1 points, still one-sided 14/14. A row-sized margin straddles its
        target in simulation even under heavy clustering, so something else is one-sided --
        winner's-curse threshold selection and calib-vs-eval shift within one scene are candidates,
        and NEITHER is fixed by resizing n. Those 14 cells also come from 5 seeds whose test sets
        overlap at IoU 0.448. Attribute nothing until the re-run separates them.

    Even with block grouping, exchangeability HERE is only APPROXIMATE. The reason is that every
    block comes from ONE Indian Pines scene, so adjacent blocks are spatially autocorrelated and
    differ in class composition. A third-split BY BLOCK makes calib and eval blocks disjoint, but
    disjoint is not the same as exchangeable draws from a population -- these are a partition of a
    single image.

    What is NOT the reason, though an earlier version of this docstring said it was: "all rows of a
    trial share ONE drop mask, which no grouping can repair". Sharing a mask does not by itself
    break exchangeability -- CONDITIONAL on the mask set the blocks can still be exchangeable, and a
    bound holding conditionally on every mask set holds marginally. What sharing costs is MONTE
    CARLO SUPPORT in the mask dimension.

    The arrangement that DID break exchangeability for the BLOCK arm was the opposite one, and it
    was here: calib and eval used INDEPENDENT mask streams. Write Z_i = (block_i, masks). With one
    shared set M the Z_i are conditionally i.i.d. given M, hence jointly exchangeable. With M_cal
    for calibration and a different M_eval, the calibration units are mutually correlated THROUGH
    M_cal while the eval unit is not, so swapping Z_1 with Z_{n+1} changes the joint law. Simulated
    against the real conformal_risk_control at 500 reps with a dominant mask effect: separate
    streams put 20.4% of runs above alpha with 5x the spread, shared streams 0.0%. The block arm now
    shares stream 0. (Per-block independent masks -- what phase9 adopted -- tested equivalently well
    but models a sensor that loses different bands in each 10x10 patch, which does not exist.)

    THE MASK ARM NEEDS THE INDEPENDENT DRAWS, for the same reason mirrored: there the unit IS the
    mask, so the test unit must be a mask calibration never saw. Both arrangements are computed.

    crc_calib_bound is an APPROXIMATE finite-sample estimate, NOT a rigorous certificate; the
    RIGOROUS certified result is phase8R (ROI-level split on CloudSEN12). crc_calib_bound (the
    bound) and crc_eval_joint_risk (the realization) are separate outputs because the bound is
    <= target BY CONSTRUCTION whenever feasible and so carries no evidence; only the realization
    does. That is NOT the same as saying a realization above the bound is the control failing --
    Theorem 2.1 bounds E[L_{n+1}] over the JOINT draw of calibration and test, so a single
    realization may exceed alpha while the theorem holds exactly. Testing it means averaging over
    many independent calib/eval draws and checking THAT mean.
    AURC/AUGRC/AUROC are rank-based descriptive metrics (no exchangeability needed)."""
    T = fit_temperature(logits_t, labels_t)
    conf_c = confidence_msp(logits_c / T); corr_c = (logits_c.argmax(1) == labels_c).astype(int)
    conf_e = confidence_msp(logits_e / T); corr_e = (logits_e.argmax(1) == labels_e).astype(int)
    cov, risk = risk_coverage_curve(corr_e, conf_e)
    # APPROXIMATE finite-sample risk control, spatial BLOCK as the exchangeable unit (see docstring)
    crc = conformal_risk_control(corr_c, conf_c, corr_e, conf_e, alpha=target_risk,
                                 calib_group=blk_c, eval_group=blk_e)
    # APPROXIMATE plug-in operating point (NOT a guarantee). Same exchangeable unit as CRC above:
    # the margin is a finite-sample correction and must be sized from the clustering, not from rows.
    conf = conformal_at_risk(corr_c, conf_c, corr_e, conf_e, target_risk=target_risk,
                             calib_group=blk_c)
    # NOT a bracket check on n_eff. `n_clusters <= n_eff <= n_rows` is provable for any integer
    # cluster sizes with n > G, so asserting it can never fail -- it looked like a guard and was a
    # tautology. What actually breaks the design effect is CLUSTER-SIZE IMBALANCE: Kish's k0 is a
    # harmonic-flavoured mean, so one dominant cluster collapses it. Sizes [8614, 1x26] with the
    # outcome constant inside every block (rho = 1 exactly) give k0 = 2.0 instead of 320, hence
    # n_eff = 4327 instead of 27 -- a 160x overstatement that sails through the bracket. The
    # checkerboard makes near-equal blocks today, so this is a guard against a future split, which
    # is exactly when a silent 160x would not be noticed.
    _k0, _nblk, _nrow = conf["k0"], crc["n_calib_units"], len(corr_c)
    if _nblk >= 2 and _k0 < 0.5 * (_nrow / _nblk):
        raise ValueError(
            f"calibration blocks are too unequal for the design-effect margin: Kish k0={_k0:.1f} "
            f"against a mean block size of {_nrow / _nblk:.1f}. One dominant block has collapsed k0, "
            f"so n_eff overstates the information and the margin is sized too small")

    # ---- per-mask and per-class breakdowns at the CERTIFIED threshold ------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN slice -> NaN, not a crash
        _mk_acc, _mk_cov, _mk_sel = _per_mask_stats(corr_e, conf_e, msk_e, crc["threshold"])
        _cl_cov, _cl_sel = _per_class_stats(corr_e, conf_e, labels_e, crc["threshold"], NUM_CLASSES)
    if _cl_cov.size == 0:
        raise ValueError("no class present in the evaluation split -- the per-class breakdown "
                         "would be vacuous and the split is degenerate")

    # ---- MASK-LEVEL arm: exchangeable unit = the DROP MASK (see docstring) --------------------
    # n here is the number of distinct failure patterns, which is `trials`, NOT the block count.
    n_mu = int(np.unique(msk_c).size)
    # B/(n+1) with B = conformal_risk_control's loss bound, which is 1.0 for the 0/1 loss used here.
    # Written as B/(n+1) rather than 1/(n+1) so it stays correct if a caller ever passes B != 1.
    _B = 1.0
    alpha_floor = _B / (n_mu + 1.0)           # no bound below this alpha at ANY threshold
    if n_mu >= 2:
        conf_ei = confidence_msp(logits_ei / T)
        corr_ei = (logits_ei.argmax(1) == labels_ei).astype(int)
        cm = conformal_risk_control(corr_c, conf_c, corr_ei, conf_ei, alpha=target_risk,
                                    calib_group=msk_c, eval_group=msk_ei)
        cm_bound = cm["calib_crc_bound"] * 100
        cm_joint = cm["eval_group_joint_risk"] * 100
        cm_cov = cm["eval_group_coverage"] * 100
        cm_sel = cm["eval_group_selective_risk"] * 100
        cm_feas = float(cm["feasible"])
        n_mu_out = float(n_mu)
    else:
        # Fewer than two mask DRAWS, which is m=0 in any normal run (exactly one "drop nothing"
        # pattern exists, so pooled_logits takes a single pass) and also --trials 1 at any m. Either
        # way there is no failure-pattern variation to average over, so the quantity is UNDEFINED --
        # not infeasible, which would read as a result about the method.
        cm_bound = cm_joint = cm_cov = cm_sel = float("nan")
        cm_feas = float("nan")
        # The unit count and the floor must go undefined TOO. Left finite they wrote '1.00' and
        # '50.00' into a CSV row whose every other mask cell was blank -- two confidently quoted
        # numbers asserting "1 distinct failure pattern" and "no bound below 50%" for a quantity the
        # code declares undefined one line above.
        n_mu_out = float("nan")
        alpha_floor = float("nan")
    return {
        "acc": float(corr_e.mean()) * 100,
        "aurc": aurc(corr_e, conf_e) * 100,
        "augrc": augrc(corr_e, conf_e) * 100,
        "auroc": selective_auroc(corr_e, conf_e) * 100,
        "plugin_coverage": conf["coverage"] * 100,   # plug-in operating-point coverage
        "plugin_cond_risk": conf["risk"] * 100,      # plug-in P(wrong|accepted); NaN if nothing accepted
        # TWO DIFFERENT ESTIMANDS, kept apart on purpose. `crc_calib_bound` is the CRC certificate
        # computed on the calibration set ((n/(n+1))*Rhat + B/(n+1) at the chosen threshold, <= alpha
        # iff crc_feasible); `crc_eval_joint_risk` is the joint mass the EVAL set actually realized
        # at that threshold. They used to share one key that carried the realization while its name
        # promised the bound. Here the bound is APPROXIMATE anyway (see docstring: the units are only
        # approximately exchangeable), so it is not a certificate either -- but realized-above-bound
        # is still the only way to see the control failing, and that needs both numbers.
        # All three from the GROUP (spatial-block) weighted family: that is what the bound is about,
        # and the only family in which joint = selective x coverage. The sample-weighted numbers the
        # core also returns are NOT interchangeable with these once blocks differ in size.
        "crc_calib_bound": crc["calib_crc_bound"] * 100,   # APPROXIMATE bound, NOT a certificate here
        "crc_eval_joint_risk": crc["eval_group_joint_risk"] * 100,  # REALIZED P(accepted AND wrong)
        "crc_coverage": crc["eval_group_coverage"] * 100,  # coverage -- quote WITH the risks
        "crc_sel_risk": crc["eval_group_selective_risk"] * 100,     # = joint / coverage exactly
        # NEAR-VACUOUS, kept only so the CSV shows it is: lambda_max=+inf gives Rhat=0, so the
        # criterion there is just B/(n+1), which clears alpha for any n >= 1/alpha - 1 (9 at
        # alpha=0.10). With ~27 blocks this is 1.0 in EVERY run -- it reports sample size, not
        # success, and conditioning aggregates on it excluded nothing.
        "crc_feasible": float(crc["feasible"]),
        # the flag that actually discriminates: did CRC end up predicting ANYTHING? A run that
        # abstained on everything scores joint risk 0.0, which is not a low risk but an absent one.
        "crc_positive_coverage": float(crc["eval_group_coverage"] > 0.0),
        "crc_units": float(crc["n_calib_units"]),    # exchangeable BLOCKS behind the bound, not rows
        # ---- MASK-level arm: unit = drop mask, n = trials. The operational estimand. ----
        "crcmask_calib_bound": cm_bound,
        "crcmask_eval_joint_risk": cm_joint,
        "crcmask_coverage": cm_cov,
        "crcmask_sel_risk": cm_sel,
        "crcmask_feasible": cm_feas,
        # i.i.d. mask DRAWS -- this IS n for Theorem 2.1, because independent draws are
        # exchangeable whether or not two of them coincide. It is NOT the number of distinct
        # failure patterns: masks are drawn i.i.d. per trial, so duplicates are routine (32 draws
        # cover at most 10 patterns at m=1, where only C(10,1) exist). The banner reports both.
        "crcmask_units": float(n_mu_out),
        # mirrors crc_positive_coverage. Without it every crcmask_* number went through an
        # UNCONDITIONED mean, so an arm that abstained on everything contributed joint risk 0.0 --
        # which is not a low risk but an absent one. That is guaranteed at --smoke (trials=3 puts
        # the floor at 25% against alpha=10%), so every smoke deliverable would have carried a
        # perfect-looking 0.00 at every m>=1.
        "crcmask_positive_coverage": float(cm_cov > 0.0) if np.isfinite(cm_cov) else float("nan"),
        "crcmask_alpha_floor": alpha_floor * 100,     # no bound is possible below this alpha
        # diagnostics that were computed and then dropped on the floor
        "temperature": float(T),
        "plugin_threshold": float(conf["threshold"]),
        # DESIGN-EFFECT effective n, NOT a count of blocks and NOT a count of rows -- it lands
        # between them at whatever the estimated intra-cluster correlation says. Named _neff so it
        # cannot be read as "units" next to crc_units, which IS a block count: the two are
        # different quantities and were briefly identical only while this used the cluster count.
        "plugin_neff": float(conf["n_calib_units"]),
        "plugin_icc": float(conf["icc"]),
        "plugin_eff_target": float(conf["eff_target"]) * 100,
        # mirrors crc_positive_coverage: did the plug-in accept anything at all? plugin_coverage
        # averages a 0.0 from an abstain-all seed while plugin_cond_risk NaN-drops it, so the two
        # columns are conditioned on DIFFERENT seed sets unless this rate is read alongside them.
        "plugin_positive_coverage": float(conf["coverage"] > 0.0),
        "plugin_degenerate": float(conf["degenerate"]),
        "crc_threshold": float(crc["threshold"]),
        # ---- per-MASK spread at the certified operating point (see _per_mask_stats) ----
        # Quoted at crc["threshold"], the threshold actually certified, so these describe the
        # operating point the paper reports rather than some other one. "worst" is the worst mask
        # OBSERVED among --trials draws, which at high m is a small sample of C(G,m): it is a lower
        # bound on how bad the true worst mask is, never an estimate of it.
        "mask_acc_worst": float(np.min(_mk_acc)) * 100,
        "mask_acc_median": float(np.median(_mk_acc)) * 100,
        "mask_acc_p05": float(np.percentile(_mk_acc, 5)) * 100,
        "mask_cov_worst": float(np.min(_mk_cov)) * 100,
        "mask_sel_risk_worst": float(np.nanmax(_mk_sel)) * 100 if np.isfinite(_mk_sel).any() else float("nan"),
        "mask_sel_risk_median": float(np.nanmedian(_mk_sel)) * 100 if np.isfinite(_mk_sel).any() else float("nan"),
        # ---- per-CLASS reliability at the same operating point (see _per_class_stats) ----
        "cls_cov_min": float(np.min(_cl_cov)) * 100,
        "cls_cov_macro": float(np.mean(_cl_cov)) * 100,
        "cls_zero_cov_n": float((_cl_cov <= 0.0).sum()),
        "cls_sel_risk_max": float(np.nanmax(_cl_sel)) * 100 if np.isfinite(_cl_sel).any() else float("nan"),
        "cls_sel_risk_macro": float(np.nanmean(_cl_sel)) * 100 if np.isfinite(_cl_sel).any() else float("nan"),
        "curve": (cov, risk),
    }


# DEFAULT_TARGET_RISK is the argparse DEFAULT for alpha and nothing else. The value a run actually
# uses is `args.target_risk`, threaded explicitly into reliability_at_m (KEYWORD-ONLY, NO default, so
# omitting it is a TypeError rather than a silent revert to 0.10) and into run_seed through
# parallel.run_jobs' `shared` dict -- workers re-import this module, so a module-level constant would
# NOT have carried a CLI override into them.
#
# The old name lied. A comment on `TARGET_RISK = 0.10` called it "the single source of truth for
# alpha" and said the bug it fixed was "a bare 0.10 default on reliability_at_m plus a hardcoded 10%
# in the header" -- but only the header had been fixed. `reliability_at_m(..., target_risk=0.10)` was
# still there and run_seed still passed nothing, so TARGET_RISK = 0.05 would have printed 5%, written
# 5% into the LaTeX and recorded 5% in provenance while every number was computed at 10%.
DEFAULT_TARGET_RISK = 0.10

# The configuration whose outputs ARE the paper's deliverables. Anything else is redirected by
# _config_suffix(). jobs/device are absent: they change speed, not results, and pinning a device
# would make this unrunnable elsewhere (provenance records them). nondeterministic IS here.
#
# trials=32, raised from 8, because the MASK-level arm needs it: its n is the number of distinct
# failure patterns, and CRC cannot state any bound unless B/(n+1) <= alpha, i.e. n >= 1/alpha - 1 = 9
# at alpha=10%. At trials=8 that arm was structurally infeasible at EVERY threshold -- not a finding
# about the method, an artefact of having observed 8 failures. 32 puts the floor at 1/33 = 3.0%,
# leaving 7 points of the 10% budget for the empirical risk. Inference is cheap next to training.
CANONICAL = {"seeds": [0, 1, 2, 3, 4], "groups": 10, "max_missing": 6, "trials": 32,
             "epochs": 60, "nondeterministic": False, "target_risk": DEFAULT_TARGET_RISK}
_ABBR = {"seeds": "s", "groups": "g", "max_missing": "m", "trials": "t", "epochs": "e",
         "nondeterministic": "nondet", "target_risk": "a"}

# b2m is the MATCHED-INFORMATION control: B2's exact network and training budget, plus the
# group-presence mask. proposed-vs-b2 confounds architecture with being told what is missing;
# proposed-vs-b2m isolates the architecture, and b2m-vs-b2 isolates the mask.
METHODS = ["proposed", "b2", "b2m"]
# Function NAMES, resolved through globals() at call time rather than bound function objects.
# Binding the objects here silently defeats monkeypatching, and tests/test_reliability_guards.py
# patches `logits_mlp` to exercise pooled_logits without a real model -- it would have kept
# calling the original and failed on a stub. Late lookup keeps the seam the tests rely on.
_LOGITS_FN = {"proposed": "logits_proposed", "b2": "logits_mlp", "b2m": "logits_maskaware"}
# names spell out the ESTIMAND: plugin_* is CONDITIONAL P(wrong|accepted), crc_* is JOINT
# P(accepted AND wrong) -- and within crc_*, `_calib_bound` is the bound while `_eval_joint_risk` is
# its realization. Never quote either risk without crc_coverage.
METRIC_NAMES = ["acc", "aurc", "augrc", "auroc", "plugin_coverage", "plugin_cond_risk",
                "crc_calib_bound", "crc_eval_joint_risk", "crc_coverage", "crc_sel_risk",
                "crc_feasible", "crc_units", "crc_positive_coverage",
                # mask-level arm: unit = drop mask, n = trials (the operational estimand)
                "crcmask_calib_bound", "crcmask_eval_joint_risk", "crcmask_coverage",
                "crcmask_sel_risk", "crcmask_feasible", "crcmask_units", "crcmask_alpha_floor",
                "crcmask_positive_coverage",
                "temperature", "plugin_neff", "plugin_icc", "plugin_eff_target",
                "plugin_positive_coverage", "plugin_degenerate",
                # per-mask spread: the mixture mean can look survivable while one mask is not
                "mask_acc_worst", "mask_acc_median", "mask_acc_p05", "mask_cov_worst",
                "mask_sel_risk_worst", "mask_sel_risk_median",
                # per-class: a flattering global coverage can be bought by dropping small classes
                "cls_cov_min", "cls_cov_macro", "cls_zero_cov_n",
                "cls_sel_risk_max", "cls_sel_risk_macro"]

# Per-seed ONLY -- written to the raw CSV, never aggregated. A threshold is an OPERATING POINT, not
# a metric: it is +inf when the target was unreachable, and a mean containing one +inf is +inf,
# which reads as "every seed abstained" when only one did. Keeping them out also keeps non-finite
# cells out of the deliverable CSV, which experiments/integrity_check.py rejects -- it would have
# failed a phase that ran perfectly.
RAW_ONLY_METRICS = ["plugin_threshold", "crc_threshold"]
ALL_METRICS = METRIC_NAMES + RAW_ONLY_METRICS

COV_GRID = np.linspace(0.02, 1.0, 100)      # common coverage grid for risk-coverage curves
RC_MS = [0, 3, 6]                            # missing-group levels plotted for proposed

# LaTeX macro names spell the missing-band level out in words. They were hardcoded to "...Six" while
# the VALUE was indexed at args.max_missing, so `--max-missing 4` wrote m=4's number into
# \aurcPropSix. m=6 still maps to "Six", so canonical macro names are unchanged.
_NUMWORD = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
            6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def _mtag(m):
    return _NUMWORD.get(int(m), f"M{int(m)}")


def _config_suffix(args, ignore=()):
    """'' iff `args` matches CANONICAL exactly; otherwise a tag naming what differs.

    `ignore` drops keys whose value is dictated by another flag (--smoke fixes seeds/epochs/trials),
    so the tag names only what the USER chose. Redirect rather than refuse: exploration stays cheap
    and the artefact says on its face that it is not the canonical one.
    """
    diffs = []
    for k in sorted(CANONICAL):
        if k in ignore:
            continue
        want, got = CANONICAL[k], getattr(args, k)
        if isinstance(want, list):
            got = list(got)
        if got != want:
            diffs.append((k, got))
    if not diffs:
        return ""
    parts = []
    for k, v in diffs:
        v = "".join(str(x) for x in v) if isinstance(v, list) else v
        parts.append(f"{_ABBR.get(k, k)}{str(v).replace('.', 'p')}")
    tag = "-".join(parts)
    if len(tag) > 48:                      # keep filenames sane; provenance JSON holds the detail
        tag = hashlib.sha256(tag.encode()).hexdigest()[:12]
    return "_nc-" + tag


def _csv_num(v, fmt=".2f"):
    """Format for a CSV cell, or EMPTY when the quantity is undefined.

    An empty cell and a "nan" cell are not the same statement. The mask-level arm is undefined at
    m=0 -- there is exactly one "nothing missing" pattern, so there is no failure-pattern randomness
    to average over -- and plugin_cond_risk is undefined whenever nothing was accepted. "Not
    applicable here" is what an empty cell means in a CSV, and it is what pandas/numpy read back as
    NaN anyway; "nan" additionally asserts that a numeric computation returned a non-number.

    It also matters operationally: experiments/integrity_check.py parses every cell with float() and
    FAILS the phase on any non-finite value, while skipping cells float() cannot parse at all. A
    "nan" from a legitimately undefined quantity would therefore have failed a run that was correct.
    The _n companion column already says how many seeds stood behind each mean.
    """
    return format(v, fmt) if np.isfinite(v) else ""


def _tex_num(v, fmt=".1f"):
    """Format for a \\newcommand body, or emit an UNDEFINED macro if not finite.

    f"{float('nan'):.1f}" is the string "nan", a perfectly valid \\newcommand that typesets the word
    "nan" into the paper. An undefined control sequence is a hard LaTeX error instead -- the failure
    has to be louder than the thing it is reporting.
    """
    return format(v, fmt) if np.isfinite(v) else r"\phaseFourRUndefined"


def _block_ids(shape, block, offset):
    """(id map, n_bj) for the checkerboard tiling bandsim.io.disjoint_block_split uses.

    Local on purpose: bandsim.io owns the SPLIT, this file only needs to NAME the blocks that split
    produced. Rather than trust that this copy of the formula stays in step with io.py's, the caller
    checks the ids against the split's own checkerboard PARITY -- a copy that has drifted fails that
    check, whereas the previous guard (matching pixel COUNT) stays green while every id shifts.
    """
    H, W = shape[:2]
    b = int(block)
    bi = (np.arange(H)[:, None] + int(offset)) // b
    bj = (np.arange(W)[None, :] + int(offset)) // b
    n_bj = int(np.max(bj)) + 1
    return (bi * n_bj + bj), n_bj


def run_seed(seed, cube, gt, groups_n, max_missing, trials, epochs, target_risk):
    """One seed: train B2 + proposed, then selective-prediction metrics per m.

    NOT an independent replicate, despite what this docstring used to say. `seed` drives the model
    init, the checkerboard OFFSET of the train/test split, the block third-split and the drop-mask
    streams at once, and the offset only shifts the grid by `seed` pixels: measured over seeds 0-4
    at block=10 the mean pairwise test-set IoU is 0.448 (0.333 = independent half-splits, 1.0 =
    identical) and adjacent seeds sit at 0.632. Every across-seed std here is an UNDERESTIMATE.

    Returns {"store": {method: {metric: array-over-m}}, "rc": {m: interp risk curve}}."""
    ms = list(range(0, max_missing + 1))
    Xtr, ytr, Xte, yte = prep(cube, gt, block=10, offset=seed)
    groups = contiguous_groups(Xtr.shape[1], groups_n)
    cwl = group_center_wavelengths(AVIRIS_WL_NM, groups)

    m_b2 = train_mlp(Xtr, ytr, groups, seed, group_dropout=True, epochs=epochs)
    m_b2m = train_mlp_maskaware(Xtr, ytr, groups, seed, epochs=epochs)
    m_prop = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES)
    pretrain_sgmae(m_prop, Xtr, groups, seed, epochs=max(1, epochs // 2))
    finetune_proposed(m_prop, Xtr, ytr, groups, seed, epochs=epochs)
    # NOTE ON BUDGET, since "strongest baseline" is a claim about compute as well as design:
    # b2 and b2m each get `epochs` supervised passes; proposed gets epochs//2 SGMAE pretraining
    # PLUS `epochs` finetuning, i.e. ~1.5x. That asymmetry is real and is NOT corrected here --
    # correcting it means either starving the proposed model or over-training the baselines, and
    # either choice needs to be argued in the paper rather than buried in a training call.
    models = {"proposed": m_prop, "b2": m_b2, "b2m": m_b2m}

    # SPATIAL-BLOCK id per test pixel -> the exchangeable unit handed to CRC. prep() splits
    # train/test on the same `block`x`block` checkerboard (guard=1) and cube[te_mask] enumerates the
    # mask in C order, so argwhere(te_mask) lists each test pixel's (row,col) in Xte's own order.
    n_pix = len(Xte)
    _, te_mask = disjoint_block_split(gt, block=10, guard=1, offset=seed)
    rcs = np.argwhere(te_mask)
    # Explicit raises, not `assert`: `python -O` strips assert statements, and these two guard the
    # exchangeability the conformal numbers rest on. A validation that disappears under an
    # optimisation flag is not a validation.
    if len(rcs) != n_pix:
        raise ValueError(f"block-id order does not match Xte ({len(rcs)} vs {n_pix}) -- "
                         f"prep()/disjoint_block_split drifted")
    id_map, n_bj = _block_ids(gt.shape, block=10, offset=seed)
    blk_all = id_map[rcs[:, 0], rcs[:, 1]]
    # Standing assertion, and NOT a restatement of the formula: it decodes the ids back to (bi, bj)
    # and checks them against the CHECKERBOARD PARITY disjoint_block_split actually used to build
    # te_mask (train = even parity, so every test pixel must be odd). If bandsim.io changes how it
    # tiles or which side is train, this fires; a formula copied from it would not.
    if not np.all(((blk_all // n_bj) + (blk_all % n_bj)) % 2 == 1):
        raise ValueError("block ids disagree with disjoint_block_split's checkerboard parity: test "
                         "pixels must lie in odd-parity cells -- the tiling convention in "
                         "bandsim.io changed and these ids no longer name the split's blocks")

    # 3-way split of TEST PIXELS **BY BLOCK**, not by pixel: temperature is fit on `temp`, conformal
    # is calibrated on `calib`, metrics are evaluated on `eval`.
    #
    # The block is what we hand CRC as its exchangeable unit, so splitting pixels at random would put
    # the SAME block on both sides -- telling CRC it holds ~80 independent calibration units while
    # calibration and evaluation in fact share every one of them. That is not a bookkeeping detail:
    # it is the assumption the guarantee rests on, and with the pixel split the realised joint risk
    # exceeded alpha at the easiest missing levels (10.47% and 10.08% against alpha=10%). Splitting
    # whole blocks makes the calibration and evaluation units genuinely disjoint. It costs sample
    # size -- n_calib falls to about a third of the blocks -- and that cost shows up honestly as a
    # larger B/(n+1) correction rather than as a bound that looks tight because it was computed on
    # units that were not independent.
    ublk = np.unique(blk_all)
    bperm = np.random.default_rng(50000 + seed).permutation(ublk)
    cut1, cut2 = len(bperm) // 3, 2 * (len(bperm) // 3)
    temp_b, calib_b, eval_b = bperm[:cut1], bperm[cut1:cut2], bperm[cut2:]
    temp_pix = np.where(np.isin(blk_all, temp_b))[0]
    calib_pix = np.where(np.isin(blk_all, calib_b))[0]
    eval_pix = np.where(np.isin(blk_all, eval_b))[0]
    Xt, yt = Xte[temp_pix], yte[temp_pix]
    Xc, yc = Xte[calib_pix], yte[calib_pix]
    Xe, ye = Xte[eval_pix], yte[eval_pix]
    blk_c, blk_e = blk_all[calib_pix], blk_all[eval_pix]
    shared = set(blk_c.tolist()) & set(blk_e.tolist())
    if shared:
        raise ValueError(f"calibration and evaluation share {len(shared)} spatial block(s) -- that is "
                         f"exactly the exchangeability CRC needs, so the numbers would be unearned")

    store = {mth: {} for mth in METHODS}
    rc = {}
    for mth in METHODS:
        per_m = {mn: [] for mn in ALL_METRICS}
        for m in ms:
            # THE TWO ARMS NEED OPPOSITE MASK ARRANGEMENTS, so both are built.
            #
            # BLOCK arm (unit = block): calib and eval SHARE stream 0. They used to use independent
            # streams (0 and 1), which reads like leakage hygiene and is the opposite: with
            # different mask sets the calibration units are mutually correlated through THEIR mask
            # set while the eval unit is not, so the sequence is not exchangeable and CRC's theorem
            # does not apply. Simulated at 500 reps with a dominant mask effect, separate streams
            # put 20.4% of runs above alpha against 0.0% shared. Reusing the mask is not leakage:
            # the PIXELS are disjoint and the mask is drawn, not fitted.
            #
            # MASK arm (unit = mask): calib and eval must draw INDEPENDENT masks (stream 1), because
            # there the test unit IS a mask and it has to be one calibration never saw.
            #
            # temp keeps its own stream (2): the temperature is part of the score function and is
            # fixed before calibration, so it must not be tuned on calibration's masks. All streams
            # are identical ACROSS methods, which is what makes proposed-vs-b2 paired.
            lg_t, lb_t, _ = pooled_logits(mth, models[mth], Xt, yt, blk_all[temp_pix], groups, m,
                                          trials, np.random.default_rng([seed, m, 2]))
            lg_c, lb_c, bk_c = pooled_logits(mth, models[mth], Xc, yc, blk_c, groups, m, trials,
                                             np.random.default_rng([seed, m, 0]))
            lg_e, lb_e, bk_e = pooled_logits(mth, models[mth], Xe, ye, blk_e, groups, m, trials,
                                             np.random.default_rng([seed, m, 0]))
            lg_ei, lb_ei, _ = pooled_logits(mth, models[mth], Xe, ye, blk_e, groups, m, trials,
                                            np.random.default_rng([seed, m, 1]))
            # Row i of a pool belongs to trial i // n_pixels, so the MASK id needs no extra return
            # value from pooled_logits (whose signature is pinned by tests/test_reliability_guards).
            msk_c = np.arange(len(lg_c)) // len(Xc)
            msk_e = np.arange(len(lg_e)) // len(Xe)        # eval draws, SHARED stream (block arm)
            msk_ei = np.arange(len(lg_ei)) // len(Xe)      # eval draws, INDEPENDENT (mask arm)
            r = reliability_at_m(lg_t, lb_t, lg_c, lb_c, bk_c, lg_e, lb_e, bk_e,
                                 target_risk=target_risk, msk_c=msk_c, msk_e=msk_e,
                                 logits_ei=lg_ei, labels_ei=lb_ei, msk_ei=msk_ei)
            for mn in ALL_METRICS:
                per_m[mn].append(r[mn])
            if mth == "proposed" and m in RC_MS and m <= max_missing:
                cov, risk = r["curve"]
                rc[m] = np.interp(COV_GRID, cov, risk)
        for mn in ALL_METRICS:
            store[mth][mn] = np.array(per_m[mn])
    return {"store": store, "rc": rc}


def main():
    ap = argparse.ArgumentParser()
    # Defaults are READ FROM CANONICAL, not repeated. They were duplicate literals, and only
    # target_risk was linked -- so raising the --trials default without touching CANONICAL would
    # have made a plain no-flag run non-canonical, redirect itself to _nc-t<N>, and quietly stop
    # regenerating the deliverables while the stale ones stayed in place looking stamped.
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL["seeds"]))
    ap.add_argument("--groups", type=int, default=CANONICAL["groups"])
    ap.add_argument("--max-missing", type=int, default=CANONICAL["max_missing"])
    ap.add_argument("--trials", type=int, default=CANONICAL["trials"])
    ap.add_argument("--epochs", type=int, default=CANONICAL["epochs"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent seed workers (default: adaptive; also BANDSIM_WORKERS)")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                    help="device for the workers (default: auto; also BANDSIM_DEVICE)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="disable deterministic algorithms for a small speedup")
    ap.add_argument("--target-risk", type=float, default=DEFAULT_TARGET_RISK,
                    help=f"alpha for BOTH conformal arms (default {DEFAULT_TARGET_RISK}); "
                         f"a non-default value redirects the outputs")
    args = ap.parse_args()

    # Validate BEFORE training. All of these were reachable, and the cheapest (--groups 5
    # --max-missing 6) surfaced from inside rng.choice AFTER all five seeds had finished training.
    if len(set(args.seeds)) != len(args.seeds):
        ap.error(f"--seeds must be unique, got {args.seeds}")
    if any(s < 0 for s in args.seeds):
        ap.error("--seeds must be >= 0: the seed is also the checkerboard OFFSET and a pixel index")
    if args.groups < 2:
        ap.error("--groups must be >= 2 (band-group dropout degenerates to no dropout at G=1)")
    if not 0 <= args.max_missing <= args.groups:
        ap.error(f"--max-missing must lie in [0, --groups={args.groups}], got {args.max_missing}")
    if args.trials < 1:
        ap.error(f"--trials must be >= 1, got {args.trials}")
    if args.epochs < 1:
        ap.error(f"--epochs must be >= 1, got {args.epochs}")
    if args.jobs is not None and args.jobs < 1:
        ap.error(f"--jobs must be >= 1, got {args.jobs}")
    if not 0.0 < args.target_risk < 1.0:
        ap.error(f"--target-risk must lie strictly in (0, 1), got {args.target_risk}")
    # This file's committed numbers were ONCE a --smoke (1 seed / 12 epochs / 3 trials) quoted as
    # converged, because a smoke run wrote the very CSV and .tex the paper reads. The provenance
    # stamp at the end says which run produced a file; this suffix stops the collision happening.
    if args.smoke:
        args.seeds = [0]; args.epochs = 12; args.trials = 3
        sfx = "_smoke"
        print("[smoke] 1 seed / 12 epochs / 3 trials — writing *_smoke artefacts, NOT the deliverables")
    else:
        sfx = ""
    # The config tag is computed AFTER the smoke overrides and ignores the three settings smoke
    # dictates. Computing it first made `--smoke --epochs 30` write "_smoke_nc-e30" from a run that
    # actually used epochs=12 -- a filename describing a configuration that never ran.
    sfx += _config_suffix(args, ignore=("seeds", "epochs", "trials") if args.smoke else ())
    if sfx and not args.smoke:
        print("!" * 96)
        print(f"! NON-CANONICAL CONFIG — writing '{sfx}' artefacts, NOT the paper deliverables.")
        for k in sorted(CANONICAL):
            got = list(getattr(args, k)) if isinstance(CANONICAL[k], list) else getattr(args, k)
            if got != CANONICAL[k]:
                print(f"!   {k}: {got}   (canonical: {CANONICAL[k]})")
        print("! Re-run with the canonical settings to regenerate paper/results_phase4R_*.")
        print("!" * 96)
    # The MASK arm's floor is a property of --trials alone, so it is knowable before any compute.
    _mask_floor = 1.0 / (args.trials + 1.0)
    if _mask_floor > args.target_risk:
        print("!" * 96)
        print(f"! MASK-LEVEL CRC IS INFEASIBLE AT alpha={args.target_risk:.0%} WITH --trials {args.trials}.")
        print("!   CRC needs B/(n+1) <= alpha and the mask arm's n IS --trials, so the floor is")
        print(f"!   1/{args.trials + 1} = {_mask_floor:.1%}. No threshold can produce a bound; the arm will")
        print("!   report infeasible for reasons of SAMPLE SIZE, not model quality.")
        print(f"!   Need --trials >= {int(np.ceil(1.0 / args.target_risk - 1))} for a bound to exist at all.")
        print("!" * 96)
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device); print('HW:', hw.info())

    cube, gt = load_data()
    ms = list(range(0, args.max_missing + 1))
    methods = METHODS
    metric_names = METRIC_NAMES              # AGGREGATED into the summary CSV
    all_metrics = ALL_METRICS                # collected per seed; thresholds are raw-CSV only
    store = {mth: {mn: [] for mn in all_metrics} for mth in methods}
    cov_grid = COV_GRID                      # risk-coverage curves: proposed at m in {0,3,6}, per seed
    rc_curves = {m: [] for m in RC_MS if m <= args.max_missing}

    import time
    t0 = time.time()
    results = parallel.run_jobs(
        run_seed, args.seeds,
        # target_risk travels through `shared`, NOT as a module constant: run_jobs may run each
        # seed in a separate process that re-imports this module, where a module-level global would
        # carry the file's default rather than the CLI override.
        shared=dict(cube=cube, gt=gt, groups_n=args.groups, max_missing=args.max_missing,
                    trials=args.trials, epochs=args.epochs, target_risk=args.target_risk),
        prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
        label="phase4R/seed")
    for sd, res in zip(args.seeds, results):
        for mth in methods:
            for mn in all_metrics:
                store[mth][mn].append(res["store"][mth][mn])
        for m in rc_curves:
            if res["rc"].get(m) is not None:
                rc_curves[m].append(res["rc"][m])
        print(f"seed {sd}: prop AURC@0={store['proposed']['aurc'][-1][0]:.1f} "
              f"AURC@{args.max_missing}={store['proposed']['aurc'][-1][-1]:.1f} "
              f"AUROC@{args.max_missing}={store['proposed']['auroc'][-1][-1]:.1f} "
              f"| b2 AURC@{args.max_missing}={store['b2']['aurc'][-1][-1]:.1f}")
    print(f"(all {len(args.seeds)} seeds in {time.time()-t0:.1f}s)")

    # ---- aggregate mean/std over seeds, CONDITIONING the CRC numbers on FEASIBILITY -------------
    # When no threshold certifies, CRC falls back to abstain-all: coverage 0.0 and a realized joint
    # loss of 0.0. That zero means "could not certify, so predicted nothing" -- NOT "certified and
    # made no confident errors" -- and a plain mean over all seeds averages the two together, pulling
    # the reported risk DOWN precisely at the m where the method failed, and the coverage down with
    # it. So the three numbers that only exist when a certificate exists are averaged over the
    # FEASIBLE seeds only; crc_feasible stays a rate over ALL seeds and is what tells the reader how
    # much of the sample the other three describe. An m where NO seed certified is NaN, never 0.0 --
    # the same rule sel_risk already follows for an empty accepted set, for the same reason.
    _CRC_COND = ("crc_calib_bound", "crc_eval_joint_risk", "crc_coverage", "crc_sel_risk")
    # The mask arm needs the SAME guard and did not have it. It is the more exposed of the two:
    # its n is --trials, so B/(n+1) > alpha whenever trials < 1/alpha - 1, and it then abstains on
    # everything and returns joint risk 0.0 at coverage 0.0. At --smoke (trials=3) that happens in
    # every run at every m>=1. Averaged unconditioned, the deliverable reads 0.00 confidently-wrong
    # mass for an arm that predicted nothing.
    _CRCMASK_COND = ("crcmask_calib_bound", "crcmask_eval_joint_risk", "crcmask_coverage",
                     "crcmask_sel_risk")

    # THE GUARD IS COVERAGE > 0, NOT crc_feasible. `feasible` only asks whether SOME threshold
    # certifies -- and lambda_max = +inf always does, because there Rhat = 0 and the criterion is
    # just B/(n+1) = 1/(n+1). At alpha=0.10 that certifies for any n >= 9, and this has ~27
    # calibration blocks, so crc_feasible is TRUE IN EVERY RUN by construction: it reports sample
    # size, not success. Conditioning on it therefore excluded NOTHING, and the degenerate case it
    # was written to exclude -- CRC falls back to abstain-all, coverage 0.0, joint loss 0.0, and
    # that 0.0 averaged in as though it were a low risk -- passed straight through, because
    # abstain-all IS feasible. Verified directly against conformal_risk_control.
    _crc_usable = {mth: np.stack(store[mth]["crc_coverage"]) > 0.0 for mth in methods}
    # NaN > 0.0 is False, so m=0 (where the mask arm is undefined) is excluded here too -- which is
    # what we want: undefined and zero-coverage both mean "no risk to report".
    _crcmask_usable = {mth: np.stack(store[mth]["crcmask_coverage"]) > 0.0 for mth in methods}

    def _agg_over_seeds(mth, mn, feasible_only=True):
        """(mean, std, n_valid) over seeds. n_valid is not a formality: with one surviving seed
        nanstd returns 0.0, which reads as a perfectly stable result rather than a sample of one."""
        A = np.stack(store[mth][mn]).astype(float)                  # (n_seeds, n_m)
        if feasible_only and mn in _CRC_COND:
            A = np.where(_crc_usable[mth], A, np.nan)               # coverage > 0; see above
        elif feasible_only and mn in _CRCMASK_COND:
            A = np.where(_crcmask_usable[mth], A, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN column -> NaN, no crash
            # non-NaN, not isfinite: an infinite threshold is a real (abstain-all) outcome, whereas
            # NaN means the quantity was never defined for that seed.
            return np.nanmean(A, 0), np.nanstd(A, 0), (~np.isnan(A)).sum(0)

    agg = {mth: {mn: _agg_over_seeds(mth, mn) for mn in metric_names} for mth in methods}
    # OPERATIONAL twins: the same two CRC numbers over ALL seeds, a zero-coverage seed contributing
    # the abstain-all outcome it actually produced. Both are needed and neither is sufficient: CRC's
    # guarantee covers the procedure INCLUDING its fallback, so only the unconditional mean can be
    # checked against alpha -- but averaging in a 0.0 that means "predicted nothing" pulls the
    # reported risk DOWN exactly where the method failed. For coverage the bias runs the other way.
    agg_oper = {mth: {mn: _agg_over_seeds(mth, mn, feasible_only=False)
                      for mn in ("crc_eval_joint_risk", "crc_coverage")} for mth in methods}
    n_infeas = {mth: (~_crc_usable[mth]).sum(0) for mth in methods}
    if any(v.sum() for v in n_infeas.values()):
        print("\n[CRC] some seeds produced ZERO COVERAGE (CRC fell back to abstaining on everything).")
        print("      Their 0.0 joint loss means 'predicted nothing', not 'made no confident errors',")
        print("      so it is EXCLUDED from the crc_* means, which are therefore conditional on the")
        print("      run having predicted at all. NOTE crc_feasible is ~always 1.0 here, so it is")
        print("      NOT the guard -- coverage > 0 is. Zero-coverage seeds per m:")
        for mth in methods:
            print(f"      {mth:<9} " + " ".join(f"m={m}:{int(c)}/{len(args.seeds)}"
                                                for m, c in zip(ms, n_infeas[mth]) if c))

    # ---- csv ---- (column names carry the estimand; see METRIC_NAMES)
    # Every metric ships as mean/std/n. The std was computed and then thrown away -- the file kept
    # only column [0] -- so the one artefact the paper reads could not state a dispersion, and the
    # figures plotted a band the CSV could not corroborate.
    with open(P(f"results_phase4R_reliability{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = [f"{mn}{s}" for mn in metric_names for s in ("", "_std", "_n")]
        w.writerow(["method", "missing_groups"] + cols
                   + ["crc_eval_joint_risk_oper", "crc_coverage_oper"])
        for mth in methods:
            for i, m in enumerate(ms):
                row = [mth, m]
                for mn in metric_names:
                    mean, std, nval = agg[mth][mn]
                    row += [_csv_num(mean[i]), _csv_num(std[i]), int(nval[i])]
                row += [_csv_num(agg_oper[mth]['crc_eval_joint_risk'][0][i]),
                        _csv_num(agg_oper[mth]['crc_coverage'][0][i])]
                w.writerow(row)

    # ---- raw per-seed csv ---- LONG FORM. The aggregate is lossy and the run that produced it
    # costs GPU-hours, so the per-seed values are kept: without them a paired test, a confidence
    # interval, or a check of which seed drove an outlier all require re-running the experiment.
    with open(P(f"results_phase4R_raw{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "method", "missing_groups", "metric", "value"])
        for si, sd in enumerate(args.seeds):
            for mth in methods:
                for mn in all_metrics:          # includes plugin_threshold / crc_threshold
                    for i, m in enumerate(ms):
                        w.writerow([sd, mth, m, mn, f"{store[mth][mn][si][i]:.6g}"])

    # ---- paired proposed-minus-b2 csv ----
    # Both methods see the SAME seeds, block split and drop-mask draws, so every (seed, m) cell is a
    # matched pair. Two mean +/- std curves discard that pairing and compare through the across-seed
    # variance, which is the larger and mostly shared component. NO p-value: 5 seeds whose test sets
    # overlap at IoU 0.448 are not 5 independent observations.
    with open(P(f"results_phase4R_paired{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "metric", "missing_groups", "mean_diff", "std_diff",
                    "n_seeds", "n_seeds_first_better"])
        # "who won" is only defined for metrics that do not trade against a second quantity.
        # plugin_coverage and plugin_cond_risk are a COUPLED PAIR -- either can be improved alone by
        # moving the operating point -- so their win column is left EMPTY rather than filled with an
        # integer that would be read as a verdict.
        _LOWER_IS_BETTER = {"aurc", "augrc"}
        _HIGHER_IS_BETTER = {"auroc", "acc", "mask_acc_worst", "cls_cov_min"}
        # THREE contrasts, because proposed-minus-b2 alone cannot separate the architecture from
        # the information. b2m is B2's exact network and budget plus the group-presence mask:
        #   proposed - b2m : the ARCHITECTURE (cross-band attention, wavelength PE, SGMAE), with
        #                    missingness information held equal
        #   b2m      - b2  : the MASK alone, with architecture held equal
        #   proposed - b2  : the headline comparison, which is the SUM of the two above and must
        #                    not be attributed to the architecture on its own
        _CONTRASTS = [("proposed", "b2m"), ("b2m", "b2"), ("proposed", "b2")]
        for a, b in _CONTRASTS:
            for mn in ("aurc", "augrc", "auroc", "acc", "plugin_coverage", "plugin_cond_risk",
                       "mask_acc_worst", "cls_cov_min"):
                D = np.stack(store[a][mn]).astype(float) - np.stack(store[b][mn]).astype(float)
                for i, m in enumerate(ms):
                    d = D[:, i]; d = d[~np.isnan(d)]
                    if mn in _LOWER_IS_BETTER:
                        better = int((d < 0).sum())
                    elif mn in _HIGHER_IS_BETTER:
                        better = int((d > 0).sum())
                    else:
                        better = ""
                    w.writerow([f"{a}-minus-{b}", mn, m, f"{d.mean():.4f}" if d.size else "nan",
                                f"{d.std():.4f}" if d.size else "nan", d.size, better])

    # ---- summary tex ----
    with open(P(f"results_phase4R_summary{sfx}.tex"), "w") as f:
        f.write(f"% Phase 4R — Indian Pines, {args.groups} groups, seeds={args.seeds}, "
                f"alpha={args.target_risk}, trials={args.trials}\n")
        mm = args.max_missing
        # The macro name now SPELLS the level it holds. It was hardcoded to "Six" while the value
        # was indexed at args.max_missing, so `--max-missing 4` wrote m=4 into \\aurcPropSix and the
        # paper read it as m=6. m=6 -> "Six", so canonical macro names are unchanged.
        mt = _mtag(mm)
        for mth, tag in [("proposed", "Prop"), ("b2", "Btwo"), ("b2m", "BtwoM")]:
            f.write(f"\\newcommand{{\\aurc{tag}Clean}}{{{_tex_num(agg[mth]['aurc'][0][0])}}}\n")
            f.write(f"\\newcommand{{\\aurc{tag}{mt}}}{{{_tex_num(agg[mth]['aurc'][0][mm])}}}\n")
            f.write(f"\\newcommand{{\\auroc{tag}{mt}}}{{{_tex_num(agg[mth]['auroc'][0][mm])}}}\n")
            f.write(f"\\newcommand{{\\plugincov{tag}{mt}}}{{{_tex_num(agg[mth]['plugin_coverage'][0][mm])}}}\n")
            # APPROXIMATE (CRC) JOINT risk P(accepted AND wrong) -- macro name says "joint" so the tex
            # cannot describe it as a selective/conditional risk, and now also says whether it is the
            # BOUND or the REALIZATION, because the old single macro carried the realization under a
            # name the prose read as the bound. Its coverage macro ships WITH them (the joint mass is
            # trivially small when almost everything is abstained on), plus the exchangeable-unit
            # count and the fraction of seeds where a threshold was feasible.
            f.write(f"\\newcommand{{\\crcjointriskeval{tag}{mt}}}{{{_tex_num(agg[mth]['crc_eval_joint_risk'][0][mm])}}}\n")
            f.write(f"\\newcommand{{\\crcjointriskbound{tag}{mt}}}{{{_tex_num(agg[mth]['crc_calib_bound'][0][mm])}}}\n")
            f.write(f"\\newcommand{{\\crccov{tag}{mt}}}{{{_tex_num(agg[mth]['crc_coverage'][0][mm])}}}\n")
            f.write(f"\\newcommand{{\\crcunits{tag}{mt}}}{{{_tex_num(agg[mth]['crc_units'][0][mm], '.0f')}}}\n")
            # crcfeas is NEAR-VACUOUS (lambda=inf certifies whenever n >= 1/alpha-1) so it reads 100
            # regardless; crcposcov is the one that means something.
            f.write(f"\\newcommand{{\\crcfeas{tag}{mt}}}{{{_tex_num(agg[mth]['crc_feasible'][0][mm]*100, '.0f')}}}\n")
            f.write(f"\\newcommand{{\\crcposcov{tag}{mt}}}{{{_tex_num(agg[mth]['crc_positive_coverage'][0][mm]*100, '.0f')}}}\n")
            # MASK-level arm: the operational estimand ("a NEW failure pattern").
            f.write(f"\\newcommand{{\\crcmaskjointrisk{tag}{mt}}}{{{_tex_num(agg[mth]['crcmask_eval_joint_risk'][0][mm])}}}\n")
            f.write(f"\\newcommand{{\\crcmaskcov{tag}{mt}}}{{{_tex_num(agg[mth]['crcmask_coverage'][0][mm])}}}\n")
            f.write(f"\\newcommand{{\\crcmaskunits{tag}{mt}}}{{{_tex_num(agg[mth]['crcmask_units'][0][mm], '.0f')}}}\n")
            f.write(f"\\newcommand{{\\crcmaskfeas{tag}{mt}}}{{{_tex_num(agg[mth]['crcmask_feasible'][0][mm]*100, '.0f')}}}\n")
            f.write(f"\\newcommand{{\\plugincondrisk{tag}{mt}}}{{{_tex_num(agg[mth]['plugin_cond_risk'][0][mm])}}}\n")
            f.write(f"\\newcommand{{\\pluginneff{tag}{mt}}}{{{_tex_num(agg[mth]['plugin_neff'][0][mm], '.0f')}}}\n")

    # ---- fig 1: risk-coverage (proposed, m=0/3/6) ----
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    cmap = {0: "#1f6f3a", 3: "#e67e22", 6: "#c0392b"}
    for m in sorted(rc_curves):
        if not rc_curves[m]:
            continue
        arr = np.stack(rc_curves[m]); mean = arr.mean(0); std = arr.std(0)
        ax.plot(cov_grid, mean * 100, "-", color=cmap[m], lw=1.8, label=f"{m} missing groups")
        ax.fill_between(cov_grid, np.clip(mean - std, 0, 1) * 100, np.clip(mean + std, 0, 1) * 100,
                        color=cmap[m], alpha=0.15, lw=0)     # a risk outside [0,100]% is undrawable
    ax.set_xlabel("Coverage (fraction of pixels predicted)")
    ax.set_ylabel("Selective risk (error %)")
    ax.set_title("Risk-coverage under missing bands (Proposed)", fontsize=8.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(P(f"figs/fig_risk_coverage{sfx}.pdf")); plt.close(fig)

    # ---- fig 2: AURC & selective-AUROC vs #missing (proposed vs b2) ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    mcol = {"proposed": "#1f6f3a", "b2": "#e67e22", "b2m": "#2980b9"}
    mlab = {"proposed": "Proposed", "b2": "B2 dropout", "b2m": "B2M dropout+mask"}
    for mth in methods:
        # +/-1 std CLIPPED to the range the metric can occupy; an unclipped band runs past 100% and
        # draws a region no run could have produced. It is a dispersion, NOT a confidence interval.
        me, st, _ = agg[mth]["aurc"]
        a1.plot(ms, me, "-o", color=mcol[mth], lw=1.6, ms=3, label=mlab[mth])
        a1.fill_between(ms, np.clip(me - st, 0, 100), np.clip(me + st, 0, 100),
                        color=mcol[mth], alpha=0.15, lw=0)
        me2, st2, _ = agg[mth]["auroc"]
        a2.plot(ms, me2, "-o", color=mcol[mth], lw=1.6, ms=3, label=mlab[mth])
        a2.fill_between(ms, np.clip(me2 - st2, 0, 100), np.clip(me2 + st2, 0, 100),
                        color=mcol[mth], alpha=0.15, lw=0)
    a1.set_xlabel("Missing groups"); a1.set_ylabel("AURC (lower=better)"); a1.grid(alpha=0.3)
    a1.set_title("Selective risk (AURC)", fontsize=8.5); a1.legend(fontsize=7, frameon=False)
    a2.set_xlabel("Missing groups"); a2.set_ylabel("Misclassif.-detection AUROC")
    a2.set_title("Confidence quality", fontsize=8.5); a2.grid(alpha=0.3); a2.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(P(f"figs/fig_reliability_vs_missing{sfx}.pdf")); plt.close(fig)

    # ---- console summary ----
    print("\n===== Phase 4R reliability (mean over {} seeds) =====".format(len(args.seeds)))
    print("[plug-in operating point -- CONDITIONAL P(wrong|accepted), APPROXIMATE, not a guarantee]")
    # alpha comes from TARGET_RISK in every label. A literal "10%" here is the same drift bug the
    # module comment claims to have fixed: change TARGET_RISK and the header would still say 10%.
    _a = f"{args.target_risk:.0%}"
    print(f"m   | prop:acc  AURC  AUROC  opCov@{_a}risk | b2:acc  AURC  AUROC  opCov@{_a}risk")
    for i, m in enumerate(ms):
        p = agg["proposed"]; b = agg["b2"]
        print(f"{m}   | {p['acc'][0][i]:6.1f} {p['aurc'][0][i]:5.1f} {p['auroc'][0][i]:5.1f} "
              f"{p['plugin_coverage'][0][i]:6.1f}        | "
              f"{b['acc'][0][i]:5.1f} {b['aurc'][0][i]:5.1f} {b['auroc'][0][i]:5.1f} {b['plugin_coverage'][0][i]:6.1f}")
    print("\n[APPROXIMATE split-conformal risk control (CRC, Angelopoulos+ 2022) on the JOINT mass")
    print(f" P(accepted AND wrong) <= {_a}. NOT the conditional error rate among accepted pixels, and")
    print(" NOT readable without the coverage column beside it -- abstaining alone drives it to 0.")
    print(" The BOUND column is the calibration-side quantity ((n/(n+1))Rhat + B/(n+1)), which is")
    print(" <= alpha BY CONSTRUCTION whenever feasible and therefore carries no evidence either way;")
    print(" the REALIZED column is what the eval set actually incurred at that threshold.")
    print(" Theorem 2.1 bounds E[L_{n+1}] over the JOINT draw of calibration and test, so a single")
    print(" realization above alpha does NOT contradict it and is not 'the control failing'. Testing")
    print(" the claim means averaging over many independent calib/eval draws and checking THAT mean;")
    print(" the per-row marks below are DESCRIPTIVE, not verdicts.")
    print(" UNIT = SPATIAL BLOCK, so this answers 'a new REGION under the SAME failures we saw'.")
    print(" For 'a NEW failure', see the mask-level table below -- a much harder question.]")
    print("m   | prop: bound% realized%  cov%  units feas% | b2: bound% realized%  cov%  units feas%   (* = over alpha)")
    a_pct = 100 * args.target_risk
    for i, m in enumerate(ms):
        p = agg["proposed"]; b = agg["b2"]
        # Mark rows whose realised joint risk exceeds alpha. CRC bounds a MARGINAL expectation, so an
        # individual realisation may exceed it without contradicting the theorem -- but an unmarked
        # 10.47 next to a header promising "<= 10%" is exactly the kind of thing a reader skims past.
        pm = "*" if p["crc_eval_joint_risk"][0][i] > a_pct else " "
        bm = "*" if b["crc_eval_joint_risk"][0][i] > a_pct else " "
        print(f"{m}   | {p['crc_calib_bound'][0][i]:8.2f} {p['crc_eval_joint_risk'][0][i]:8.2f}{pm} "
              f"{p['crc_coverage'][0][i]:5.1f} {p['crc_units'][0][i]:6.0f} {p['crc_feasible'][0][i]*100:5.0f} | "
              f"{b['crc_calib_bound'][0][i]:8.2f} {b['crc_eval_joint_risk'][0][i]:8.2f}{bm} "
              f"{b['crc_coverage'][0][i]:5.1f} {b['crc_units'][0][i]:6.0f} {b['crc_feasible'][0][i]*100:5.0f}")
    over = [m for i, m in enumerate(ms)
            if any(agg[mth]["crc_eval_joint_risk"][0][i] > a_pct for mth in methods)]
    if over:
        print(f"    -> realised joint risk exceeds alpha={a_pct:.0f}% at m={over}. This DESCRIBES these draws;")
        print("       it is not a detected violation. CRC bounds a marginal expectation, one realization")
        print("       above alpha is consistent with the theorem, and the units are only approximately")
        print("       exchangeable anyway. Do not quote these as certified -- in either direction.")

    # ---- MASK-LEVEL arm: the operational estimand -------------------------------------------
    print("\n[MASK-LEVEL CRC -- exchangeable unit is the DROP MASK, not the spatial block.")
    print(" CONDITIONAL ON THIS CALIBRATION REGION -- the pixel-set dimension has n=1, so a")
    print(" calibration-vs-evaluation shift breaks it (80% of runs over alpha at a 0.5-logit gap,")
    print(" and MORE --trials makes it worse by shrinking the B/(n+1) slack). Not a certificate.")
    print(" This moves toward the question an instrument-failure argument asks: can the confidently-")
    print(" wrong mass be bounded for a band loss NEVER SEEN in calibration? n is --trials, i.e. the")
    print(" number of i.i.d. mask DRAWS -- NOT the number of distinct patterns, which is smaller")
    print(" (32 draws cover ~9.7 of the 10 masks at m=1 but only ~29.7 of 210 at m=6). The floor")
    print(" alone requires")
    print(f" n >= 1/alpha - 1 = {int(np.ceil(1.0/args.target_risk - 1))} before any bound exists. m=0 is NaN by definition:")
    print(" there is exactly one 'nothing missing' pattern, so there is no failure randomness.]")
    print("m   | prop: bound% realized%  cov% masks floor% feas% | b2: bound% realized%  cov% masks floor% feas%")
    for i, m in enumerate(ms):
        p = agg["proposed"]; b = agg["b2"]
        def _row(a):
            return (f"{a['crcmask_calib_bound'][0][i]:8.2f} {a['crcmask_eval_joint_risk'][0][i]:8.2f} "
                    f"{a['crcmask_coverage'][0][i]:5.1f} {a['crcmask_units'][0][i]:5.0f} "
                    f"{a['crcmask_alpha_floor'][0][i]:6.2f} {a['crcmask_feasible'][0][i]*100:5.0f}")
        print(f"{m}   | {_row(p)} | {_row(b)}")
    _mf = agg["proposed"]["crcmask_feasible"][0]
    if np.isfinite(_mf).any() and np.nanmax(_mf) < 0.5:
        print(f"    -> the mask arm certified in NO seed at ANY m. With --trials {args.trials} the floor is")
        print(f"       {100.0/(args.trials+1):.2f}% against alpha={a_pct:.0f}%; this is a SAMPLE-SIZE verdict about how")
        print("       many failure patterns were observed, NOT evidence about the models.")

    # ---- plug-in arm vs its target ----------------------------------------------------------
    print(f"\n[plug-in operating point: achieved CONDITIONAL P(wrong|accepted) vs alpha={a_pct:.0f}%]")
    print(" achieved% is conditional on the seed having ACCEPTED something (blank otherwise); acc%")
    print(" is the share of seeds that did. n_eff is the n the RETURNED margin was sized from --")
    print(" after the refinement pass that is the ACCEPTED count, not the larger full-set value.")
    print("m   | prop: achieved% effTgt%  n_eff  icc acc% | b2: achieved% effTgt%  n_eff  icc acc%   (* = over target)")
    for i, m in enumerate(ms):
        p = agg["proposed"]; b = agg["b2"]
        def _prow(a):
            v = a["plugin_cond_risk"][0][i]
            mk = "*" if np.isfinite(v) and v > a_pct else " "
            return (f"{v:9.2f}{mk} {a['plugin_eff_target'][0][i]:7.2f} {a['plugin_neff'][0][i]:6.0f} "
                    f"{a['plugin_icc'][0][i]:4.2f} {a['plugin_positive_coverage'][0][i]*100:4.0f}")
        print(f"{m}   | {_prow(p)} | {_prow(b)}")
    _pdg = np.nanmax([np.nanmax(agg[mth]["plugin_degenerate"][0]) for mth in methods])
    if np.isfinite(_pdg) and _pdg > 0:
        print("    -> at least one cell hit margin >= target, where the selection stops targeting the")
        print("       risk and starts fitting the longest error-free calibration prefix. Those cells")
        print("       ABSTAIN by construction rather than report a meaningless threshold.")
    # ---- the limitation, where nobody can miss it ----
    # ---- MATCHED-INFORMATION ABLATION ------------------------------------------------------
    print("\n[ABLATION -- is the gain the ARCHITECTURE or just being TOLD what is missing?")
    print(" b2m is B2's exact network, budget and dropout stream PLUS the group-presence mask.")
    print(" proposed-b2m isolates the architecture; b2m-b2 isolates the mask; proposed-b2 is their")
    print(" SUM and must not be attributed to the architecture alone.]")
    print("m   |   AURC prop   b2m    b2 | dAURC arch  mask |  AUROC prop   b2m    b2")
    for i, m in enumerate(ms):
        A = {k: agg[k]["aurc"][0][i] for k in methods}
        R = {k: agg[k]["auroc"][0][i] for k in methods}
        print(f"{m}   | {A['proposed']:10.1f} {A['b2m']:5.1f} {A['b2']:5.1f} | "
              f"{A['proposed']-A['b2m']:10.1f} {A['b2m']-A['b2']:5.1f} | "
              f"{R['proposed']:10.1f} {R['b2m']:5.1f} {R['b2']:5.1f}")
    _arch = np.array([agg["proposed"]["aurc"][0][i] - agg["b2m"]["aurc"][0][i] for i in range(len(ms))])
    _mask = np.array([agg["b2m"]["aurc"][0][i] - agg["b2"]["aurc"][0][i] for i in range(len(ms))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _sa, _sm = np.nansum(np.abs(_arch)), np.nansum(np.abs(_mask))
    if _sa + _sm > 0:
        print(f"    -> summed |dAURC|: architecture {_sa:.1f}, mask {_sm:.1f} "
              f"({100*_sa/(_sa+_sm):.0f}% / {100*_sm/(_sa+_sm):.0f}%). Lower AURC is better, so a")
        print("       NEGATIVE architecture column is the proposed model winning on architecture alone.")

    # ---- PER-MASK WORST CASE ---------------------------------------------------------------
    print("\n[PER-MASK spread at the certified threshold. The pooled rows above average over the")
    print(" mask MIXTURE -- 'how bad is a random band loss'. An instrument failure is a PARTICULAR")
    print(" loss, so the worst mask is the operative number. 'worst' is the worst mask OBSERVED in")
    print(f" --trials={args.trials} draws, a LOWER BOUND on the true worst, not an estimate of it.]")
    print("m   | prop: acc_med acc_wst cov_wst sel_wst | b2m: acc_wst sel_wst | b2: acc_wst sel_wst")
    for i, m in enumerate(ms):
        p = agg["proposed"]
        print(f"{m}   | {p['mask_acc_median'][0][i]:11.1f} {p['mask_acc_worst'][0][i]:7.1f} "
              f"{p['mask_cov_worst'][0][i]:7.1f} {p['mask_sel_risk_worst'][0][i]:7.1f} | "
              f"{agg['b2m']['mask_acc_worst'][0][i]:8.1f} {agg['b2m']['mask_sel_risk_worst'][0][i]:7.1f} | "
              f"{agg['b2']['mask_acc_worst'][0][i]:7.1f} {agg['b2']['mask_sel_risk_worst'][0][i]:7.1f}")

    # ---- PER-CLASS RELIABILITY -------------------------------------------------------------
    print("\n[PER-CLASS at the certified threshold. Indian Pines is severely imbalanced, so a")
    print(" flattering GLOBAL coverage can be bought by abstaining almost entirely on the small")
    print(" classes -- the abstention lands where the pixels are few and barely moves the pooled")
    print(" number. cov_min is the worst class; zeroN counts classes dropped from the map entirely.]")
    print("m   | prop: cov_min cov_macro zeroN sel_max | b2m: cov_min zeroN | b2: cov_min zeroN")
    for i, m in enumerate(ms):
        p = agg["proposed"]
        print(f"{m}   | {p['cls_cov_min'][0][i]:11.1f} {p['cls_cov_macro'][0][i]:9.1f} "
              f"{p['cls_zero_cov_n'][0][i]:5.1f} {p['cls_sel_risk_max'][0][i]:7.1f} | "
              f"{agg['b2m']['cls_cov_min'][0][i]:8.1f} {agg['b2m']['cls_zero_cov_n'][0][i]:5.1f} | "
              f"{agg['b2']['cls_cov_min'][0][i]:7.1f} {agg['b2']['cls_zero_cov_n'][0][i]:5.1f}")
    _zc = max(float(np.nanmax(agg[k]["cls_zero_cov_n"][0])) for k in methods)
    if _zc > 0:
        print(f"    -> up to {_zc:.0f} class(es) received ZERO coverage at some m: those classes are")
        print("       absent from the predicted map, and no global coverage number says so.")

    u = agg["proposed"]["crc_units"][0]
    umin, umax = (np.nanmin(u), np.nanmax(u)) if np.isfinite(u).any() else (float("nan"),) * 2
    from math import comb
    _mask_counts = [comb(args.groups, x) for x in ms]
    print("\n" + "!" * 96)
    print("! LIMITATION -- these CRC numbers are APPROXIMATE, NOT a certificate. Do not call them one.")
    print(f"!  * BLOCK arm: unit = spatial checkerboard BLOCK, n ~ {umin:.0f}-{umax:.0f}, not the pooled pixel-rows")
    print(f"!    (one pixel recurs once per --trials (={args.trials}) draw and a block's pixels are correlated).")
    print(f"!    n~{umin:.0f} means B/(n+1) is ~{100.0/(umin+1):.1f} points of alpha={a_pct:.0f}% -- a third of the budget is")
    print("!    the finite-sample term alone, so this arm has little power.")
    print("!  * calib and eval blocks are DISJOINT (third-split by block, not by pixel), but they still")
    print("!    come from ONE scene, so ADJACENT blocks remain spatially correlated. THIS is the")
    print("!    exchangeability gap. The shared drop mask is NOT -- it costs Monte-Carlo support in")
    print("!    the mask dimension, a different and smaller problem, and calib/eval SHARE the stream")
    print("!    precisely so the block units stay exchangeable.")
    print(f"!  * MASK COVERAGE: --trials={args.trials} masks per m out of C({args.groups},m) = {_mask_counts}")
    print(f"!    ({sum(_mask_counts)} patterns in total). These are means over a UNIFORM MASK MIXTURE, not")
    print("!    per-pattern results, and say nothing about the WORST band group to lose -- which is")
    print("!    what an instrument-failure argument needs. phase9 varies the mask POLICY; neither")
    print("!    enumerates. Do NOT read these rows as 'any m missing bands is survivable'.")
    print("!  * SEEDS ARE NOT INDEPENDENT REPLICATES: offset=seed shifts the checkerboard by `seed`")
    print("!    pixels, giving mean pairwise test-set IoU 0.448 (0.333 = independent, 1.0 = identical)")
    print("!    and 0.632 for adjacent seeds. Every std here is an UNDERESTIMATE.")
    print("!  => the RIGOROUS certificate is phase8R: ROI-level split on real Sentinel-2 CloudSEN12,")
    print("!     where a whole LOCATION is the exchangeable unit and calib/eval ROIs are DISJOINT.")
    print("!" * 96)
    # The provenance stamp exists because this file's committed numbers were once --smoke output
    # (1 seed, 12 epochs, 3 trials) quoted as if converged, with nothing in the CSV to say so.
    # EVERY deliverable CSV is stamped, not just the headline one: scripts/doctor.py asserts exactly
    # that over paper/results_*.csv, and a sidecar-less file is one nobody can attribute to a run.
    _prov = {"target_risk": args.target_risk, "missing_levels": list(ms),
             "exchangeable_unit_block_arm": "spatial checkerboard block",
             "exchangeable_unit_mask_arm": "drop mask (n = trials)",
             "mask_arm_alpha_floor": 1.0 / (args.trials + 1.0),
             "calib_eval_split": "by block (disjoint)",
             "block_arm_calib_eval_share_mask_stream": True,
             "mask_arm_calib_eval_independent_masks": True,
             "plugin_margin_unit": "design-effect effective n (not rows, not clusters)",
             "canonical": sfx == "", "config_suffix": sfx}
    for _name, _role in [("reliability", "aggregate mean/std/n over seeds"),
                         ("raw", "long form, one row per (seed, method, m, metric)"),
                         ("paired", "per-m proposed-minus-b2 differences")]:
        stamp(P(f"results_phase4R_{_name}{sfx}.csv"), args, extra={**_prov, "role": _role})
    print(f"\nwrote: {P(f'figs/fig_risk_coverage{sfx}.pdf')}")
    print(f"       {P(f'figs/fig_reliability_vs_missing{sfx}.pdf')}")
    print(f"       {P(f'results_phase4R_reliability{sfx}.csv')}  {P(f'results_phase4R_summary{sfx}.tex')}")
    print(f"       {P(f'results_phase4R_raw{sfx}.csv')}  {P(f'results_phase4R_paired{sfx}.csv')}")


if __name__ == "__main__":
    main()
