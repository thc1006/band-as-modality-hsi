"""Reliability / selective-prediction metrics for missing-band abstention (Phase 4R).

The paper's honest-ace contribution: quantify WHEN NOT TO TRUST a prediction under missing
bands. Nobody has drawn risk-coverage / misclassification-detection AUROC as a function of
#missing bands for HSI segmentation (see docs/guide/EXP_PHASE4R_PLAN.md).

Metrics
-------
- confidence_msp        : max-softmax-probability confidence
- risk_coverage_curve   : (coverage, risk) by abstaining on low-confidence samples
- aurc / augrc          : area under risk-coverage / generalized risk-coverage (NeurIPS'24)
- selective_auroc       : AUROC of confidence detecting CORRECT vs incorrect predictions
- fit_temperature       : temperature scaling (calibrated confidence)
- conformal_risk_control: CRC certificate on the JOINT risk P(accepted AND wrong)
- conformal_at_risk     : plug-in operating point at a target CONDITIONAL risk P(wrong | accepted)

ESTIMANDS DIFFER — do not paraphrase one as the other. `conformal_risk_control` certifies the JOINT
"confidently-wrong" mass P(accepted AND wrong); `conformal_at_risk` targets the CONDITIONAL selective
risk P(wrong | accepted), i.e. the error rate among the predictions actually made. They are related
by P(wrong | accepted) = P(accepted AND wrong) / P(accepted), so the joint quantity is driven to 0 by
abstaining and is meaningless without its coverage — ALWAYS report the two together.

A BOUND IS NOT ITS REALIZATION either. `conformal_risk_control` returns BOTH, under names that say
which is which: `calib_crc_bound` is the certificate computed on calibration (<= alpha whenever
`feasible`), `eval_group_joint_risk` is the joint mass actually realized on the evaluation set. They
used to share one key called `cert_risk` that carried the REALIZATION while its name promised the
bound, so every downstream plot/CSV label was describing the wrong estimand. A breach is exactly
`eval_group_joint_risk > calib_crc_bound`, only visible if the two are reported side by side.

NOR DO A GROUP-WEIGHTED AND A SAMPLE-WEIGHTED NUMBER SHARE A DENOMINATOR. With grouping on, the
identity above holds within `eval_group_*` and within `eval_sample_*`, and NOT across them; only the
group family is what CRC bounds. See conformal_risk_control's docstring for the worked counterexample.

Pure numpy (self-contained AUROC), so unit-testable without sklearn.
"""
from __future__ import annotations
import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _validate_correct_conf(correct, confidence):
    """Shared input contract for the public selective-prediction metrics (H-5). Returns
    (correct_bool, confidence_float). A public entry point fed by long pipelines must fail closed on
    malformed input rather than return a plausible-but-impossible number (a `correct` of 2/-1 makes
    risk_coverage_curve report a NEGATIVE risk; a NaN confidence makes selective_auroc return 1.0)."""
    correct = np.asarray(correct)
    conf = np.asarray(confidence, float)
    if correct.shape != conf.shape:
        raise ValueError(f"correct/confidence length mismatch: {correct.shape} vs {conf.shape}")
    if conf.ndim != 1:
        raise ValueError(f"correct/confidence must be 1-D, got shape {conf.shape}")
    if conf.size == 0:
        raise ValueError("empty input: no samples to score")
    if not np.isfinite(conf).all():
        raise ValueError("confidence contains non-finite values (NaN/Inf) -- an all-(-inf) logit row "
                         "softmaxes to NaN; validate the model output before scoring")
    cc = correct.astype(float)
    if not np.all((cc == 0.0) | (cc == 1.0)):
        raise ValueError("`correct` must be boolean or 0/1 (an argmax-equals-label indicator), got "
                         f"values outside {{0,1}} (e.g. {cc[(cc != 0) & (cc != 1)][:3].tolist()})")
    return correct.astype(bool), conf


def confidence_msp(logits):
    """Max softmax probability per sample (the standard selective-prediction score)."""
    return softmax(np.asarray(logits, float), axis=-1).max(axis=-1)


def risk_coverage_curve(correct, confidence):
    """Return (coverage, risk) at each UNIQUE confidence threshold, most-confident first.

    Evaluating at unique thresholds (rather than one point per sample) makes the curve
    PERMUTATION-INVARIANT: tied-confidence samples enter/leave the kept set together, so the result
    no longer depends on the arbitrary order among ties. coverage[k]=fraction kept; risk[k]=error
    rate among the kept (most-confident) set at that threshold.
    """
    # Fail closed on malformed input (H-5): a length mismatch would silently drop the longer array's
    # tail; a `correct` of 2/-1 would produce a NEGATIVE selective risk; a non-finite confidence
    # (all-(-inf) logits softmax to NaN) would sort arbitrarily. Validate all of it in one place.
    correct, conf = _validate_correct_conf(correct, confidence)
    correct = correct.astype(float)
    n = conf.size
    order = np.argsort(-conf, kind="mergesort")    # most confident first (stable)
    csorted = conf[order]
    cum_correct = np.cumsum(correct[order])
    kept = np.arange(1, n + 1)
    risk = 1.0 - cum_correct / kept
    coverage = kept / n
    last = np.concatenate((csorted[1:] != csorted[:-1], [True]))   # last index of each tie block
    return coverage[last], risk[last]


def aurc(correct, confidence):
    """Finite-sample area under the risk-coverage curve (lower = better). Coverage-weighted mean of
    the selective risk: sum over tie-blocks of risk * Δcoverage. For distinct confidences this is the
    standard mean-of-cumulative-risks estimator (so a single wrong sample gives AURC=1, NOT 0 as a
    single-point trapezoid would); with ties it is permutation-invariant."""
    cov, risk = risk_coverage_curve(correct, confidence)
    w = np.diff(np.concatenate(([0.0], cov)))      # coverage increment of each (tie-block) step
    return float((risk * w).sum())


def augrc(correct, confidence):
    """Finite-sample area under the GENERALIZED risk-coverage curve (Traub et al., NeurIPS'24):
    generalized risk g = risk * coverage, integrated over coverage (coverage-weighted mean)."""
    cov, risk = risk_coverage_curve(correct, confidence)
    w = np.diff(np.concatenate(([0.0], cov)))
    return float((risk * cov * w).sum())


def selective_auroc(correct, confidence):
    """AUROC of confidence ranking correct (label=1) above incorrect (label=0).

    = P(conf(correct) > conf(incorrect)); rank-based, base-rate insensitive. Returns NaN if
    only one class present.
    """
    correct, conf = _validate_correct_conf(correct, confidence)   # H-5: reject NaN conf / non-0/1 correct
    pos = conf[correct]; neg = conf[~correct]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney U -> AUROC, with tie handling via rank averaging
    order = np.argsort(conf, kind="mergesort")
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, conf.size + 1)
    # average ties
    _, inv, counts = np.unique(conf, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(counts.size); np.add.at(sum_ranks, inv, ranks)
    ranks = (sum_ranks / counts)[inv]
    r_pos = ranks[correct].sum()
    n_pos = pos.size; n_neg = neg.size
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_temperature(logits, labels, iters=200, lr=0.05):
    """Temperature scaling: find T>0 minimizing NLL of softmax(logits / T).

    Simple gradient descent on log T (keeps T positive). Returns scalar T.
    """
    logits = np.asarray(logits, float)
    labels = np.asarray(labels, int)
    # H-4: a public optimiser fed by every reliability run. Guard the inputs so it fails closed
    # instead of returning NaN/Inf T (Inf logits -> NaN T; extreme logits -> Inf T; an out-of-range
    # label used to raise a bare IndexError; empty input a reduction error), which would then silently
    # rescale every downstream confidence and reorder which samples clear a threshold.
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError(f"logits must be (N, C) with N >= 1, got shape {logits.shape}")
    if labels.shape != (logits.shape[0],):
        raise ValueError(f"labels shape {labels.shape} != logits rows {logits.shape[0]}")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN/Inf; temperature scaling would diverge")
    C = logits.shape[1]
    if labels.size and (labels.min() < 0 or labels.max() >= C):
        raise ValueError(f"labels out of range [0,{C}): min {labels.min()} max {labels.max()}")
    n = labels.size
    LOGT_CLAMP = float(np.log(1e3))          # T in [1e-3, 1e3]; a fit beyond this is degenerate, not signal
    logT = 0.0
    for _ in range(iters):
        T = np.exp(logT)
        p = softmax(logits / T)
        # d NLL / d logT   (chain through T = e^{logT})
        true_logit = logits[np.arange(n), labels]
        mean_logit = (p * logits).sum(1)
        grad_logT = ((true_logit - mean_logit) / T).mean()  # dNLL/d(logT)
        if not np.isfinite(grad_logT):                       # overflow on pathological logits -> stop
            break
        logT = float(np.clip(logT - lr * grad_logT, -LOGT_CLAMP, LOGT_CLAMP))  # DESCENT, clamped
    return float(np.exp(logT))


def _validated(correct, conf, group, side):
    """Validate one (correct, confidence, group) triple and return it as (bool array, float array).

    These are public entry points fed by long experiment pipelines, where a silent shape or dtype
    coercion surfaces only as a wrong number in a paper table. Each check below is a way a caller
    can hand this function something it will happily compute a meaningless answer from."""
    c = np.asarray(correct)
    k = np.asarray(conf, float)
    if c.ndim != 1 or k.ndim != 1:
        raise ValueError(f"{side}: correct and confidence must be 1-D, got {c.shape} and {k.shape}")
    if c.size != k.size:
        raise ValueError(f"{side}: {c.size} correctness flags for {k.size} confidences")
    if c.size == 0:
        raise ValueError(f"{side}: empty — CRC cannot calibrate on or evaluate over no points")
    if not np.isfinite(k).all():
        # lambda_max = +inf is the ONLY threshold that abstains on every point. A non-finite
        # confidence would satisfy `conf >= inf` and be ACCEPTED there, silently voiding the
        # abstain-all fallback that the feasibility guarantee rests on.
        raise ValueError(f"{side}: confidences must be finite (got NaN/inf); a non-finite "
                         f"confidence is accepted even at the abstain-everything threshold")
    cf = c.astype(float)
    if not np.isin(cf, (0.0, 1.0)).all():
        raise ValueError(f"{side}: `correct` must be boolean or 0/1, got values like "
                         f"{np.unique(cf)[:4]} — any non-zero float silently casts to True and "
                         f"would be counted as a CORRECT prediction")
    if group is not None:
        g = np.asarray(group)
        if g.ndim != 1 or g.size != c.size:
            raise ValueError(f"{side}: {g.size} group ids for {c.size} points — the group array "
                             f"must label every row or the units desynchronise from their losses")
    return c.astype(bool), k


def conformal_risk_control(calib_correct, calib_conf, eval_correct, eval_conf, alpha=0.10,
                           B=1.0, n_grid=256, calib_group=None, eval_group=None):
    """Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster 2022, arXiv:2208.02814).

    ESTIMAND — READ BEFORE QUOTING ANY NUMBER. CRC here certifies the JOINT mass
        E[L] = P(accepted AND wrong),
    NOT the CONDITIONAL selective risk P(wrong | accepted). A certified JOINT risk of 10% at 20%
    coverage is a 50% error rate among the predictions actually made, and the joint risk is driven to
    0 by simply abstaining. So `calib_crc_bound` / `eval_group_joint_risk` must always be quoted WITH
    their coverage, and only ever in joint language ("confidently-wrong mass", "P(accepted and
    wrong)") — never as "selective risk".

    TWO DENOMINATORS, NEVER MIXED. When `eval_group` is given, the loss is averaged WITHIN each group
    and then over groups with equal weight — that is the estimand CRC's theorem is about (draw a
    fresh GROUP), and it is NOT the sample-weighted mean whenever groups differ in size. The return
    therefore carries two complete, internally consistent triples:

        eval_group_joint_risk / eval_group_coverage / eval_group_selective_risk   <- CERTIFIED family
        eval_sample_joint_risk / eval_sample_coverage / eval_sample_selective_risk <- descriptive

    Within EITHER family, joint = selective * coverage holds exactly. ACROSS families it does not:
    one wrong sample alone in its group and nine correct samples in another gives a group joint risk
    of 0.50 and a sample joint risk of 0.10. This function previously returned the group-weighted
    joint loss alongside a SAMPLE-weighted coverage and selective risk, so those three numbers could
    not be combined even though the docstring's own identity invited it. `calib_crc_bound` bounds the
    GROUP family only; the sample family is descriptive and carries no guarantee.

    For the loss  L(lambda) = 1{ confidence >= lambda  AND  prediction wrong }  the threshold is
    (their eq. 4)
        lambda_hat = inf{ lambda : (n/(n+1)) * Rhat_n(lambda) + B/(n+1) <= alpha },
    Rhat_n = mean calibration loss, B = loss bound (=1 here); the paper sets lambda_hat = lambda_max
    when that set is empty. Their Theorem 2.1 states: "Assume that Li(lambda) is non-increasing in
    lambda, right-continuous, and Li(lambda_max) <= alpha, sup_lambda Li(lambda) <= B < infinity
    almost surely. Then E[L_{n+1}(lambda_hat)] <= alpha." That is a MARGINAL expectation over the
    joint draw of calibration AND test under EXCHANGEABILITY of L_1..L_{n+1} — not a per-run promise,
    so a single realisation may exceed alpha without contradicting the theorem.

    lambda_max = +inf here, so the condition L_i(lambda_max) <= alpha holds for EVERY point, test
    included. A data-dependent lambda_max (e.g. just above max(calib confidence)) satisfies it only on
    CALIBRATION: an eval point more confident than anything seen in calibration is still ACCEPTED at
    the supposedly abstain-everything threshold — exactly the naive-under-shift regime this study is
    about — so the "abstain-all" fallback would silently realise a large joint risk while claiming to
    have abstained.

    Caveat: with `>=` acceptance L is left- rather than right-continuous, so the theorem's infimum
    need not itself be feasible. We never take that infimum — we return a GRID point that satisfies
    the constraint, hence >= it, hence (L non-increasing) no riskier. The two differ only on the
    measure-zero event confidence == lambda exactly.

    Exchangeable unit = whatever `calib_group` says (use a patch-/block-level group so within-patch
    correlation, or the same pixel repeated across trials, cannot inflate n); `n_calib_units` reports
    the n that actually entered the bound."""
    cc, ck = _validated(calib_correct, calib_conf, calib_group, "calibration")
    ec, ek = _validated(eval_correct, eval_conf, eval_group, "evaluation")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    if not float(B) >= 1.0:
        # The loss here is the BINARY accepted-and-wrong indicator (sup L = 1), so the CRC bound's
        # B/(n+1) correction is only valid for B >= 1. A B < 1 shrinks that term below what a 0/1
        # loss entitles it to and can "certify" a threshold whose true held-out joint risk is 1.0
        # (verified: B=0.1 reports a 0.033 calibration bound at 100% eval loss). B is kept as a
        # parameter for a future bounded non-binary loss, but guarded so it can never understate it.
        raise ValueError(f"B (the loss bound) must be >= 1 for the binary accepted-and-wrong loss, "
                         f"got {B}")
    if int(n_grid) < 1:
        raise ValueError(f"n_grid must be >= 1, got {n_grid}")
    wrong_c = (~cc).astype(float)
    # H-7: candidate thresholds at the ACTUAL calibration score boundaries, not a linspace over
    # [min,max]. A linspace wastes candidates in empty score regions and can skip the true acceptance
    # boundaries between them, so its "most coverage that certifies" is only grid-optimal. Every
    # boundary is a calib confidence value: use the UNIQUE calib scores when few enough (the selected
    # threshold is then EXACT), else their QUANTILES (candidates placed at score DENSITY, which a
    # linspace is not) capped at n_grid so the (n, n_grid) loss matrix stays bounded.
    uck = np.unique(ck)
    grid = uck if uck.size <= n_grid else np.unique(np.quantile(uck, np.linspace(0.0, 1.0, n_grid)))
    grid = np.concatenate((grid, [np.inf]))   # lambda_max = +inf: the ONLY threshold that abstains on
                                              # EVERY point, calib and eval alike however far the eval
                                              # confidences have shifted above the calibration range.
                                              # A finite lambda_max (e.g. nextafter(max calib conf))
                                              # breaks Theorem 2.1's L_i(lambda_max) <= alpha on test.
    # Acceptance is conf >= lambda (NOT strict >). A 2026-07-22 review asked for strict > on a
    # continuous-lambda right-continuity argument, but this is a DISCRETE-grid CRC whose abstain-all is
    # lambda_max = +inf, and >= is what makes +inf the UNIQUE abstain-all: at the grid's top point
    # (max calib conf) >= still accepts that boundary calib sample, so it is never a free abstain-all,
    # and the procedure must fall through to +inf, which abstains on EVAL too however far eval shifted
    # above the calibration range. Strict > lets thr=max-calib-conf abstain on calib (Rhat=0, feasible)
    # while ACCEPTING every up-shifted eval pixel -> the exact "certified but 100% eval loss" failure
    # test_crc_abstain_all_actually_abstains_on_shifted_eval guards. On float MSP confidences a grid
    # point coinciding with a confidence is measure-zero, so >= vs > is numerically moot; the shift-
    # robust abstain-all is not. Keep >=.
    ploss_c = wrong_c[:, None] * (ck[:, None] >= grid[None, :])       # (n, n_grid+1) per-sample loss

    def _agg(ploss, group):
        """(mean-loss-per-grid, n_units). group=None -> each SAMPLE is a unit; else aggregate PER
        GROUP first (each group = one exchangeable unit -> strict on spatially-correlated data
        e.g. pixels within a patch, which are NOT exchangeable; patches are)."""
        if group is None:
            return ploss.mean(axis=0), ploss.shape[0]
        _, inv = np.unique(np.asarray(group), return_inverse=True)
        G = int(inv.max()) + 1
        gsum = np.zeros((G, ploss.shape[1])); np.add.at(gsum, inv, ploss)
        gcnt = np.bincount(inv, minlength=G)[:, None].clip(min=1)
        return (gsum / gcnt).mean(axis=0), G                         # per-group mean, then over groups

    Rhat, n = _agg(ploss_c, calib_group)
    crc = (n / (n + 1.0)) * Rhat + B / (n + 1.0)
    ok = np.where(crc <= alpha)[0]                                    # crc non-increasing in lambda
    feasible = bool(ok.size)                                         # any threshold (incl. abstain-all) certifies?
    j = int(ok.min()) if feasible else int(grid.size - 1)           # most coverage that certifies; else abstain-all
    thr = float(grid[j])
    kept = ek >= thr                                                 # matches the calib loss (>=); see the shift-robust abstain-all note above
    # Aggregate the joint loss AND the acceptance indicator through the SAME `_agg`, so the returned
    # risk and coverage share a denominator and `joint = selective * coverage` holds exactly within
    # each family. The previous return aggregated the loss per GROUP but took coverage and selective
    # risk as plain SAMPLE means, so the three could not be combined at all once groups had unequal
    # sizes -- and both experiments here group (phase4R by spatial block, phase8R by ROI).
    pair_e = np.stack([(~ec).astype(float) * kept, kept.astype(float)], axis=1)   # (m, 2)
    g_pair, n_eval_units = _agg(pair_e, eval_group)
    g_joint, g_cov = float(g_pair[0]), float(g_pair[1])
    s_joint, s_cov = float(pair_e[:, 0].mean()), float(kept.mean())
    _sel = lambda joint, cov: (joint / cov) if cov > 0 else float("nan")  # undefined, never 0
    # THE BOUND AND ITS REALIZATION, under names that say which is which. They were previously
    # conflated under one key called `cert_risk`, which returned the EVALUATION realization while its
    # name and every plot label promised a bound -- so the y-axis of the headline figure was not the
    # quantity it claimed to be.
    #   calib_crc_bound       : (n/(n+1))*Rhat_calib + B/(n+1) at the chosen threshold. THIS is the
    #                           certificate: an upper bound on E[L] over the joint calib+test draw
    #                           under exchangeability -- and it bounds the GROUP family only.
    #   eval_group_joint_risk : the group-weighted joint mass actually observed on eval at that
    #                           threshold. A realization, not a promise; it may exceed the bound in
    #                           any single run, and does so systematically when exchangeability is
    #                           violated -- precisely the effect this study reports.
    #   eval_sample_*         : the same three quantities with a SAMPLE denominator. Descriptive
    #                           only: CRC says nothing about them when groups differ in size.
    return {"threshold": thr,
            "calib_crc_bound": float(crc[j]),   # the CERTIFICATE (a bound). <= alpha whenever feasible.
            # --- CERTIFIED family: group-weighted, self-consistent, what calib_crc_bound bounds ---
            "eval_group_joint_risk": g_joint,   # REALIZED E[L]=P(accepted AND wrong). NOT a bound.
            "eval_group_coverage": g_cov,       # MUST accompany the risk -- abstaining drives it to 0
            "eval_group_selective_risk": _sel(g_joint, g_cov),   # P(wrong|accepted); NaN if nothing kept
            # --- DESCRIPTIVE family: sample-weighted. NOT bounded by calib_crc_bound. -------------
            "eval_sample_joint_risk": s_joint,
            "eval_sample_coverage": s_cov,
            "eval_sample_selective_risk": _sel(s_joint, s_cov),
            "n_calib_units": int(n),            # exchangeable units that entered the bound (NOT rows, if grouped)
            "n_eval_units": int(n_eval_units),  # units the eval numbers were averaged over
            "feasible": feasible}               # False => NOT certifiable at alpha (e.g. too few calib units)


def effective_sample_size(correct, group):
    """Design-effect effective sample size for a CLUSTERED binary outcome.

    n_eff = n_rows / DEFF,  DEFF = 1 + (k0 - 1) * rho,
    with k0 Kish's average cluster size (which handles UNEQUAL clusters) and rho the intra-cluster
    correlation estimated by one-way ANOVA on `correct`.

    Why not just count rows or count clusters. A finite-sample margin of the form
    sqrt(p(1-p)/n) is a BERNOULLI formula: it assumes each of the n units is one Bernoulli(p) draw.
      * n = rows assumes rho = 0 -- that correlated pixels, and the same pixel repeated once per
        missing-band trial, are independent evidence. It UNDER-corrects.
      * n = clusters assumes rho = 1 -- that a whole block behaves as a single coin flip. But a
        block is a MEAN of k correlated Bernoullis, whose variance is p(1-p)[1+(k-1)rho]/k, strictly
        below p(1-p) unless rho = 1. It OVER-corrects.
    Measured on phase4R-shaped data the two extremes bracket the truth by 5.7x below and 2.7x above,
    and the over-correction is not free: it collapsed the plug-in operating point's coverage from
    0.62 to 0.24, gutting exactly the high-missing-band cells the abstention story rests on.
    Estimating rho puts n_eff where the data says it is instead of at whichever extreme was assumed.

    Returns (n_eff, rho, k0). Falls back to the row count when there is one cluster or no variance.
    """
    correct = np.asarray(correct, float)
    g = np.asarray(group)
    _, inv = np.unique(g, return_inverse=True)
    k = np.bincount(inv).astype(float)
    n = float(k.sum()); G = int(k.size)
    if G < 2 or n <= G:
        return float(n), 0.0, float(n / max(1, G))
    ybar = correct.mean()
    gmean = np.bincount(inv, weights=correct) / k
    msb = float((k * (gmean - ybar) ** 2).sum() / (G - 1))
    msw = float(((correct - gmean[inv]) ** 2).sum() / (n - G))
    k0 = float((n - (k ** 2).sum() / n) / (G - 1))
    denom = msb + (k0 - 1.0) * msw
    if denom <= 0:
        # msb = msw = 0, i.e. the outcome is CONSTANT across every row (an all-correct or all-wrong
        # calibration set). The data then carries zero information about clustering, and returning
        # rho=0 would silently pick n_eff = n_rows -- the maximally anti-conservative answer in the
        # one case where nothing is known. Assume the worst instead: one effective unit per cluster.
        return float(G), 1.0, k0
    rho = (msb - msw) / denom
    # A negative ANOVA rho means the cluster means vary LESS than independent sampling; clamping to
    # 0 (n_eff = n_rows) is conservative there, not anti-conservative. The clamp's real exposure is
    # a truly positive rho estimated negative by noise -- measured at 18% of runs when rho=0.001 but
    # 0.00% at the rho~0.05 seen here, and the correction it suppresses is negligible at that size.
    rho = float(min(1.0, max(0.0, rho)))
    deff = 1.0 + (k0 - 1.0) * rho
    return float(max(1.0, n / max(deff, 1e-12))), rho, k0


def conformal_at_risk(calib_correct, calib_conf, eval_correct, eval_conf, target_risk=0.10,
                      conservative=True, z=1.0, calib_group=None):
    """RCPS-style target-risk operating point via a calibration set.

    ESTIMAND: the CONDITIONAL selective risk P(wrong | accepted) — the error rate among the
    predictions actually made. This is the complement of conformal_risk_control's JOINT
    P(accepted AND wrong); the two are NOT interchangeable in prose or in column names.

    Pick the confidence threshold on the CALIB set that keeps as many samples as possible
    while calib selective risk <= an (optionally conservative) target, then apply it to EVAL.
    Calib and eval should be from the same missing-band level m and DISJOINT samples.

    conservative=True subtracts a one-sided finite-sample margin
        margin = z * sqrt(target(1-target)/n_calib)
    from the target before thresholding (RCPS/UCB flavour), so the achieved EVAL risk stays
    <= target with higher probability. Set conservative=False for the plain plug-in point.
    Even so, this is an approximate control, not an exact conformal guarantee — report the
    returned coverage/risk descriptively.

    `n_calib` IS AN EFFECTIVE SAMPLE SIZE when `calib_group` is given -- not a row count and not a
    cluster count; see effective_sample_size for why both of those are wrong, in opposite
    directions. Pass the same block/ROI id array `conformal_risk_control` takes whenever the
    calibration rows are correlated. Without it the row count is kept, so every existing caller is
    byte-identical.

    WHAT THIS MARGIN DOES NOT FIX, so do not read a corrected n as a guarantee:
      * the threshold is chosen as the LOWEST one clearing the target, i.e. an extremum over many
        candidates of a noisy empirical curve. Paying for that needs a bound UNIFORM over the
        threshold (RCPS); a pointwise 1-sigma margin cannot;
      * z=1.0 is a ~84% one-sided level, not a confidence anyone would quote;
      * a calibration-vs-evaluation shift is a bias, and no margin shrinks a bias.
    (The full-set-denominator caveat that used to sit here is fixed by the refinement pass below,
    which resizes n by the accepted fraction. Leaving it listed would have told the reader a defect
    was live in the same function that closes it.)
    This stays an APPROXIMATE, descriptive operating point. conformal_risk_control is the arm with
    an actual theorem behind it.

    Returns dict(threshold, coverage, risk, n_calib_units, margin, eff_target, icc, degenerate)
    measured on EVAL, `risk` being that CONDITIONAL rate. When the target is unreachable the
    threshold is +inf: nothing is accepted, coverage is 0 and `risk` is NaN (undefined — no
    predictions were made), NEVER 0, which would read as "no errors".
    """
    _cb, ck = _validated(calib_correct, calib_conf, calib_group, "calibration")
    _eb, ek = _validated(eval_correct, eval_conf, None, "evaluation")
    if not (0.0 <= float(target_risk) <= 1.0):
        raise ValueError(f"target_risk must lie in [0, 1], got {target_risk}")
    cc = _cb.astype(float)
    rho, k0_cal = 0.0, 1.0
    if calib_group is not None:
        n_cal, rho, k0_cal = effective_sample_size(cc, calib_group)
    else:
        n_cal = float(cc.size)
    n_cal = max(1.0, n_cal)
    eff_target = target_risk
    margin = 0.0
    degenerate = False
    if conservative:
        margin = float(z * np.sqrt(target_risk * (1.0 - target_risk) / n_cal))
        # A margin at or above the target leaves eff_target <= 0, and `risk_u <= 0` then selects the
        # longest all-correct confidence prefix -- noise fitting that reports risk 0.0 and is
        # INSENSITIVE to target_risk (every alpha below the critical one returns the same
        # threshold). Say so and abstain instead. The exact condition is
        #     margin >= target  <=>  target <= z^2 / (n_eff + z^2),
        # NOT 1/(n_eff+1), which is only the z=1 case -- and z is a public parameter. At z=1.645 (a
        # 95% one-sided level) with 27 units, alpha=0.10 is ALREADY inside the degenerate region.
        # The refinement below moves the boundary further out, to z^2 / (n_eff*coverage + z^2).
        if margin >= target_risk:
            degenerate = True
        eff_target = max(0.0, target_risk - margin)
    # Evaluate calib risk at UNIQUE thresholds so a whole tied-confidence block is kept together --
    # the old per-sample position could pick a threshold that split a calib tie, then re-include the
    # full block via >= on eval (calib risk measured on a partial block, applied to the whole block).
    order = np.argsort(-ck, kind="mergesort")
    csorted = ck[order]
    kept = np.arange(1, cc.size + 1)
    risk = 1.0 - np.cumsum(cc[order]) / kept
    last = np.concatenate((csorted[1:] != csorted[:-1], [True]))     # last index of each tie block
    thr_u = csorted[last]; risk_u = risk[last]                       # whole-block-kept risk per unique threshold
    def _select(target):
        idx = np.where(risk_u <= target)[0]
        return (int(idx.max()) if idx.size else None)

    j = None if degenerate else _select(eff_target)
    # ONE REFINEMENT PASS on the denominator. The estimator being controlled is #wrong / #accepted,
    # so its variance goes as p(1-p) / n_ACCEPTED, not / n_total -- and at the low coverages this
    # study operates at (12% at six missing groups) that difference is ~3x, all of it in the
    # anti-conservative direction. The accepted count depends on the threshold, which depends on the
    # margin, so this cannot be solved in closed form; instead the first pass fixes a threshold, the
    # accepted fraction at that threshold rescales n, and the second pass re-selects. Monotone and
    # therefore non-oscillating: n_acc <= n_cal, so margin only grows and the threshold only rises.
    #
    # STOPPING AT ONE STEP IS LOAD-BEARING, NOT INCIDENTAL. Iterated to convergence this map is
    # abstain-all in ~74% of low-coverage runs (true risk 0.30, 27 clusters), and one step agrees
    # with the fixed point in 0% of them -- the operating point exists BECAUSE the iteration is
    # truncated. That is a deliberate choice of a bounded, monotone correction over a fixed point
    # that collapses; it is not an approximation that happens to be close. At phase4R's n_eff (~550)
    # the whole refinement costs ~1.5pp of coverage, well clear of the cliff below n_eff ~ 15.
    #
    # One caveat this cannot fix: rho is measured over ALL calibration rows and then applied to the
    # ACCEPTED subset. When acceptance tracks block difficulty the accepted set is LESS clustered
    # than the whole, so this over-corrects (up to ~2.8x at 5% coverage); when confidence is a pure
    # block score it under-corrects by 5-8%. The direction is data-dependent and unsigned.
    n_margin = n_cal              # the n the RETURNED margin is sized from; updated by refinement
    # GATED ON `calib_group`, not on `conservative` alone. Gated the loose way this silently changed
    # every existing caller -- phase8R and phase8E both call without a group -- by ~1pp of coverage,
    # falsifying this function's own "byte-identical" promise and staling their committed CSVs. A
    # caller that has not told us its rows are clustered gets exactly the old behaviour.
    if j is not None and conservative and calib_group is not None:
        cov_cal = float(kept[last][j] / cc.size)          # calib coverage at the first threshold
        n_acc = max(1.0, n_cal * cov_cal)
        margin = float(z * np.sqrt(target_risk * (1.0 - target_risk) / n_acc))
        n_margin = n_acc
        if margin >= target_risk:
            # eff_target must move too. Leaving it at the first-pass value reported a target that
            # was never used, next to a margin that was.
            degenerate, j, eff_target = True, None, 0.0
        else:
            eff_target = max(0.0, target_risk - margin)
            j = _select(eff_target)
    thr = float(thr_u[j]) if j is not None else np.inf
    ec = _eb.astype(float)
    sel = ek >= thr
    coverage = float(sel.mean())
    risk_eval = float(1.0 - ec[sel].mean()) if sel.any() else float("nan")   # undefined, not 0
    # n_calib_units is the n the RETURNED margin was actually sized from -- after the refinement
    # pass that is the ACCEPTED count, not the full-set design-effect n. Reporting the full-set
    # figure next to a margin computed from a smaller one made the column unauditable: the two
    # disagreed by ~2-3x, which is the whole quantity the column exists to expose. n_calib_full
    # keeps the pre-refinement value for anyone who wants to see both.
    # H-6: an explicit status, because `degenerate=False` alone was ambiguous -- it read identically
    # whether a threshold was found (ok) or the target was simply UNREACHABLE (all calib wrong -> no
    # threshold clears it -> thr=+inf, coverage 0, risk NaN). Those are different outcomes and a reader
    # must be able to tell them apart. (invalid_input never returns here -- `_validated` raises.)
    status = ("margin_degenerate" if degenerate else
              "target_unreachable" if j is None else "ok")
    return {"threshold": float(thr), "coverage": coverage, "risk": risk_eval,
            "n_calib_units": int(round(n_margin)), "n_calib_full": int(round(n_cal)),
            "margin": float(margin), "eff_target": float(eff_target),
            "icc": float(rho), "k0": float(k0_cal), "degenerate": bool(degenerate),
            "status": status}
