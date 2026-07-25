#!/usr/bin/env python
"""Weighted-conformal ablation (offline, flagship logit dumps): can a covariate-shift-aware conformal
threshold restore the certificate WITHOUT recalibrating on L2A labels, or is target relabelling (Mondrian)
truly necessary? We build the Tibshirani et al. (2019) likelihood-ratio-weighted split-conformal threshold,
weighting each clean calibration pixel by w(s) = p_L2A(s)/p_clean(s) (the confidence-density ratio, using
only the UNLABELLED L2A confidences an operator can see), then deploy that threshold on L2A. If the true
L2A joint risk stays above target, the failure is not pure covariate shift -- the confidence-to-error
relationship itself moves under L2A -- so no reweighting of clean labels can fix it, which is exactly why
Mondrian (which uses L2A labels) is required. This is the standard 'is Mondrian the only fix?' ablation.

Sanity: with uniform weights the construction must reproduce the naive (clean-threshold) joint risk exactly.
"""
import glob
import os
import sys

import numpy as np
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8R_reliability as P8R
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.reliability import fit_temperature

DUMPS = sorted(glob.glob(os.path.join(_HERE, "..", "paper", "scenedump_flagship", "*.npz")))
ALPHA = 0.10
T9 = 2.262


def weighted_threshold(s_cal, wrong_cal, w, alpha):
    """Smallest confidence threshold whose WEIGHTED joint risk on the calibration set is <= alpha.
    Weighted joint risk at tau = sum_i w_i [s_i>=tau] wrong_i / sum_i w_i. As tau falls (accept more) the
    risk rises, so we scan candidate thresholds high->low and stop just before it would exceed alpha."""
    order = np.argsort(-s_cal)                       # high confidence first
    s, wr, ww = s_cal[order], wrong_cal[order], w[order]
    cum_bad = np.cumsum(ww * wr)                     # weighted confidently-wrong mass among accepted
    W = ww.sum()
    risk = cum_bad / W                               # weighted JOINT risk if we accept down to each point
    ok = np.where(risk <= alpha)[0]
    if len(ok) == 0:
        return np.inf                                # cannot meet target even accepting nothing beyond top
    k = ok[-1]                                        # accept the k+1 highest-confidence points
    return s[k]


def density_ratio(s_cal, s_tgt, nbins=25):
    """w(s) = p_tgt(s)/p_cal(s) via shared-edge histograms with Laplace smoothing, evaluated at each s_cal."""
    lo, hi = 0.0, 1.0
    edges = np.linspace(lo, hi, nbins + 1)
    hc, _ = np.histogram(np.clip(s_cal, lo, hi), bins=edges)
    ht, _ = np.histogram(np.clip(s_tgt, lo, hi), bins=edges)
    pc = (hc + 1.0) / (hc.sum() + nbins)
    pt = (ht + 1.0) / (ht.sum() + nbins)
    ratio = pt / pc
    idx = np.clip(np.digitize(np.clip(s_cal, lo, hi), edges) - 1, 0, nbins - 1)
    return ratio[idx]


def joint_at(s, wrong, thr):
    """Unweighted joint risk (accept & wrong) at a fixed threshold, per-pixel mean (component-equal handled
    by the caller when it passes component-averaged inputs)."""
    return float(((s >= thr) & wrong).mean()) * 100


def comp_joint_at(s, wrong, comp, thr):
    aw = (s >= thr) & wrong
    return float(np.mean([aw[comp == c].mean() for c in np.unique(comp)])) * 100


def main():
    print(f"loaded {len(DUMPS)} flagship dumps")
    naive, weighted, mond, sanity = [], [], [], []
    ncov, wcov, mcov = [], [], []
    for si, f in enumerate(DUMPS):
        d = np.load(f)
        lc, ll, y, comp = d["logits_clean"], d["logits_l2a"], d["y"], d["comp"]
        for ss in range(10):
            mt, mc, me = P8R.split_test_rois(comp, ss)
            Tc = fit_temperature(lc[mt], y[mt])
            # clean calibration confidences + errors
            pc = softmax(lc[mc] / Tc, axis=1)
            s_cal, wrong_cal = pc.max(1), (pc.argmax(1) != y[mc])
            # L2A evaluation confidences (unlabelled for the weight; labels used only to score)
            pe = softmax(ll[me] / Tc, axis=1)
            s_eval, wrong_eval = pe.max(1), (pe.argmax(1) != y[me])
            comp_e = comp[me]
            # naive: clean-calibrated unweighted threshold
            thr_naive = weighted_threshold(s_cal, wrong_cal, np.ones_like(s_cal), ALPHA)
            naive.append((si, ss, comp_joint_at(s_eval, wrong_eval, comp_e, thr_naive)))
            sanity.append(comp_joint_at(s_eval, wrong_eval, comp_e, thr_naive))
            ncov.append(float((s_eval >= thr_naive).mean()) * 100)
            # weighted: reweight clean calib by the L2A/clean confidence-density ratio
            w = density_ratio(s_cal, s_eval)
            thr_w = weighted_threshold(s_cal, wrong_cal, w, ALPHA)
            weighted.append((si, ss, comp_joint_at(s_eval, wrong_eval, comp_e, thr_w)))
            wcov.append(float((s_eval >= thr_w).mean()) * 100)
            # Mondrian reference: threshold calibrated on L2A itself
            Tl = fit_temperature(ll[mt], y[mt])
            pl = softmax(ll[mc] / Tl, axis=1)
            thr_m = weighted_threshold(pl.max(1), (pl.argmax(1) != y[mc]), np.ones(len(pl)), ALPHA)
            pe_m = softmax(ll[me] / Tl, axis=1)
            mond.append((si, ss, comp_joint_at(pe_m.max(1), pe_m.argmax(1) != y[me], comp_e, thr_m)))
            mcov.append(float((pe_m.max(1) >= thr_m).mean()) * 100)

    for name, rows, cov in [("naive (clean thr)", naive, ncov), ("weighted-conformal", weighted, wcov),
                            ("Mondrian (L2A thr)", mond, mcov)]:
        m, se = two_way_se(rows)
        flag = "BREACH" if m - T9 * se > 10 else "controlled"
        print(f"  {name:22s} L2A joint {m:6.2f} +/- {se:4.2f}  [{m - T9 * se:5.1f},{m + T9 * se:5.1f}]  "
              f"cov {np.mean(cov):4.0f}%  {flag}")
    print(f"\n  sanity: uniform-weight construction reproduces naive "
          f"({'OK' if abs(np.mean(sanity) - two_way_se(naive)[0]) < 1e-6 else 'MISMATCH'})")
    print("  reading: if weighted-conformal still breaches, the L2A shift is not pure covariate shift, so")
    print("  no reweighting of clean labels restores the certificate -- Mondrian's use of L2A labels is required.")


if __name__ == "__main__":
    main()
