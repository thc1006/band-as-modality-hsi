#!/usr/bin/env python
"""Offline analyses on the flagship logit dumps (no retraining):
  (3.3) a scene-component bootstrap of the L2A naive joint risk -- resample the 184 exchangeable units
        with replacement and recompute the mean within-component confidently-wrong loss, to check the
        headline against scene-composition uncertainty independently of the two-way cluster SE;
  (3.6) the full temperature x threshold 2x2 factorial -- the missing B arm (source temperature, target
        threshold) alongside naive (A), control (C) and Mondrian (D), to attribute the failure properly.
"""
import glob
import os
import sys
import statistics as st

import numpy as np
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8R_reliability as P8R
from bandsim.reliability import fit_temperature, conformal_risk_control

DUMPS = sorted(glob.glob(os.path.join(_HERE, "..", "paper", "scenedump_flagship", "*.npz")))
ALPHA = 0.10
rng = np.random.default_rng(20260723)


def crc_threshold(logits, T, y, mask, comp):
    p = softmax(logits[mask] / T, axis=1)
    corr = (p.argmax(1) == y[mask])
    return conformal_risk_control(corr, p.max(1), corr, p.max(1), alpha=ALPHA,
                                  calib_group=comp[mask], eval_group=comp[mask])["threshold"]


def comp_equal_joint(logits, T, y, mask, comp, thr):
    p = softmax(logits[mask] / T, axis=1)
    aw = (p.max(1) >= thr) & (p.argmax(1) != y[mask])
    ce = comp[mask]
    return float(np.mean([aw[ce == c].mean() for c in np.unique(ce)])) * 100


def main():
    print(f"loaded {len(DUMPS)} flagship dumps")
    fac = {k: [] for k in ("A_naive", "B_srcT_tgtThr", "C_control", "D_mondrian")}
    boot = []
    for f in DUMPS:
        d = np.load(f)
        lc, ll, y, comp = d["logits_clean"], d["logits_l2a"], d["y"], d["comp"]

        # ---- 2x2 factorial over the 10 calibration splits ----
        for ss in range(10):
            mt, mc, me = P8R.split_test_rois(comp, ss)
            Tc = fit_temperature(lc[mt], y[mt])                 # source (clean) temperature
            Tl = fit_temperature(ll[mt], y[mt])                 # target (L2A) temperature
            # A threshold lives on the temperature-scaled confidence, so each arm's threshold MUST be
            # calibrated with the SAME temperature it is applied with; only the calibration DATA (source
            # clean vs target L2A) varies between "source threshold" and "target threshold".
            thr_A = crc_threshold(lc, Tc, y, mc, comp)          # source data, source temp
            thr_B = crc_threshold(ll, Tc, y, mc, comp)          # target data, source temp
            thr_C = crc_threshold(lc, Tl, y, mc, comp)          # source data, target temp (= control)
            thr_D = crc_threshold(ll, Tl, y, mc, comp)          # target data, target temp (= Mondrian)
            fac["A_naive"].append(comp_equal_joint(ll, Tc, y, me, comp, thr_A))
            fac["B_srcT_tgtThr"].append(comp_equal_joint(ll, Tc, y, me, comp, thr_B))
            fac["C_control"].append(comp_equal_joint(ll, Tl, y, me, comp, thr_C))
            fac["D_mondrian"].append(comp_equal_joint(ll, Tl, y, me, comp, thr_D))

        # ---- scene-component bootstrap of the L2A naive joint (fixed clean operating point) ----
        mt, mc, me = P8R.split_test_rois(comp, 0)
        Tc = fit_temperature(lc[mt], y[mt])
        thr = crc_threshold(lc, Tc, y, mc, comp)
        p = softmax(ll / Tc, axis=1)
        aw = (p.max(1) >= thr) & (p.argmax(1) != y)
        uc = np.unique(comp)
        Lc = np.array([aw[comp == c].mean() for c in uc])       # per-component confidently-wrong loss
        for _ in range(4000):
            b = rng.integers(0, len(uc), len(uc))               # resample components with replacement
            boot.append(Lc[b].mean() * 100)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n(3.3) scene-component bootstrap, L2A naive joint: mean {np.mean(boot):.1f}%, "
          f"95% CI [{lo:.1f}, {hi:.1f}]  ({'excludes' if lo > 10 else 'includes'} the 10% target)")
    print("\n(3.6) temperature x threshold 2x2 factorial (L2A joint risk, %):")
    print(f"  {'':22s} {'source threshold':>18s} {'target threshold':>18s}")
    print(f"  {'source temperature':22s} {st.mean(fac['A_naive']):15.1f} (A) {st.mean(fac['B_srcT_tgtThr']):15.1f} (B)")
    print(f"  {'target temperature':22s} {st.mean(fac['C_control']):15.1f} (C) {st.mean(fac['D_mondrian']):15.1f} (D)")
    print(f"  -> naive (A) {st.mean(fac['A_naive']):.1f}; fixing only temperature (C) {st.mean(fac['C_control']):.1f}; "
          f"fixing only threshold (B) {st.mean(fac['B_srcT_tgtThr']):.1f}; both (D) {st.mean(fac['D_mondrian']):.1f}")


if __name__ == "__main__":
    main()
