#!/usr/bin/env python
"""Alpha-sensitivity of the flagship breach (offline, from the 10 flagship logit dumps, no retraining).
Is the naive-certificate failure an artefact of the 10% operating point, or does it hold across target
risks? We re-run the CRC at alpha in {5,10,15,20}% and report naive (clean-calibrated) vs Mondrian
(L2A-calibrated) component-equal joint risk with the two-way cluster-robust SE over the seed x split
design, so a reviewer asking 'why alpha=10%?' sees the phenomenon is not target-specific."""
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
from bandsim.reliability import fit_temperature, conformal_risk_control

DUMPS = sorted(glob.glob(os.path.join(_HERE, "..", "paper", "scenedump_flagship", "*.npz")))
T9 = 2.262


def crc_thr(logits, T, y, mask, comp, alpha):
    p = softmax(logits[mask] / T, axis=1)
    corr = (p.argmax(1) == y[mask])
    return conformal_risk_control(corr, p.max(1), corr, p.max(1), alpha=alpha,
                                  calib_group=comp[mask], eval_group=comp[mask])["threshold"]


def ce_joint(logits, T, y, mask, comp, thr):
    p = softmax(logits[mask] / T, axis=1)
    aw = (p.max(1) >= thr) & (p.argmax(1) != y[mask])
    ce = comp[mask]
    return float(np.mean([aw[ce == c].mean() for c in np.unique(ce)])) * 100


def main():
    print(f"loaded {len(DUMPS)} flagship dumps")
    print(f"  {'target':>7s} {'naive joint (t9 CI)':>26s} {'breach?':>8s} {'Mondrian joint':>16s}")
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        rn, rm = [], []
        for si, f in enumerate(DUMPS):
            d = np.load(f)
            lc, ll, y, comp = d["logits_clean"], d["logits_l2a"], d["y"], d["comp"]
            for ss in range(10):
                mt, mc, me = P8R.split_test_rois(comp, ss)
                Tc = fit_temperature(lc[mt], y[mt])
                Tl = fit_temperature(ll[mt], y[mt])
                thr_n = crc_thr(lc, Tc, y, mc, comp, alpha)     # naive: clean-calibrated threshold
                thr_m = crc_thr(ll, Tl, y, mc, comp, alpha)     # Mondrian: L2A-calibrated threshold
                rn.append((si, ss, ce_joint(ll, Tc, y, me, comp, thr_n)))
                rm.append((si, ss, ce_joint(ll, Tl, y, me, comp, thr_m)))
        mn, sen = two_way_se(rn)
        mm, sem = two_way_se(rm)
        tgt = alpha * 100
        breach = "YES" if mn - T9 * sen > tgt else "no"
        print(f"  {tgt:6.0f}% {mn:6.2f} +/- {sen:4.2f} [{mn - T9 * sen:5.1f},{mn + T9 * sen:5.1f}] "
              f"{breach:>8s}  {mm:6.2f} +/- {sem:4.2f}", flush=True)
    print("\n  (naive is calibrated to hold each target on CLEAN, then deployed on L2A; Mondrian recalibrates on L2A)")


if __name__ == "__main__":
    main()
