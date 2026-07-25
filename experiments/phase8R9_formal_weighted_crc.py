#!/usr/bin/env python
"""E1 (round-5 review 3.1): the FORMAL covariate-shift weighted conformal, not the confidence-density
heuristic. The Tibshirani et al. weight is a real likelihood ratio w(x)=dP_target/dP_source(x); we estimate
it with a clean-vs-L2A domain classifier on the model's representation (the temperature-scaled softmax, a
richer statistic than the 1-D confidence), reweight the clean calibration by w, find the CRC threshold, and
deploy on L2A. We report the effective sample size after weighting and clip the weights. Offline from
scenedump_flagship. First-principles prediction: it ALSO fails, because the L1C->L2A shift moves the
confidence-to-error relationship P(Y|X), violating the covariate-shift assumption weighted conformal needs
-- so 'no source-only reweighting repairs it' is defensible, while 'weighted conformal does not repair it'
(the old over-claim) is not."""
import glob
import os
import sys

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8R_reliability as P8R
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.reliability import fit_temperature

DUMPS = sorted(glob.glob(os.path.join(_HERE, "..", "paper", "scenedump_flagship", "*.npz")))
ALPHA = 0.10
T9 = 2.262


def wthr(s, wrong, w, alpha):
    """Smallest confidence threshold whose w-weighted joint risk on the calibration set is <= alpha."""
    o = np.argsort(-s)
    ss, wr, ww = s[o], wrong[o], w[o]
    risk = np.cumsum(ww * wr) / ww.sum()
    ok = np.where(risk <= alpha)[0]
    return ss[ok[-1]] if len(ok) else np.inf


def cjoint(s, wrong, comp, thr):
    aw = (s >= thr) & wrong
    return float(np.mean([aw[comp == c].mean() for c in np.unique(comp)])) * 100


def cover(s, thr):
    return float((s >= thr).mean()) * 100


def main():
    print(f"loaded {len(DUMPS)} flagship dumps; FORMAL weighted conformal (domain-classifier w)", flush=True)
    naive, weighted, ncov, wcov, ess_all = [], [], [], [], []
    for si, f in enumerate(DUMPS):
        d = np.load(f)
        lc, ll, y, comp = d["logits_clean"], d["logits_l2a"], d["y"], d["comp"]
        for ss in range(10):
            mt, mc, me = P8R.split_test_rois(comp, ss)
            Tc = fit_temperature(lc[mt], y[mt])
            pc = softmax(lc[mc] / Tc, axis=1); s_cal, wrong_cal = pc.max(1), (pc.argmax(1) != y[mc])
            pe = softmax(ll[me] / Tc, axis=1); s_ev, wrong_ev = pe.max(1), (pe.argmax(1) != y[me])
            comp_e = comp[me]
            # domain classifier on the TEMP split (disjoint from calib/eval): clean=0, L2A=1
            Xd = np.vstack([softmax(lc[mt] / Tc, axis=1), softmax(ll[mt] / Tc, axis=1)])
            yd = np.r_[np.zeros(int(mt.sum())), np.ones(int(mt.sum()))]
            clf = LogisticRegression(max_iter=300).fit(Xd, yd)
            p1 = clf.predict_proba(pc)[:, 1]
            w = np.clip(p1 / (1 - p1 + 1e-9), 1e-3, 1e3)           # likelihood ratio, clipped
            ess_all.append((w.sum() ** 2 / (w ** 2).sum()) / len(w) * 100)
            thr_n = wthr(s_cal, wrong_cal, np.ones_like(s_cal), ALPHA)
            thr_w = wthr(s_cal, wrong_cal, w, ALPHA)
            naive.append((si, ss, cjoint(s_ev, wrong_ev, comp_e, thr_n))); ncov.append(cover(s_ev, thr_n))
            weighted.append((si, ss, cjoint(s_ev, wrong_ev, comp_e, thr_w))); wcov.append(cover(s_ev, thr_w))
    for name, rows, cov in [("naive (clean thr)", naive, ncov), ("FORMAL weighted", weighted, wcov)]:
        m, se = two_way_se(rows)
        print(f"  {name:18s} L2A joint {m:5.2f} +/- {se:.2f}  [{m - T9 * se:.1f},{m + T9 * se:.1f}]  "
              f"cov {np.mean(cov):3.0f}%  {'BREACH' if m - T9 * se > 10 else 'controlled'}", flush=True)
    mw, sew = two_way_se(weighted)
    print(f"  effective sample size after weighting: {np.mean(ess_all):.0f}% of calibration")
    print(f"  -> the FORMAL covariate-shift correction "
          f"{'ALSO FAILS -- the shift is not pure covariate shift (P(Y|X) moved), so no source-only reweighting repairs it' if mw - T9 * sew > 10 else 'restores control'}")


if __name__ == "__main__":
    main()
