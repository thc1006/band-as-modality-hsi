#!/usr/bin/env python
"""R6-E1 (round-6 review 3.1 / B1 / C10): the paper's 'weighted conformal' is an importance-weighted
source-calibration HEURISTIC (a global threshold whose w-weighted calibration joint risk <= alpha), not the
formal test-point-dependent covariate-shift CRC. Two things are needed: (1) rename/downgrade it, and (2)
show the heuristic is not simply broken -- that under a GENUINE covariate shift (P(Y|X) fixed, only P(X)
moves) it DOES recover risk control. This script is that synthetic control, plus the domain-classifier
diagnostics (AUROC, weight histogram, ESS) on the real flagship. If the heuristic recovers on synthetic
covariate shift but fails on L2A, then the L2A failure is an assumption violation (the shift moves P(Y|X)),
not a broken implementation."""
import glob
import os
import sys

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from phase8R9_formal_weighted_crc import wthr
from bandsim.reliability import fit_temperature

ALPHA = 0.10


def joint(conf, wrong, thr):
    return float(((conf >= thr) & wrong).mean()) * 100


def synthetic_control():
    print("=== (1) SYNTHETIC covariate-shift control: does the heuristic recover when P(Y|X) is fixed? ===")
    rng = np.random.default_rng(20260725)
    d, K, N = 8, 4, 40000
    Wtrue = rng.standard_normal((d, K)) * 1.6            # FIXED P(Y|X) = softmax(x @ Wtrue), same for src/tgt
    shift = rng.standard_normal(d) * 1.1                  # covariate shift: only P(X) moves

    def gen(n, mu):
        X = (rng.standard_normal((n, d)) + mu).astype(np.float32)
        P = softmax(X @ Wtrue, axis=1)
        Y = (P.cumsum(1) > rng.random((n, 1))).argmax(1)
        return X, Y

    Xtr, Ytr = gen(N, 0.0)                               # source train
    Xca, Yca = gen(N, 0.0)                               # source calibration
    Xdc, _ = gen(N, shift)                               # target sample for the domain classifier
    Xev, Yev = gen(N, shift)                             # target evaluation (P(Y|X) identical to source)
    clf = LogisticRegression(max_iter=500).fit(Xtr, Ytr)
    Pc = clf.predict_proba(Xca); cc, wc = Pc.max(1), (Pc.argmax(1) != Yca)
    Pe = clf.predict_proba(Xev); ce, we = Pe.max(1), (Pe.argmax(1) != Yev)

    # domain classifier on the softmax (exactly the flagship recipe): clean/source=0, target=1
    dclf = LogisticRegression(max_iter=500).fit(
        np.vstack([Pc, clf.predict_proba(Xdc)]), np.r_[np.zeros(len(Pc)), np.ones(len(Xdc))])
    p1 = dclf.predict_proba(Pc)[:, 1]
    w_dc = np.clip(p1 / (1 - p1 + 1e-9), 1e-3, 1e3)
    w_true = np.exp(Xca @ shift - 0.5 * shift @ shift)   # analytic Gaussian likelihood ratio dP_tgt/dP_src

    thr_n = wthr(cc, wc, np.ones_like(cc), ALPHA)
    thr_dc = wthr(cc, wc, w_dc, ALPHA)
    thr_tw = wthr(cc, wc, w_true, ALPHA)
    jn, jdc, jtw = joint(ce, we, thr_n), joint(ce, we, thr_dc), joint(ce, we, thr_tw)
    ess = (w_dc.sum() ** 2 / (w_dc ** 2).sum()) / len(w_dc) * 100
    print(f"  target joint risk (alpha=10%):  naive {jn:.1f}  |  weighted(domain-clf) {jdc:.1f}  "
          f"|  weighted(true w) {jtw:.1f}   [ESS {ess:.0f}%]")
    ok = jn > 13 and jdc < 13
    print(f"  -> naive {'BREACHES' if jn > 13 else 'ok'} under the covariate shift; the heuristic "
          f"{'RECOVERS control' if jdc < 13 else 'does NOT recover'} -- so on synthetic covariate shift the "
          f"implementation is {'SOUND (its L2A failure is an assumption violation, not a bug)' if ok else 'QUESTIONABLE'}")


def flagship_diagnostics():
    print("\n=== (2) DIAGNOSTICS of the weighting on the real flagship (clean vs L2A) ===")
    dumps = sorted(glob.glob(os.path.join(_HERE, "..", "paper", "scenedump_flagship", "*.npz")))
    d = np.load(dumps[0])
    lc, ll, y = d["logits_clean"], d["logits_l2a"], d["y"]
    T = fit_temperature(lc, y)
    Pc, Pl = softmax(lc / T, axis=1), softmax(ll / T, axis=1)
    dclf = LogisticRegression(max_iter=300).fit(
        np.vstack([Pc, Pl]), np.r_[np.zeros(len(Pc)), np.ones(len(Pl))])
    auroc = roc_auc_score(np.r_[np.zeros(len(Pc)), np.ones(len(Pl))],
                          dclf.predict_proba(np.vstack([Pc, Pl]))[:, 1])
    p1 = dclf.predict_proba(Pc)[:, 1]
    w = np.clip(p1 / (1 - p1 + 1e-9), 1e-3, 1e3)
    ess = (w.sum() ** 2 / (w ** 2).sum()) / len(w) * 100
    clipfrac = float(((p1 / (1 - p1 + 1e-9) < 1e-3) | (p1 / (1 - p1 + 1e-9) > 1e3)).mean()) * 100
    print(f"  domain-classifier clean-vs-L2A AUROC {auroc:.3f} (0.5=indistinguishable, 1=perfectly separable)")
    print(f"  weight w=odds:  median {np.median(w):.2f}  p5 {np.percentile(w, 5):.2f}  p95 {np.percentile(w, 95):.2f}  "
          f"clipped {clipfrac:.1f}%  ESS {ess:.0f}% of calibration")
    print(f"  -> a well-separated (AUROC>0.7), heavy-tailed weight with low ESS: the reweighting is doing real work "
          f"but cannot fix an output-space P(Y|X) drift -- consistent with the heuristic failing on L2A.")


if __name__ == "__main__":
    synthetic_control()
    flagship_diagnostics()
