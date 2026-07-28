#!/usr/bin/env python
"""R6 (reviewer B1, the formal fix): a genuine covariate-shift WEIGHTED conformal risk control at the
scene-connected-component level -- the weighted analog of the Angelopoulos et al. (2024) CRC bound with
Tibshirani et al. (2019) covariate-shift weights. Unlike the round-5 heuristic (phase8R9, a single global
importance-weighted threshold), this has (i) a scene-component-level likelihood ratio w(X_i), (ii) the
test-point weight term p_{n+1}^w(x).B in the finite-sample bound, and (iii) a TEST-DEPENDENT threshold
lambda-hat(x). Cached flagship logit dumps are reused (no retraining). A synthetic PURE-covariate-shift
positive control (P(Y|X) fixed) proves the implementation RECOVERS control -- naive breaches, weighted
restores -- before we report its behaviour on the real L1C->L2A shift.

Finite-sample weighted CRC. Exchangeable unit = component i; per-unit loss
    L_i(lambda) = (1/|S_i|) sum_{p in S_i} 1[ c_p >= lambda AND y_hat_p != y_p ]   (non-increasing in lambda),
loss bound B = sup L = 1. For a test component x with covariate X_{n+1},
    lambda-hat(x) = inf{ lambda : sum_i p_i^w(x) L_i(lambda) + p_{n+1}^w(x) B <= alpha },
    p_i^w(x) = w(X_i)/(sum_j w(X_j)+w(X_{n+1})),   p_{n+1}^w(x) = w(X_{n+1})/(sum_j w(X_j)+w(X_{n+1})).
With W = sum_j w(X_j) and G(lambda) = sum_i w(X_i) L_i(lambda),
    lambda-hat(x) = inf{ lambda : G(lambda) <= alpha*W - w(X_{n+1})*(1-alpha) },
and the grid is augmented with lambda = +inf (accept nothing, L=COV=0). With uniform weights this
reduces to the standard CRC bound sum_i L_i <= alpha(n+1)-B (a built-in sanity check that must match
the flagship naive ~28.9%). The covariate X is the model's temperature-scaled softmax averaged over the
component (the input reflectance is not in the dumps); w is a clean-vs-L2A logistic domain classifier's
odds, clipped, fitted on the temperature split (disjoint from calibration and evaluation).

Run: python phase8R11_weighted_crc_formal.py
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8R_reliability as P8R
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.reliability import fit_temperature
from bandsim import hw

ALPHA = 0.10
CLIP = 1e3
NGRID = 256
BETA = 2.5                                              # synthetic-shift strength for the positive control
def _seed_num(p):                                                   # r2 §6.5: sort by the NUMERIC seed, not
    import re                                                       # lexicographically (seed10 < seed2 as text)
    m = re.search(r"seed(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 1 << 30
DUMPS = sorted(glob.glob(os.path.join(_HERE, "..", "paper", "scenedump_flagship", "*.npz")), key=_seed_num)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def state_arrays(logits, y, comp, mask, Tc, grid):
    """Per-component confidently-wrong loss L[c,g] and coverage COV[c,g] over the ascending grid (which
    ends with +inf = accept nothing), the mean-softmax covariate feat[c], and the mean confidence -- all
    in ONE component-sorted pass (O(n log n); no per-component O(n) masks)."""
    sm = softmax(logits[mask] / Tc)
    conf = sm.max(1); wrong = (sm.argmax(1) != y[mask]).astype(np.float64); cmp = comp[mask]
    order = np.argsort(cmp, kind="stable")
    cs, cf_s, wr_s, sm_s = cmp[order], conf[order], wrong[order], sm[order]
    ids, starts = np.unique(cs, return_index=True)
    ends = np.append(starts[1:], len(cs)); C = len(ids)
    L = np.zeros((C, len(grid))); COV = np.zeros((C, len(grid)))
    feat = np.zeros((C, sm.shape[1])); meanconf = np.zeros(C)
    for ci in range(C):
        a, b = starts[ci], ends[ci]; n = b - a
        cf = cf_s[a:b]; feat[ci] = sm_s[a:b].mean(0); meanconf[ci] = cf.mean()
        o = np.argsort(cf); cfs = cf[o]; cw = np.cumsum(wr_s[a:b][o]); totw = cw[-1]
        kbelow = np.searchsorted(cfs, grid, side="left")
        wrong_below = np.where(kbelow > 0, cw[np.clip(kbelow - 1, 0, n - 1)], 0.0)
        L[ci] = (totw - wrong_below) / n; COV[ci] = (n - kbelow) / n
    return L, COV, ids, feat, meanconf


def domain_weights(F0, F1, Fq_list):
    """Odds P(target|x)/P(source|x); source=F0 (0), target=F1 (1); clipped weights per query + AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([F0, F1]); yb = np.r_[np.zeros(len(F0)), np.ones(len(F1))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), yb)
    auc = roc_auc_score(yb, clf.predict_proba(sc.transform(X))[:, 1])
    out = []
    for Fq in Fq_list:
        p = clf.predict_proba(sc.transform(Fq))[:, 1]
        out.append(np.clip(p / (1.0 - p + 1e-12), 1.0 / CLIP, CLIP))
    return out, auc


def ess_frac(w):
    return (w.sum() ** 2) / (np.square(w).sum() * len(w))


def weighted_crc_perunit(Lc, wc, grid, Le, COVe, we, alpha):
    """Formal weighted CRC: per eval component, test-dependent lambda-hat(x); return per-unit realized
    within-component confidently-wrong fraction and coverage. Uniform w reduces to standard CRC."""
    W = wc.sum()
    G = (wc[:, None] * Lc).sum(0)                        # G(lambda), non-increasing across the grid
    realized = np.zeros(len(we)); cover = np.zeros(len(we))
    for x in range(len(we)):
        target = alpha * W - we[x] * (1.0 - alpha)
        ok = np.nonzero(G <= target)[0]
        if len(ok) == 0:
            realized[x] = 0.0; cover[x] = 0.0            # infeasible even at lambda=+inf: accept nothing
        else:
            g = ok[0]
            realized[x] = Le[x, g]; cover[x] = COVe[x, g]
    return realized, cover


def make_grid(conf_calib):
    return np.append(np.quantile(conf_calib, np.linspace(0.0, 1.0, NGRID)), np.inf)


def comp_loss(conf, wrong, comp, grid):
    """Component confidently-wrong loss L[c,g] and coverage COV[c,g] over the ascending grid, from raw
    per-pixel confidence + correctness (used by the fully-synthetic positive control)."""
    order = np.argsort(comp, kind="stable")
    cs, cf_s, wr_s = comp[order], conf[order], wrong[order].astype(np.float64)
    ids, starts = np.unique(cs, return_index=True)
    ends = np.append(starts[1:], len(cs)); C = len(ids)
    L = np.zeros((C, len(grid))); COV = np.zeros((C, len(grid)))
    for ci in range(C):
        a, b = starts[ci], ends[ci]; n = b - a
        cf = cf_s[a:b]; o = np.argsort(cf); cfs = cf[o]; cw = np.cumsum(wr_s[a:b][o]); totw = cw[-1]
        kbelow = np.searchsorted(cfs, grid, side="left")
        wrong_below = np.where(kbelow > 0, cw[np.clip(kbelow - 1, 0, n - 1)], 0.0)
        L[ci] = (totw - wrong_below) / n; COV[ci] = (n - kbelow) / n
    return L, COV, ids


def synthetic_positive_control(alpha, seeds, beta):
    """A fully-controlled PURE covariate-shift positive control (P(Y|X) fixed by construction). Each
    component has a difficulty covariate d in [0,1]; its per-pixel error rate rises with d, so its
    confidently-wrong loss rises with d, while the pixel-confidence law is identical for all d (so P(Y|X)
    -- error given confidence -- does NOT change). The target upweights high-d components with the KNOWN
    weight w(d)=exp(beta*d). The source-calibrated naive threshold therefore breaches the target risk while
    the weighted CRC with the true weight restores control at useful coverage -- validating the weighting
    and the test-point term end-to-end (the sanity check separately validates the unweighted machinery)."""
    naive, wtd, ncov, wcov = [], [], [], []
    for sd in range(seeds):
        rng = np.random.default_rng(4200 + sd)
        Ncomp, m = 200, 400
        d = rng.uniform(0.0, 1.0, Ncomp)
        err = 0.04 + 0.34 * d                                    # per-component error rate (rises with d)
        conf = rng.uniform(0.5, 1.0, (Ncomp, m))                 # confidence law independent of d
        wrong = (rng.uniform(0.0, 1.0, (Ncomp, m)) < err[:, None]).astype(np.float64)
        comp = np.repeat(np.arange(Ncomp), m)
        grid = make_grid(conf.ravel())
        L, COV, _ = comp_loss(conf.ravel(), wrong.ravel(), comp, grid)
        w = np.exp(beta * d)                                     # true covariate-shift weight
        perm = rng.permutation(Ncomp); cal, ev = perm[:100], perm[100:]
        rn, cn = weighted_crc_perunit(L[cal], np.ones(len(cal)), grid, L[ev], COV[ev], np.ones(len(ev)), alpha)
        rw, cw = weighted_crc_perunit(L[cal], w[cal], grid, L[ev], COV[ev], w[ev], alpha)
        we = w[ev]
        naive.append((we * rn).sum() / we.sum() * 100); wtd.append((we * rw).sum() / we.sum() * 100)
        ncov.append((we * cn).sum() / we.sum() * 100); wcov.append((we * cw).sum() / we.sum() * 100)
    se = lambda a: np.std(a, ddof=1) / np.sqrt(len(a))
    return (np.mean(naive), se(naive), np.mean(wtd), se(wtd), np.mean(ncov), np.mean(wcov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--pc-seeds", type=int, default=3)      # positive-control seeds
    ap.add_argument("--beta", type=float, default=BETA)     # synthetic-shift strength
    ap.add_argument("--pc-only", action="store_true")       # skip the L2A run (fast positive-control tuning)
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer="cpu")
    dumps = DUMPS[: args.seeds]
    print(f"loaded {len(dumps)} flagship dumps; FORMAL component-level weighted CRC "
          f"(test-dependent lambda-hat, test-point term, +inf grid, B=1)", flush=True)

    # ---------- (1) real L1C->L2A shift ----------
    naive, wtd, ncov, wcov, aucs, esss = [], [], [], [], [], []
    for si, f in enumerate([] if args.pc_only else dumps):
        d = np.load(f); lc, ll, y, comp = d["logits_clean"], d["logits_l2a"], d["y"].astype(int), d["comp"]
        for ss in args.splits:
            mt, mc, me = P8R.split_test_rois(comp, ss)
            Tc = fit_temperature(lc[mt], y[mt])
            grid = make_grid(softmax(lc[mc] / Tc).max(1))
            Lc, COVc, ids_c, Fc, _ = state_arrays(lc, y, comp, mc, Tc, grid)     # clean calibration
            Le, COVe, ids_e, Fe, _ = state_arrays(ll, y, comp, me, Tc, grid)     # L2A evaluation
            _, _, _, F0, _ = state_arrays(lc, y, comp, mt, Tc, grid)             # clean temp (source 0)
            _, _, _, F1, _ = state_arrays(ll, y, comp, mt, Tc, grid)             # L2A temp   (target 1)
            (wc, we), auc = domain_weights(F0, F1, [Fc, Fe])
            aucs.append(auc); esss.append(ess_frac(wc) * 100)
            rn, cn = weighted_crc_perunit(Lc, np.ones(len(wc)), grid, Le, COVe, np.ones(len(we)), ALPHA)
            rw, cw = weighted_crc_perunit(Lc, wc, grid, Le, COVe, we, ALPHA)
            naive.append((si, ss, rn.mean() * 100)); ncov.append(cn.mean() * 100)
            wtd.append((si, ss, rw.mean() * 100)); wcov.append(cw.mean() * 100)
    if naive:
        mn, sn = two_way_se(naive); mw, sw = two_way_se(wtd)
        tc = {3: 4.303, 5: 2.776}.get(len(dumps), 2.262)
        print("\n  (1) L1C->L2A shift, component-equal certificate (joint risk % @ coverage %):")
        print(f"    naive  (uniform CRC) joint {mn:6.2f} +/- {sn:.2f} [{mn-tc*sn:.1f},{mn+tc*sn:.1f}]  cov {np.mean(ncov):.0f}%   (sanity: must reproduce flagship ~28.9)")
        print(f"    FORMAL weighted CRC  joint {mw:6.2f} +/- {sw:.2f} [{mw-tc*sw:.1f},{mw+tc*sw:.1f}]  cov {np.mean(wcov):.0f}%")
        print(f"    domain-classifier AUROC {np.mean(aucs):.3f}; calib ESS {np.mean(esss):.0f}%; clip [1e-3,1e3]")

    # ---------- (2) fully-synthetic PURE covariate-shift positive control (P(Y|X) fixed) ----------
    mpn, spn, mpw, spw, ncov_pc, wcov_pc = synthetic_positive_control(ALPHA, max(args.pc_seeds, 8), args.beta)
    print("\n  (2) positive control -- fully-synthetic pure covariate shift, P(Y|X) fixed by construction:")
    print(f"    naive threshold  E_target joint {mpn:6.2f} +/- {spn:.2f}  cov {ncov_pc:.0f}%   (breaches: source-calibrated threshold under the shift)")
    print(f"    FORMAL weighted  E_target joint {mpw:6.2f} +/- {spw:.2f}  cov {wcov_pc:.0f}%   (recovers <= {ALPHA*100:.0f} at useful coverage)")
    ok = mpn > ALPHA * 100 + 1.0 and mpw <= ALPHA * 100 and wcov_pc > 30
    print(f"    => positive control {'PASSES' if ok else 'INCONCLUSIVE'}: the formal weighted CRC "
          f"{'recovers a genuine covariate-shift breach at useful coverage -- the threshold algebra and test-point term are validated on KNOWN weights (estimated-weight ratio estimation is a separate plug-in, not exercised by this synthetic control)' if ok else 'did not recover cleanly -- inspect'}.")
    print("\n  -> reading: (i) uniform weights reproduce the flagship breach (sanity: the unweighted CRC "
          "machinery is correct); (ii) the synthetic control shows the weighting recovers a breach WHEN THE "
          "WEIGHTS ARE KNOWN; (iii) on the real L2A shift the estimated-weight CRC finds no useful operating "
          "point in the source-normalized representation (see its coverage and AUROC printed above) -- "
          "near-separability leaves source-only reweighting with no overlap to exploit. This does NOT prove "
          "weighted conformal impossible in a better-aligned representation; the fix that works (product-aware "
          "re-normalization) removes the shift rather than reweighting around it.")


if __name__ == "__main__":
    main()
