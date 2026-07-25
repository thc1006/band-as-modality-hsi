#!/usr/bin/env python
"""Second-benchmark (val) robustness: scene-component bootstrap of the naive L2A breach + the AURC ranking
degradation, offline from scenedump_val -- the same checks the flagship gets, now on the independent
CloudSEN12 validation scene set."""
import glob
import os
import sys

import numpy as np
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from phase8R5_secondbench import split3, crc_thr
from bandsim.reliability import fit_temperature

DUMPS = sorted(glob.glob(os.path.join(_HERE, "..", "paper", "scenedump_val", "*.npz")))
rng = np.random.default_rng(20260724)


def aurc(conf, wrong):
    """Area under the risk-coverage curve: selective risk (error among accepted) integrated over coverage
    as the acceptance threshold sweeps the confidences high->low."""
    order = np.argsort(-conf)
    w = wrong[order].astype(float)
    cov = np.arange(1, len(w) + 1) / len(w)
    sel = np.cumsum(w) / np.arange(1, len(w) + 1)
    return float(np.trapz(sel, cov)) * 100


def main():
    print(f"loaded {len(DUMPS)} val dumps")
    boot, point, ac, al = [], [], [], []
    for f in DUMPS:
        d = np.load(f)
        lc, ll, y, comp = d["logits_clean"], d["logits_l2a"], d["y"], d["comp"]
        mt, mc, me = split3(comp, 0)                    # a representative deployment split for the operating point
        Tc = fit_temperature(lc[mt], y[mt])
        thr = crc_thr(lc, Tc, y, mc, comp)              # naive: clean-calibrated threshold at alpha=10%
        p = softmax(ll / Tc, axis=1)
        aw = (p.max(1) >= thr) & (p.argmax(1) != y)     # confidently-wrong on L2A
        uc = np.unique(comp)
        Lc = np.array([aw[comp == c].mean() for c in uc])
        point.append(Lc.mean() * 100)
        for _ in range(4000):
            b = rng.integers(0, len(uc), len(uc))
            boot.append(Lc[b].mean() * 100)
        pc, pl = softmax(lc / Tc, axis=1), softmax(ll / Tc, axis=1)
        ac.append(aurc(pc.max(1), pc.argmax(1) != y))
        al.append(aurc(pl.max(1), pl.argmax(1) != y))

    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  val naive scene-component bootstrap: mean {np.mean(point):.1f}%, 95% CI [{lo:.1f}, {hi:.1f}] "
          f"({'excludes' if lo > 10 else 'includes'} the 10% target)")
    print(f"  val AURC (selective risk over coverage): clean {np.mean(ac):.2f} -> L2A {np.mean(al):.2f} "
          f"({'degrades' if np.mean(al) > np.mean(ac) else 'stable'}, x{np.mean(al) / np.mean(ac):.2f})")


if __name__ == "__main__":
    main()
