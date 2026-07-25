#!/usr/bin/env python
"""E5 (round-5 review 3.5): NESTED scene-component bootstrap of the SURFACE breach. Unlike the fixed-
operating-point bootstrap (which only resampled evaluation components), this resamples the source (dark)
AND target (bright) components with replacement AND re-runs the whole calibration pipeline per draw --
re-split temp/calib, re-fit temperature, re-compute the CRC threshold -- so the interval propagates
CALIBRATION uncertainty, exactly the reviewer's concern. Only the surface (near the 10% boundary) needs
this; the flagship is far from 10. Offline from scenedump_surface."""
import glob
import os
import sys

import numpy as np
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.reliability import fit_temperature, conformal_risk_control

DUMPS = sorted(glob.glob(os.path.join(_HERE, "..", "paper", "scenedump_surface", "*.npz")))
ALPHA = 0.10
B = 1500
rng = np.random.default_rng(20260724)


def gather(ids, cidx, lg, y):
    """Pixels of the (possibly duplicated) component ids via a precomputed {component: pixel-indices} map
    (avoids an O(pixels) comp==c scan per component); each drawn instance gets a fresh unique group id so
    bootstrap-duplicated components count as separate exchangeable units. Identical output to a comp==c
    gather, just far cheaper inside the 1500-draw loop."""
    idxs, gs = [], []
    for k, c in enumerate(ids):
        ii = cidx[c]
        idxs.append(ii); gs.append(np.full(len(ii), k))
    ii = np.concatenate(idxs)
    return lg[ii], y[ii], np.concatenate(gs)


def main():
    print(f"loaded {len(DUMPS)} surface dumps; NESTED bootstrap B={B}", flush=True)
    boot = []
    seeds = DUMPS[:3]                                        # a few model seeds; nested calibration per seed
    per = B // len(seeds)
    for si, f in enumerate(seeds):
        d = np.load(f)
        lg, y, comp, it = d["logits"], d["y"].astype(np.int64), d["comp"], d["is_target"].astype(bool)
        uc = np.unique(comp)
        frac = {c: float(it[comp == c].mean()) for c in uc}     # bright fraction of each component
        dark = np.array([c for c in uc if frac[c] == 0.0])      # PURELY-dark components (source)
        bright = np.array([c for c in uc if frac[c] == 1.0])    # PURELY-bright components (target)
        cidx = {c: np.flatnonzero(comp == c) for c in uc}       # precompute pixel indices per component ONCE
        # components mixing a dark and a bright ROI (merged via a shared Sentinel-2 product) are dropped, so
        # calibration (dark) and evaluation (bright) never share an acquisition and comp==c is single-surface
        for bi in range(per):
            perm = rng.permutation(dark)                            # split the UNIQUE dark into DISJOINT
            nt = max(2, int(round(len(dark) * 0.4)))                # temp / calib components (paper requires
            temp_c, calib_c = perm[:nt], perm[nt:]                  # temperature fit disjoint from CRC calib)
            tb = rng.choice(temp_c, len(temp_c), replace=True)      # then bootstrap each set's composition
            cb = rng.choice(calib_c, len(calib_c), replace=True)    # (temp_c and calib_c never share an
            bb = rng.choice(bright, len(bright), replace=True)      # original component, so no leak)
            xt, yt, _ = gather(tb, cidx, lg, y)
            T = fit_temperature(xt, yt)
            xc, yc, gc = gather(cb, cidx, lg, y)
            pc = softmax(xc / T, axis=1); corr = (pc.argmax(1) == yc)
            thr = conformal_risk_control(corr, pc.max(1), corr, pc.max(1), alpha=ALPHA,
                                         calib_group=gc, eval_group=gc)["threshold"]
            xe, ye, ge = gather(bb, cidx, lg, y)
            p = softmax(xe / T, axis=1); aw = (p.max(1) >= thr) & (p.argmax(1) != ye)
            boot.append(float(np.mean([aw[ge == g].mean() for g in np.unique(ge)])) * 100)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  surface NESTED bootstrap: mean {np.mean(boot):.1f}%, 95% CI [{lo:.1f}, {hi:.1f}]  "
          f"({'excludes' if lo > 10 else 'INCLUDES'} the 10% target)", flush=True)
    print(f"  (fixed-operating-point bootstrap was [9.3, 13.6])")
    print(f"  -> surface verdict: {'still a breach even under full calibration resampling' if lo > 10 else 'point above target but inferentially INCONCLUSIVE under full calibration resampling'}")


if __name__ == "__main__":
    main()
