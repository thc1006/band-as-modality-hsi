#!/usr/bin/env python
"""Scene-component bootstrap of the SURFACE naive breach (reviewer 3.3, the near-boundary result).
The naive arm calibrates its threshold on the DARK (source) surface units and is deployed on the unseen
BRIGHT (target) surface units; we resample the bright evaluation components with replacement and recompute
the mean within-component confidently-wrong loss, to check the 11.4% surface breach against
scene-composition uncertainty. Runs offline from the per-seed surface logit dumps (no retraining)."""
import glob
import os
import sys

import numpy as np
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8R2_landcover_reliability as LC
from bandsim.reliability import fit_temperature, conformal_risk_control

DUMPS = sorted(glob.glob(os.path.join(_HERE, "..", "paper", (sys.argv[1] if len(sys.argv)>1 else "scenedump_surface"), "*.npz")))
ALPHA = 0.10
rng = np.random.default_rng(20260723)


def main():
    print(f"loaded {len(DUMPS)} surface dumps")
    boot, point = [], []
    for f in DUMPS:
        d = np.load(f)
        lg, y, comp, is_tgt = d["logits"], d["y"], d["comp"], d["is_target"].astype(bool)
        bright_u = np.unique(comp[is_tgt])                      # target (deployed) exchangeable units
        # dark (source) two-way temp/calib split, seed 0, matching phase8R2_landcover
        sp = LC._split_units(comp[~is_tgt], 0, salt=2, n_parts=2)
        if sp is None:
            print(f"  {os.path.basename(f)}: too few dark units -- skip"); continue
        temp_d, calib_d = sp
        mt = np.isin(comp, temp_d) & ~is_tgt
        mc = np.isin(comp, calib_d) & ~is_tgt
        T = fit_temperature(lg[mt], y[mt])                      # source (dark) temperature
        pc = softmax(lg[mc] / T, axis=1)
        corr_c = (pc.argmax(1) == y[mc])
        thr = conformal_risk_control(corr_c, pc.max(1), corr_c, pc.max(1), alpha=ALPHA,
                                     calib_group=comp[mc], eval_group=comp[mc])["threshold"]
        # per bright-component confidently-wrong loss at the dark-calibrated operating point
        p = softmax(lg / T, axis=1)
        aw = (p.max(1) >= thr) & (p.argmax(1) != y)
        Lc = np.array([aw[comp == c].mean() for c in bright_u])
        point.append(Lc.mean() * 100)
        for _ in range(4000):
            b = rng.integers(0, len(bright_u), len(bright_u))
            boot.append(Lc[b].mean() * 100)
        print(f"  {os.path.basename(f)}: {len(bright_u)} bright units, naive joint {Lc.mean()*100:.1f}%",
              flush=True)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nsurface naive scene-component bootstrap: point mean {np.mean(point):.1f}%, "
          f"95% CI [{lo:.1f}, {hi:.1f}]  ({'excludes' if lo > 10 else 'includes'} the 10% target)")


if __name__ == "__main__":
    main()
