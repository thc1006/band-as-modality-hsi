#!/usr/bin/env python
"""R6-E3 (round-6 review 3.4 / C11): the paper contrasts a surface breach (11.35%) with a geographic
non-breach (9.79%), but 'one significant + one not' is NOT the same as 'the two differ'. We directly
estimate R_surface - R_geography and its interval. The two axes are SEPARATE experiments (different source
strata -> different models), so this is an UNPAIRED difference: two-way cluster-robust SEs combined in
quadrature, plus a model-seed bootstrap resampling each axis independently. Offline from the raw per-run
CSVs (10 model seeds x 10 split seeds, proposed/naive, crc_group_joint_risk)."""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from phase8R_perclass_weighting_agg import two_way_se

T9 = 2.262


def load(tag):
    df = pd.read_csv(os.path.join(_HERE, "..", "paper", f"results_phase8R2_{tag}_raw_10seed.csv"))
    d = df[(df["method"] == "proposed") & (df["arm"] == "naive")]
    return d[["model_seed", "split_seed", "crc_group_joint_risk"]].to_numpy(float)


def main():
    surf = load("landcover")     # bright-surface deployment (the 'surface gap')
    geo = load("geography")      # Southern-hemisphere deployment
    print(f"surface {len(surf)} runs, geography {len(geo)} runs (proposed/naive)")
    ms, ss = two_way_se([(int(a), int(b), c) for a, b, c in surf])
    mg, sg = two_way_se([(int(a), int(b), c) for a, b, c in geo])
    print(f"  surface naive joint    {ms:.2f} +/- {ss:.2f}  (t9 [{ms - T9 * ss:.1f}, {ms + T9 * ss:.1f}])")
    print(f"  geography naive joint  {mg:.2f} +/- {sg:.2f}  (t9 [{mg - T9 * sg:.1f}, {mg + T9 * sg:.1f}])")

    diff = ms - mg
    se_diff = np.hypot(ss, sg)                                       # independent axes -> quadrature
    print(f"\n  DIFFERENCE R_surface - R_geography = {diff:+.2f} +/- {se_diff:.2f}  "
          f"(quadrature t9 CI [{diff - T9 * se_diff:+.2f}, {diff + T9 * se_diff:+.2f}])")

    # model-seed bootstrap, resampling each axis independently (unpaired)
    rng = np.random.default_rng(20260725)
    sseeds, gseeds = np.unique(surf[:, 0]), np.unique(geo[:, 0])
    boot = []
    for _ in range(5000):
        sp = np.concatenate([surf[surf[:, 0] == s, 2] for s in rng.choice(sseeds, len(sseeds), replace=True)])
        gp = np.concatenate([geo[geo[:, 0] == s, 2] for s in rng.choice(gseeds, len(gseeds), replace=True)])
        boot.append(sp.mean() - gp.mean())
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    print(f"  model-seed bootstrap 95% CI [{blo:+.2f}, {bhi:+.2f}]")

    sig = blo > 0 or bhi < 0
    print(f"\n  -> surface vs geography: {'genuinely DIFFERENT (difference CI excludes 0)' if sig else 'NOT distinguishable (difference CI includes 0)'}; "
          f"surface point {ms:.1f} (>10, suggestive excess), geography {mg:.1f} (near 10, no clear breach). "
          f"State the contrast via this difference, not two separate significance verdicts.")


if __name__ == "__main__":
    main()
