#!/usr/bin/env python
"""R6-E2 (round-6 review 3.5 / C8): the paper says ACOLITE and Sen2Cor break the certificate to a
'statistically indistinguishable degree', but close point estimates (28.56 vs 28.64) are NOT an equivalence
claim. We compute the PAIRED difference R_ACOLITE - R_Sen2Cor per (seed, split) on the identical retained
subset, its two-way cluster-robust CI and a seed-cluster bootstrap CI, and a TOST equivalence test against a
pre-declared margin. Offline from results_phase8R3_acolite10.csv (10 seeds x 10 splits, naive arm)."""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from phase8R_perclass_weighting_agg import two_way_se

T9 = 2.262
MARGIN = 2.0   # pre-declared equivalence margin (percentage points), ~7% of the ~28.6% risk


def main():
    df = pd.read_csv(os.path.join(_HERE, "..", "paper", "results_phase8R3_acolite10.csv"))
    nai = df[df["arm"] == "naive"]
    piv = nai.pivot_table(index=["seed", "split"], columns="state", values="joint").dropna()
    d = (piv["ACOLITE"] - piv["L2A"]).reset_index()          # paired diff per (seed, split)
    d.columns = ["seed", "split", "diff"]
    print(f"loaded {len(d)} paired (seed,split) cells; ACOLITE vs Sen2Cor-L2A naive joint on the same subset")
    print(f"  means: ACOLITE {piv['ACOLITE'].mean():.2f}, Sen2Cor-L2A {piv['L2A'].mean():.2f}", flush=True)

    rows = list(zip(d["seed"].astype(int), d["split"].astype(int), d["diff"].astype(float)))
    m, se = two_way_se(rows)
    lo, hi = m - T9 * se, m + T9 * se
    print(f"\n  paired difference R_ACOLITE - R_Sen2Cor = {m:+.2f} +/- {se:.2f}  "
          f"(two-way t9 CI [{lo:+.2f}, {hi:+.2f}])")

    # seed-cluster bootstrap of the paired difference (resample the 10 model seeds with replacement)
    rng = np.random.default_rng(20260725)
    seeds = d["seed"].unique()
    boot = []
    for _ in range(5000):
        pick = rng.choice(seeds, len(seeds), replace=True)
        vals = np.concatenate([d[d["seed"] == s]["diff"].to_numpy() for s in pick])
        boot.append(vals.mean())
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    print(f"  seed-cluster bootstrap 95% CI [{blo:+.2f}, {bhi:+.2f}]")

    # TOST: equivalent iff the whole CI lies within (-MARGIN, +MARGIN)
    equiv_t9 = (lo > -MARGIN) and (hi < MARGIN)
    equiv_boot = (blo > -MARGIN) and (bhi < MARGIN)
    print(f"\n  TOST equivalence at margin +/-{MARGIN} pts: "
          f"t9 {'PASS' if equiv_t9 else 'NOT established'}, bootstrap {'PASS' if equiv_boot else 'NOT established'}")
    print(f"  -> {'statistically equivalent within a ' + str(MARGIN) + '-point margin' if equiv_t9 and equiv_boot else 'similar but not formally equivalent'}; "
          f"report the paired difference and CI, not 'indistinguishable'.")


if __name__ == "__main__":
    main()
