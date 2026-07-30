#!/usr/bin/env python3
"""AURC on the CloudSEN12 FLAGSHIP (band-as-modality) core — does the flagship confidence stay INFORMATIVE
under the realistic L1C->L2A shift, unlike the extreme HSI-6S/EMIT shifts where band confidence went
UNINFORMATIVE (aurc_matched_coverage.py)? This checks the new AURC finding does not backwash onto the paper's
flagship.

Runs entirely OFFLINE from the banked scenedump_rich (raw reflectance + per-seed clean/L2A logits) — no
retraining. AURC is only APPROXIMATELY temperature-invariant (for K>2 a single scalar T can slightly reorder
max-softmax; a direct check found <=2 pp impact and no verdict change), so raw logits are used here. The dumped L2A logits
already use the STALE L1C-train normalization (the naive silent-failure deployment). Pixel-level.

  python experiments/aurc_cloudsen12_core.py            # reads paper/scenedump_rich/ (build it first via
                                                         # phase8R_scenedump_rich.py --seeds 0..9)
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.special import softmax
from scipy.integrate import trapezoid

_HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(_HERE, "..", "paper", "scenedump_rich")


def aurc(conf, corr):
    o = np.argsort(-conf); c = corr[o].astype(float); k = np.arange(1, len(c) + 1)
    return float(trapezoid(1.0 - np.cumsum(c) / k, k / len(c))) * 100


def sel_risk_at(conf, corr, cov):
    o = np.argsort(-conf); k = max(1, int(cov * len(conf)))
    return float(1.0 - corr[o][:k].mean()) * 100


def main():
    inp = os.path.join(DUMP, "inputs_test.npz")
    if not os.path.exists(inp):
        raise FileNotFoundError(f"no rich dump at {DUMP} -- build first: "
                                f"python experiments/phase8R_scenedump_rich.py --seeds 0 1 2 3 4 5 6 7 8 9")
    y = np.load(inp)["y"].astype(int)
    covs = [0.4, 0.6, 0.8]
    rows = []
    for f in sorted(glob.glob(os.path.join(DUMP, "logits_seed*.npz"))):
        z = np.load(f)
        pc = softmax(z["logits_clean_test"], axis=1); cc = pc.argmax(1) == y
        pl = softmax(z["logits_l2a_test"], axis=1); cl = pl.argmax(1) == y     # STALE-normalized L2A (naive)
        rows.append(dict(seed=os.path.basename(f),
                         aurc_clean=aurc(pc.max(1), cc), aurc_l2a=aurc(pl.max(1), cl),
                         err_clean=float((~cc).mean()) * 100, err_l2a=float((~cl).mean()) * 100,
                         sr_clean={f"{c}": sel_risk_at(pc.max(1), cc, c) for c in covs},
                         sr_l2a={f"{c}": sel_risk_at(pl.max(1), cl, c) for c in covs}))
    m = lambda k: float(np.mean([r[k] for r in rows]))
    al, el = m("aurc_l2a"), m("err_l2a")
    print(f"CloudSEN12 flagship (band-as-modality), {len(rows)} seeds, offline from scenedump_rich (pixel-level):")
    print(f"  AURC clean {m('aurc_clean'):.1f} (err {m('err_clean'):.0f}) -> L2A(naive/stale) {al:.1f} (err {el:.0f})")
    gap = el - al
    verdict = ("INFORMATIVE — confidence still ranks L2A errors (AURC well below error), UNLIKE the extreme "
               "HSI-6S/EMIT shifts") if gap > 5 else "UNINFORMATIVE — AURC ~ error (confidence died)"
    print(f"  => L2A confidence is {verdict}  (error - AURC = {gap:.0f} pp)")
    for c in covs:
        scc = np.mean([r["sr_clean"][f"{c}"] for r in rows]); scl = np.mean([r["sr_l2a"][f"{c}"] for r in rows])
        print(f"  matched-cov {int(c*100)}%: sel-risk clean {scc:.1f}  L2A {scl:.1f}")
    out = os.path.join(_HERE, "..", "paper", "results_aurc_cloudsen12_core.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
