#!/usr/bin/env python
"""E4 (round-5 review 3.3): the official CloudSEN12 validation split is RANDOM from no-test, not spatially
block-separated, so it can share Sentinel-2 products with TRAIN. Audit found 2 val products also in train.
Drop those pixels and recompute the second-benchmark breach so the replication is genuinely calibration-
disjoint from training. Offline from scenedump_val + the val npz's per-pixel s2_id."""
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from phase8R5_secondbench import components, split3, crc_thr, ce_joint
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.reliability import fit_temperature

T9 = 2.262


def main():
    d = np.load(os.path.join(_HERE, "..", "data", "cloudsen12_val_secondbench.npz"), allow_pickle=True)
    s2 = np.array([str(x) for x in d["s2_id"]])
    roi = d["roi_id"]
    tr_s2 = set(str(x) for x in pd.read_csv(os.path.join(_HERE, "..", "data", "cloudsen12", "train",
                                                         "metadata.csv"))["s2_id"].dropna())
    leaked = np.array([x in tr_s2 for x in s2])
    print(f"leaked products (val∩train): {sorted(set(s2[leaked]))}", flush=True)
    print(f"leaked pixels {leaked.sum()}/{len(leaked)} ({leaked.sum() / len(leaked) * 100:.2f}%); "
          f"ROIs affected {sorted(set(str(x) for x in roi[leaked]))}", flush=True)
    keep = ~leaked
    comp = components(roi[keep], d["s2_id"][keep])
    print(f"after drop: {keep.sum()} px, {len(np.unique(comp))} scene-components (was 104)", flush=True)

    dumps = sorted(glob.glob(os.path.join(_HERE, "..", "paper", "scenedump_val", "*.npz")))
    rn, rm = [], []
    for si, f in enumerate(dumps):
        dd = np.load(f)
        lc, ll, y = dd["logits_clean"][keep], dd["logits_l2a"][keep], dd["y"][keep].astype(np.int64)
        for ss in range(10):
            mt, mc, me = split3(comp, ss)
            Tc = fit_temperature(lc[mt], y[mt])
            Tl = fit_temperature(ll[mt], y[mt])
            rn.append((si, ss, ce_joint(ll, Tc, y, me, comp, crc_thr(lc, Tc, y, mc, comp))))
            rm.append((si, ss, ce_joint(ll, Tl, y, me, comp, crc_thr(ll, Tl, y, mc, comp))))
        print(f"  seed {si} done", flush=True)
    mn, sen = two_way_se(rn)
    mm, sem = two_way_se(rm)
    print(f"\n=== E4 fix: second benchmark with train-leaked products removed ===")
    print(f"  L2A naive    {mn:5.2f} +/- {sen:.2f}  t9 CI[{mn - T9 * sen:.1f},{mn + T9 * sen:.1f}]  "
          f"(was 29.44 [27.5,31.4])")
    print(f"  L2A mondrian {mm:5.2f} +/- {sem:.2f}  (was 7.87)")
    print(f"  -> {'breach intact (excludes 10); replication now calibration-disjoint from train' if mn - T9 * sen > 10 else 'CHECK: changed materially'}")


if __name__ == "__main__":
    main()
