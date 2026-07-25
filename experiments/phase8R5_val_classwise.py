#!/usr/bin/env python
"""Per-class decomposition of the second-benchmark (val) breach, offline from scenedump_val -- the Table 2
analysis on the independent validation scene set. Does the minority-cloud-class collapse (thin cloud,
shadow) that drives the test-split joint risk replicate on held-out scenes?"""
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
CLASSES = ["clear", "thick", "thin", "shadow"]


def iou(pred, y, c):
    tp = ((pred == c) & (y == c)).sum()
    fp = ((pred == c) & (y != c)).sum()
    fn = ((pred != c) & (y == c)).sum()
    return tp / (tp + fp + fn + 1e-9) * 100


def main():
    print(f"loaded {len(DUMPS)} val dumps")
    iou_c = [[] for _ in range(4)]
    iou_l = [[] for _ in range(4)]
    sel = [[] for _ in range(4)]
    supp = [[] for _ in range(4)]
    miou_c, miou_l = [], []
    for f in DUMPS:
        d = np.load(f)
        lc, ll, y, comp = d["logits_clean"], d["logits_l2a"], d["y"], d["comp"]
        pc, pl = lc.argmax(1), ll.argmax(1)
        mt, mc, me = split3(comp, 0)
        Tc = fit_temperature(lc[mt], y[mt])
        thr = crc_thr(lc, Tc, y, mc, comp)                       # clean-calibrated operating point
        acc = softmax(ll / Tc, axis=1).max(1) >= thr             # accepted pixels on L2A
        ic, il = [], []
        for c in range(4):
            ic.append(iou(pc, y, c)); il.append(iou(pl, y, c))
            iou_c[c].append(ic[-1]); iou_l[c].append(il[-1])
            m = (y == c) & acc                                   # accepted pixels whose TRUE class is c
            sel[c].append((pl[m] != y[m]).mean() * 100 if m.sum() else np.nan)
            supp[c].append((y == c).mean() * 100)
        miou_c.append(np.mean(ic)); miou_l.append(np.mean(il))

    print(f"  mean IoU: clean {np.mean(miou_c):.1f} -> L2A {np.mean(miou_l):.1f}")
    print(f"  {'class':8s} {'IoU clean->L2A':>16s} {'L2A sel-risk':>13s} {'support':>8s}")
    for c in range(4):
        print(f"  {CLASSES[c]:8s} {np.mean(iou_c[c]):6.1f} -> {np.mean(iou_l[c]):5.1f}   "
              f"{np.nanmean(sel[c]):11.1f}%  {np.mean(supp[c]):6.0f}%")


if __name__ == "__main__":
    main()
