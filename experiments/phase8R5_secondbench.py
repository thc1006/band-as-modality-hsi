#!/usr/bin/env python
"""Second-benchmark flagship run: does the naive conformal certificate breach on an INDEPENDENT scene set?

Train the identical band-as-modality model on the flagship train split (same seed, same normalisation);
calibrate + evaluate the CRC on the OFFICIAL CloudSEN12 validation split (535 patches / 107 ROIs, high-
quality expert labels, disjoint from both train and test), loaded by phase8R5_val_loader into
cloudsen12_val_secondbench.npz with paired L1C and Sen2Cor L2A. Everything downstream is the flagship's:
the component-equal joint confidently-wrong risk estimand, the scene-connected-component exchangeable unit
(ROIs unioned when they share a Sentinel-2 product), the naive (clean-calibrated) vs Mondrian (state-
calibrated) arms, and the two-way cluster-robust SE over the seed x split design.

Run AFTER the loader finishes:  CUDA_VISIBLE_DEVICES=1 python phase8R5_secondbench.py --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim.reliability import fit_temperature, conformal_risk_control
from bandsim import hw

ALPHA = 0.10


def components(roi_id, s2_id):
    """ROIs unioned when they share any Sentinel-2 product -> scene-connected components, exactly as the
    flagship maps 195 test ROIs to 184 components. Returns a 0..K-1 integer code per pixel."""
    uc = np.unique(roi_id)
    parent = {r: r for r in uc}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    s2_rois = defaultdict(set)
    for r, s in zip(roi_id, s2_id):
        s2_rois[s].add(r)
    for rs in s2_rois.values():
        rs = list(rs)
        for r in rs[1:]:
            parent[find(r)] = find(rs[0])
    root = {r: find(r) for r in uc}
    code = {c: i for i, c in enumerate(sorted(set(root.values())))}
    return np.array([code[root[r]] for r in roi_id], np.int32)


def split3(comp, ss):
    """Deterministic 25%/37.5%/37.5% temperature/calibration/evaluation split of the components, the same
    proportions as the flagship's 46/69/69 of 184."""
    uc = np.unique(comp)
    perm = np.random.default_rng(1000 + ss).permutation(uc)
    n = len(uc)
    a = int(round(n * 0.25))
    b = a + int(round(n * 0.375))
    return np.isin(comp, perm[:a]), np.isin(comp, perm[a:b]), np.isin(comp, perm[b:])


def crc_thr(logits, T, y, mask, comp):
    p = softmax(logits[mask] / T, axis=1)
    corr = (p.argmax(1) == y[mask])
    return conformal_risk_control(corr, p.max(1), corr, p.max(1), alpha=ALPHA,
                                  calib_group=comp[mask], eval_group=comp[mask])["threshold"]


def ce_joint(logits, T, y, mask, comp, thr):
    p = softmax(logits[mask] / T, axis=1)
    aw = (p.max(1) >= thr) & (p.argmax(1) != y[mask])
    ce = comp[mask]
    return float(np.mean([aw[ce == c].mean() for c in np.unique(ce)])) * 100


def cover(logits, T, mask, thr):
    p = softmax(logits[mask] / T, axis=1)
    return float((p.max(1) >= thr).mean()) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val", default="data/cloudsen12_val_secondbench.npz")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)

    d = np.load(args.val, allow_pickle=True)
    Xl1c, Xl2a, y = d["X_l1c"], d["X_l2a"], d["y"].astype(np.int64)
    comp = components(d["roi_id"], d["s2_id"])
    print(f"val second benchmark: {len(y)} px, {len(np.unique(comp))} scene-components, "
          f"class {np.bincount(y, minlength=4)}", flush=True)

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")     # L2A drops the B10 group, as the flagship

    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)          # val normalised with TRAIN statistics
    Xtr_n = norm(Xtr)
    bs = P2.auto_bs(Xtr_n.shape[0])
    Xc, Xa = norm(Xl1c), norm(Xl2a)

    rows = []
    for seed in args.seeds:
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)
        lc = P8R.logits_at("proposed", m, Xc, groups, [])        # clean L1C
        ll = P8R.logits_at("proposed", m, Xa, groups, [g_b10])   # real L2A, B10 group dropped
        for ss in range(10):
            mt, mc, me = split3(comp, ss)
            Tc = fit_temperature(lc[mt], y[mt])                  # clean (source) temperature
            Tl = fit_temperature(ll[mt], y[mt])                  # L2A (target) temperature
            thr_n = crc_thr(lc, Tc, y, mc, comp)                 # naive: calibrate on clean
            thr_m = crc_thr(ll, Tl, y, mc, comp)                 # Mondrian: calibrate on L2A
            rows.append(dict(seed=seed, split=ss, state="clean", arm="naive",
                             joint=ce_joint(lc, Tc, y, me, comp, thr_n), cov=cover(lc, Tc, me, thr_n)))
            rows.append(dict(seed=seed, split=ss, state="L2A", arm="naive",
                             joint=ce_joint(ll, Tc, y, me, comp, thr_n), cov=cover(ll, Tc, me, thr_n)))
            rows.append(dict(seed=seed, split=ss, state="L2A", arm="mondrian",
                             joint=ce_joint(ll, Tl, y, me, comp, thr_m), cov=cover(ll, Tl, me, thr_m)))
        print(f"  seed {seed}: clean acc {(lc.argmax(1) == y).mean() * 100:.1f}, "
              f"L2A acc {(ll.argmax(1) == y).mean() * 100:.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("paper/results_phase8R5_secondbench_raw.csv", index=False)
    t = 2.262 if len(args.seeds) >= 10 else (4.303 if len(args.seeds) >= 3 else 12.71)
    print(f"\n=== Second benchmark: CloudSEN12 validation split, {len(np.unique(comp))} independent "
          f"scene-components ===")
    for (st, arm), g in df.groupby(["state", "arm"]):
        # bracket access throughout: the column is named 'cov', which collides with DataFrame.cov() under
        # attribute access (g.cov returns the covariance METHOD, not the column) -- the same trap for any
        # column sharing a DataFrame method name
        mm, se = two_way_se(list(zip(g["seed"], g["split"], g["joint"])))
        flag = "breach" if mm - t * se > 10 else ("no clear breach" if mm + t * se > 10 else "at/below target")
        print(f"  {st:6s} {arm:9s} joint={mm:5.2f} +/- {se:.2f}  t-CI[{mm - t * se:5.2f},{mm + t * se:5.2f}] "
              f"cov={g['cov'].mean():4.0f}%  ({flag})")


if __name__ == "__main__":
    main()
