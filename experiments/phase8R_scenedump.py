#!/usr/bin/env python
"""Dump per-pixel logits + labels + scene-component ids for the flagship (clean L1C and L2A), one file
per model seed, so a scene-component OUTER bootstrap and the AURC/AUGRC/2x2 analyses can be run offline
without retraining. The model is fixed per seed; only the calibration/temperature/evaluation partition
is resampled offline, which is exactly the scene-reuse dependency reviewer 3.3 asks us to propagate.

Run: CUDA_VISIBLE_DEVICES=0 python phase8R_scenedump.py --seeds 0 1 2 3 4 5 6 7 8 9 --out scenedump_flagship
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim import hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="scenedump_flagship")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    drop = {"clean": [], "L2A": [g_b10]}

    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leaked = meta.index[meta["s2_id"].isin(train_prod)].to_numpy()
    ids = np.setdiff1d(np.arange(len(meta)), leaked)

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=(prd == "L1C"))
    X_l1c, y_te, pid = load("L1C")
    X_l2a, _ = load("L2A")
    # scene-component id per pixel -> integer codes (the ids are strings like 'ROI_0001', which cannot
    # be cast to int; np.unique(...return_inverse) gives a stable 0..K-1 code the offline bootstrap groups on)
    comp = np.unique(P8R.scene_component_ids("test")[pid], return_inverse=True)[1].astype(np.int32)

    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr_n = norm(Xtr)
    Xc = {"clean": norm(X_l1c), "L2A": norm(X_l2a)}
    bs = P2.auto_bs(Xtr_n.shape[0])

    outdir = P8R.P(args.out)
    os.makedirs(outdir, exist_ok=True)
    for seed in args.seeds:
        hw.seed_model(seed)                                          # P0-2: seed before the constructor
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)
        lc = P8R.logits_at("proposed", m, Xc["clean"], groups, drop["clean"])
        ll = P8R.logits_at("proposed", m, Xc["L2A"], groups, drop["L2A"])
        np.savez_compressed(os.path.join(outdir, f"seed{seed}.npz"),
                            logits_clean=lc.astype(np.float32), logits_l2a=ll.astype(np.float32),
                            y=y_te.astype(np.int16), comp=comp.astype(np.int32))
        acc_c = float((lc.argmax(1) == y_te).mean())
        acc_l = float((ll.argmax(1) == y_te).mean())
        print(f"  dumped seed {seed}: clean acc {acc_c*100:.1f}, L2A acc {acc_l*100:.1f}, "
              f"{len(np.unique(comp))} components, {len(y_te)} px", flush=True)
    print(f"wrote {len(args.seeds)} seed dumps to {outdir}")


if __name__ == "__main__":
    main()
