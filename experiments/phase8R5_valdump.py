#!/usr/bin/env python
"""Dump per-seed val logits (clean L1C + real L2A) for the second benchmark, so the AURC/AUGRC, class-wise
risk-coverage, and scene-component bootstrap can be run offline on the INDEPENDENT CloudSEN12 validation
scene set -- exactly as scenedump_flagship enables those analyses on the test set. Same training recipe,
seed and normalisation as phase8R5_secondbench, so the dumped logits are the very ones its CRC scores."""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R
from phase8R5_secondbench import components
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim import hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val", default="data/cloudsen12_val_secondbench.npz")
    ap.add_argument("--out", default="scenedump_val")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)

    d = np.load(args.val, allow_pickle=True)
    Xl1c, Xl2a, y = d["X_l1c"], d["X_l2a"], d["y"].astype(np.int64)
    comp = components(d["roi_id"], d["s2_id"])
    print(f"val: {len(y)} px, {len(np.unique(comp))} scene-components", flush=True)

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr_n = norm(Xtr)
    bs = P2.auto_bs(Xtr_n.shape[0])
    Xc, Xa = norm(Xl1c), norm(Xl2a)

    outdir = P8R.P(args.out)
    os.makedirs(outdir, exist_ok=True)
    for seed in args.seeds:
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)
        lc = P8R.logits_at("proposed", m, Xc, groups, [])
        ll = P8R.logits_at("proposed", m, Xa, groups, [g_b10])
        np.savez_compressed(os.path.join(outdir, f"seed{seed}.npz"),
                            logits_clean=lc.astype(np.float32), logits_l2a=ll.astype(np.float32),
                            y=y.astype(np.int16), comp=comp.astype(np.int32))
        print(f"  dumped seed {seed}: clean acc {(lc.argmax(1) == y).mean() * 100:.1f}, "
              f"L2A acc {(ll.argmax(1) == y).mean() * 100:.1f}", flush=True)
    print(f"wrote {len(args.seeds)} val dumps to {outdir}")


if __name__ == "__main__":
    main()
