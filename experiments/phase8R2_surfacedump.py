#!/usr/bin/env python
"""Dump clean-L1C logits for the surface deployment axis (train + calibrate on DARK surfaces, deploy on
unseen BRIGHT surfaces), one file per model seed, so the surface scene-component bootstrap (reviewer 3.3,
the near-boundary breach) runs offline without retraining. We reuse phase8R2_landcover's exact data flow:
train on DARK train pixels, normalise with ALL-train statistics, evaluate on the TEST set, and record the
bright/dark (target/source) membership so the offline bootstrap can resample the bright evaluation units.
"""
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
import phase8R2_landcover_reliability as LC
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim import hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--axis", default="surface", choices=["surface", "geography"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    if args.out is None:
        args.out = f"scenedump_{args.axis}"
    hw.setup(deterministic=True, prefer=args.device)

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)

    # train (DARK source only); normalise with ALL-train statistics, exactly as phase8R2_landcover
    Xtr, ytr, ptr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345,
                                  return_patch_id=True)
    is_tgt_tr = LC._target_mask(args.axis, "train", ptr)
    Xtr_src, ytr_src = Xtr[~is_tgt_tr], ytr[~is_tgt_tr]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr_n = norm(Xtr_src)

    # test (all L1C); bright = TARGET (unseen deploy), dark = SOURCE (naive-calibrate)
    X, y, pid = P8.load_split("test", "L1C", pixels_per_patch=400, seed=54321, return_patch_id=True)
    is_target = LC._target_mask(args.axis, "test", pid)
    comp = np.unique(P8R.scene_component_ids("test")[pid], return_inverse=True)[1].astype(np.int32)
    Xn = norm(X)
    bs = P2.auto_bs(Xtr_n.shape[0])

    outdir = P8R.P(args.out)
    os.makedirs(outdir, exist_ok=True)
    for seed in args.seeds:
        hw.seed_model(seed)                                          # P0-2: seed before the constructor
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr_src, groups, seed, epochs=args.epochs, bs=bs)
        lg = P8R.logits_at("proposed", m, Xn, groups, [])       # clean L1C, no band drop
        np.savez_compressed(os.path.join(outdir, f"seed{seed}.npz"),
                            logits=lg.astype(np.float32), y=y.astype(np.int16), comp=comp,
                            is_target=is_target.astype(bool))
        ab = float((lg[is_target].argmax(1) == y[is_target]).mean())
        ad = float((lg[~is_target].argmax(1) == y[~is_target]).mean())
        print(f"  dumped seed {seed}: bright(target) acc {ab*100:.1f}, dark(source) acc {ad*100:.1f}, "
              f"{int(np.unique(comp[is_target]).size)} bright units / {int(np.unique(comp[~is_target]).size)} dark",
              flush=True)
    print(f"wrote {len(args.seeds)} surface dumps to {outdir}")


if __name__ == "__main__":
    main()
