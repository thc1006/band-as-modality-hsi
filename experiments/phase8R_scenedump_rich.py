#!/usr/bin/env python
"""RICH flagship dump: bank the RAW reflectance inputs (once) + per-seed clean/L2A logits, for BOTH the
test and val splits, so every future reliability analysis -- re-normalization contracts, self-trained vs
frozen calibration (the R19 gap), weighted-CRC, AURC/AUGRC, scene-component bootstrap -- runs OFFLINE on
CPU without ever retraining the 2M-pixel flagship again.

Superset of phase8R_scenedump.py (which dumps test logits only). The RAW per-band reflectance is the
reusable part: the frozen model is fixed per seed, so any input re-normalization / product-aware contract
can be re-evaluated offline by re-standardizing these arrays and pushing them back through the saved model.
We therefore ALSO save each seed's model state_dict.

Run: python phase8R_scenedump_rich.py --seeds 0 1 2 3 4 5 6 7 8 9 --val --out scenedump_rich
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import torch

import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim import hw


def _split_inputs(split, ids_seed=54321):
    """Raw (pre-normalization) L1C + L2A reflectance, labels, and scene-component codes for one split,
    with the train-product leak-guard applied (patches whose s2_id appears in train are dropped)."""
    meta = pd.read_csv(os.path.join(P8.DATA, split, "metadata.csv"))
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leaked = meta.index[meta["s2_id"].isin(train_prod)].to_numpy()
    ids = np.setdiff1d(np.arange(len(meta)), leaked)

    def load(prd):
        return P8.load_split(split, prd, pixels_per_patch=400, patch_ids=ids, seed=ids_seed,
                             return_patch_id=(prd == "L1C"))
    X_l1c, y, pid = load("L1C")
    X_l2a, _ = load("L2A")
    comp = np.unique(P8R.scene_component_ids(split)[pid], return_inverse=True)[1].astype(np.int32)
    return dict(X_l1c=X_l1c.astype(np.float32), X_l2a=X_l2a.astype(np.float32),
                y=y.astype(np.int16), comp=comp, n_leaked=int(len(leaked)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val", action="store_true", help="also dump the val split")
    ap.add_argument("--out", default="scenedump_rich")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    drop = {"clean": [], "L2A": [g_b10]}                                  # L2A has no cirrus band B10

    outdir = P8R.P(args.out)
    os.makedirs(outdir, exist_ok=True)
    splits = ["test"] + (["val"] if args.val else [])
    banks = {}
    for sp in splits:
        d = _split_inputs(sp)
        np.savez_compressed(os.path.join(outdir, f"inputs_{sp}.npz"),
                            X_l1c=d["X_l1c"], X_l2a=d["X_l2a"], y=d["y"], comp=d["comp"])
        banks[sp] = d
        print(f"  banked RAW inputs [{sp}]: {len(d['y'])} px, {len(np.unique(d['comp']))} components, "
              f"{d['n_leaked']} leaked patches dropped", flush=True)

    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr_n = norm(Xtr)
    bs = P2.auto_bs(Xtr_n.shape[0])
    np.savez_compressed(os.path.join(outdir, "norm_stats.npz"), mu=mu.astype(np.float64), sd=sd.astype(np.float64))

    for seed in args.seeds:
        hw.seed_model(seed)                                              # seed BEFORE the constructor
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)
        payload = {}
        line = [f"seed {seed}:"]
        for sp in splits:
            d = banks[sp]
            lc = P8R.logits_at("proposed", m, norm(d["X_l1c"]), groups, drop["clean"])
            ll = P8R.logits_at("proposed", m, norm(d["X_l2a"]), groups, drop["L2A"])
            payload[f"logits_clean_{sp}"] = lc.astype(np.float32)
            payload[f"logits_l2a_{sp}"] = ll.astype(np.float32)
            line.append(f"[{sp}] clean {float((lc.argmax(1)==d['y']).mean())*100:.1f} / "
                        f"L2A {float((ll.argmax(1)==d['y']).mean())*100:.1f}")
        np.savez_compressed(os.path.join(outdir, f"logits_seed{seed}.npz"), **payload)
        torch.save(m.state_dict(), os.path.join(outdir, f"model_seed{seed}.pt"))
        print("  " + " ".join(line) + " acc%", flush=True)
    print(f"wrote {len(args.seeds)} seed logit/model dumps + raw inputs to {outdir}")


if __name__ == "__main__":
    main()
