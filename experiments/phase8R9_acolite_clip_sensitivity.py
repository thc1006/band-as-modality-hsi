#!/usr/bin/env python
"""E3 (round-5 review 3.2): is the 28.6% ACOLITE breach an artefact of the reflectance CLIP [-0.1,1.6]?
ACOLITE's dark-spectrum inversion over-corrects a small fraction of bright pixels to unphysical reflectance;
phase8R3_acolite clips ALL products to a physical range before the comparison. The reviewer asks whether
that clip drives the 28.6. We re-run the ENTIRE pipeline (train + eval) under three clip regimes -- NONE
(raw, keeps every ACOLITE outlier incl. the int16 saturation at 3.2767), the paper's [-0.1,1.6], and an
aggressive [0.0,1.0] -- identically applied to the training stats and all eval states. If ACOLITE-L2A naive
joint stays ~28 across all three, the breach is not a clipping artefact. Same model, unit, split design, and
CRC as phase8R3_acolite; fewer model seeds (this is a sensitivity sweep, not the headline estimate).

Run: CUDA_VISIBLE_DEVICES=0 python phase8R9_acolite_clip_sensitivity.py --seeds 0 1 2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R
from phase8R3_acolite import ACOLITE_BANDS, overall_metrics
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim.reliability import fit_temperature, conformal_risk_control
from bandsim import hw

CLIPS = {"none(raw)": None, "paper[-0.1,1.6]": (-0.1, 1.6), "tight[0,1.0]": (0.0, 1.0)}
ALPHA = 0.10
T2 = 4.303   # Student-t 0.975 at df = min(3 seeds, 10 splits) - 1 = 2 (this is a 3-seed sweep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)

    P8.PRODUCTS["ACOLITE"] = ACOLITE_BANDS
    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b9 = P8._assert_singleton(groups, P8.B9_IDX, "B9")
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    drop = {"clean": [], "L2A": [g_b10], "ACOLITE": [g_b9, g_b10]}

    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    b4 = np.fromfile(os.path.join(P8.DATA, "test", "ACOLITE_B4.dat"), dtype="<i2").reshape(-1, 512, 512)
    has_acolite = np.array([b4[i, 1:510, 1:510].any() for i in range(len(meta))])
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leaked = meta["s2_id"].isin(train_prod).to_numpy()
    ids = np.flatnonzero(has_acolite & ~leaked)

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=(prd == "L1C"))
    X_l1c, y_te, pid = load("L1C")                              # RAW reflectance (load_split does not clip)
    X_l2a, _ = load("L2A")
    X_aco, _ = load("ACOLITE")
    comp_all = P8R.scene_component_ids("test")[pid]
    Xtr_raw, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    bs = P2.auto_bs(Xtr_raw.shape[0])
    # how many ACOLITE eval pixels does the paper's clip actually move? (context for the sweep)
    moved = float(((X_aco < -0.1) | (X_aco > 1.6)).any(1).mean()) * 100
    print(f"  eval pixels {len(y_te)} / {len(np.unique(comp_all))} components; "
          f"paper clip touches {moved:.2f}% of ACOLITE eval pixels\n", flush=True)

    summary = {}
    for cname, crange in CLIPS.items():
        if crange is None:
            clipf = lambda A: A
        else:
            clipf = lambda A, lo=crange[0], hi=crange[1]: np.clip(A, lo, hi)
        Xtr_c = clipf(Xtr_raw)
        mu, sd = Xtr_c.mean(0), Xtr_c.std(0) + 1e-8
        norm = lambda A: ((clipf(A) - mu) / sd).astype(np.float32)
        Xtr_n = ((Xtr_c - mu) / sd).astype(np.float32)
        Xc = {"clean": norm(X_l1c), "L2A": norm(X_l2a), "ACOLITE": norm(X_aco)}

        rows = {"clean": [], "L2A": [], "ACOLITE": []}
        for seed in args.seeds:
            m = GroupedCrossBandAttention(groups, cwl, 4)
            P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
            P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)

            def logits(state, mask):
                return P8R.logits_at("proposed", m, Xc[state][mask], groups, drop[state])

            for ss in args.split_seeds:
                mt, mc, me = P8R.split_test_rois(comp_all, ss)
                Tclean = fit_temperature(logits("clean", mt), y_te[mt])
                pc = softmax(logits("clean", mc) / Tclean, axis=1)
                corr_c = (pc.argmax(1) == y_te[mc])
                thr = conformal_risk_control(corr_c, pc.max(1), corr_c, pc.max(1), alpha=ALPHA,
                                             calib_group=comp_all[mc], eval_group=comp_all[mc])["threshold"]
                for st in ("clean", "L2A", "ACOLITE"):
                    p = softmax(logits(st, me) / Tclean, axis=1)      # naive: clean temperature + clean thr
                    j, _, _ = overall_metrics(p.argmax(1) == y_te[me], p.max(1), comp_all[me], thr)
                    rows[st].append((seed, ss, j))
            print(f"  [{cname:16s}] seed {seed} done", flush=True)
        summary[cname] = {st: two_way_se(rows[st]) for st in rows}
        a = summary[cname]["ACOLITE"]
        print(f"  == {cname:16s}: ACOLITE-L2A naive joint {a[0]:.2f} +/- {a[1]:.2f} "
              f"[{a[0] - T2 * a[1]:.1f},{a[0] + T2 * a[1]:.1f}]  "
              f"(clean {summary[cname]['clean'][0]:.1f}, Sen2Cor-L2A {summary[cname]['L2A'][0]:.1f})\n", flush=True)

    print("=" * 78)
    print("CLIP SENSITIVITY of the ACOLITE-L2A naive joint risk (target 10%):")
    for cname in CLIPS:
        a = summary[cname]["ACOLITE"]
        print(f"  {cname:16s}  {a[0]:5.2f} +/- {a[1]:.2f}")
    vals = [summary[c]["ACOLITE"][0] for c in CLIPS]
    print(f"  -> spread across clip regimes: {max(vals) - min(vals):.2f} points; "
          f"{'STABLE -- 28.6 is not a clipping artefact' if max(vals) - min(vals) < 4 else 'CLIP-SENSITIVE -- investigate'}")
    print("     (paper 10-seed headline under the [-0.1,1.6] clip: 29.60, from results_phase8R3_acolite10.csv)")


if __name__ == "__main__":
    main()
