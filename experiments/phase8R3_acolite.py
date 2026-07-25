#!/usr/bin/env python
"""Second-processor control (reviewer: single Sen2Cor processor). Re-run the flagship CRC with a
SECOND, independent atmospheric processor -- ACOLITE (dark-spectrum fixed-AOT) -- as the L2A product,
and ask whether the source-calibrated certificate breaks under ACOLITE-L2A the same way it does under
Sen2Cor-L2A. If both independent processors break it, the breach is not a Sen2Cor artefact.

ACOLITE ships 11 bands (B1-B8, B8A, B11, B12; no B9 water-vapour, no B10 cirrus), so ACOLITE-L2A drops
band-groups g4(B9) and g5(B10); Sen2Cor-L2A drops only g5(B10). We monkey-patch PRODUCTS at runtime so
the vetted load_split reads ACOLITE_<band>.dat and places the bands at their L1C indices (no core edit),
and we evaluate on the intersection of the leak-guarded test patches with those ACOLITE processed
successfully (355 of 975 scenes failed ACOLITE and are excluded, on the SAME patches for every state).

Run: CUDA_VISIBLE_DEVICES=1 python phase8R3_acolite.py --seeds 0 1 2 3 4 --out-tag acolite
"""
import argparse
import csv
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
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim.reliability import fit_temperature, conformal_risk_control
from bandsim import hw

ACOLITE_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]  # no B9, no B10


def overall_metrics(corr, conf, comp, thr):
    acc = conf >= thr
    aw = acc & (~corr)
    uc = np.unique(comp)
    L = np.array([aw[comp == c].mean() for c in uc])
    Cov = np.array([acc[comp == c].mean() for c in uc])
    joint = float(L.mean()) * 100
    cov = float(Cov.mean()) * 100
    sel = joint / cov * 100 if cov > 0 else float("nan")
    return joint, sel, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out-tag", default="acolite")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)

    P8.PRODUCTS["ACOLITE"] = ACOLITE_BANDS                      # runtime only, no core edit

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b9 = P8._assert_singleton(groups, P8.B9_IDX, "B9")
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    drop = {"clean": [], "L2A": [g_b10], "ACOLITE": [g_b9, g_b10]}

    # which test patches did ACOLITE process (non-zero)? evaluate all states on this SAME set.
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    b4 = np.fromfile(os.path.join(P8.DATA, "test", "ACOLITE_B4.dat"), dtype="<i2").reshape(-1, 512, 512)
    has_acolite = np.array([b4[i, 1:510, 1:510].any() for i in range(len(meta))])
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leaked = meta["s2_id"].isin(train_prod).to_numpy()
    ids = np.flatnonzero(has_acolite & ~leaked)                # leak-guarded AND ACOLITE-present
    print(f"  test patches: {len(meta)} | ACOLITE-present {has_acolite.sum()} | "
          f"leak-guarded+present {len(ids)}", flush=True)

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=(prd == "L1C"))
    X_l1c, y_te, pid = load("L1C")
    X_l2a, _ = load("L2A")
    X_aco, _ = load("ACOLITE")
    comp_all = P8R.scene_component_ids("test")[pid]
    print(f"  eval pixels {len(y_te)} over {len(np.unique(comp_all))} scene-components", flush=True)

    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    # ACOLITE's dark-spectrum inversion over-corrects a small fraction of bright cloud/snow pixels to
    # unphysical reflectance (0.7% saturate the int16 store at 3.27; ~3% exceed |z|>10 once normalised).
    # Clip ALL products to a physical reflectance range so the cross-processor comparison is not driven
    # by these outliers; applied identically to L1C/L2A/ACOLITE (and the training stats) for consistency.
    clip = lambda A: np.clip(A, -0.1, 1.6)
    Xtr = clip(Xtr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr_n = norm(Xtr)
    Xc = {"clean": norm(clip(X_l1c)), "L2A": norm(clip(X_l2a)), "ACOLITE": norm(clip(X_aco))}
    bs = P2.auto_bs(Xtr_n.shape[0])

    rows = []
    for seed in args.seeds:
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)

        def logits(state, mask):
            return P8R.logits_at("proposed", m, Xc[state][mask], groups, drop[state])

        for ss in args.split_seeds:
            mt, mc, me = P8R.split_test_rois(comp_all, ss)
            T = {st: fit_temperature(logits(st, mt), y_te[mt]) for st in Xc}

            def cp(state, mask, temp):
                p = softmax(logits(state, mask) / temp, axis=1)
                return p.argmax(1), p.max(1)

            # naive threshold from clean; Mondrian threshold from the state itself
            pc, cc = cp("clean", mc, T["clean"])
            corr_c = (pc == y_te[mc])
            thr_naive = conformal_risk_control(corr_c, cc, corr_c, cc, alpha=args.alpha,
                                               calib_group=comp_all[mc], eval_group=comp_all[mc])["threshold"]
            for st in ("clean", "L2A", "ACOLITE"):
                pcs, ccs = cp(st, mc, T[st])
                corr_s = (pcs == y_te[mc])
                thr_m = conformal_risk_control(corr_s, ccs, corr_s, ccs, alpha=args.alpha,
                                               calib_group=comp_all[mc], eval_group=comp_all[mc])["threshold"]
                comp_ev = comp_all[me]; yy = y_te[me]
                for arm, thr, temp in (("naive", thr_naive, T["clean"]), ("mondrian", thr_m, T[st])):
                    pred, conf = cp(st, me, temp)
                    j, sel, cov = overall_metrics(pred == yy, conf, comp_ev, thr)
                    rows.append(dict(seed=seed, split=ss, state=st, arm=arm,
                                     joint=j, selective=sel, coverage=cov,
                                     acc=float((pred == yy).mean()) * 100))
            r = [x for x in rows if x["seed"] == seed and x["split"] == ss and x["arm"] == "naive"]
            g = {x["state"]: x["joint"] for x in r}
            print(f"  seed {seed} split {ss}: naive joint  clean={g['clean']:.1f}  "
                  f"Sen2Cor-L2A={g['L2A']:.1f}  ACOLITE-L2A={g['ACOLITE']:.1f}", flush=True)

    out = P8R.P(f"results_phase8R3_{args.out_tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
