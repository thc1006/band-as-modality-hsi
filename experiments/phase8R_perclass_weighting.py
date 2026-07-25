#!/usr/bin/env python
"""Per-class breakdown (reviewer S3.5) AND same-weighting comparison (reviewer S3.1), across the
full 10 split x 10 seed design, from ONE set of trained models (train once, compute both).

Reviewer S3.5 -- per-class over runs. The headline pixel accuracy hides a per-class collapse on the
imbalanced 4-class task; we report class-wise IoU/F1/PA/UA and class-wise CRC joint/selective/coverage
across all runs (mean + two-way cluster-robust SE), not just one descriptive seed.

Reviewer S3.1 -- same-weighting. The certified reliability triple is component-equal-weighted while the
segmentation accuracy is pixel-level, so a reader cannot line them up. Here, at the SAME CRC operating
point, we report the joint/coverage/selective triple AND the accuracy under BOTH weightings:
  * component-equal (each scene-connected component weighted equally -- what CRC certifies), and
  * pixel-pooled (sample-weighted -- what a per-pixel accuracy uses),
so the headline can be read on one footing. The CRC threshold is selected exactly as the flagship
selects it (conformal_risk_control); we only EVALUATE the two weightings at that fixed threshold, so no
conformal machinery is re-implemented.

Run as two instances to use both GPUs, e.g.
  CUDA_VISIBLE_DEVICES=0 python phase8R_perclass_weighting.py --seeds 0 1 2 3 4 --out-tag gpu0
  CUDA_VISIBLE_DEVICES=1 python phase8R_perclass_weighting.py --seeds 5 6 7 8 9 --out-tag gpu1
then concatenate the per-cell CSVs and aggregate.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R
from phase8R_classwise import perclass_seg, perclass_risk, CLASSES, NC
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim.reliability import fit_temperature, conformal_risk_control
from bandsim import hw


def weighting_metrics(corr, conf, comp, thr):
    """The joint/coverage/selective triple + accuracy at a FIXED threshold `thr`, under two weightings.
    corr = (pred == y) per eval pixel; conf = max-softmax per pixel; comp = component id per pixel.
    Returns component-equal (*_ce) and pixel-pooled (*_pp) values, all in %.
    Identity within a weighting: joint = selective * coverage (we form selective = joint / coverage)."""
    acc = conf >= thr                      # accepted
    wrong = ~corr
    aw = acc & wrong                       # accepted AND wrong
    # pixel-pooled (sample-weighted): pool every pixel equally
    cov_pp = float(acc.mean())
    joint_pp = float(aw.mean())
    sel_pp = joint_pp / cov_pp if cov_pp > 0 else float("nan")
    acc_pixel = float(corr.mean())
    # component-equal: each component contributes its within-component fraction with equal weight
    uc = np.unique(comp)
    L = np.array([aw[comp == c].mean() for c in uc])           # within-component confidently-wrong
    Cov = np.array([acc[comp == c].mean() for c in uc])        # within-component coverage
    A = np.array([corr[comp == c].mean() for c in uc])         # within-component accuracy
    cov_ce = float(Cov.mean())
    joint_ce = float(L.mean())
    sel_ce = joint_ce / cov_ce if cov_ce > 0 else float("nan")
    acc_ce = float(A.mean())
    return dict(joint_ce=joint_ce * 100, cov_ce=cov_ce * 100, sel_ce=sel_ce * 100, acc_ce=acc_ce * 100,
                joint_pp=joint_pp * 100, cov_pp=cov_pp * 100, sel_pp=sel_pp * 100,
                acc_pixel=acc_pixel * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--smoke", action="store_true", help="1 seed x 1 split, sanity only")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.split_seeds = args.seeds[:1], args.split_seeds[:1]
    hw.setup(deterministic=True, prefer=args.device)

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    drop = {"clean": [], "L2A": [g_b10]}                       # L2A: B10 absent

    # test L1C + L2A, leak-guarded (mirror phase8R/classwise), fixed pixel sampling
    import pandas as pd
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leaked = meta.index[meta["s2_id"].isin(train_prod)].to_numpy()
    ids = np.setdiff1d(np.arange(len(meta)), leaked)

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=(prd == "L1C"))
    X_l1c, y_te, pid = load("L1C")
    X_l2a, _ = load("L2A")
    comp_all = P8R.scene_component_ids("test")[pid]            # exchangeable unit per pixel

    # train (source L1C) is seed-INDEPENDENT (fixed subsample seed 12345) -- load & normalise ONCE,
    # not once per model seed, so the 10-seed run does not re-read the training patches ten times.
    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr_n = norm(Xtr)
    Xc = {"clean": norm(X_l1c), "L2A": norm(X_l2a)}
    bs = P2.auto_bs(Xtr_n.shape[0])

    rows = []
    for seed in args.seeds:
        m = GroupedCrossBandAttention(groups, cwl, NC)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)

        def logits(state, mask):
            return P8R.logits_at("proposed", m, Xc[state][mask], groups, drop[state])

        for ss in args.split_seeds:
            mt, mc, me = P8R.split_test_rois(comp_all, ss)     # temp / calib / eval masks
            # per-state temperature (Mondrian uses its own; naive uses clean's)
            T = {st: fit_temperature(logits(st, mt), y_te[mt]) for st in ("clean", "L2A")}

            def cp(state, mask, temp):
                p = softmax(logits(state, mask) / temp, axis=1)
                return p.argmax(1), p.max(1)

            # thresholds: naive = clean-calibrated; Mondrian = per-state-calibrated
            pc_cl, cc_cl = cp("clean", mc, T["clean"])
            corr_cl = (pc_cl == y_te[mc])
            thr_naive = conformal_risk_control(corr_cl, cc_cl, corr_cl, cc_cl, alpha=args.alpha,
                                               calib_group=comp_all[mc], eval_group=comp_all[mc])["threshold"]
            thr_mond = {}
            for st in ("clean", "L2A"):
                pcs, ccs = cp(st, mc, T[st])                    # state-own temperature on the calib split
                corr_s = (pcs == y_te[mc])
                thr_mond[st] = conformal_risk_control(corr_s, ccs, corr_s, ccs, alpha=args.alpha,
                                                      calib_group=comp_all[mc],
                                                      eval_group=comp_all[mc])["threshold"]

            comp_ev = comp_all[me]
            yy = y_te[me]
            for st in ("clean", "L2A"):
                for arm in ("naive", "mondrian"):
                    temp = T["clean"] if arm == "naive" else T[st]
                    thr = thr_naive if arm == "naive" else thr_mond[st]
                    pred, conf = cp(st, me, temp)
                    corr = (pred == yy)
                    w = weighting_metrics(corr, conf, comp_ev, thr)
                    seg = perclass_seg(yy, pred)
                    rk = perclass_risk(yy, pred, conf, thr)
                    base = dict(seed=seed, split=ss, state=st, arm=arm, thr=float(thr),
                                oa=float((pred == yy).mean()) * 100,
                                miou=float(np.nanmean([seg[c]["iou"] for c in range(NC)])) * 100,
                                **w)
                    ov = dict(base, scope="overall", cls="ALL")
                    rows.append(ov)
                    if st == "L2A" and arm == "naive":
                        _probe = ov                            # for the progress line below
                    for c in range(NC):
                        rows.append(dict(base, scope="perclass", cls=CLASSES[c],
                                         iou=seg[c]["iou"] * 100, f1=seg[c]["f1"] * 100,
                                         pa=seg[c]["pa"] * 100, ua=seg[c]["ua"] * 100,
                                         support=seg[c]["support"],
                                         cls_joint=rk[c]["joint"] * 100, cls_sel=rk[c]["selective"] * 100,
                                         cls_cov=rk[c]["coverage"] * 100))
            print(f"  seed {seed} split {ss}: L2A naive  joint_ce={_probe['joint_ce']:.2f} "
                  f"joint_pp={_probe['joint_pp']:.2f} | cov_ce={_probe['cov_ce']:.1f} "
                  f"cov_pp={_probe['cov_pp']:.1f} | acc_pixel={_probe['acc_pixel']:.1f} "
                  f"acc_ce={_probe['acc_ce']:.1f}", flush=True)

    tag = f"_{args.out_tag}" if args.out_tag else ""
    out = P8R.P(f"results_phase8R_perclass_weighting{tag}.csv")
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows, {len(args.seeds)} seeds x {len(args.split_seeds)} splits)")


if __name__ == "__main__":
    main()
