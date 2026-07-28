#!/usr/bin/env python
"""R7-C7 (round-7 review): the FULL 10x10 class-wise decomposition.

The round-6 class-wise table (phase8R_classwise) used 10 model seeds but a SINGLE fixed calibration split.
IoU / F1 are threshold-free, so the per-class collapse they show is credible; but the class-conditional
CRC operating-point metrics -- joint risk P(accepted & wrong | class), selective risk P(wrong | accepted,
class) and coverage -- depend on the clean-calibrated acceptance THRESHOLD, which moves with the calibration
split. To support "minority classes drive the headline joint risk ACROSS THE FULL EXPERIMENTAL DESIGN" we
therefore re-run the decomposition over the SAME 10 model seeds x 10 scene-component split seeds as the
flagship, with two-way cluster-robust SEs on every per-class number, and report the class-support-weighted
sum so it can be checked against the headline.

Efficiency: each model seed is trained ONCE; full-test logits are cached once per state; the 10 splits then
only re-fit the temperature, re-derive the CRC threshold and re-index -- 10 trainings, 100 evaluations.

Run: CUDA_VISIBLE_DEVICES=0 python phase8R16_classwise_10x10.py --seeds 0 1 2 3 4 5 6 7 8 9
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
from phase8R_classwise import perclass_seg, perclass_risk, CLASSES, NC
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim.reliability import fit_temperature, conformal_risk_control
from bandsim import hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8R16_classwise_10x10"))
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)

    import pandas as pd
    from scipy.special import softmax
    from scipy.stats import t as student_t

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    drop = {"clean": [], "L2A": [g_b10]}

    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr = norm(Xtr)

    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leaked = meta.index[meta["s2_id"].isin(train_prod)].to_numpy()
    ids = np.setdiff1d(np.arange(len(meta)), leaked)

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=(prd == "L1C"))
    X_l1c, y_te, pid = load("L1C")
    X_l2a, y_l2a = load("L2A")
    y_te = y_te.astype(np.int64)
    np.testing.assert_array_equal(y_te, y_l2a.astype(np.int64))       # L1C/L2A must sample the identical pixels/labels (same seed)
    if X_l1c.shape != X_l2a.shape:
        raise ValueError(f"L1C/L2A shape mismatch {X_l1c.shape} vs {X_l2a.shape}")
    comp = P8R.scene_component_ids("test")[pid]
    Xc = {"clean": norm(X_l1c), "L2A": norm(X_l2a)}
    bs = P2.auto_bs(Xtr.shape[0])
    print(f"class-wise 10x10: {len(y_te)} test px / {len(np.unique(comp))} scene-components; "
          f"{len(args.seeds)} model seeds x {len(args.split_seeds)} split seeds; alpha {args.alpha:.0%}", flush=True)

    # long-form rows tagged (seed, split); aggregated per (state, class, metric) with two-way SEs
    METRICS = ["iou", "f1", "pa", "ua", "joint", "selective", "coverage"]
    rec = {(st, c, mtr): [] for st in ("clean", "L2A") for c in range(NC) for mtr in METRICS}
    wsum = {st: [] for st in ("clean", "L2A")}                       # support-weighted joint per cell
    per_cell = []
    for seed in args.seeds:
        hw.seed_model(seed)                                         # P0-2: seed before the constructor
        m = GroupedCrossBandAttention(groups, cwl, NC)
        P2.pretrain_sgmae(m, Xtr, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr, ytr, groups, seed, epochs=args.epochs, bs=bs)
        raw = {st: P8R.logits_at("proposed", m, Xc[st], groups, drop[st]) for st in ("clean", "L2A")}
        for ss in args.split_seeds:
            mt, mc, me = P8R.split_test_rois(comp, ss)
            T = fit_temperature(raw["clean"][mt], y_te[mt])
            pc = softmax(raw["clean"][mc] / T, axis=1)
            thr = float(conformal_risk_control((pc.argmax(1) == y_te[mc]), pc.max(1),
                                               (pc.argmax(1) == y_te[mc]), pc.max(1), alpha=args.alpha,
                                               calib_group=comp[mc], eval_group=comp[mc])["threshold"])
            if not np.isfinite(thr):                                # abstain-all/infeasible would fake joint=0 via conf>=inf
                raise RuntimeError(f"CRC selected a non-finite threshold at seed {seed} split {ss}")
            for st in ("clean", "L2A"):
                p = softmax(raw[st][me] / T, axis=1)
                pred, conf, yy = p.argmax(1), p.max(1), y_te[me]
                seg = perclass_seg(yy, pred); rk = perclass_risk(yy, pred, conf, thr)
                tot = sum(seg[c]["support"] for c in range(NC))
                wj = 0.0
                for c in range(NC):
                    vals = dict(iou=seg[c]["iou"] * 100, f1=seg[c]["f1"] * 100, pa=seg[c]["pa"] * 100,
                                ua=seg[c]["ua"] * 100, joint=rk[c]["joint"] * 100,
                                selective=rk[c]["selective"] * 100, coverage=rk[c]["coverage"] * 100)
                    for mtr in METRICS:
                        if np.isfinite(vals[mtr]):
                            rec[(st, c, mtr)].append((seed, ss, vals[mtr]))
                    if tot and seg[c]["support"]:                    # skip zero-support class (0*NaN would poison wj)
                        wj += (seg[c]["support"] / tot) * (rk[c]["joint"] * 100)
                    per_cell.append(dict(state=st, cls=CLASSES[c], model_seed=int(seed), split_seed=int(ss),
                                         support=seg[c]["support"], **vals))
                wsum[st].append((seed, ss, wj))
        print(f"  seed {seed} done ({len(args.split_seeds)} splits)", flush=True)

    df = min(len(set(args.seeds)), len(set(args.split_seeds))) - 1
    tcrit = float(student_t.ppf(0.975, df)) if df >= 1 else float("nan")

    def agg(cells):
        """two-way SE + min/max over the (seed,split) grid (min/max so a saturated 100.0 can be qualified)."""
        if not cells:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        mm, se = two_way_se(cells)
        v = np.array([c[2] for c in cells])
        return mm, se, float(v.min()), float(v.max()), len(cells)

    print(f"\n===== class-wise 10x10 (proposed; two-way SE, t df={df}) =====")
    for st in ("clean", "L2A"):
        wm, wse = two_way_se(wsum[st])
        print(f"\n[{st}]  support-weighted sum_c w_c*joint_c = {wm:.2f} +/- {wse:.2f} "
              f"[{wm - tcrit * wse:.1f},{wm + tcrit * wse:.1f}]   (compare headline: L2A ~27.8, clean ~8.6)")
        print(f"  {'class':13s} {'IoU':>11s} {'joint':>13s} {'selective(min..max)':>22s} {'coverage':>12s}  support")
        for c in range(NC):
            iou_m, iou_se, *_ = agg(rec[(st, c, "iou")])
            j_m, j_se, *_ = agg(rec[(st, c, "joint")])
            s_m, s_se, s_lo, s_hi, s_n = agg(rec[(st, c, "selective")])
            cov_m, cov_se, *_ = agg(rec[(st, c, "coverage")])
            sup = int(np.median([r["support"] for r in per_cell if r["state"] == st and r["cls"] == CLASSES[c]]))
            print(f"  {CLASSES[c]:13s} {iou_m:5.1f}+/-{iou_se:4.1f} {j_m:6.2f}+/-{j_se:5.2f} "
                  f"{s_m:6.1f}+/-{s_se:4.1f}({s_lo:.0f}..{s_hi:.0f}) {cov_m:6.1f}+/-{cov_se:4.1f}  {sup}", flush=True)

    pd.DataFrame(per_cell).to_csv(args.out + "_percell.csv", index=False)
    # machine-readable aggregate
    out_agg = {"alpha": args.alpha, "df": df, "tcrit": tcrit,
               "weighted_joint": {st: dict(zip(("mean", "se"), two_way_se(wsum[st]))) for st in ("clean", "L2A")},
               "perclass": {}}
    for st in ("clean", "L2A"):
        for c in range(NC):
            for mtr in METRICS:
                mm, se, lo, hi, n = agg(rec[(st, c, mtr)])
                out_agg["perclass"][f"{st}|{CLASSES[c]}|{mtr}"] = dict(mean=mm, se=se, min=lo, max=hi, n=n)
    import json
    with open(args.out + "_summary.json", "w") as f:
        json.dump(out_agg, f, indent=2)
    print(f"\nwrote {args.out}_percell.csv + {args.out}_summary.json")


if __name__ == "__main__":
    main()
