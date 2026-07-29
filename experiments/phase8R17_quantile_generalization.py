#!/usr/bin/env python
"""Phase 8R17 -- operational-generalization probes for the label-free quantile transport (reviewer 3.2).

Does the per-band monotone quantile transport behave like a STABLE product-state transform (H1: a mapping
estimated on some target components transfers to others) or a mixture-specific batch adaptation (H2: it only
works for the particular 69-component calibration mixture / H4: a different surface regime needs a different
mapping)? We re-use phase8R10's exact data, training and clean-source certificate, and vary ONLY which/how
many target components ESTIMATE the mapping. Every arm stays evaluation-disjoint: the mapping is estimated
on calibration components and deployed on a disjoint evaluation set, exactly as the flagship disjoint arm.

Probes (identical clean-calibrated threshold + temperature, no target labels):
  (0) full        : estimate on ALL calibration components  == phase8R10 disjoint-quantile headline (anchor)
  (1) n{08..40}   : estimate on n RANDOM calibration components; does near-target converge as n grows? (H2)
  (2) halfA/halfB : two DISJOINT random halves of the calibration components; do they agree?             (H2)
  (3) calib_bright/calib_dark : estimate on the bright / dark half of calibration components (split by
                    per-component L1C brightness); does a mapping from one surface regime still reach
                    near target on the SAME eval set?                                                    (H4)

Emits DATA only (per-cell CSV + summary JSON); no verdict is hardcoded. Reuses phase8R10's transport, band
statistics and component accounting so every number is bit-comparable to the released normalization control.
The cross-SEASON, cross-DATE and cross-PROCESSOR-baseline transfers are explicitly NOT tested here.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import t as student_t

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase8R_reliability as P8R
import phase2_degradation as P2
import phase8R10_normalization_control as R10
from phase8R3_acolite import overall_metrics
from phase8R_perclass_weighting_agg import two_way_se
from bandsim import hw
from bandsim.grouping import group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention
from bandsim.reliability import fit_temperature, conformal_risk_control

band_stats = R10.band_stats
quantile_match = R10.quantile_match
seed_all = R10.seed_all
ALPHA = R10.ALPHA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 40])
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8R17_quantile_generalization"))
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)
    seeds, splits = args.seeds, args.split_seeds

    # ------ data + reference stats: replicate phase8R10.main() exactly (deterministic loaders) ------
    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    DROP_L2A = [g_b10]

    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    s2 = meta["s2_id"]
    if s2.isna().any():
        raise ValueError("test scenes with NaN s2_id -- resolve provenance before trusting this split")
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    ids = np.flatnonzero(~s2.isin(train_prod).to_numpy())

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=True, return_pixel_index=True)
    X_l1c, y_te, pid, pix = load("L1C")
    X_l2a, y_l2a, pid2, pix2 = load("L2A")
    if X_l1c.shape != X_l2a.shape:
        raise ValueError(f"L1C/L2A shape mismatch {X_l1c.shape} vs {X_l2a.shape}")
    np.testing.assert_array_equal(pid, pid2)
    np.testing.assert_array_equal(pix, pix2)
    np.testing.assert_array_equal(y_te, y_l2a)
    comp_all = P8R.scene_component_ids("test")[pid]
    Xtr_l1c, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    ytr, y_te = ytr.astype(np.int64), y_te.astype(np.int64)

    clip = lambda A: np.clip(A, -0.1, 1.6)
    keep_b = [i for i in range(X_l1c.shape[1]) if i != P8.B10_IDX]
    Xtr_c, Xl2a_c, Xl1c_c = clip(Xtr_l1c), clip(X_l2a), clip(X_l1c)
    mu_tr, sd_tr = band_stats("L1C-train", Xtr_c, keep_b)
    Xtr_n = ((Xtr_c - mu_tr) / sd_tr).astype(np.float32)

    # per-component L1C brightness (data-derived surface proxy; no external labels) for the cross-surface probe
    bright_px = Xl1c_c[:, keep_b].mean(axis=1)
    uniq_comp = np.unique(comp_all)
    comp_bright = {str(c): float(bright_px[comp_all == c].mean()) for c in uniq_comp}
    bmed = float(np.median(list(comp_bright.values())))
    bright_comp = np.array([c for c in uniq_comp if comp_bright[str(c)] >= bmed])
    dark_comp = np.array([c for c in uniq_comp if comp_bright[str(c)] < bmed])
    print(f"  eval {len(y_te)} px / {len(uniq_comp)} components; brightness median {bmed:.3f}; "
          f"bright {len(bright_comp)} / dark {len(dark_comp)} comps; {len(seeds)} seeds x {len(splits)} splits",
          flush=True)

    bs = P2.auto_bs(Xtr_n.shape[0])
    rows, covs = {}, {}
    def rec(arm, seed, ss, j, cov):
        rows.setdefault(arm, []).append((seed, ss, j)); covs.setdefault(arm, []).append(cov)

    for seed in seeds:
        seed_all(seed + 101)
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)

        def clean_logits(mask):
            Xe = ((Xl1c_c[mask] - mu_tr) / sd_tr).astype(np.float32)
            return P8R.logits_at("proposed", m, Xe, groups, [])

        def qt_risk(cal_mask, eval_mask, Tc, thr):
            """cal_mask: boolean pixel mask of the calibration components used to ESTIMATE the map (already
            intersected with the calibration partition -> evaluation-disjoint). Returns (joint%, coverage%)."""
            if cal_mask.sum() < 50:
                return np.nan, np.nan
            Xqm = quantile_match(Xl2a_c[eval_mask], Xl2a_c[cal_mask], Xtr_c, mu_tr, sd_tr, keep_b)
            lg = P8R.logits_at("proposed", m, Xqm.astype(np.float32), groups, DROP_L2A)
            p = softmax(lg / Tc, axis=1)
            corr = p.argmax(1) == y_te[eval_mask]
            j, sel, cov = overall_metrics(corr, p.max(1), comp_all[eval_mask], thr)
            return float(j), float(cov)

        rng = np.random.default_rng(seed + 202)
        for ss in splits:
            mt, mc, me = P8R.split_test_rois(comp_all, ss)           # boolean pixel masks, split BY component
            Tc = fit_temperature(clean_logits(mt), y_te[mt])
            pc = softmax(clean_logits(mc) / Tc, axis=1)
            corr_c = pc.argmax(1) == y_te[mc]
            thr = float(conformal_risk_control(corr_c, pc.max(1), corr_c, pc.max(1), alpha=ALPHA,
                                               calib_group=comp_all[mc], eval_group=comp_all[mc])["threshold"])
            if not np.isfinite(thr):
                raise RuntimeError("CRC selected a non-finite threshold")
            calib_comps = np.unique(comp_all[mc])                    # components in the calibration partition

            def cal_mask_of(comps):                                  # pixels of `comps` INSIDE the calib partition
                return mc & np.isin(comp_all, np.asarray(comps))

            # (0) full-calibration anchor  == phase8R10 disjoint-quantile headline
            j, cov = qt_risk(mc, me, Tc, thr); rec("full", seed, ss, j, cov)

            # (1) sample-size stability: n random calibration components
            for n in args.sizes:
                if n >= len(calib_comps):
                    continue
                sub = rng.choice(calib_comps, size=n, replace=False)
                j, cov = qt_risk(cal_mask_of(sub), me, Tc, thr); rec(f"n{n:02d}", seed, ss, j, cov)

            # (2) two disjoint random halves of the calibration components
            perm = rng.permutation(calib_comps)
            jA, cA = qt_risk(cal_mask_of(perm[:len(perm) // 2]), me, Tc, thr); rec("halfA", seed, ss, jA, cA)
            jB, cB = qt_risk(cal_mask_of(perm[len(perm) // 2:]), me, Tc, thr); rec("halfB", seed, ss, jB, cB)

            # (3) cross-surface: estimate on the bright / dark calibration components (disjoint from eval)
            jbr, cbr = qt_risk(cal_mask_of(np.intersect1d(calib_comps, bright_comp)), me, Tc, thr)
            rec("calib_bright", seed, ss, jbr, cbr)
            jdk, cdk = qt_risk(cal_mask_of(np.intersect1d(calib_comps, dark_comp)), me, Tc, thr)
            rec("calib_dark", seed, ss, jdk, cdk)

        done = [r[2] for r in rows["full"] if r[0] == seed and np.isfinite(r[2])]
        print(f"  seed {seed}: full-calib quantile mean {np.mean(done):.2f}", flush=True)

    # ------ aggregate: two-way cluster-robust SE (seed x split), t at df=min(#seed,#split)-1 ------
    df = min(len(set(seeds)), len(set(splits))) - 1
    tcrit = float(student_t.ppf(0.975, df))
    print(f"\n  two-way SE, t df={df} (tcrit {tcrit:.3f}); ALPHA target {ALPHA*100:.0f}%")
    summary = {"alpha": ALPHA, "df": df, "tcrit": tcrit, "n_components": int(len(uniq_comp)),
               "brightness_median": bmed, "n_bright": int(len(bright_comp)), "n_dark": int(len(dark_comp)),
               "sizes": list(args.sizes), "arms": {}}
    order = ["full"] + [f"n{n:02d}" for n in args.sizes] + ["halfA", "halfB", "calib_bright", "calib_dark"]
    for arm in order:
        if arm not in rows:
            continue
        trip = [(s, r, v) for s, r, v in rows[arm] if np.isfinite(v)]
        if len(trip) < 3:
            print(f"  {arm:14s} insufficient finite cells ({len(trip)})"); continue
        mm, se = two_way_se(trip)
        cov = float(np.nanmean(covs[arm]))
        lo, hi = mm - tcrit * se, mm + tcrit * se
        summary["arms"][arm] = {"mean": mm, "se": se, "lo": lo, "hi": hi, "coverage": cov, "n_cells": len(trip)}
        print(f"  {arm:14s} joint {mm:6.2f} +/- {se:.2f}  [{lo:5.1f},{hi:5.1f}]  cov {cov:4.0f}%  (n={len(trip)})",
              flush=True)

    if "full" in summary["arms"]:
        f = summary["arms"]["full"]["mean"]
        print(f"\n  SUMMARY (data, not a verdict): full-calibration quantile transport {f:.2f}% (anchor vs the "
              "phase8R10 10-seed headline 9.6%). Sample-size arms show whether the near-target level is reached "
              "with fewer estimation components; halfA/halfB and calib_bright/calib_dark show whether the "
              "estimated mapping depends on WHICH (or which surface regime of) calibration components produced "
              "it. Read transfer as robust only where a reduced/paired arm's interval overlaps the full arm. "
              "Cross-SEASON, cross-DATE and cross-PROCESSOR-baseline transfer are NOT tested here.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    long = [{"arm": a, "seed": s, "split": r, "joint": v} for a in rows for (s, r, v) in rows[a]]
    pd.DataFrame(long).to_csv(args.out + "_percell.csv", index=False)
    print(f"\n  wrote {args.out}_summary.json + _percell.csv")


if __name__ == "__main__":
    main()
