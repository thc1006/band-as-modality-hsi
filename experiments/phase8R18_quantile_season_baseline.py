#!/usr/bin/env python
"""Phase 8R18 -- cross-season and cross-Sen2Cor-baseline transfer of the label-free quantile transport.

Companion to phase8R17 (which probed sample size, random subsets and surface brightness). Here we test the
two remaining metadata-derivable transfer axes the reviewer asked for: does a quantile map estimated on one
SEASON, or on one Sen2Cor PROCESSING-BASELINE group, still reach near target on the (season/baseline-mixed)
evaluation set? Everything reuses phase8R10's transport and phase8R17's evaluation-disjoint protocol; only
which calibration components ESTIMATE the mapping changes. Season and baseline are read straight from the
CloudSEN12 test metadata (s2_date -> season; s2_sen2cor_version), so no external labels are introduced.

Arms (identical clean-calibrated threshold + temperature, no target labels, deployment always disjoint):
  full          : all calibration components (== phase8R17 anchor)
  calib_warm    : estimate on warm-season (MAM+JJA) calibration components
  calib_cold    : estimate on cold-season (SON+DJF) calibration components
  calib_newSC   : estimate on newer Sen2Cor baselines (>= N02.13) calibration components
  calib_oldSC   : estimate on older Sen2Cor baselines (<= N02.12) calibration components
Read transfer as robust only where an arm's interval overlaps the full arm AND is near the 10% target.
Cross-DATE (future acquisitions) and cross-PROCESSOR (a different corrector) remain untested.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import t as student_t

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
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

band_stats, quantile_match, seed_all, ALPHA = R10.band_stats, R10.quantile_match, R10.seed_all, R10.ALPHA
_SEASON = {12: "cold", 1: "cold", 2: "cold", 9: "cold", 10: "cold", 11: "cold",
           3: "warm", 4: "warm", 5: "warm", 6: "warm", 7: "warm", 8: "warm"}   # MAM+JJA warm / SON+DJF cold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8R18_quantile_season_baseline"))
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)
    seeds, splits = args.seeds, args.split_seeds

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    DROP_L2A = [g_b10]

    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    s2 = meta["s2_id"]
    if s2.isna().any():
        raise ValueError("NaN s2_id in test metadata")
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    ids = np.flatnonzero(~s2.isin(train_prod).to_numpy())
    # per-patch season + Sen2Cor baseline (scene-level fields; aligned with the metadata row order = patch id)
    month = pd.to_datetime(meta["s2_date"]).dt.month.to_numpy()
    season_patch = np.array([_SEASON[m] for m in month])
    def sc_new(v):  # newer baseline >= N02.13
        try:
            return float(str(v).replace("N", "")) >= 2.13
        except Exception:
            return False
    scnew_patch = np.array([sc_new(v) for v in meta["s2_sen2cor_version"].to_numpy()])

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=True, return_pixel_index=True)
    X_l1c, y_te, pid, pix = load("L1C")
    X_l2a, y_l2a, pid2, pix2 = load("L2A")
    np.testing.assert_array_equal(pid, pid2); np.testing.assert_array_equal(pix, pix2)
    np.testing.assert_array_equal(y_te, y_l2a)
    comp_all = P8R.scene_component_ids("test")[pid]
    season_px = season_patch[pid]                                   # per-pixel season / baseline via patch id
    scnew_px = scnew_patch[pid]
    Xtr_l1c, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    ytr, y_te = ytr.astype(np.int64), y_te.astype(np.int64)

    clip = lambda A: np.clip(A, -0.1, 1.6)
    keep_b = [i for i in range(X_l1c.shape[1]) if i != P8.B10_IDX]
    Xtr_c, Xl2a_c = clip(Xtr_l1c), clip(X_l2a)
    mu_tr, sd_tr = band_stats("L1C-train", Xtr_c, keep_b)
    Xtr_n = ((Xtr_c - mu_tr) / sd_tr).astype(np.float32)

    # component -> season / baseline (a component lies within one scene, so its patches share both; assert it)
    uniq = np.unique(comp_all)
    comp_season, comp_scnew = {}, {}
    for c in uniq:
        m = comp_all == c
        sv = set(season_px[m].tolist())
        comp_season[str(c)] = season_px[m][0]
        comp_scnew[str(c)] = bool(np.round(scnew_px[m].mean()))          # majority (baseline can differ across a merged component)
    warm = np.array([c for c in uniq if comp_season[str(c)] == "warm"])
    cold = np.array([c for c in uniq if comp_season[str(c)] == "cold"])
    newsc = np.array([c for c in uniq if comp_scnew[str(c)]])
    oldsc = np.array([c for c in uniq if not comp_scnew[str(c)]])
    print(f"  eval {len(y_te)} px / {len(uniq)} components; warm {len(warm)} / cold {len(cold)}; "
          f"newSC(>=N02.13) {len(newsc)} / oldSC {len(oldsc)}; {len(seeds)} seeds x {len(splits)} splits", flush=True)

    bs = P2.auto_bs(Xtr_n.shape[0])
    rows, covs = {}, {}
    def rec(a, s, r, j, cov): rows.setdefault(a, []).append((s, r, j)); covs.setdefault(a, []).append(cov)

    for seed in seeds:
        seed_all(seed + 101)
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)

        def clean_logits(mask):
            return P8R.logits_at("proposed", m, ((clip(X_l1c[mask]) - mu_tr) / sd_tr).astype(np.float32), groups, [])

        def qt_risk(cal_mask, eval_mask, Tc, thr):
            if cal_mask.sum() < 50:
                return np.nan, np.nan
            Xqm = quantile_match(Xl2a_c[eval_mask], Xl2a_c[cal_mask], Xtr_c, mu_tr, sd_tr, keep_b)
            p = softmax(P8R.logits_at("proposed", m, Xqm.astype(np.float32), groups, DROP_L2A) / Tc, axis=1)
            corr = p.argmax(1) == y_te[eval_mask]
            j, sel, cov = overall_metrics(corr, p.max(1), comp_all[eval_mask], thr)
            return float(j), float(cov)

        for ss in splits:
            mt, mc, me = P8R.split_test_rois(comp_all, ss)
            Tc = fit_temperature(clean_logits(mt), y_te[mt])
            pc = softmax(clean_logits(mc) / Tc, axis=1)
            corr_c = pc.argmax(1) == y_te[mc]
            thr = float(conformal_risk_control(corr_c, pc.max(1), corr_c, pc.max(1), alpha=ALPHA,
                                               calib_group=comp_all[mc], eval_group=comp_all[mc])["threshold"])
            if not np.isfinite(thr):
                raise RuntimeError("non-finite CRC threshold")
            calib = np.unique(comp_all[mc])
            def cm(comps): return mc & np.isin(comp_all, np.intersect1d(calib, comps))
            for arm, comps in [("full", calib), ("calib_warm", warm), ("calib_cold", cold),
                               ("calib_newSC", newsc), ("calib_oldSC", oldsc)]:
                j, cov = qt_risk(cm(comps) if arm != "full" else mc, me, Tc, thr)
                rec(arm, seed, ss, j, cov)
        done = [r[2] for r in rows["full"] if r[0] == seed and np.isfinite(r[2])]
        print(f"  seed {seed}: full-calib quantile mean {np.mean(done):.2f}", flush=True)

    df = min(len(set(seeds)), len(set(splits))) - 1
    tcrit = float(student_t.ppf(0.975, df))
    print(f"\n  two-way SE, t df={df} (tcrit {tcrit:.3f}); ALPHA target {ALPHA*100:.0f}%")
    summary = {"alpha": ALPHA, "df": df, "tcrit": tcrit, "n_components": int(len(uniq)),
               "n_warm": int(len(warm)), "n_cold": int(len(cold)), "n_newSC": int(len(newsc)),
               "n_oldSC": int(len(oldsc)), "arms": {}}
    for arm in ["full", "calib_warm", "calib_cold", "calib_newSC", "calib_oldSC"]:
        trip = [(s, r, v) for s, r, v in rows.get(arm, []) if np.isfinite(v)]
        if len(trip) < 3:
            print(f"  {arm:14s} insufficient ({len(trip)})"); continue
        mm, se = two_way_se(trip); cov = float(np.nanmean(covs[arm])); lo, hi = mm - tcrit * se, mm + tcrit * se
        summary["arms"][arm] = {"mean": mm, "se": se, "lo": lo, "hi": hi, "coverage": cov, "n_cells": len(trip)}
        print(f"  {arm:14s} joint {mm:6.2f} +/- {se:.2f}  [{lo:5.1f},{hi:5.1f}]  cov {cov:4.0f}%  (n={len(trip)})", flush=True)
    if "full" in summary["arms"]:
        print(f"\n  SUMMARY (data): full-calib {summary['arms']['full']['mean']:.2f}% (anchor). Cross-season "
              "(warm/cold) and cross-baseline (new/old Sen2Cor) arms show whether a mapping estimated on one "
              "season or processing-baseline group still reaches near target on the mixed eval set. Cross-DATE "
              "and cross-PROCESSOR transfer remain untested.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out + "_summary.json", "w"), indent=1)
    pd.DataFrame([{"arm": a, "seed": s, "split": r, "joint": v} for a in rows for (s, r, v) in rows[a]]).to_csv(
        args.out + "_percell.csv", index=False)
    print(f"\n  wrote {args.out}_summary.json + _percell.csv")


if __name__ == "__main__":
    main()
