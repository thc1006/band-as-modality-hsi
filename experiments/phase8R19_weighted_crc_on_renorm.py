#!/usr/bin/env python
"""Phase 8R19 -- does product-aware re-normalization restore the OVERLAP that the formal weighted CRC needs?

The formal component-level weighted CRC (phase8R11) can only ABSTAIN on the real L1C->L2A shift (joint
1.9% at 7% coverage) because clean and L2A are near-separable in the source representation (domain-classifier
AUROC ~0.99): source-only reweighting has essentially no overlap to exploit. The label-free product-aware
re-normalization collapses that separability (paper: AUROC 1.00 -> 0.56). So the pointed question (reviewer
bonus): on the RE-NORMALIZED representation, can the SAME formal weighted CRC now obtain overlap, control AND
a useful coverage at once?

We run the identical formal weighted CRC of phase8R11, changing ONLY the target: L2A vs quantile-re-normalized
L2A (the re-normalization estimated on the DISJOINT calibration components -- no evaluation transduction).
Per (seed, split) we report, for each target, the clean-vs-target domain AUROC, the calibration ESS, and the
formal weighted CRC's realized joint risk and coverage (component-equal, mean +/- two-way SE, df=9). Emits
DATA only; no verdict hardcoded. The plug-in likelihood ratio still inherits no EXACT finite-sample guarantee
and the re-normalization is estimated from deployment data -- this measures usable control, not an exact
re-derived certificate.
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
import phase8R11_weighted_crc_formal as R11
from phase8R_perclass_weighting_agg import two_way_se
from bandsim import hw
from bandsim.grouping import group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention
from bandsim.reliability import fit_temperature

band_stats, quantile_match, seed_all, ALPHA = R10.band_stats, R10.quantile_match, R10.seed_all, R11.ALPHA
state_arrays, domain_weights, weighted_crc_perunit, make_grid, ess_frac = (
    R11.state_arrays, R11.domain_weights, R11.weighted_crc_perunit, R11.make_grid, R11.ess_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8R19_weighted_crc_on_renorm"))
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
        raise ValueError("NaN s2_id")
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    ids = np.flatnonzero(~s2.isin(train_prod).to_numpy())

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=True, return_pixel_index=True)
    X_l1c, y_te, pid, pix = load("L1C")
    X_l2a, y_l2a, pid2, pix2 = load("L2A")
    np.testing.assert_array_equal(pid, pid2); np.testing.assert_array_equal(pix, pix2)
    np.testing.assert_array_equal(y_te, y_l2a)
    comp = P8R.scene_component_ids("test")[pid]
    Xtr_l1c, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    ytr, y_te = ytr.astype(np.int64), y_te.astype(np.int64)

    clip = lambda A: np.clip(A, -0.1, 1.6)
    keep_b = [i for i in range(X_l1c.shape[1]) if i != P8.B10_IDX]
    Xtr_c, Xl2a_c = clip(Xtr_l1c), clip(X_l2a)
    mu_tr, sd_tr = band_stats("L1C-train", Xtr_c, keep_b)
    Xtr_n = ((Xtr_c - mu_tr) / sd_tr).astype(np.float32)
    Xl1c_n = ((clip(X_l1c) - mu_tr) / sd_tr).astype(np.float32)
    Xl2a_src_n = ((Xl2a_c - mu_tr) / sd_tr).astype(np.float32)         # stale source-normalized L2A
    print(f"  eval {len(y_te)} px / {len(np.unique(comp))} components; {len(seeds)} seeds x {len(splits)} splits",
          flush=True)

    bs = P2.auto_bs(Xtr_n.shape[0])
    rows = {a: [] for a in ["src_uniform", "src_weighted", "ren_uniform", "ren_weighted"]}
    covs = {a: [] for a in rows}
    diag = {"src_auc": [], "src_ess": [], "ren_auc": [], "ren_ess": []}

    for seed in seeds:
        seed_all(seed + 101)
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)
        lc = P8R.logits_at("proposed", m, Xl1c_n, groups, [])          # clean logits (all px)
        ll = P8R.logits_at("proposed", m, Xl2a_src_n, groups, DROP_L2A)  # source-normalized L2A logits (all px)

        for ss in splits:
            mt, mc, me = P8R.split_test_rois(comp, ss)
            Tc = fit_temperature(lc[mt], y_te[mt])
            grid = make_grid(softmax(lc[mc] / Tc).max(1))
            Lc, COVc, _, Fc, _ = state_arrays(lc, y_te, comp, mc, Tc, grid)      # CLEAN calibration
            _, _, _, F0, _ = state_arrays(lc, y_te, comp, mt, Tc, grid)          # source domain (clean temp)

            def run(ltarget, tag):
                Le, COVe, _, Fe, _ = state_arrays(ltarget, y_te, comp, me, Tc, grid)
                _, _, _, F1, _ = state_arrays(ltarget, y_te, comp, mt, Tc, grid)
                (wc, we), auc = domain_weights(F0, F1, [Fc, Fe])
                rn, cn = weighted_crc_perunit(Lc, np.ones(len(wc)), grid, Le, COVe, np.ones(len(we)), ALPHA)
                rw, cw = weighted_crc_perunit(Lc, wc, grid, Le, COVe, we, ALPHA)
                rows[f"{tag}_uniform"].append((seed, ss, float(np.mean(rn)) * 100)); covs[f"{tag}_uniform"].append(float(np.mean(cn)) * 100)
                rows[f"{tag}_weighted"].append((seed, ss, float(np.mean(rw)) * 100)); covs[f"{tag}_weighted"].append(float(np.mean(cw)) * 100)
                return auc, ess_frac(we)

            # source target (== phase8R11): reproduces the ~1.9%@7% abstention
            a_s, e_s = run(ll, "src")
            diag["src_auc"].append(a_s); diag["src_ess"].append(e_s)
            # re-normalized target: quantile transport estimated on the DISJOINT calibration L2A (mc), applied to all
            Xren = quantile_match(Xl2a_c, Xl2a_c[mc], Xtr_c, mu_tr, sd_tr, keep_b).astype(np.float32)
            lr = P8R.logits_at("proposed", m, Xren, groups, DROP_L2A)
            a_r, e_r = run(lr, "ren")
            diag["ren_auc"].append(a_r); diag["ren_ess"].append(e_r)
        print(f"  seed {seed}: src wtd {np.mean([r[2] for r in rows['src_weighted'] if r[0]==seed]):.1f}@"
              f"{np.mean(covs['src_weighted'][-len(splits):]):.0f}%  ren wtd "
              f"{np.mean([r[2] for r in rows['ren_weighted'] if r[0]==seed]):.1f}@"
              f"{np.mean(covs['ren_weighted'][-len(splits):]):.0f}%  (AUROC src {a_s:.2f} ren {a_r:.2f})", flush=True)

    df = min(len(set(seeds)), len(set(splits))) - 1
    tcrit = float(student_t.ppf(0.975, df))
    print(f"\n  two-way SE, t df={df}; ALPHA {ALPHA*100:.0f}%  (joint risk % @ coverage %)")
    summary = {"alpha": ALPHA, "df": df, "tcrit": tcrit, "arms": {},
               "diagnostics": {k: float(np.mean(v)) for k, v in diag.items()}}
    for arm in ["src_uniform", "src_weighted", "ren_uniform", "ren_weighted"]:
        trip = [(s, r, v) for s, r, v in rows[arm] if np.isfinite(v)]
        mm, se = two_way_se(trip); cov = float(np.mean(covs[arm])); lo, hi = mm - tcrit * se, mm + tcrit * se
        summary["arms"][arm] = {"mean": mm, "se": se, "lo": lo, "hi": hi, "coverage": cov, "n_cells": len(trip)}
        print(f"  {arm:14s} joint {mm:6.2f} +/- {se:.2f} [{lo:5.1f},{hi:5.1f}]  cov {cov:4.0f}%  (n={len(trip)})", flush=True)
    d = summary["diagnostics"]
    print(f"\n  diagnostics: domain AUROC  source {d['src_auc']:.3f} -> re-normalized {d['ren_auc']:.3f}; "
          f"calibration ESS  source {d['src_ess']*100:.0f}% -> re-normalized {d['ren_ess']*100:.0f}%")
    print("  SUMMARY (data): on the SOURCE representation the formal weighted CRC abstains (high AUROC, tiny ESS, "
          "low coverage); read whether RE-NORMALIZING first restores overlap (AUROC down, ESS up) and lets the "
          "SAME weighted CRC hold control at a useful coverage. Plug-in weights carry no exact finite-sample "
          "guarantee and the re-normalization is estimated from deployment data.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out + "_summary.json", "w"), indent=1)
    pd.DataFrame([{"arm": a, "seed": s, "split": r, "joint": v} for a in rows for (s, r, v) in rows[a]]).to_csv(
        args.out + "_percell.csv", index=False)
    print(f"\n  wrote {args.out}_summary.json + _percell.csv")


if __name__ == "__main__":
    main()
