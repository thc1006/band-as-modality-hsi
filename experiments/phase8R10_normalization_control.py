#!/usr/bin/env python
"""R6-E4 (round-6 review 3.3 / B3): is the L1C->L2A breach dominated by a STALE INPUT-NORMALIZATION contract
rather than a genuine confidence-error drift -- and, if so, is that specifically a PRODUCT-normalization
mismatch, not merely generic test-batch adaptation or a train->test scene-composition change?

The flagship standardises every product with its L1C-TRAIN per-band statistics and reuses them on L2A. We
train the band-as-modality model once per seed on L1C, then evaluate the identical clean-calibrated naive
certificate under SIX input standardisations of the SAME trained model (only the standardisation differs),
so the 'normalization' explanation is decomposed rather than confounded:

  A  clean          L1C test, L1C-TRAIN stats                 -- in-domain baseline
  B  clean_selfnorm L1C test, L1C-TEST  stats                 -- generic test-batch renorm control (no product change)
  C  L2A_src        L2A test, L1C-TRAIN stats                 -- the flagship's stale normalization (the breach)
  D  L2A_pairedL1C  L2A test, L1C-TEST  stats                 -- product offset ONLY (train->test composition removed)
  E  L2A_prod       L2A test, L2A-TEST  stats                 -- product-aware, transductive (full-test stats)
  *  L2A_prod_disj  L2A test, L2A-CALIB-split stats (per split) -- product-aware, EVALUATION-DISJOINT (PRIMARY arm)

Read as PAIRED differences over the seed x split grid (two-way cluster-robust SE, proper t df):
  * C - disj : operational, label-free repair magnitude (PRIMARY -- disjoint stats rule out transduction)
  * E - disj : transductive advantage (should include 0)
  * C - D    : train->test COMPOSITION share of the fix
  * D - E    : PRODUCT-normalization share of the fix (the 'normalization contract' proper)
  * B - A    : generic test-batch renorm on the in-domain case (should ~0 if the fix is product-specific)

The label-free repair restores the empirical target joint-risk LEVEL on the tested scenes; it does NOT
re-derive a finite-sample conformal guarantee (the normalization is estimated from deployment data, which
breaks source-calibration exchangeability). This script emits DATA (per-cell CSV + paired-difference CIs);
it does not auto-issue a scientific verdict from arbitrary numeric cutoffs.

Run: CUDA_VISIBLE_DEVICES=0 python phase8R10_normalization_control.py --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import t as student_t

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R
from phase8R3_acolite import overall_metrics
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim.reliability import fit_temperature, conformal_risk_control
from bandsim import hw

ALPHA = 0.10


def seed_all(s):
    """Seed IMMEDIATELY before each model constructor. The project's training functions only reseed INSIDE
    pretrain/finetune (after the module is built), so without this the initial weights depend on execution
    order (running seed k alone != seed k within a batch). Matches the +101 convention in phase2_degradation."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def require_finite(name, x):
    if not np.isfinite(x).all():
        raise FloatingPointError(f"{name}: {int((~np.isfinite(x)).sum())} non-finite value(s) -- a NaN/Inf "
                                 "confidence is treated as abstention by `conf >= thr` and would fake a low risk")


def band_stats(name, x, keep_idx, floor=1e-6):
    """Per-band mean/std with a finiteness gate; RAISE if a RETAINED (non-B10) band is near-constant (its
    z-score would blow up and could masquerade as abstention). B10 is 0 in L2A and dropped downstream, so we
    floor it rather than alarm."""
    require_finite(name, x)
    mu, sd = x.mean(0), x.std(0)
    bad = [int(i) for i in keep_idx if sd[i] < floor]
    if bad:
        raise ValueError(f"{name}: near-constant retained band(s) {bad} (sd={sd[bad]}); z-score unstable")
    return mu, np.maximum(sd, floor)


def robust_transport(Xeval, Xcal_l2a, Xsrc, mu_tr, sd_tr, keep_idx, floor=1e-6):
    """Robust 2-moment transport (label-free, evaluation-disjoint). Align each L2A eval band's robust
    location/scale (median, 1.4826*MAD, estimated on the DISJOINT calibration L2A) onto the L1C-train
    band's robust location/scale, THEN apply the model's own train z-score. This down-weights the heavy,
    over-corrected bright-pixel tails that inflate plain mean/std, while keeping the final scale EXACTLY
    the training normalisation -- so there is no statistic-family mismatch with training (which used
    mean/std). It sits between the plain mean/std disjoint arm (2 moments, non-robust) and the full
    quantile transport (all moments). Dropped bands (B10, zeroed in L2A and masked in the forward pass)
    fall back to the plain train z-score so the tensor stays finite."""
    out = ((Xeval - mu_tr) / sd_tr).astype(np.float32)
    keep = {int(i) for i in keep_idx}
    for b in range(Xeval.shape[1]):
        if b not in keep:
            continue
        med_c = np.median(Xcal_l2a[:, b]); mad_c = float(np.median(np.abs(Xcal_l2a[:, b] - med_c)) * 1.4826)
        med_s = np.median(Xsrc[:, b]);     mad_s = float(np.median(np.abs(Xsrc[:, b] - med_s)) * 1.4826)
        if mad_c < floor or mad_s < floor:
            raise ValueError(f"robust_transport band {b}: near-constant (mad_cal={mad_c:.2e}, mad_src={mad_s:.2e})")
        xm = (Xeval[:, b] - med_c) / mad_c * mad_s + med_s   # L2A robust moments -> L1C-train robust moments
        out[:, b] = (xm - mu_tr[b]) / sd_tr[b]               # exact training normalisation on the transported band
    return out


def quantile_match(Xeval, Xcal_l2a, Xsrc, mu_tr, sd_tr, keep_idx):
    """Per-band monotone quantile mapping (label-free, evaluation-disjoint): transform each eval L2A band so
    its marginal matches the SOURCE (L1C-train) marginal, using the disjoint CALIBRATION L2A as the reference
    for the L2A CDF -- no eval transduction. This aligns the FULL per-band distribution (all moments), not
    only mean/std, to the scale the certificate was calibrated on; the result is then z-scored with the
    model's own training statistics, exactly as at training. Formally v' = Q_src(F_{L2A-cal}(x)), band by
    band, with plotting-position CDFs and linear interpolation (a proper monotone transport map). Dropped
    bands (B10, zeroed in L2A and masked in the forward pass) fall back to the plain train z-score."""
    out = ((Xeval - mu_tr) / sd_tr).astype(np.float32)       # safe finite default for every band
    keep = {int(i) for i in keep_idx}
    for b in range(Xeval.shape[1]):
        if b not in keep:
            continue
        cal = np.sort(Xcal_l2a[:, b]); src = np.sort(Xsrc[:, b])
        cdf_cal = (np.arange(len(cal)) + 0.5) / len(cal)     # plotting-position empirical CDF of calib L2A
        cdf_src = (np.arange(len(src)) + 0.5) / len(src)
        u = np.interp(Xeval[:, b], cal, cdf_cal)             # F_{L2A-cal}(x) in (0,1), monotone
        v = np.interp(u, cdf_src, src)                       # Q_src(u): source quantile at that level
        out[:, b] = (v - mu_tr[b]) / sd_tr[b]                # z-score on the model's own training scale
    return out


def comp_equal_acc(corr, comp):
    """Component-equal (not pixel-pooled) accuracy, to match the certified triple's weighting."""
    order = np.argsort(comp, kind="stable")
    cs, vs = comp[order], corr[order].astype(float)
    _, starts = np.unique(cs, return_index=True)
    sums = np.add.reduceat(vs, starts)
    counts = np.diff(np.append(starts, len(cs)))
    return float(np.mean(sums / counts)) * 100


def paired_delta(rows_a, rows_b, seeds, splits):
    """Per-cell (a - b) over the shared seed x split grid, then two-way cluster-robust SE + t CI."""
    da = {(s, r): v for s, r, v in rows_a}
    db = {(s, r): v for s, r, v in rows_b}
    diff = [(s, r, da[(s, r)] - db[(s, r)]) for s in seeds for r in splits if (s, r) in da and (s, r) in db]
    m, se = two_way_se(diff)
    df = min(len(set(seeds)), len(set(splits))) - 1
    tc = float(student_t.ppf(0.975, df)) if df >= 1 else float("nan")
    return m, se, m - tc * se, m + tc * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8R10_normalization_control"))
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)
    seeds, splits = args.seeds, args.split_seeds

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    DROP_L2A = [g_b10]                                                # L2A drops the cirrus band regardless

    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    s2 = meta["s2_id"]
    if s2.isna().any():                                              # NaN s2_id silently survives ~isin()
        raise ValueError(f"{int(s2.isna().sum())} test scenes have NaN s2_id -- the exact-product-overlap "
                         "guard cannot clear them; resolve provenance before trusting this split")
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    ids = np.flatnonzero(~s2.isin(train_prod).to_numpy())            # exact-product-overlap-guarded test

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=True, return_pixel_index=True)
    X_l1c, y_te, pid, pix = load("L1C")
    X_l2a, y_l2a, pid2, pix2 = load("L2A")
    # ALIGNMENT ASSERT -- the paired design relies on identical (patch, pixel, label) ordering across products
    if X_l1c.shape != X_l2a.shape:
        raise ValueError(f"L1C/L2A tensor shape mismatch {X_l1c.shape} vs {X_l2a.shape}")
    np.testing.assert_array_equal(pid, pid2)
    np.testing.assert_array_equal(pix, pix2)
    np.testing.assert_array_equal(y_te, y_l2a)
    comp_all = P8R.scene_component_ids("test")[pid]
    Xtr_l1c, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    ytr, y_te = ytr.astype(np.int64), y_te.astype(np.int64)

    clip = lambda A: np.clip(A, -0.1, 1.6)
    keep_b = [i for i in range(X_l1c.shape[1]) if i != P8.B10_IDX]   # B10 is 0 in L2A and dropped; exclude
    for nm, A in [("L1C", X_l1c), ("L2A", X_l2a)]:                   # clip-rate audit over retained bands
        lo = float((A[:, keep_b] < -0.1).mean()) * 100
        hi = float((A[:, keep_b] > 1.6).mean()) * 100
        print(f"  clip audit {nm}: {lo:.3f}% < -0.1, {hi:.3f}% > 1.6 (over {len(keep_b)} retained bands)", flush=True)

    Xtr_c, Xl2a_c = clip(Xtr_l1c), clip(X_l2a)
    mu_tr, sd_tr = band_stats("L1C-train", Xtr_c, keep_b)           # A/C reference (the model's own, stale)
    mu_1t, sd_1t = band_stats("L1C-test", clip(X_l1c), keep_b)      # B/D reference (composition-controlled)
    mu_2t, sd_2t = band_stats("L2A-test", Xl2a_c, keep_b)          # E reference (product-aware, transductive)
    Xtr_n = ((Xtr_c - mu_tr) / sd_tr).astype(np.float32)

    # arm -> (full input array, mu, sd, drop-groups). One trained model per seed; only standardisation differs.
    ARMS = {"clean":          (X_l1c, mu_tr, sd_tr, []),
            "clean_selfnorm": (X_l1c, mu_1t, sd_1t, []),
            "L2A_src":        (X_l2a, mu_tr, sd_tr, DROP_L2A),
            "L2A_pairedL1C":  (X_l2a, mu_1t, sd_1t, DROP_L2A),
            "L2A_prod":       (X_l2a, mu_2t, sd_2t, DROP_L2A)}
    trk = list(ARMS) + ["L2A_prod_disj", "L2A_prod_disj_robust", "L2A_prod_disj_quantile"]
    print(f"  eval {len(y_te)} px / {len(np.unique(comp_all))} components; L1C-train vs L2A-test per-band mean "
          f"shift {np.abs(mu_tr - mu_2t)[keep_b].mean():.3f}; {len(seeds)} seeds x {len(splits)} splits", flush=True)

    bs = P2.auto_bs(Xtr_n.shape[0])
    rows = {s: [] for s in trk}
    covs = {s: [] for s in trk}
    sels = {s: [] for s in trk}
    pacc = {s: [] for s in trk}
    cacc = {s: [] for s in trk}
    for seed in seeds:
        seed_all(seed + 101)                                        # FIX P0-1: seed BEFORE the constructor
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)

        def arm_logits(spec, mask):
            Xfull, mu, sd, drop = spec
            Xe = ((clip(Xfull[mask]) - mu) / sd).astype(np.float32)
            require_finite("input", Xe)
            lg = P8R.logits_at("proposed", m, Xe, groups, drop)
            require_finite("logits", lg)
            return lg

        def record(arm, spec, me, Tc, thr):
            p = softmax(arm_logits(spec, me) / Tc, axis=1)
            require_finite(f"prob[{arm}]", p)
            corr_e = p.argmax(1) == y_te[me]
            j, sel, cov = overall_metrics(corr_e, p.max(1), comp_all[me], thr)
            rows[arm].append((seed, ss, j)); covs[arm].append(cov); sels[arm].append(sel)
            pacc[arm].append(float(corr_e.mean()) * 100); cacc[arm].append(comp_equal_acc(corr_e, comp_all[me]))

        def record_normalized(arm, Xnorm, drop, me, Tc, thr):
            """As record(), but Xnorm is ALREADY on the model's training z-score scale (a transport map
            produced it), so we bypass arm_logits' (x-mu)/sd. Same (Tc, thr) from the clean source
            calibration as every other arm -- only the deployment input's standardisation differs."""
            require_finite(f"input[{arm}]", Xnorm)
            lg = P8R.logits_at("proposed", m, Xnorm.astype(np.float32), groups, drop)
            require_finite(f"logits[{arm}]", lg)
            p = softmax(lg / Tc, axis=1)
            require_finite(f"prob[{arm}]", p)
            corr_e = p.argmax(1) == y_te[me]
            j, sel, cov = overall_metrics(corr_e, p.max(1), comp_all[me], thr)
            rows[arm].append((seed, ss, j)); covs[arm].append(cov); sels[arm].append(sel)
            pacc[arm].append(float(corr_e.mean()) * 100); cacc[arm].append(comp_equal_acc(corr_e, comp_all[me]))

        for ss in splits:
            mt, mc, me = P8R.split_test_rois(comp_all, ss)
            Tc = fit_temperature(arm_logits(ARMS["clean"], mt), y_te[mt])
            pc = softmax(arm_logits(ARMS["clean"], mc) / Tc, axis=1)
            require_finite("calib-prob", pc)
            corr_c = pc.argmax(1) == y_te[mc]
            thr = float(conformal_risk_control(corr_c, pc.max(1), corr_c, pc.max(1), alpha=ALPHA,
                                               calib_group=comp_all[mc], eval_group=comp_all[mc])["threshold"])
            if not np.isfinite(thr):
                raise RuntimeError("CRC selected a non-finite threshold (control only by full abstention)")
            for arm in ARMS:
                record(arm, ARMS[arm], me, Tc, thr)
            # disjoint product-aware: L2A stats from the CALIBRATION components, applied to EVAL components
            mu_d, sd_d = band_stats("L2A-calib", Xl2a_c[mc], keep_b)
            record("L2A_prod_disj", (X_l2a, mu_d, sd_d, DROP_L2A), me, Tc, thr)
            # richer LABEL-FREE, EVAL-DISJOINT normalisation: can a stronger transport (beyond mean/std)
            # push the disjoint arm to <=alpha? calibration L2A is the reference; L1C-train is the target.
            Xrob = robust_transport(Xl2a_c[me], Xl2a_c[mc], Xtr_c, mu_tr, sd_tr, keep_b)
            record_normalized("L2A_prod_disj_robust", Xrob, DROP_L2A, me, Tc, thr)
            Xqm = quantile_match(Xl2a_c[me], Xl2a_c[mc], Xtr_c, mu_tr, sd_tr, keep_b)
            record_normalized("L2A_prod_disj_quantile", Xqm, DROP_L2A, me, Tc, thr)
        cur = {s: np.mean([r[2] for r in rows[s] if r[0] == seed]) for s in ("clean", "L2A_src", "L2A_prod")}
        print(f"  seed {seed}: clean {cur['clean']:.1f}  L2A_src {cur['L2A_src']:.1f}  "
              f"L2A_prod {cur['L2A_prod']:.1f}", flush=True)

    # --- aggregate: proper t df + balanced-grid assert (two_way_se silently drops missing/NaN cells) ---
    df = min(len(set(seeds)), len(set(splits))) - 1
    if df < 1:
        raise ValueError("need >= 2 model seeds AND >= 2 split seeds for two-way inference")
    tcrit = float(student_t.ppf(0.975, df))
    expected = {(s, r) for s in seeds for r in splits}
    agg = {}
    for st in trk:
        if {(s, r) for s, r, _ in rows[st]} != expected:
            raise ValueError(f"arm {st}: unbalanced/incomplete grid ({len(rows[st])}/{len(expected)} cells)")
        mm, se = two_way_se(rows[st])
        agg[st] = (mm, se)
        print(f"  {st:14s} joint {mm:6.2f} +/- {se:.2f}  [{mm - tcrit * se:5.1f},{mm + tcrit * se:5.1f}]  "
              f"cov {np.mean(covs[st]):4.0f}%  sel {np.mean(sels[st]):4.1f}%  "
              f"acc(px {np.mean(pacc[st]):4.1f} / comp {np.mean(cacc[st]):4.1f})", flush=True)

    print("\n  PAIRED differences (mean +/- two-way SE [95% t CI], same seed x split cells):")
    for a, b, label in [
        ("L2A_src", "L2A_prod_disj", "PRIMARY operational label-free repair (stale - product-aware disjoint)"),
        ("L2A_prod", "L2A_prod_disj", "transductive advantage (full-test - disjoint; expect ~0)"),
        ("L2A_src", "L2A_pairedL1C", "train->test COMPOSITION share (stale - L1C-test stats)"),
        ("L2A_pairedL1C", "L2A_prod", "PRODUCT-normalization share (L1C-test - L2A-test stats)"),
        ("clean_selfnorm", "clean", "generic test-batch renorm on the in-domain case (expect ~0)"),
        ("L2A_src", "L2A_prod", "total fix (stale - product-aware transductive)"),
        ("L2A_prod_disj", "L2A_prod_disj_robust", "richer: robust 2-moment transport vs mean/std disjoint"),
        ("L2A_prod_disj", "L2A_prod_disj_quantile", "richer: full quantile transport vs mean/std disjoint")]:
        md, se, lo, hi = paired_delta(rows[a], rows[b], seeds, splits)
        z0 = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"    d[{a} - {b}] = {md:+6.2f} +/- {se:.2f}  [{lo:+5.1f},{hi:+5.1f}]  ({z0})  {label}", flush=True)

    # --- persist per-cell metrics + a machine-readable summary (audit / paper numbers) ---
    per_cell = []
    for st in trk:
        for (s, r, j), cov, sel, pa, ca in zip(rows[st], covs[st], sels[st], pacc[st], cacc[st]):
            per_cell.append(dict(arm=st, model_seed=int(s), split_seed=int(r), joint=j, coverage=cov,
                                 selective=sel, pixel_acc=pa, component_acc=ca))
    pd.DataFrame(per_cell).to_csv(args.out + "_percell.csv", index=False)
    summary = {"alpha": ALPHA, "df": df, "tcrit": tcrit,
               "arms": {st: {"joint_mean": agg[st][0], "joint_se": agg[st][1],
                             "coverage": float(np.mean(covs[st])), "selective": float(np.mean(sels[st])),
                             "pixel_acc": float(np.mean(pacc[st])), "component_acc": float(np.mean(cacc[st]))}
                        for st in trk}}
    with open(args.out + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    dj = agg["L2A_prod_disj"]
    # does ANY label-free, eval-disjoint transport bring the UPPER CI at/below the alpha target?
    tgt = ALPHA * 100
    disj_arms = [(st, agg[st][0], agg[st][1], agg[st][0] + tcrit * agg[st][1])
                 for st in ("L2A_prod_disj", "L2A_prod_disj_robust", "L2A_prod_disj_quantile")]
    best = min(disj_arms, key=lambda t: t[1])
    controlled = [st for st, m, se, hi in disj_arms if hi <= tgt]
    print(f"\n  SUMMARY (data, not an auto-verdict): stale L2A joint {agg['L2A_src'][0]:.1f}; product-aware "
          f"DISJOINT (mean/std) {dj[0]:.1f} +/- {dj[1]:.1f} at {np.mean(covs['L2A_prod_disj']):.0f}% coverage. "
          f"Richer label-free eval-disjoint transports -> "
          + ", ".join(f"{st.split('disj_')[-1] if 'disj_' in st else 'meanstd'} {m:.2f}(hi {hi:.1f})"
                      for st, m, se, hi in disj_arms) + f". Target {tgt:.0f}%. "
          + (f"CONTROL ACHIEVED (upper CI <= target) by: {controlled} -> 'restores control' is defensible."
             if controlled else
             f"NO arm's upper CI reaches <= target (best = {best[0]} at {best[1]:.2f}, hi {best[3]:.1f}); "
             "the honest claim is 'reduces to near-target, residual above alpha'.") +
          " Label-free repair restores the empirical LEVEL but does NOT re-derive a finite-sample guarantee.")
    print(f"  wrote {args.out}_percell.csv + {args.out}_summary.json")


if __name__ == "__main__":
    main()
