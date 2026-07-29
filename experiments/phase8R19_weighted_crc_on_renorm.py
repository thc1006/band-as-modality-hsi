#!/usr/bin/env python
"""Phase 8R19 -- does product-aware re-normalization make the formal weighted CRC useful again?

A 2x2 factorial on the identical clean-source certificate: {source-normalized, quantile-re-normalized L2A}
x {uniform, plug-in estimated-weight CRC}. It isolates four things a single 4-arm table cannot: the effect
of re-normalization under a uniform certificate, the effect of weighting on each representation, and---the
question of interest---the INTERACTION (does re-normalization turn weighting from useless into useful?).

Publication-grade guards (a max-standard review of the first draft demanded all of these):
 * per-row softmax reused from phase8R11 (NOT scipy's flatten-default), imported by name;
 * mid-distribution empirical CDF for the quantile map (the reflectance is ~98% tied, so np.interp on the raw
   sorted array uses an ill-defined first-rank convention) + reported endpoint-clamp rates;
 * FOUR disjoint scene-component roles per split -- temperature / CRC-calibration / target-reference / eval --
   so the re-normalization reference is disjoint from the CRC calibration set, not just from evaluation;
 * held-out domain AUROC (classifier fit on the temperature fold, scored on a disjoint fold), reported next to
   the in-sample AUROC, since in-sample AUROC over-states separability;
 * the formal weighted CRC returns a FEASIBILITY flag; an infeasible test unit (alpha*W - w*(1-alpha) < 0) is a
   deployment abstain-all fallback, NOT a certified operating point -- we report the feasible fraction and the
   conditional-on-feasible risk/coverage separately from the fallback-policy mean;
 * calibration ESS uses the CALIBRATION weights (evaluation ESS reported separately);
 * a strict balanced-grid check fails closed on any missing/non-finite (seed,split) cell;
 * paired two-way-cluster-robust contrasts (joint AND coverage) for the four factorial effects.

This is NOT numerically identical to phase8R11 (the flagship dumps lack the input reflectance needed to
re-normalize, so R19 trains its own model under a clipped pipeline); the source arm is a consistency check,
not a reproduction. Plug-in weights inherit no exact finite-sample guarantee, and the re-normalization is
estimated from deployment data -- this measures usable control on the deployment scenes, not a re-derived
certificate. Emits DATA only; every printed reading is computed from the run, none hardcoded.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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

from phase8R11_weighted_crc_formal import softmax as row_softmax   # per-row; NOT scipy flatten-default
band_stats, ALPHA, CLIP = R10.band_stats, R11.ALPHA, R11.CLIP
state_arrays, make_grid, ess_frac = R11.state_arrays, R11.make_grid, R11.ess_frac


def require_finite(name, x):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        bad = np.argwhere(~np.isfinite(x))[:5]
        raise FloatingPointError(f"{name}: non-finite at {bad.tolist()}")


def empirical_cdf_knots(x):
    """Unique values + mid-distribution CDF  P(X<v)+0.5 P(X=v)  -- a proper ECDF that is well defined under
    the heavy value-ties of quantised reflectance (np.interp on the raw sorted array is first-rank)."""
    x = np.asarray(x, np.float64)
    require_finite("cdf-sample", x)
    v, c = np.unique(x, return_counts=True)
    if v.size < 2:
        raise ValueError("degenerate (constant) band for quantile map")
    cdf = (np.cumsum(c) - 0.5 * c) / x.size
    return v, cdf


def quantile_map(Xeval, Xref, Xsrc, mu_tr, sd_tr, keep):
    """Per-band monotone quantile transport eval-L2A -> L1C-train marginal via the reference-L2A ECDF, then
    z-score with the train stats (dropped bands fall back to a train z-score, as in phase8R10). Returns the
    normalized input and, per band, the fraction of eval values clamped to the reference extremes."""
    out = np.zeros_like(Xeval, np.float64)
    clamp_lo = np.zeros(Xeval.shape[1]); clamp_hi = np.zeros(Xeval.shape[1])
    for b in range(Xeval.shape[1]):
        if b not in keep:
            out[:, b] = (Xeval[:, b] - mu_tr[b]) / sd_tr[b]; continue
        cv, cc = empirical_cdf_knots(Xref[:, b])
        sv, sc = empirical_cdf_knots(Xsrc[:, b])
        clamp_lo[b] = float(np.mean(Xeval[:, b] < cv[0])); clamp_hi[b] = float(np.mean(Xeval[:, b] > cv[-1]))
        u = np.interp(Xeval[:, b], cv, cc, left=cc[0], right=cc[-1])   # eval value -> reference CDF level
        mapped = np.interp(u, sc, sv, left=sv[0], right=sv[-1])        # CDF level -> source quantile
        out[:, b] = (mapped - mu_tr[b]) / sd_tr[b]
    require_finite("quantile-map", out)
    return out.astype(np.float32), float(clamp_lo[keep].mean() * 100), float(clamp_hi[keep].mean() * 100)


def wcrc(Lc, wc, grid, Le, COVe, we, alpha):
    """Formal weighted CRC with the test-point term, returning FEASIBILITY. Infeasible (rhs<0) => no bound =>
    deployment abstain-all fallback (realized 0, cover 0, feasible False), reported apart from the mean."""
    W = wc.sum(); G = (wc[:, None] * Lc).sum(0)
    realized = np.zeros(len(we)); cover = np.zeros(len(we)); feas = np.zeros(len(we), bool)
    for x in range(len(we)):
        rhs = alpha * W - we[x] * (1.0 - alpha)
        ok = np.flatnonzero(G <= rhs)
        if rhs < 0 or ok.size == 0:                          # infeasible -> uncertified fallback
            continue
        g = ok[0]; realized[x] = Le[x, g]; cover[x] = COVe[x, g]; feas[x] = True
    return realized, cover, feas


def domain_weights_heldout(F0_tr, F1_tr, F0_ho, F1_ho, Fq_list):
    """Fit clean-vs-target logistic weights on the TRAINING fold (F*_tr); report both in-sample AUROC and a
    HELD-OUT AUROC (scored on the disjoint F*_ho fold); return clipped odds for each query set."""
    X = np.vstack([F0_tr, F1_tr]); yb = np.r_[np.zeros(len(F0_tr)), np.ones(len(F1_tr))]
    sc = StandardScaler().fit(X); clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), yb)
    auc_tr = roc_auc_score(yb, clf.predict_proba(sc.transform(X))[:, 1])
    Xho = np.vstack([F0_ho, F1_ho]); yho = np.r_[np.zeros(len(F0_ho)), np.ones(len(F1_ho))]
    auc_ho = roc_auc_score(yho, clf.predict_proba(sc.transform(Xho))[:, 1]) if len(set(yho)) == 2 else float("nan")
    out = [np.clip(clf.predict_proba(sc.transform(Fq))[:, 1] / (1 - clf.predict_proba(sc.transform(Fq))[:, 1] + 1e-12),
                   1.0 / CLIP, CLIP) for Fq in Fq_list]
    return out, float(auc_tr), float(auc_ho)


def four_way_split(comp, seed):
    """Four DISJOINT scene-component sets -> boolean pixel masks: temperature / CRC-calibration /
    target-reference / evaluation. Re-normalization uses only 'ref' (disjoint from crc-cal and eval)."""
    uro = np.unique(comp)
    if len(uro) < 8:
        raise ValueError(f"need >= 8 components for a 4-way split, got {len(uro)}")
    perm = np.random.default_rng(90000 + seed).permutation(uro)
    n = len(perm); a, b, c = int(0.15 * n), int(0.15 * n) + int(0.35 * n), int(0.15 * n) + int(0.35 * n) + int(0.25 * n)
    sets = (perm[:a], perm[a:b], perm[b:c], perm[c:])
    for s in sets:
        if len(s) < 2:
            raise ValueError("a 4-way fold is too small")
    return tuple(np.isin(comp, s) for s in sets)


def validate_grid(rows, seeds, splits, name):
    cells = {}
    for s, r, v in rows:
        if (s, r) in cells:
            raise ValueError(f"{name}: duplicate cell {(s, r)}")
        if not np.isfinite(v):
            raise FloatingPointError(f"{name}: non-finite at seed={s} split={r}")
        cells[(s, r)] = v
    want = {(s, r) for s in seeds for r in splits}
    if set(cells) != want:
        raise ValueError(f"{name}: incomplete grid, missing {sorted(want - set(cells))[:5]}")
    return [(s, r, cells[(s, r)]) for s in seeds for r in splits]


def agg(rows, seeds, splits, tcrit, name):
    trip = validate_grid(rows, seeds, splits, name)
    mm, se = two_way_se(trip)
    return {"mean": mm, "se": se, "lo": mm - tcrit * se, "hi": mm + tcrit * se}


def paired(rows_a, rows_b, seeds, splits, tcrit):
    da = {(s, r): v for s, r, v in rows_a}; db = {(s, r): v for s, r, v in rows_b}
    d = [(s, r, da[(s, r)] - db[(s, r)]) for s in seeds for r in splits]
    mm, se = two_way_se(d)
    return {"mean": mm, "se": se, "lo": mm - tcrit * se, "hi": mm + tcrit * se}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8R19_weighted_crc_on_renorm"))
    args = ap.parse_args()
    seeds, splits = args.seeds, args.split_seeds
    if not seeds or not splits:
        raise ValueError("need >=1 seed and >=1 split")
    for nm, vv in (("seeds", seeds), ("splits", splits)):
        if len(set(vv)) != len(vv):
            raise ValueError(f"duplicate {nm}")
    if args.epochs < 1:
        raise ValueError("--epochs >= 1")
    df = min(len(set(seeds)), len(set(splits))) - 1
    if df < 2:
        raise ValueError("need >=3 clusters in each crossed dimension for two-way inference")
    tcrit = float(student_t.ppf(0.975, df))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    hw.setup(deterministic=True, prefer=args.device)

    groups = P8.s2_physical_groups(); cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10"); DROP = [g_b10]
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    if meta["s2_id"].isna().any():
        raise ValueError("NaN s2_id")
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    ids = np.flatnonzero(~meta["s2_id"].isin(train_prod).to_numpy())

    def load(p):
        return P8.load_split("test", p, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=True, return_pixel_index=True)
    X_l1c, y_te, pid, pix = load("L1C"); X_l2a, y_l2a, p2, x2 = load("L2A")
    np.testing.assert_array_equal(pid, p2); np.testing.assert_array_equal(pix, x2); np.testing.assert_array_equal(y_te, y_l2a)
    comp = P8R.scene_component_ids("test")[pid]
    Xtr_l1c, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    ytr, y_te = ytr.astype(np.int64), y_te.astype(np.int64)
    for nm, A in [("X_l1c", X_l1c), ("X_l2a", X_l2a), ("Xtr", Xtr_l1c)]:
        require_finite(nm, A)
    clip = lambda A: np.clip(A, -0.1, 1.6)
    keep = [i for i in range(X_l1c.shape[1]) if i != P8.B10_IDX]
    Xtr_c, Xl2a_c = clip(Xtr_l1c), clip(X_l2a)
    mu_tr, sd_tr = band_stats("L1C-train", Xtr_c, keep)
    Xtr_n = ((Xtr_c - mu_tr) / sd_tr).astype(np.float32)
    Xl1c_n = ((clip(X_l1c) - mu_tr) / sd_tr).astype(np.float32)
    Xl2a_src_n = ((Xl2a_c - mu_tr) / sd_tr).astype(np.float32)
    for nm, A in [("Xtr_n", Xtr_n), ("Xl1c_n", Xl1c_n), ("Xl2a_src_n", Xl2a_src_n)]:
        require_finite(nm, A)
    print(f"  eval {len(y_te)} px / {len(np.unique(comp))} components; {len(seeds)} seeds x {len(splits)} splits; df={df}",
          flush=True)

    bs = P2.auto_bs(Xtr_n.shape[0])
    ARMS = ["src_uniform", "src_weighted", "ren_uniform", "ren_weighted"]
    joint = {a: [] for a in ARMS}; cov = {a: [] for a in ARMS}; cjoint = {a: [] for a in ARMS}
    ccov = {a: [] for a in ARMS}; feas = {a: [] for a in ARMS}
    diag = {k: [] for k in ["src_auc_tr", "src_auc_ho", "ren_auc_tr", "ren_auc_ho",
                            "src_calib_ess", "src_eval_ess", "ren_calib_ess", "ren_eval_ess",
                            "ren_clamp_lo", "ren_clamp_hi", "Tc"]}
    percell = []

    for seed in seeds:
        hw.seed_model(seed)
        m = GroupedCrossBandAttention(groups, cwl, 4)
        P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
        P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)
        lc = P8R.logits_at("proposed", m, Xl1c_n, groups, []); require_finite("clean logits", lc)
        ll = P8R.logits_at("proposed", m, Xl2a_src_n, groups, DROP); require_finite("L2A logits", ll)

        for ss in splits:
            temp, ccal, ref, ev = four_way_split(comp, ss)
            Tc = fit_temperature(lc[temp], y_te[temp])
            if np.isclose(Tc, 1e-3) or np.isclose(Tc, 1e3):
                raise RuntimeError(f"degenerate temperature Tc={Tc} seed={seed} split={ss}")
            grid = make_grid(row_softmax(lc[ccal] / Tc).max(1))
            Lc, COVc, _, Fc, _ = state_arrays(lc, y_te, comp, ccal, Tc, grid)          # CLEAN CRC calibration
            _, _, _, F0_tr, _ = state_arrays(lc, y_te, comp, temp, Tc, grid)           # source clf-train fold
            _, _, _, F0_ho, _ = state_arrays(lc, y_te, comp, ref, Tc, grid)            # source clf held-out fold

            # re-normalize on the REFERENCE fold only (disjoint from ccal AND ev)
            Xren, clo, chi = quantile_map(Xl2a_c, Xl2a_c[ref], Xtr_c, mu_tr, sd_tr, keep)
            lr = P8R.logits_at("proposed", m, Xren, groups, DROP); require_finite("renorm logits", lr)

            for tag, lt in [("src", ll), ("ren", lr)]:
                Le, COVe, _, Fe, _ = state_arrays(lt, y_te, comp, ev, Tc, grid)
                _, _, _, F1_tr, _ = state_arrays(lt, y_te, comp, temp, Tc, grid)
                _, _, _, F1_ho, _ = state_arrays(lt, y_te, comp, ref, Tc, grid)
                (wc, we), auc_tr, auc_ho = domain_weights_heldout(F0_tr, F1_tr, F0_ho, F1_ho, [Fc, Fe])
                for wname, w_c, w_e in [("uniform", np.ones(len(wc)), np.ones(len(we))), ("weighted", wc, we)]:
                    r, c, fe = wcrc(Lc, w_c, grid, Le, COVe, w_e, ALPHA)
                    arm = f"{tag}_{wname}"
                    jm = float(np.mean(r) * 100); cm = float(np.mean(c) * 100); fr = float(np.mean(fe) * 100)
                    cj = float(np.mean(r[fe]) * 100) if fe.any() else float("nan")
                    cc = float(np.mean(c[fe]) * 100) if fe.any() else float("nan")
                    joint[arm].append((seed, ss, jm)); cov[arm].append((seed, ss, cm))
                    feas[arm].append((seed, ss, fr)); cjoint[arm].append((seed, ss, cj)); ccov[arm].append((seed, ss, cc))
                    percell.append(dict(seed=seed, split=ss, arm=arm, representation=tag, weighting=wname,
                                        joint_risk=jm, coverage=cm, feasible_frac=fr, cond_joint=cj, cond_cov=cc,
                                        temperature=float(Tc)))
                diag[f"{tag}_auc_tr"].append(auc_tr); diag[f"{tag}_auc_ho"].append(auc_ho)
                diag[f"{tag}_calib_ess"].append(float(ess_frac(wc))); diag[f"{tag}_eval_ess"].append(float(ess_frac(we)))
            diag["ren_clamp_lo"].append(clo); diag["ren_clamp_hi"].append(chi); diag["Tc"].append(float(Tc))
        print(f"  seed {seed}: src_u {np.mean([v for s,_,v in joint['src_uniform'] if s==seed]):.1f}@"
              f"{np.mean([v for s,_,v in cov['src_uniform'] if s==seed]):.0f}  "
              f"src_w {np.mean([v for s,_,v in joint['src_weighted'] if s==seed]):.1f}@"
              f"{np.mean([v for s,_,v in cov['src_weighted'] if s==seed]):.0f}  "
              f"ren_u {np.mean([v for s,_,v in joint['ren_uniform'] if s==seed]):.1f}@"
              f"{np.mean([v for s,_,v in cov['ren_uniform'] if s==seed]):.0f}  "
              f"ren_w {np.mean([v for s,_,v in joint['ren_weighted'] if s==seed]):.1f}@"
              f"{np.mean([v for s,_,v in cov['ren_weighted'] if s==seed]):.0f}", flush=True)

    print(f"\n  two-way SE df={df} tcrit {tcrit:.3f}; ALPHA {ALPHA*100:.0f}%  (joint% @ cov%, feasible%)")
    summary = {"alpha": ALPHA, "df": df, "tcrit": tcrit, "arms": {}, "contrasts": {},
               "diagnostics": {k: float(np.nanmean(v)) for k, v in diag.items()},
               "provenance": {"args": vars(args)}}
    try:
        summary["provenance"]["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE).decode().strip()
    except Exception:
        summary["provenance"]["git"] = "unknown"
    for a in ARMS:
        j = agg(joint[a], seeds, splits, tcrit, a + ".joint"); c = agg(cov[a], seeds, splits, tcrit, a + ".cov")
        f = agg(feas[a], seeds, splits, tcrit, a + ".feas")
        summary["arms"][a] = {"joint": j, "coverage": c, "feasible_frac": f,
                              "cond_joint": float(np.nanmean([v for _, _, v in cjoint[a]])),
                              "cond_cov": float(np.nanmean([v for _, _, v in ccov[a]]))}
        print(f"  {a:14s} joint {j['mean']:5.2f} [{j['lo']:5.1f},{j['hi']:5.1f}]  cov {c['mean']:4.0f} "
              f"[{c['lo']:4.0f},{c['hi']:4.0f}]  feasible {f['mean']:4.0f}%  (cond {summary['arms'][a]['cond_joint']:.1f}@"
              f"{summary['arms'][a]['cond_cov']:.0f})", flush=True)

    print("\n  PAIRED factorial contrasts (mean [two-way t CI]):")
    for label, a, b in [("renorm|uniform (ren_u - src_u)", "ren_uniform", "src_uniform"),
                        ("weight|source (src_w - src_u)", "src_weighted", "src_uniform"),
                        ("weight|renorm (ren_w - ren_u)", "ren_weighted", "ren_uniform")]:
        pj = paired(joint[a], joint[b], seeds, splits, tcrit); pc = paired(cov[a], cov[b], seeds, splits, tcrit)
        summary["contrasts"][label] = {"joint": pj, "coverage": pc}
        print(f"    {label:34s} joint {pj['mean']:+5.2f} [{pj['lo']:+5.1f},{pj['hi']:+5.1f}]  "
              f"cov {pc['mean']:+5.1f} [{pc['lo']:+5.1f},{pc['hi']:+5.1f}]")
    # interaction: (ren_w-ren_u) - (src_w-src_u)
    inter = [(s, r, (dict(((x, y), v) for x, y, v in joint["ren_weighted"])[(s, r)]
                     - dict(((x, y), v) for x, y, v in joint["ren_uniform"])[(s, r)])
                    - (dict(((x, y), v) for x, y, v in joint["src_weighted"])[(s, r)]
                       - dict(((x, y), v) for x, y, v in joint["src_uniform"])[(s, r)]))
             for s in seeds for r in splits]
    im, ise = two_way_se(inter)
    summary["contrasts"]["interaction (weight effect: renorm - source)"] = {"mean": im, "se": ise,
                                                                            "lo": im - tcrit * ise, "hi": im + tcrit * ise}
    print(f"    {'interaction (weight: ren - src)':34s} joint {im:+5.2f} [{im-tcrit*ise:+5.1f},{im+tcrit*ise:+5.1f}]  "
          "(<0 => weighting helps MORE on the re-normalized representation)")
    d = summary["diagnostics"]
    print(f"\n  diagnostics: domain AUROC train {d['src_auc_tr']:.3f}->{d['ren_auc_tr']:.3f}  "
          f"held-out {d['src_auc_ho']:.3f}->{d['ren_auc_ho']:.3f}; calib ESS {d['src_calib_ess']*100:.0f}%->"
          f"{d['ren_calib_ess']*100:.0f}% (eval ESS {d['src_eval_ess']*100:.0f}%->{d['ren_eval_ess']*100:.0f}%); "
          f"renorm clamp lo/hi {d['ren_clamp_lo']:.2f}/{d['ren_clamp_hi']:.2f}%")
    print("  (reading is the paired contrasts + held-out AUROC above; nothing here is hardcoded.)")

    json.dump(summary, open(args.out + "_summary.json", "w"), indent=1)
    pd.DataFrame(percell).to_csv(args.out + "_percell.csv", index=False)
    print(f"\n  wrote {args.out}_summary.json + _percell.csv")


if __name__ == "__main__":
    main()
