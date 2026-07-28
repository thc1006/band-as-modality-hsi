#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8R2 — a SECOND operational shift axis for the reliability result: a SURFACE-DOMAIN-GAP shift on
real Sentinel-2 (CloudSEN12), same cloud-segmentation task as the phase8R atmospheric flagship.

WHY (and why NOT the naive land-cover version). phase8R shows naive conformal COVERAGE fails under the
L1C->L2A ATMOSPHERIC-CORRECTION shift; a single axis is a single diagnosis. A first attempt -- train
globally, then stratify the TEST set by ESA WorldCover land cover -- was too weak: a model that has
already seen every surface in training is well enough calibrated across them that naive conformal
barely breaks (the smoke confirmed it). The flagship's shift bites precisely because L2A is an input
regime the model NEVER trained on. So this axis reproduces that condition with a surface DOMAIN GAP:

  TRAIN + naive-calibrate on DARK surfaces (vegetation, cropland, water) only;
  DEPLOY on BRIGHT surfaces (built, bare soil, snow/ice) the model has NEVER seen.

Bright, high-albedo surfaces are THE operational cloud-confusion case, so a detector trained where
labels are plentiful (vegetated land) and pushed onto snow/desert/urban is a real deployment, and a
different shift TYPE from atmospheric correction -- exactly the generalisation the reliability law needs.

DESIGN (conformal core bandsim/reliability.py reused verbatim; unit + SE fixes mirror phase8R):
  * naive    -- temperature + CRC threshold calibrated on held-out DARK (source) test units, evaluated
                on BRIGHT (target) test units: a source-calibrated detector deployed on an unseen surface.
  * mondrian -- temperature + CRC threshold calibrated WITHIN BRIGHT, evaluated on a disjoint BRIGHT set.
Both arms evaluate on the SAME bright eval units, so the contrast is purely source- vs target-calibration.
Exchangeable unit = SCENE-CONNECTED COMPONENT (P0-2: ROIs sharing any s2_id unioned), on the split AND
the CRC grouping. SE over the crossed (split_seed x model_seed) design is two-way cluster-robust (P0-3).
Temperature is fitted on units DISJOINT from the CRC calibration units (phase8R's exchangeability lesson).

READ WITH COVERAGE (a certified JOINT mass goes to 0 by abstaining): naive_group_coverage /
naive_group_joint_risk on bright = what a dark-calibrated threshold achieves on the unseen surface;
mondrian_group_* = what recalibrating on bright recovers. The estimand is the MEAN over independent
(split_seed, model_seed) draws; raw per-run rows are written.

HONESTY: this is a covariate/domain-gap shift, a SECOND axis, not the atmospheric certificate; the model
is genuinely less accurate on the unseen surface, so the naive breach mixes accuracy loss with
calibration shift -- that is the operational reality of deploying on an unseen surface, reported as such.

Outputs (../paper/): results_phase8R2_landcover.csv + _raw.csv. --smoke writes *_smoke only.
Usage: python experiments/phase8R2_landcover_reliability.py --split-seeds 0 1 2 3 4 --model-seeds 0 1 2
"""
import os, sys, csv, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R           # logits_at / METHODS / NUM_CLASSES / P()
from bandsim.grouping import group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention
from bandsim.reliability import confidence_msp, fit_temperature, conformal_risk_control, aurc, selective_auroc
from bandsim import hw, parallel
from bandsim.provenance import stamp

TARGET_RISK = P8R.TARGET_RISK
NUM_CLASSES = P8R.NUM_CLASSES
METHODS = P8R.METHODS

# ESA WorldCover code -> {dark source (trained on) | bright target (unseen, deployed on)}. bright groups
# the high-albedo surfaces that are the operational cloud-confusion case (built 50, bare 60, snow/ice 70).
DARK_CODES = {10, 20, 30, 40, 80, 90, 100}     # tree/shrub/grass/crop/water/wetland/moss
BRIGHT_CODES = {50, 60, 70}                     # built / bare / snow-ice
TEMP_FRAC, CALIB_FRAC = 0.25, 0.375             # three-way ROI budget (same as phase8R)


def _landcover_of_patch(split):
    import pandas as pd
    meta = pd.read_csv(os.path.join(P8.DATA, split, "metadata.csv"))
    if "land_cover" not in meta.columns:
        raise KeyError(f"{split}/metadata.csv has no land_cover column")
    return meta["land_cover"].to_numpy()


def _is_bright(codes):
    return np.array([int(c) in BRIGHT_CODES for c in codes], dtype=bool)


# The shift axis picks which patches are the TARGET (deployment, held out of training) vs the SOURCE
# (trained + naive-calibrated on). `surface`: TARGET = bright/high-albedo land cover (cloud-confusion
# case). `geography`: TARGET = the Southern Hemisphere (train on the Northern, a spatial domain gap --
# the canonical "labelled where you have data, deployed elsewhere" scenario, a shift TYPE distinct from
# surface AND from atmospheric correction). Both reuse the identical source-vs-target-calibration machinery.
AXIS = {"surface":   {"stem": "landcover", "source": "dark", "target": "bright"},
        "geography":  {"stem": "geography", "source": "northern", "target": "southern"}}


def _target_mask(axis, split, patch_ids):
    """Boolean is_TARGET per patch (in patch_ids order) for the chosen shift axis."""
    if axis == "surface":
        return _is_bright(_landcover_of_patch(split)[patch_ids])
    if axis == "geography":
        import pandas as pd, re
        meta = pd.read_csv(os.path.join(P8.DATA, split, "metadata.csv"))

        def _lat(s):
            m = re.search(r"POINT\s*\(([-\d.]+)\s+([-\d.]+)\)", str(s))
            return float(m.group(2)) if m else np.nan
        lat = meta["proj_centroid"].map(_lat).to_numpy()[patch_ids]
        return lat < 0.0                         # Southern Hemisphere = TARGET (train on Northern)
    raise ValueError(f"unknown --axis {axis!r} (surface | geography)")


def _split_units(units, split_seed, salt, n_parts):
    """Disjoint unit sets (n_parts of them) from an array of exchangeable-unit ids."""
    uni = np.unique(units)
    if len(uni) < n_parts:
        return None
    perm = np.random.default_rng(90000 + 1000 * salt + split_seed).permutation(uni)
    if n_parts == 3:
        n_t = max(1, int(round(TEMP_FRAC * len(perm))))
        n_c = max(1, int(round(CALIB_FRAC * len(perm))))
        if n_t + n_c >= len(perm):
            return None
        return perm[:n_t], perm[n_t:n_t + n_c], perm[n_t + n_c:]
    else:  # 2-way temp/calib for the source (dark) side
        n_t = max(1, int(round(0.4 * len(perm))))
        if n_t >= len(perm):
            return None
        return perm[:n_t], perm[n_t:]


def _mask(unit, keep):
    return np.isin(unit, keep)


def domain_gap_reliability(kind, model, X, y, unit, is_target, split_seed, target, groups):
    """One trained model (trained on the SOURCE domain only): naive (source-calibrated) vs mondrian
    (target-calibrated) CRC on the unseen TARGET eval units. Axis-generic: for --axis surface the
    target is bright land cover, for --axis geography it is the Southern hemisphere. NOTE: the output
    fields keep the historical names acc_bright/aurc_bright/auroc_bright, but they hold the
    TARGET-domain accuracy/AURC/AUROC for whichever axis is run (not specifically 'bright')."""
    lg = P8R.logits_at(kind, model, X, groups, [])              # clean L1C; the shift is WHERE deployed, not the bands
    corr = (lg.argmax(1) == y).astype(int)
    src_u = np.unique(unit[~is_target]); tgt_u = np.unique(unit[is_target])
    sp_b = _split_units(unit[is_target], split_seed, salt=1, n_parts=3)   # temp_b, calib_b, eval_b
    sp_d = _split_units(unit[~is_target], split_seed, salt=2, n_parts=2)  # temp_d, calib_d
    if sp_b is None or sp_d is None:
        return []
    tB, cB, eB = sp_b
    tD, cD = sp_d

    def crc_arm(temp_u, calib_u, eval_u):
        mt = _mask(unit, temp_u)
        T = fit_temperature(lg[mt], y[mt])
        mc, me = _mask(unit, calib_u), _mask(unit, eval_u)
        cf_c = confidence_msp(lg[mc] / T); cf_e = confidence_msp(lg[me] / T)
        crc = conformal_risk_control(corr[mc], cf_c, corr[me], cf_e, alpha=target,
                                     calib_group=unit[mc], eval_group=unit[me])
        return crc, float(T)

    me_eval = _mask(unit, eB)
    base = {"acc_bright": float(corr[me_eval].mean()) * 100,
            "aurc_bright": aurc(corr[me_eval], confidence_msp(lg[me_eval])) * 100,
            "auroc_bright": selective_auroc(corr[me_eval], confidence_msp(lg[me_eval])) * 100,
            "n_source_units": int(src_u.size), "n_target_units": int(tgt_u.size),
            "n_eval_units": int(np.unique(unit[me_eval]).size), "n_eval_px": int(me_eval.sum())}
    naive_crc, T_naive = crc_arm(tD, cD, eB)                    # SOURCE calibration -> TARGET eval (naive: deploy source-calibrated on unseen target)
    mond_crc, T_mond = crc_arm(tB, cB, eB)                      # TARGET calibration -> TARGET eval (mondrian: recalibrate on the deployment domain)
    rows = []
    for arm, crc, T in (("naive", naive_crc, T_naive), ("mondrian", mond_crc, T_mond)):
        rows.append(dict(base, arm=arm, temperature=T,
                         crc_threshold=float(crc["threshold"]),
                         crc_group_joint_risk=crc["eval_group_joint_risk"] * 100,
                         crc_group_coverage=crc["eval_group_coverage"] * 100,
                         crc_group_selective_risk=crc["eval_group_selective_risk"] * 100,
                         crc_feasible=int(crc["feasible"]),
                         n_calib_units=int(crc["n_calib_units"])))
    return rows


def run_seed(job, Xtr_src, ytr_src, X, y, unit, is_target, groups, cwl, subsample_frac, epochs, target):
    """One (split_seed, model_seed): train on a DARK-surface subsample, evaluate the domain-gap arms."""
    split_seed, model_seed = job
    rs = np.random.default_rng(model_seed)
    k = max(1, int(round(subsample_frac * Xtr_src.shape[0])))
    sub = rs.choice(Xtr_src.shape[0], size=k, replace=False)
    Xs, ys = Xtr_src[sub], ytr_src[sub]
    bs = P2.auto_bs(Xs.shape[0])
    hw.seed_model(model_seed)                                       # P0-2: seed before the B2 constructor
    m_b2 = P2.train_mlp(Xs, ys, groups, model_seed, group_dropout=True, epochs=epochs,
                        num_classes=NUM_CLASSES, bs=bs)
    hw.seed_model(model_seed)                                       # P0-2: and before the proposed model
    m_prop = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES)
    P2.pretrain_sgmae(m_prop, Xs, groups, model_seed, epochs=max(1, epochs // 2), bs=bs)
    P2.finetune_proposed(m_prop, Xs, ys, groups, model_seed, epochs=epochs, bs=bs)
    models = {"proposed": m_prop, "b2": m_b2}
    out = []
    for meth in METHODS:
        for r in domain_gap_reliability(meth, models[meth], X, y, unit, is_target, split_seed, target, groups):
            out.append(dict(r, method=meth, split_seed=split_seed, model_seed=model_seed))
    return out


def _two_way_se(a, splits, models):
    a = np.asarray(a, float); N = a.size
    if N < 2:
        return 0.0
    e = a - a.mean()
    def V(lab):
        lab = np.asarray(lab, dtype=object)
        return sum(float(e[lab == g].sum()) ** 2 for g in set(lab.tolist())) / N ** 2
    viid = float((e ** 2).sum()) / N ** 2
    core = V(splits) + V(models) - viid
    return float(np.sqrt(core if core > 0 else viid))       # iid floor: never a misleading 0 on non-constant data


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--subsample-frac", type=float, default=0.8)
    ap.add_argument("--patches-train", type=int, default=3000)
    ap.add_argument("--px-train", type=int, default=300)
    ap.add_argument("--px-test", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=TARGET_RISK)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    ap.add_argument("--axis", default="surface", choices=["surface", "geography"],
                    help="the shift axis: surface (train dark, deploy bright) or geography "
                         "(train Northern Hemisphere, deploy Southern)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-tag", default="", help="filename suffix for ALL outputs (e.g. _10seed) so a "
                    "re-run does not clobber the canonical results; must be filename-safe. --smoke forces _smoke.")
    args = ap.parse_args()
    if not (0.0 < args.subsample_frac <= 1.0) or not (0.0 < args.alpha <= 1.0):
        raise ValueError("--subsample-frac and --alpha must be in (0, 1]")
    _safe = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    if args.out_tag and (set(args.out_tag) - _safe):
        raise ValueError(f"--out-tag must be filename-safe [A-Za-z0-9_.-], got {args.out_tag!r}")

    sfx = args.out_tag
    if args.smoke:
        args.split_seeds, args.model_seeds, args.epochs = [0, 1], [0], 10
        args.patches_train, args.px_train, args.px_test = 600, 200, 300
        sfx = "_smoke"
        print("[smoke] 2 splits x 1 model seed / 10 epochs — *_smoke artefacts only")
    hw.setup(deterministic=True, prefer=args.device); print("HW:", hw.info())

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)

    src_name, tgt_name = AXIS[args.axis]["source"], AXIS[args.axis]["target"]
    print(f"loading CloudSEN12 train (filter to {src_name.upper()} = source) + test | axis={args.axis} ...")
    Xtr, ytr, ptr = P8.load_split("train", "L1C", pixels_per_patch=args.px_train,
                                  n_patches=args.patches_train, seed=12345, return_patch_id=True)
    is_tgt_tr = _target_mask(args.axis, "train", ptr)          # TRAIN pixels that are the TARGET domain
    Xtr_src, ytr_src = Xtr[~is_tgt_tr], ytr[~is_tgt_tr]        # TRAIN on the SOURCE domain ONLY

    X, y, pid = P8.load_split("test", "L1C", pixels_per_patch=args.px_test, seed=54321,
                              return_patch_id=True)
    # SCENE-LEVEL train/test separation (C6 leak-guard, mirrors phase8R): drop test pixels whose
    # patch's Sentinel-2 product also appears in TRAIN (4 products overlap the shipped CloudSEN12 splits).
    import pandas as pd
    _train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    _tmeta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    _leaked = _tmeta.index[_tmeta["s2_id"].isin(_train_prod)].to_numpy()
    _keep = ~np.isin(pid, _leaked)
    if not _keep.all():
        print(f"[leak-guard] dropping {int((~_keep).sum())} test pixels from {len(_leaked)} "
              f"train-overlap products (scene-level train/test separation)")
        X, y, pid = X[_keep], y[_keep], pid[_keep]
    unit = P8.scene_component_ids("test")[pid]                 # P0-2: scene-component unit
    is_target = _target_mask(args.axis, "test", pid)          # TEST pixels in the unseen TARGET domain

    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xtr_src, X = norm(Xtr_src), norm(X)
    nb_u = int(np.unique(unit[is_target]).size); nd_u = int(np.unique(unit[~is_target]).size)
    print(f"  TRAIN {src_name} {Xtr_src.shape[0]} px ({tgt_name} TRAIN excluded) | TEST {X.shape[0]} px: "
          f"{nd_u} {src_name}-source units / {nb_u} {tgt_name}-TARGET units (unseen) | alpha {args.alpha:.0%}")
    if nb_u < 3:
        raise SystemExit(f"only {nb_u} {tgt_name} test units -- need >= 3 for the deployment split")

    jobs = [(ss, ms) for ss in args.split_seeds for ms in args.model_seeds]
    results = parallel.run_jobs(
        run_seed, jobs,
        shared=dict(Xtr_src=Xtr_src, ytr_src=ytr_src, X=X, y=y, unit=unit, is_target=is_target,
                    groups=groups, cwl=cwl, subsample_frac=args.subsample_frac, epochs=args.epochs,
                    target=args.alpha),
        prefer=args.device, jobs=args.jobs, deterministic=True, label="phase8R2/run")
    rows = [r for sub in results for r in sub]

    raw_fields = ["split_seed", "model_seed", "method", "arm", "acc_bright", "aurc_bright",
                  "auroc_bright", "temperature", "crc_threshold", "crc_group_joint_risk",
                  "crc_group_coverage", "crc_group_selective_risk", "crc_feasible", "n_calib_units",
                  "n_source_units", "n_target_units", "n_eval_units", "n_eval_px"]
    stem = AXIS[args.axis]["stem"]
    raw_out = P8R.P(f"results_phase8R2_{stem}_raw{sfx}.csv")
    with open(raw_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    agg_fields = ["method", "arm", "n_runs", "coverage_mean", "coverage_se", "joint_risk_mean",
                  "joint_risk_se", "selective_risk_mean", "acc_bright_mean", "threshold_mean",
                  "temperature_mean", "n_calib_units_med", "feasible_rate"]
    agg_out = P8R.P(f"results_phase8R2_{stem}{sfx}.csv")
    by = {}
    for r in rows:
        by.setdefault((r["method"], r["arm"]), []).append(r)
    with open(agg_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields); w.writeheader()
        for meth in METHODS:
            for arm in ("naive", "mondrian"):
                g = by.get((meth, arm))
                if not g:
                    continue
                ss = [r["split_seed"] for r in g]; ms = [r["model_seed"] for r in g]
                cov = [r["crc_group_coverage"] for r in g]; jr = [r["crc_group_joint_risk"] for r in g]
                w.writerow({"method": meth, "arm": arm, "n_runs": len(g),
                            "coverage_mean": f"{np.mean(cov):.4f}", "coverage_se": f"{_two_way_se(cov, ss, ms):.4f}",
                            "joint_risk_mean": f"{np.mean(jr):.4f}", "joint_risk_se": f"{_two_way_se(jr, ss, ms):.4f}",
                            "selective_risk_mean": f"{np.mean([r['crc_group_selective_risk'] for r in g]):.4f}",
                            "acc_bright_mean": f"{np.mean([r['acc_bright'] for r in g]):.4f}",
                            "threshold_mean": f"{np.mean([r['crc_threshold'] for r in g]):.4f}",
                            "temperature_mean": f"{np.mean([r['temperature'] for r in g]):.4f}",
                            "n_calib_units_med": int(np.median([r['n_calib_units'] for r in g])),
                            "feasible_rate": f"{np.mean([r['crc_feasible'] for r in g]):.3f}"})
    stamp(agg_out, args, extra={"dataset": "cloudsen12", "shift_axis": f"{args.axis}_domain_gap",
                                "source": f"{src_name}_trained", "target": f"{tgt_name}_unseen",
                                "unit": "scene_component", "se": "two-way cluster-robust",
                                "target_risk": args.alpha, "methods": METHODS})
    stamp(raw_out, args, extra={"n_rows": len(rows), "shift_axis": f"{args.axis}_domain_gap"})
    print(f"\n  wrote {os.path.basename(agg_out)} + {os.path.basename(raw_out)}")
    print(f"\n  proposed on {tgt_name.upper()} (unseen) domain (target joint risk {args.alpha:.0%}):")
    print(f"  {'arm':9s} {'coverage':>9s} {'jointRisk':>10s} {'acc':>6s}")
    for arm in ("naive", "mondrian"):
        g = by.get(("proposed", arm))
        if g:
            print(f"  {arm:9s} {np.mean([r['crc_group_coverage'] for r in g]):8.2f}% "
                  f"{np.mean([r['crc_group_joint_risk'] for r in g]):9.2f}% "
                  f"{np.mean([r['acc_bright'] for r in g]):5.1f}")


if __name__ == "__main__":
    main()
