#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8E2 -- frozen-DOFA baseline under the FLAGSHIP protocol, so its number is COMPARABLE to phase8R.

phase8E ran the DOFA baseline with the plug-in `conformal_at_risk` (a CONDITIONAL selective risk
P(wrong|accepted)) on a SINGLE calibration split with a design-effect SE. The flagship phase8R reports
the CRC JOINT risk P(accepted AND wrong) over a crossed 10-split x 3-seed design with a two-way
cluster-robust SE. Those are different estimands and different designs, so "DOFA 10.6% vs 28.9%" was not a
like-for-like comparison. This script fixes that: it runs the frozen-DOFA feature baseline through the
SAME machinery as phase8R --
  * CRC joint risk (conformal_risk_control), naive (clean-calibrated) vs Mondrian (state-recalibrated);
  * scene-connected-component exchangeable unit (P8.scene_component_ids), on the split AND the CRC group;
  * three-way component split (temperature / CRC-calibration / evaluation), re-drawn per split_seed;
  * crossed design over (split_seed in 0..9) x (model_seed in 0..2), two-way cluster-robust SE.
DOFA is frozen and its head is light, so features are cached once per state and only the head is retrained
per model_seed; the split loop is then cheap (re-partition + temperature + CRC on cached logits).

Reuses phase8E verbatim for everything DOFA-specific (load_dofa / load_spatial / extract_features /
train_head / head_logits_perpix / DofaSegHead / DOFA_BANDS / STATES / IMG), so the encoder, the pinned
checkpoint, the band set and the head are identical to the committed phase8E baseline.

Output (../paper/): results_phase8E2_dofa_crc.csv (per state, arm: joint risk mean+two-way SE, coverage,
feasibility, unit counts) + _raw.csv. --smoke writes *_smoke only.
"""
import os, sys, csv, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase8E_dofa as P8E                      # DOFA encoder/head/loaders reused verbatim
from bandsim.reliability import confidence_msp, fit_temperature, conformal_risk_control
from bandsim import hw
from bandsim.provenance import stamp

IMG = P8E.IMG
STATES = P8E.STATES                             # [(clean,L1C,()), (dropSWIR,L1C,(7,8)), (L2A_real,L2A,())]
TARGET_RISK = P8E.TARGET_RISK
NUM_CLASSES = P8.NUM_CLASSES
TEMP_FRAC, CALIB_FRAC = 0.25, 0.375             # same three-way split budget as phase8R


def _split3(units, split_seed):
    """Three DISJOINT scene-component sets (temperature / CRC-calibration / evaluation)."""
    uni = np.unique(units)
    if len(uni) < 3:
        return None
    perm = np.random.default_rng(85000 + split_seed).permutation(uni)
    n_t = max(1, int(round(TEMP_FRAC * len(perm))))
    n_c = max(1, int(round(CALIB_FRAC * len(perm))))
    if n_t + n_c >= len(perm):
        return None
    return perm[:n_t], perm[n_t:n_t + n_c], perm[n_t + n_c:]


def _two_way_se(a, splits, models):
    a = np.asarray(a, float); N = a.size
    if N < 2:
        return 0.0
    e = a - a.mean()
    def V(lab):
        lab = np.asarray(lab, dtype=object)
        return sum(float(e[lab == g].sum()) ** 2 for g in set(lab.tolist())) / N ** 2
    return float(np.sqrt(max(V(splits) + V(models) - float((e ** 2).sum()) / N ** 2, 0.0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--patches-train", type=int, default=800)
    ap.add_argument("--patches-test", type=int, default=None, help="cap on #test patches (default all)")
    ap.add_argument("--px-per-patch", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-norm", default="batch", choices=["batch", "group"])
    ap.add_argument("--alpha", type=float, default=TARGET_RISK)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    sfx = ""
    if args.smoke:
        args.split_seeds, args.model_seeds, args.epochs = [0, 1], [0], 4
        args.patches_train, args.patches_test, args.px_per_patch = 60, 80, 150
        sfx = "_smoke"
        print("[smoke] 2 splits x 1 model seed / 4 epochs -- *_smoke artefacts only")
    dev = hw.device() if args.device is None else args.device
    import pandas as pd
    n_train = len(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv")))
    n_test = len(pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv")))
    rng = np.random.default_rng(2024)
    train_ids = np.sort(rng.choice(n_train, size=min(args.patches_train, n_train), replace=False))
    comp = P8.scene_component_ids("test")                        # per-patch scene-component (the unit)
    test_ids = np.arange(n_test)
    if args.patches_test is not None and args.patches_test < n_test:
        test_ids = np.sort(np.random.default_rng(70000).choice(n_test, size=args.patches_test, replace=False))
    unit_patch = comp[test_ids]
    print(f"DOFA-CRC: train {len(train_ids)} patches | test {len(test_ids)} patches over "
          f"{np.unique(unit_patch).size} scene-components | alpha {args.alpha:.0%} | "
          f"{len(args.split_seeds)} splits x {len(args.model_seeds)} seeds")

    # ---- load DOFA, normalise on TRAIN, extract features (frozen -> cached once per state) ----
    Xtr, Ytr, _ = P8E.load_spatial("train", "L1C", train_ids, return_info=True)
    mu = Xtr.mean(axis=(0, 2, 3)); sd = Xtr.std(axis=(0, 2, 3)) + 1e-6
    norm = lambda A: ((A - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float32)
    Xtr = norm(Xtr)
    dofa = P8E.load_dofa(dev)
    keep_all = list(range(len(P8E.DOFA_BANDS)))
    feat_tr = P8E.extract_features(dofa, Xtr, keep_all, dev)

    # per-state test features (all test patches), and the shared per-pixel label subsample + unit ids
    yfull = P8E.load_spatial("test", "L1C", test_ids)[1]          # labels (same file for L1C/L2A)
    rs = np.random.default_rng(999)
    idx = np.concatenate([P8E._subsample(rs, IMG * IMG, args.px_per_patch) + p * IMG * IMG
                          for p in range(len(test_ids))])
    y_px = yfull.reshape(-1)[idx]
    unit_px = unit_patch[idx // (IMG * IMG)]                      # scene-component of every sampled pixel
    feat_te = {}
    src_prod = {"L1C": None, "L2A": None}
    for prod in ("L1C", "L2A"):
        src_prod[prod] = norm(P8E.load_spatial("test", prod, test_ids)[0])
    for name, product, drop in STATES:
        keep = [i for i in range(len(P8E.DOFA_BANDS)) if i not in drop]
        feat_te[name] = P8E.extract_features(dofa, src_prod[product], keep, dev)
    del src_prod
    print("features cached (all test patches, per state). training heads + CRC ...")

    # ---- per model_seed: train head once, cache per-pixel logits per state on all test pixels ----
    logits_state = {}   # (model_seed, state) -> (n_px, C) sampled-pixel logits
    for ms in args.model_seeds:
        head = P8E.train_head(feat_tr, Ytr, dev, args.epochs, args.bs, args.lr, ms, head_norm=args.head_norm)
        for name, _prod, _drop in STATES:
            lg_full = P8E.head_logits_perpix(head, feat_te[name], dev)     # (n_patch*IMG*IMG, C)
            logits_state[(ms, name)] = lg_full[idx]

    # ---- crossed (split_seed, model_seed): three-way component split, temperature, CRC naive vs mondrian
    def _mask(keep):
        return np.isin(unit_px, keep)

    rows = []
    for ss in args.split_seeds:
        sp = _split3(unit_px, ss)
        if sp is None:
            continue
        t_u, c_u, e_u = sp
        mt, mc, me = _mask(t_u), _mask(c_u), _mask(e_u)
        for ms in args.model_seeds:
            lg = {n: logits_state[(ms, n)] for n, _p, _d in STATES}
            # clean-state temperature + clean calib scores (naive deployment calibrates once, on clean)
            T_clean = fit_temperature(lg["clean"][mt], y_px[mt])
            corr_clean_c = (lg["clean"][mc].argmax(1) == y_px[mc]).astype(int)
            conf_clean_c = confidence_msp(lg["clean"][mc] / T_clean)
            for name, _prod, _drop in STATES:
                lgs = lg[name]
                corr_c = (lgs[mc].argmax(1) == y_px[mc]).astype(int)
                corr_e = (lgs[me].argmax(1) == y_px[me]).astype(int)
                T = fit_temperature(lgs[mt], y_px[mt])
                conf_c = confidence_msp(lgs[mc] / T); conf_e = confidence_msp(lgs[me] / T)
                conf_e_clean = confidence_msp(lgs[me] / T_clean)
                arms = {
                    "mondrian": conformal_risk_control(corr_c, conf_c, corr_e, conf_e, alpha=args.alpha,
                                                       calib_group=unit_px[mc], eval_group=unit_px[me]),
                    "naive": conformal_risk_control(corr_clean_c, conf_clean_c, corr_e, conf_e_clean,
                                                    alpha=args.alpha, calib_group=unit_px[mc],
                                                    eval_group=unit_px[me]),
                }
                for arm, crc in arms.items():
                    rows.append({"split_seed": ss, "model_seed": ms, "state": name, "arm": arm,
                                 "joint_risk": crc["eval_group_joint_risk"] * 100,
                                 "coverage": crc["eval_group_coverage"] * 100,
                                 "selective_risk": crc["eval_group_selective_risk"] * 100,
                                 "acc": float(corr_e.mean()) * 100,
                                 "feasible": int(crc["feasible"]),
                                 "n_calib_units": int(crc["n_calib_units"]),
                                 "n_eval_units": int(crc["n_eval_units"])})

    P = lambda n: os.path.join(P8E.PAPER_DIR, n)  # noqa: E731
    raw_fields = ["split_seed", "model_seed", "state", "arm", "joint_risk", "coverage",
                  "selective_risk", "acc", "feasible", "n_calib_units", "n_eval_units"]
    with open(P(f"results_phase8E2_dofa_crc_raw{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    agg_fields = ["state", "arm", "n_runs", "joint_risk_mean", "joint_risk_se", "joint_risk_lo",
                  "joint_risk_hi", "coverage_mean", "acc_mean", "n_calib_units_med", "feasible_rate"]
    by = {}
    for r in rows:
        by.setdefault((r["state"], r["arm"]), []).append(r)
    agg_out = P(f"results_phase8E2_dofa_crc{sfx}.csv")
    with open(agg_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields); w.writeheader()
        for name, _p, _d in STATES:
            for arm in ("naive", "mondrian"):
                g = by.get((name, arm))
                if not g:
                    continue
                jr = [r["joint_risk"] for r in g]
                m = float(np.mean(jr))
                se = _two_way_se(jr, [r["split_seed"] for r in g], [r["model_seed"] for r in g])
                w.writerow({"state": name, "arm": arm, "n_runs": len(g),
                            "joint_risk_mean": f"{m:.4f}", "joint_risk_se": f"{se:.4f}",
                            "joint_risk_lo": f"{m - 1.96 * se:.4f}", "joint_risk_hi": f"{m + 1.96 * se:.4f}",
                            "coverage_mean": f"{np.mean([r['coverage'] for r in g]):.4f}",
                            "acc_mean": f"{np.mean([r['acc'] for r in g]):.4f}",
                            "n_calib_units_med": int(np.median([r['n_calib_units'] for r in g])),
                            "feasible_rate": f"{np.mean([r['feasible'] for r in g]):.3f}"})
    stamp(agg_out, args, extra={"dataset": "cloudsen12", "model": "frozen_dofa",
                                "estimand": "CRC joint risk P(accepted AND wrong), comparable to phase8R",
                                "unit": "scene_component", "se": "two-way cluster-robust",
                                "design": "crossed split_seed x model_seed", "target_risk": args.alpha})

    print(f"\n  wrote {os.path.basename(agg_out)} (CRC JOINT risk, comparable to phase8R)\n")
    print(f"  {'state':12s} {'arm':9s} {'jointRisk%':>11s} {'SE':>5s} {'95%CI':>16s} {'cov%':>6s}")
    for name, _p, _d in STATES:
        for arm in ("naive", "mondrian"):
            g = by.get((name, arm))
            if g:
                jr = [r["joint_risk"] for r in g]; m = np.mean(jr)
                se = _two_way_se(jr, [r["split_seed"] for r in g], [r["model_seed"] for r in g])
                print(f"  {name:12s} {arm:9s} {m:>11.2f} {se:>5.2f} [{m-1.96*se:>6.2f},{m+1.96*se:>6.2f}] "
                      f"{np.mean([r['coverage'] for r in g]):>6.1f}")


if __name__ == "__main__":
    main()
