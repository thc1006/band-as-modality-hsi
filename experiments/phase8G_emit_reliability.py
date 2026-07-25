#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8G (EXPLORATORY / SUPPLEMENTARY) — land-cover reliability probe on REAL EMIT spectra: does
EMIT's per-pixel retrieval-uncertainty PROXY behave like a reliability signal for a downstream
land-cover classifier?

WHAT THE UNCERTAINTY IS, STATED CORRECTLY. EMIT L2A reflectance and its uncertainty come from the
SAME ISOFIT optimal-estimation retrieval: the uncertainty is the posterior standard deviation of
the very quantity the reflectance estimates. It is EXTERNAL TO OUR MODEL but not independent of the
data the model is trained on, and it is not measured error against a reference. The defensible term
is a retrieval-uncertainty PROXY (see phase8F_emit's header for the long form). Nothing here is
"physical ground truth".

TWO SPLIT ARMS, AND THE CONTRAST IS PART OF THE RESULT.
  random   the historical 70/30 pixel split. Neighbouring train/eval pixels are spatially
           correlated, so accuracy is optimistic and every reliability metric inherits the leak.
  spatial  block holdout on the lat/lon the NPZ has always carried (and this script previously
           ignored): pixels are gridded into BLOCK_DEG-sized blocks, whole blocks are assigned
           70/30 by a per-seed draw, and eval pixels within GUARD_DEG of any train-assigned block
           are dropped. The exchangeable unit is the BLOCK, so the spatial arm's intervals are
           block-cluster bootstraps, not iid ones.
The precedent for expecting the arms to differ is phase8F_multi: under a proper spatial split its
per-pixel Spearman flipped sign on two of three granules (sahara +0.089 -> -0.007).

PER-PIXEL UNCERTAINTY AGGREGATION IS A CHOICE, NOT A FACT. U[pixel, band] is per-band; a scalar
per-pixel score requires an aggregate, and the historical mean(U) is only one of several defensible
ones. AGGREGATORS lists five; `mean_sd` is PRE-DECLARED as primary (the historical choice, kept so
the arm contrast is not confounded with an aggregator change) and the others are reported beside it
so a conclusion that only holds under one aggregate is visible as such. None of them needs
retraining -- they reuse the same logits.

THE THREE CHECKS ARE NOT INDEPENDENT (they share Uev, conf and correct; AUROC and AURC are both
monotone in the same ranking), so there is NO vote. Each is reported with its own interval; the
abstention check gets an empirical null (N_PERM permuted rankings, BLOCK-permuted in the spatial
arm) instead of a single random draw against an arbitrary 1-point threshold.

UNDEFINED IS NOT NEGATIVE. selective_auroc returns NaN when eval accuracy is 0% or 100%; a NaN
metric writes an EMPTY CSV cell (never the string "nan" -- integrity_check.csv_finite_and_sane
fails the whole file on it), and the summary separates "no signal" from "not measurable".

SINGLE GRANULE. n_granule = 1 whatever the seed count; seeds vary the split and the training, not
the scene. WorldCover is 10 m point-sampled at ~60 m EMIT pixels (mixed-pixel label noise), the
class set comes from full-granule label counts (--min-count; a mild eligibility leak), and the
granule's WorldCover year / acquisition date are NOT recorded in the NPZ -- listed in provenance as
absent rather than silently omitted.

Outputs (../paper/): results_phase8G_emit_reliability{sfx}.csv          one row per (arm, seed)
                     results_phase8G_emit_reliability_summary{sfx}.csv  one row per arm
Usage:
  python experiments/phase8G_emit_reliability.py --smoke
  python experiments/phase8G_emit_reliability.py --seeds 0 1 2 --device cuda
"""
import os, sys, csv, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase2_degradation as P2
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.reliability import confidence_msp, selective_auroc, aurc
from bandsim import hw
from bandsim.provenance import stamp, file_sha256

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(PAPER_DIR, exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)
EMIT_NPZ = os.path.join(os.path.dirname(_HERE), "data", "emit", "emit_sample.npz")
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare",
      70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
WC_VALID = np.array(sorted(WC), dtype=np.int64)

BLOCK_DEG = 0.05      # ~5.5 km blocks; the granule spans ~1 deg, so a few hundred blocks
GUARD_DEG = 0.002     # ~220 m (>3 EMIT pixels) guard strip around train blocks
N_PERM = 1000         # permutation null for the abstention check
N_BOOT = 1000         # bootstrap resamples for the interval columns
TRAIN_FRAC = 0.7

# Per-pixel aggregates of the per-band posterior SD. `mean_sd` is the PRE-DECLARED primary (the
# historical choice); the rest are sensitivity companions computed from the same arrays.
AGGREGATORS = {
    "mean_sd":   lambda U, R: U.mean(1),
    "rms_sd":    lambda U, R: np.sqrt((U ** 2).mean(1)),
    "median_sd": lambda U, R: np.median(U, 1),
    "max_sd":    lambda U, R: U.max(1),
    # relative to the signal, so bright pixels are not penalised for absolute SD
    "rel_rms":   lambda U, R: np.sqrt(((U / np.maximum(np.abs(R), 1e-6)) ** 2).mean(1)),
}
PRIMARY_AGG = "mean_sd"


def load_and_validate(path):
    """Fail-closed NPZ contract. Every check guards a failure that would otherwise reach the CSV
    as a plausible number (verified on the shipped NPZ: all pass; K=5)."""
    d = np.load(path)
    need = {"reflectance", "uncertainty", "wavelengths", "worldcover", "lat", "lon"}
    missing = need - set(d.files)
    if missing:
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")
    R = d["reflectance"].astype(np.float32); U = d["uncertainty"].astype(np.float32)
    wl = d["wavelengths"].astype(float); wc = d["worldcover"].astype(int)
    lat = d["lat"].astype(float); lon = d["lon"].astype(float)
    if R.ndim != 2 or R.shape != U.shape:
        raise ValueError(f"reflectance {R.shape} / uncertainty {U.shape} must be equal 2-D shapes")
    for name, a in (("reflectance", R), ("uncertainty", U), ("wavelengths", wl),
                    ("lat", lat), ("lon", lon)):
        if not np.isfinite(a).all():
            raise ValueError(f"{name}: {int((~np.isfinite(a)).sum())} non-finite values")
    if (U < 0).any():
        raise ValueError(f"uncertainty has {int((U < 0).sum())} negative values (posterior SD >= 0)")
    if not (np.diff(wl) > 0).all() or wl.size != R.shape[1]:
        raise ValueError("wavelengths must be strictly increasing and match the band axis")
    if len({len(wc), len(lat), len(lon), R.shape[0]}) != 1:
        raise ValueError("per-pixel arrays disagree on the pixel count")
    unknown = np.setdiff1d(np.unique(wc), WC_VALID)
    if unknown.size:
        # 0/nodata, 255, or a dtype accident would otherwise become a trainable class
        raise ValueError(f"unexpected WorldCover codes {unknown.tolist()}; valid: {WC_VALID.tolist()}")
    return R, U, wl, wc, lat, lon


def spatial_blocks(lat, lon, block_deg=BLOCK_DEG):
    """Integer block id per pixel from a lat/lon grid. The exchangeable unit of the spatial arm."""
    bi = np.floor((lat - lat.min()) / block_deg).astype(np.int64)
    bj = np.floor((lon - lon.min()) / block_deg).astype(np.int64)
    return bi * (bj.max() + 1) + bj, bi, bj


def spatial_split(lat, lon, rng, train_frac=TRAIN_FRAC, block_deg=BLOCK_DEG, guard_deg=GUARD_DEG):
    """(tr_idx, ev_idx, n_tr_blocks, n_ev_blocks, ev_block_ids): whole-block 70/30 with a guard.

    Blocks are assigned, not pixels. Eval pixels within `guard_deg` of the bounding box of ANY
    train block are dropped, so an eval pixel never sits within ~3 EMIT pixels of training ground.
    The guard is geometric (distance to the train blocks' boxes), not a neighbour search."""
    block, bi, bj = spatial_blocks(lat, lon, block_deg)
    uniq = np.unique(block)
    perm = rng.permutation(uniq.size)
    n_tr = max(1, int(round(train_frac * uniq.size)))
    tr_blocks = uniq[perm[:n_tr]]
    is_tr = np.isin(block, tr_blocks)
    lat0, lon0 = lat.min(), lon.min()
    stride = bj.max() + 1
    tb_bi = tr_blocks // stride; tb_bj = tr_blocks % stride
    lo_lat, hi_lat = lat0 + tb_bi * block_deg, lat0 + (tb_bi + 1) * block_deg
    lo_lon, hi_lon = lon0 + tb_bj * block_deg, lon0 + (tb_bj + 1) * block_deg
    ev_idx = np.flatnonzero(~is_tr)
    dlat = np.maximum(0.0, np.maximum(lo_lat[None, :] - lat[ev_idx, None],
                                      lat[ev_idx, None] - hi_lat[None, :]))
    dlon = np.maximum(0.0, np.maximum(lo_lon[None, :] - lon[ev_idx, None],
                                      lon[ev_idx, None] - hi_lon[None, :]))
    ev_idx = ev_idx[np.sqrt(dlat ** 2 + dlon ** 2).min(1) >= guard_deg]
    return (np.flatnonzero(is_tr), ev_idx, int(tr_blocks.size),
            int(np.unique(block[ev_idx]).size), block[ev_idx])


def block_bootstrap(stat, groups, rng, n=N_BOOT):
    """(lo, hi) 2.5/97.5 percentile interval, resampling GROUPS (clusters) with replacement.
    `stat(idx)` computes the statistic on a pixel-index subset. NaN-safe: resamples where the
    statistic is undefined are dropped; (nan, nan) if fewer than 50 survive."""
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        v = stat(np.concatenate([by[g] for g in pick]))
        if v == v:
            vals.append(v)
    if len(vals) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _spearman(x, y):
    """rho, or NaN when undefined. Shape-checked; non-finite inputs are impossible here because
    the NPZ contract rejects them at load."""
    from scipy.stats import spearmanr
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"spearman shape mismatch {x.shape} vs {y.shape}")
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    r = spearmanr(x, y).correlation
    return float(r) if r == r else float("nan")


@torch.no_grad()
def predict(model, X, groups, bs=8192):
    model.eval()                      # the trainers leave eval mode set today; do not rely on it
    dev = next(model.parameters()).device
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"predict needs a non-empty (N, B) matrix, got {X.shape}")
    out = []
    Xt = torch.from_numpy(X)
    for s in range(0, X.shape[0], bs):
        xb = Xt[s:s + bs].to(dev)
        pm = torch.from_numpy(P2.group_present_mask(xb.shape[0], groups, [])).to(dev)
        out.append(model(xb, pm).cpu().numpy())
    logits = np.concatenate(out)
    if not np.isfinite(logits).all():
        raise RuntimeError("non-finite logits; nothing downstream is interpretable")
    return logits


def fmt(x, prec=4):
    """A number, or an EMPTY cell when it is undefined -- never the string 'nan'."""
    return "" if x != x else f"{x:.{prec}f}"


def run_arm(arm, seed, R, U, y, lat, lon, groups, cwl, K, epochs, rng):
    """Train once, evaluate every reliability quantity for one (arm, seed). Returns a dict."""
    if arm == "random":
        perm = rng.permutation(len(y)); ntr = int(TRAIN_FRAC * len(y))
        tr, ev = perm[:ntr], perm[ntr:]
        # no meaningful cluster -> every pixel its own group, so block_bootstrap degrades to the
        # iid pixel bootstrap. One shared group id would make every resample identical and the
        # interval collapse to a point (observed: CI [51.7, 51.7]).
        ev_groups = np.arange(len(ev), dtype=np.int64)
        n_btr = n_bev = 0
    else:
        tr, ev, n_btr, n_bev, ev_groups = spatial_split(lat, lon, rng)
    counts = np.bincount(y[tr], minlength=K)
    if (counts == 0).any():
        raise RuntimeError(f"{arm}/seed{seed}: class index(es) "
                           f"{np.flatnonzero(counts == 0).tolist()} absent from TRAIN; enlarge "
                           f"blocks or lower --min-count")
    mu = R[tr].mean(0); sd = R[tr].std(0) + 1e-8
    Xtr = ((R[tr] - mu) / sd).astype(np.float32)
    m = GroupedCrossBandAttention(groups, cwl, K)
    P2.pretrain_sgmae(m, Xtr, groups, seed, epochs=max(1, epochs // 2))
    P2.finetune_proposed(m, Xtr, y[tr], groups, seed, epochs=epochs)
    logits = predict(m, ((R[ev] - mu) / sd).astype(np.float32), groups)
    conf = confidence_msp(logits)
    correct = (logits.argmax(1) == y[ev]).astype(int)

    out = {"arm": arm, "seed": seed, "n_train_px": len(tr), "n_eval_px": len(ev),
           "n_train_blocks": n_btr, "n_eval_blocks": n_bev,
           "acc_pct": 100.0 * float(correct.mean()),
           "auroc_model_pct": 100.0 * selective_auroc(correct, conf),
           "aurc_model_pct": 100.0 * aurc(correct, conf)}
    for name, f in AGGREGATORS.items():
        Uev = f(U[ev], R[ev])
        out[f"auroc_emit_{name}_pct"] = 100.0 * selective_auroc(correct, -Uev)
        if name == PRIMARY_AGG:
            out["spearman_unc_conf"] = _spearman(Uev, conf)
            real_aurc = aurc(correct, -Uev)
            out["aurc_emit_pct"] = 100.0 * real_aurc
            # empirical null: in the spatial arm the permutation is BLOCK-wise (block ids remapped
            # onto other blocks' positions), preserving within-block correlation under the null the
            # way the real score carries it.
            null = np.empty(N_PERM)
            uniq = np.unique(ev_groups)
            for i in range(N_PERM):
                if arm == "spatial" and uniq.size > 1:
                    remap = dict(zip(uniq.tolist(), uniq[rng.permutation(uniq.size)].tolist()))
                    order = np.argsort(np.array([remap[g] for g in ev_groups.tolist()]),
                                       kind="stable")
                    null[i] = aurc(correct, -Uev[order])
                else:
                    null[i] = aurc(correct, -rng.permutation(Uev))
            out["aurc_null_mean_pct"] = 100.0 * float(null.mean())
            out["aurc_null_p"] = float((np.sum(null <= real_aurc) + 1) / (N_PERM + 1))
            def _stat(idx):
                return selective_auroc(correct[idx], -Uev[idx])
            lo, hi = block_bootstrap(_stat, ev_groups, rng)
            out["auroc_emit_ci_lo_pct"] = 100.0 * lo
            out["auroc_emit_ci_hi_pct"] = 100.0 * hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=300)
    ap.add_argument("--arms", nargs="+", default=["spatial", "random"],
                    choices=["spatial", "random"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    ap.add_argument("--nondeterministic", action="store_true")
    args = ap.parse_args()
    if args.epochs < 1 or args.min_count < 1 or args.groups < 4:
        ap.error("--epochs/--min-count must be >= 1 and --groups >= 4")
    if len(set(args.seeds)) != len(args.seeds):
        ap.error(f"--seeds must be distinct, got {args.seeds}")
    sfx = ""
    if args.smoke:
        args.seeds = [0]; args.epochs = 8; args.groups = 12
        sfx = "_smoke"
        print("[smoke] 1 seed / 8 epochs / 12 groups — *_smoke artefacts only")
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device)
    print("HW:", hw.info())

    R, U, wl, wc, lat, lon = load_and_validate(EMIT_NPZ)
    u, c = np.unique(wc, return_counts=True)
    keep = u[c >= args.min_count]
    if keep.size < 2:
        raise RuntimeError(f"only {keep.size} class(es) reach --min-count={args.min_count}; "
                           f"a classification probe needs >= 2")
    remap = {int(k): i for i, k in enumerate(keep)}
    m_ok = np.isin(wc, keep)
    R, U, wc, lat, lon = R[m_ok], U[m_ok], wc[m_ok], lat[m_ok], lon[m_ok]
    y = np.array([remap[int(v)] for v in wc], dtype=np.int64)
    K = len(keep)
    groups = contiguous_groups(R.shape[1], args.groups)
    cwl = group_center_wavelengths(wl, groups)
    print(f"EMIT land-cover: {len(y)} px | K={K} {[WC.get(int(k)) for k in keep]} | "
          f"{R.shape[1]} bands, {args.groups} groups | arms={args.arms}")

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            rows.append(run_arm(arm, seed, R, U, y, lat, lon, groups, cwl, K,
                                args.epochs, np.random.default_rng(seed)))
            r = rows[-1]
            print(f"  {arm}/seed{seed}: acc={r['acc_pct']:.1f} "
                  f"AUROC_emit={fmt(r[f'auroc_emit_{PRIMARY_AGG}_pct'], 1)} "
                  f"CI[{fmt(r.get('auroc_emit_ci_lo_pct', float('nan')), 1)},"
                  f"{fmt(r.get('auroc_emit_ci_hi_pct', float('nan')), 1)}] "
                  f"AURC={fmt(r['aurc_emit_pct'], 1)} null={fmt(r['aurc_null_mean_pct'], 1)} "
                  f"p={fmt(r['aurc_null_p'], 3)} rho={fmt(r['spearman_unc_conf'], 3)}")

    cols = list(rows[0].keys())
    with open(P(f"results_phase8G_emit_reliability{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if isinstance(r[c], (str, int)) else fmt(r[c]) for c in cols])

    # per-arm summary; a metric undefined on some seed reports the VALID count beside the mean
    def agg(vals):
        v = np.array([x for x in vals if x == x], float)
        return (float(v.mean()) if v.size else float("nan"), int(v.size))
    sum_cols = ["arm", "n_seeds", "acc_pct_mean",
                f"auroc_emit_{PRIMARY_AGG}_pct_mean", "auroc_emit_n_valid",
                "aurc_emit_pct_mean", "aurc_null_p_median", "spearman_mean",
                "agg_sensitivity_auroc_range_pct"]
    with open(P(f"results_phase8G_emit_reliability_summary{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(sum_cols)
        for arm in args.arms:
            ar = [r for r in rows if r["arm"] == arm]
            au, n_ok = agg([r[f"auroc_emit_{PRIMARY_AGG}_pct"] for r in ar])
            aurc_m, _ = agg([r["aurc_emit_pct"] for r in ar])
            sp_m, _ = agg([r["spearman_unc_conf"] for r in ar])
            ps = sorted(r["aurc_null_p"] for r in ar if r["aurc_null_p"] == r["aurc_null_p"])
            p_med = ps[len(ps) // 2] if ps else float("nan")
            per_agg = [agg([r[f"auroc_emit_{a}_pct"] for r in ar])[0] for a in AGGREGATORS]
            rng_pct = (max(per_agg) - min(per_agg)) if all(v == v for v in per_agg) else float("nan")
            acc_m, _ = agg([r["acc_pct"] for r in ar])
            w.writerow([arm, len(ar), fmt(acc_m, 2), fmt(au, 2), n_ok, fmt(aurc_m, 2),
                        fmt(p_med, 4), fmt(sp_m, 4), fmt(rng_pct, 2)])
            print(f"[{arm}] acc={fmt(acc_m, 1)} AUROC_emit({PRIMARY_AGG})={fmt(au, 1)} "
                  f"(valid {n_ok}/{len(ar)}) p_med={fmt(p_med, 3)} agg-range={fmt(rng_pct, 1)}pp")

    # The verdict is DERIVED and hedged; no vote, no hardcoded sibling-experiment numbers.
    sp_rows = [r for r in rows if r["arm"] == "spatial"]
    if sp_rows:
        ps = [r["aurc_null_p"] for r in sp_rows if r["aurc_null_p"] == r["aurc_null_p"]]
        if not ps:
            print("-> UNTESTED on the spatial arm: the abstention statistic was undefined on every "
                  "seed (degenerate eval accuracy). No claim either way.")
        elif all(p < 0.05 for p in ps):
            print("-> On this single granule's SPATIAL holdout, abstaining on high EMIT uncertainty "
                  "beats its permutation null on every seed. Exploratory: one granule, proxy "
                  "uncertainty, point-sampled labels.")
        else:
            print("-> WEAK/NO abstention signal on the spatial arm (permutation p >= 0.05 on "
                  f"{sum(p >= 0.05 for p in ps)}/{len(ps)} seeds). Competing explanations -- "
                  "aggregator choice, label noise, class composition -- are NOT separated by this "
                  "design; see the per-aggregator columns before reading direction into it.")

    # Two direct stamp() calls rather than a loop: the smoke-isolation analyzer associates a stamp
    # with a write site by resolving one assignment level, and a tuple-driven loop is beyond that.
    extra = _provenance_extra(args, y, R, K, keep)
    stamp(P(f"results_phase8G_emit_reliability{sfx}.csv"), args, extra=extra)
    stamp(P(f"results_phase8G_emit_reliability_summary{sfx}.csv"), args, extra=extra)
    print(f"wrote: {P(f'results_phase8G_emit_reliability{sfx}.csv')} + _summary{sfx}.csv")


def _provenance_extra(args, y, R, K, keep):
    return {
            "emit_npz": EMIT_NPZ, "emit_npz_sha256": file_sha256(EMIT_NPZ),
            "n_px": int(len(y)), "n_bands": int(R.shape[1]), "n_classes": int(K),
            "classes": [WC.get(int(k), int(k)) for k in keep],
            "arms": args.arms, "block_deg": BLOCK_DEG, "guard_deg": GUARD_DEG,
            "n_perm": N_PERM, "n_boot": N_BOOT, "primary_aggregator": PRIMARY_AGG,
            "aggregators": list(AGGREGATORS),
            "uncertainty_semantics": "ISOFIT posterior SD from the same retrieval as the "
                                     "reflectance -- an external retrieval-uncertainty proxy, not "
                                     "independent ground truth",
            "provenance_gaps": ["EMIT granule id / acquisition date not in NPZ",
                                "WorldCover year/version not in NPZ",
                                "footprint purity not computed (point-sampled labels)"]}


if __name__ == "__main__":
    main()
