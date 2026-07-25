#!/usr/bin/env python3
"""CloudSEN12 channel-sampling comparison: our band-group dropout vs ChannelViT-HCS vs DiChaViT-DCS.

The Indian-Pines phase2X --sampling-baselines run showed a clean crossover (our band-group dropout
wins at low missing, HCS wins at high missing; the sampler's missing-fraction COVERAGE sets the
robustness RANGE) and that DiChaViT-DCS does NOT beat HCS on redundant HSI bands. This script
replicates that on REAL Sentinel-2 (CloudSEN12), the operational dataset, so the decomposition
section of the paper rests on more than one scene.

WHY IT REUSES phase8 rather than reimplementing. Faithfulness is the whole point: the 'drop' arm
here must be the SAME model phase8 trains as 'proposed', so the only thing the comparison varies is
the training-time group sampler. So this script imports phase8's verified pieces unchanged --
load_split (ROI split, crop, endianness, L1C layout), s2_physical_groups (7 physical groups),
_build_grouped (isolated-RNG construction), _predict (the exact eval path, incl. standardisation
handling) and _drop_sets (exhaustive enumeration) -- and copies main()'s load+standardise and
run_seed()'s per-seed subsample verbatim. The proposed model's training depends only on
(seed, Xtr_s, bs, epochs), independent of the other five methods (each reseeds; _build_grouped
forks the RNG), so the 'drop' arm is expected to reproduce phase8's committed proposed curve
BIT-FOR-BIT under matched defaults. That reproduction is checked (--anchor) and is the proof the
replication is faithful; if it fails, the numbers are not trustworthy and the script says so.

The three arms all use the wavelength-PE + SGMAE architecture; only finetune_proposed's `sampling`
differs: 'drop' (our band-group dropout, == group_dropout=True), 'hcs' (ChannelViT), 'dcs'
(DiChaViT sampling component -- Algorithm 1 over the group PE, NO CDL/TDL; report as DCS-sampling).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase8_cloudsen12 as P8   # noqa: E402  (reused verbatim: load_split/_build_grouped/_predict/_drop_sets/groups)
import phase2_degradation as P2  # noqa: E402
from bandsim import hw           # noqa: E402
from bandsim.grouping import group_center_wavelengths  # noqa: E402
from bandsim.metrics import miou                        # noqa: E402
from bandsim.provenance import stamp                    # noqa: E402

# architecture HELD FIXED (wavelength PE + SGMAE); only the training sampler varies.
ARMS = ("drop", "hcs", "dcs")


def _curve_for_arm(sampling, Xtr_s, ytr_s, Xte, yte, Xte_raw, mu, sd, groups, cwl, wl,
                   seed, epochs, bs, max_missing, dsets):
    """Train one arm (proposed architecture + `sampling`) and score its degradation curve on the
    SAME drop sets phase8 uses, via phase8's own _predict. Mirrors phase8.run_seed's proposed
    recipe exactly except finetune_proposed's sampling mode."""
    pre = max(1, epochs // 2)
    m = P8._build_grouped(groups, cwl, seed)                 # sinusoidal (wavelength) PE, isolated RNG
    P2.pretrain_sgmae(m, Xtr_s, groups, seed, epochs=pre, bs=bs)
    P2.finetune_proposed(m, Xtr_s, ytr_s, groups, seed, epochs=epochs, bs=bs, sampling=sampling)
    per_m = {mm: [] for mm in range(max_missing + 1)}
    rows = []
    for mm, ds in dsets:
        pred = P8._predict("proposed", m, Xte, groups, list(ds), wl, X_raw=Xte_raw, mu=mu, sd=sd)
        v = float(miou(yte, pred, P8.NUM_CLASSES))
        per_m[mm].append(v)
        rows.append((sampling, seed, mm, v))
    curve = np.array([float(np.mean(per_m[mm])) for mm in range(max_missing + 1)])
    return curve, rows


def paired(rows, a, b, max_missing):
    """Per-level paired mean difference a-b (positive = a more robust) with a 95% t interval."""
    from scipy import stats
    by = {}
    for arm, seed, mm, v in rows:
        by.setdefault((arm, seed), {}).setdefault(mm, []).append(v)
    seeds = sorted({s for arm, s in by if arm == a})
    out = []
    for mm in range(max_missing + 1):
        A = np.array([np.mean(by[(a, s)][mm]) for s in seeds])
        B = np.array([np.mean(by[(b, s)][mm]) for s in seeds])
        d = A - B
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        out.append((mm, float(d.mean()), float(d.mean() - h), float(d.mean() + h)))
    return out


# Faithfulness tolerance is NOISE-AWARE, not bit-for-bit. phase8_cloudsen12.py:888 states the
# attention backward has NO deterministic CUDA kernel, so a GPU run of the SAME proposed model does
# NOT reproduce bit-for-bit run-to-run (only the CPU reference path does). The residual is tiny
# (phase2X's GPU reruns matched to 4 decimals), so a coupling bug — wrong standardisation, subsample,
# or build — announces itself as a LARGE curve delta (the smoke's mis-scaled load gave 5-26 mIoU),
# cleanly separable from ~<0.1 mIoU nondeterminism noise. ANCHOR_TOL sits between them.
ANCHOR_TOL = 0.5   # mIoU: worst per-level |drop − phase8 proposed| below this = faithful replication


def anchor_check(drop_curve_by_seed, seeds, max_missing):
    """The 'drop' arm should reproduce phase8's committed Proposed curve to within CUDA-nondeterminism
    noise. Compare against results_phase8_cloudsen12_curve.csv (proposed_mean per level).
    Returns (worst_abs_delta_or_None, detail_lines)."""
    p = Path(__file__).resolve().parents[1] / "paper" / "results_phase8_cloudsen12_curve.csv"
    if not p.exists():
        return None, f"{p.name} absent — cannot anchor (run phase8 first)"
    committed = {}
    for r in csv.DictReader(p.open()):
        committed[int(r["missing_groups"])] = float(r["proposed_mean"])
    mine = {mm: float(np.mean([drop_curve_by_seed[s][mm] for s in seeds]))
            for mm in range(max_missing + 1)}
    worst = 0.0
    detail = []
    for mm in range(max_missing + 1):
        if mm in committed:
            d = abs(mine[mm] - committed[mm]); worst = max(worst, d)
            detail.append(f"m={mm}: mine {mine[mm]:.4f} vs phase8 {committed[mm]:.4f} (|Δ|={d:.4f})")
    return worst, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)         # phase8 default
    ap.add_argument("--subsample-frac", type=float, default=0.8)
    ap.add_argument("--px-train", type=int, default=300)
    ap.add_argument("--px-test", type=int, default=300)
    ap.add_argument("--patches-train", type=int, default=None)
    ap.add_argument("--patches-test", type=int, default=None)
    ap.add_argument("--max-missing", type=int, default=5)     # phase8 default
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.epochs, args.px_train, args.px_test, args.patches_train, args.patches_test \
            = [0, 1], 2, 60, 60, 40, 25
        args.out_tag = (args.out_tag or "") + "_smoke"

    hw.setup(deterministic=True, prefer=args.device)
    print("HW:", hw.info())

    # --- load + standardise EXACTLY as phase8.main does (loader seeds 12345/54321 are phase8's) ---
    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=args.px_train,
                             n_patches=args.patches_train, seed=12345)
    Xte, yte = P8.load_split("test", "L1C", pixels_per_patch=args.px_test,
                             n_patches=args.patches_test, seed=54321)[:2]
    mu = Xtr.mean(0); sd_raw = Xtr.std(0)
    dead = np.flatnonzero(sd_raw < 1e-6)
    if dead.size:
        raise SystemExit(f"dead train band(s) {dead.tolist()} (sd<1e-6) — see phase8")
    sd = sd_raw + 1e-8
    Xte_raw = Xte.astype(np.float32)                           # RAW kept for B3 path in _predict
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)

    groups = P8.s2_physical_groups()
    wl = np.array(P8.S2_WL_NM, float)
    cwl = group_center_wavelengths(wl, groups)
    G = len(groups)
    if not (0 <= args.max_missing < G):
        raise SystemExit(f"--max-missing must be in [0,{G}), got {args.max_missing}")
    dsets = P8._drop_sets(G, args.max_missing)                 # exhaustive, deterministic
    print(f"CloudSEN12: train {Xtr.shape[0]} px / test {Xte.shape[0]} px | {G} groups | "
          f"classes={P8.NUM_CLASSES} | arms={list(ARMS)} | epochs={args.epochs} | seeds={args.seeds}")

    t0 = time.time()
    rows = []
    drop_by_seed = {}
    for i, seed in enumerate(args.seeds):
        ts = time.time()
        # per-seed training subsample -- phase8.run_seed verbatim
        rs = np.random.default_rng(seed)
        ntr = Xtr.shape[0]
        k = max(1, int(round(args.subsample_frac * ntr)))
        sub = rs.choice(ntr, size=k, replace=False)
        Xtr_s, ytr_s = Xtr[sub], ytr[sub]
        bs = P2.auto_bs(Xtr_s.shape[0])
        for arm in ARMS:
            curve, arm_rows = _curve_for_arm(arm, Xtr_s, ytr_s, Xte, yte, Xte_raw, mu, sd,
                                             groups, cwl, wl, seed, args.epochs, bs,
                                             args.max_missing, dsets)
            rows += arm_rows
            if arm == "drop":
                drop_by_seed[seed] = curve
            print(f"  seed {seed} {arm} done ({time.time() - ts:.0f}s cum, bs={bs})", flush=True)
        print(f"  seed {seed} done ({time.time() - ts:.0f}s, {i + 1}/{len(args.seeds)})", flush=True)

    # --- faithfulness anchor: the 'drop' arm must reproduce phase8's committed Proposed curve ---
    worst, anchor_detail = anchor_check(drop_by_seed, args.seeds, args.max_missing)
    if isinstance(worst, float):
        # smoke uses tiny loads so it cannot match phase8's full-load curve; only judge on a full run.
        anchored = None if args.smoke else (worst < ANCHOR_TOL)
        if args.smoke:
            verdict = "smoke (tiny loads) — not judged"
        elif worst < 1e-4:
            verdict = "✓ bit-for-bit (≤1e-4) — replication faithful"
        elif worst < ANCHOR_TOL:
            verdict = f"✓ faithful within CUDA-nondeterminism noise (< {ANCHOR_TOL} mIoU)"
        else:
            verdict = f"⚠ MISMATCH ≥ {ANCHOR_TOL} mIoU — a coupling bug, do NOT trust the arms"
        print(f"\n  anchor (drop arm vs phase8 committed Proposed): worst |Δ| = {worst:.4f}  → {verdict}")
        for d in anchor_detail:
            print("    " + d)
    else:
        anchored = None
        print(f"\n  anchor: {anchor_detail}")

    contrasts = {"hcs_minus_drop": ("hcs", "drop"),
                 "dcs_minus_drop": ("dcs", "drop"),
                 "dcs_minus_hcs":  ("dcs", "hcs")}
    summary = {n: paired(rows, a, b, args.max_missing) for n, (a, b) in contrasts.items()}

    P = lambda n: Path(__file__).resolve().parents[1] / "paper" / n  # noqa: E731
    raw_out = P(f"results_phase8_sampling_raw{args.out_tag}.csv")
    with raw_out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["kind", "seed", "missing_groups", "miou"])
        for r in rows:
            w.writerow(r)
    sum_out = P(f"results_phase8_sampling{args.out_tag}.csv")
    with sum_out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["contrast", "missing_groups", "delta_miou", "ci_lo", "ci_hi"])
        for n, vals in summary.items():
            for mm, d, lo, hi in vals:
                w.writerow([n, mm, f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    print(f"\n  paired contrasts, n={len(args.seeds)} seeds (* = 95% interval excludes zero)\n")
    print(f"  {'contrast':16s} " + " ".join(f"m={m}".rjust(8) for m in range(args.max_missing + 1)))
    for n, vals in summary.items():
        cells = [f"{d:+7.2f}" + ("*" if (lo > 0 or hi < 0) else " ") for _m, d, lo, hi in vals]
        print(f"  {n:16s} " + " ".join(c.rjust(8) for c in cells))

    import collections
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for arm, seed, mm, v in rows:
        by[arm][mm].append(v)
    print("\n  absolute mIoU by arm (mean over seeds):")
    print(f"  {'arm':6s} " + " ".join(f"m={m}".rjust(7) for m in range(args.max_missing + 1)))
    for arm in ARMS:
        print(f"  {arm:6s} " + " ".join(f"{np.mean(by[arm][mm]):6.2f}".rjust(7)
                                        for mm in range(args.max_missing + 1)))

    stamp(sum_out, args, extra={"arms": list(ARMS), "dataset": "cloudsen12",
                                "n_groups": G, "num_classes": P8.NUM_CLASSES,
                                "loader_seeds": {"train": 12345, "test": 54321},
                                "subsample_frac": args.subsample_frac,
                                "anchor_worst_abs_delta_vs_phase8_proposed":
                                    (None if not isinstance(worst, float) else round(worst, 6)),
                                "anchor_tol_miou": ANCHOR_TOL,
                                "anchored_within_cuda_noise": anchored,
                                "dcs_scope": "sampling component only (Algorithm 1), no CDL/TDL",
                                "runtime_s": round(time.time() - t0, 1)})
    stamp(raw_out, args, extra={"n_rows": len(rows)})

    if isinstance(worst, float) and not args.smoke and worst >= ANCHOR_TOL:
        print(f"\n  ANCHOR FAILED (worst |Δ| = {worst:.4f} ≥ {ANCHOR_TOL} mIoU): the drop arm does not "
              "reproduce phase8's Proposed curve within CUDA-nondeterminism noise, so the replication "
              "is not faithful and the arms must not be cited. Investigate the coupling (load params "
              "/ subsample / standardisation / build / auto_bs / epochs) before trusting.")
        return 1
    print(f"\n  wrote {sum_out.name} + {raw_out.name} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
