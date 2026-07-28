#!/usr/bin/env python
"""Aggregate the per-class (S3.5) and same-weighting (S3.1) per-cell outputs into paper-ready summaries
with two-way cluster-robust standard errors over the crossed split x seed design (Cameron-Gelbach-Miller
2011), matching the flagship's error model. Reads results_phase8R_perclass_weighting_*.csv, concatenates,
and writes two summary CSVs + prints the numbers.

Intervals in the paper use t(min(G1,G2)-1); with a balanced 10x10 design that is t(9)=2.262.
"""
import csv
import glob
import math
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(_HERE, "..", "paper"))
TCRIT = 2.262


def two_way_se(triples):
    """CGM two-way cluster-robust SE of a mean over a crossed (g1 x g2) design.
    triples = list of (g1, g2, value)."""
    cells = [(a, b) for a, b, _ in triples]
    if len(set(cells)) != len(cells):                               # r2 §3.1: a duplicated (g1,g2) cell would
        from collections import Counter                             # be double-counted -- always a bug, fail closed
        dup = [c for c, n in Counter(cells).items() if n > 1]
        raise ValueError(f"two_way_se: duplicate crossed-design cell(s) {dup[:5]} -- each (g1,g2) must appear once")
    n_nan = sum(1 for _, _, v in triples if v != v)
    if n_nan:                                                       # NaN cells were silently dropped before
        print(f"  [two_way_se] dropping {n_nan} NaN cell(s) of {len(triples)}", file=sys.stderr, flush=True)
    vals = [v for _, _, v in triples if v == v]                     # drop NaN (now reported)
    trip = [(a, b, v) for a, b, v in triples if v == v]
    if len(trip) < 3:
        return (st.mean(vals) if vals else float("nan")), float("nan")
    G1 = sorted(set(a for a, _, _ in trip)); G2 = sorted(set(b for _, b, _ in trip))
    n1, n2 = len(G1), len(G2)
    grand = st.mean(vals)
    sm = {g: st.mean(v for a, _, v in trip if a == g) for g in G1}
    dm = {g: st.mean(v for _, b, v in trip if b == g) for g in G2}
    V1 = sum((sm[g] - grand) ** 2 for g in G1) / (n1 - 1) / n1 if n1 > 1 else 0.0
    V2 = sum((dm[g] - grand) ** 2 for g in G2) / (n2 - 1) / n2 if n2 > 1 else 0.0
    Viid = st.variance(vals) / len(vals) if len(vals) > 1 else 0.0
    core = V1 + V2 - Viid
    if core > 0:
        return grand, math.sqrt(core)
    # Small-cluster CGM estimate went non-positive. Genuinely constant data legitimately has SE 0; but for
    # NON-constant data a silent 0 understates uncertainty (a degenerate anti-symmetric grid can cancel
    # V1+V2 against Viid), so fall back to the larger one-way cluster SE (conservative) and warn -- the
    # estimator must never return a misleading 0 on non-constant data. (The real 10x10 designs have core>0.)
    if st.pvariance(vals) == 0.0:
        return grand, 0.0
    se = math.sqrt(Viid)   # iid floor: >= either one-way SE when CGM is non-positive; >0 for non-constant data
    print(f"  [two_way_se] CGM two-way variance non-positive (V1+V2-Viid={core:.3g}) on non-constant data; "
          f"using the iid-floor SE={se:.3g} (never a silent 0)", file=sys.stderr, flush=True)
    return grand, se


def load_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(PAPER, "results_phase8R_perclass_weighting_*.csv"))):
        if "_agg" in f or "smoke" in f:
            continue
        rows += list(csv.DictReader(open(f)))
    return rows


def num(r, k):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    rows = load_rows()
    if not rows:
        print("no per-cell CSVs found yet"); return
    seeds = sorted(set(int(r["seed"]) for r in rows))
    splits = sorted(set(int(r["split"]) for r in rows))
    print(f"loaded {len(rows)} rows: {len(seeds)} seeds {seeds}, {len(splits)} splits")

    # ---- S3.1 same-weighting (scope=overall) ----
    ov = [r for r in rows if r.get("scope") == "overall"]
    w_out = []
    print("\n===== S3.1 same-weighting (overall, joint=selective*coverage within each weighting) =====")
    hdr = f"  {'state':6s} {'arm':9s} | {'joint_ce':>10s} {'joint_pp':>10s} | {'cov_ce':>7s} {'cov_pp':>7s}" \
          f" | {'sel_ce':>7s} {'sel_pp':>7s} | {'acc_ce':>7s} {'acc_px':>7s}"
    print(hdr)
    for state in ("clean", "L2A"):
        for arm in ("naive", "mondrian"):
            cell = [r for r in ov if r["state"] == state and r["arm"] == arm]
            if not cell:
                continue
            agg = {}
            for k in ("joint_ce", "joint_pp", "cov_ce", "cov_pp", "sel_ce", "sel_pp", "acc_ce", "acc_pixel"):
                m, se = two_way_se([(int(r["seed"]), int(r["split"]), num(r, k)) for r in cell])
                agg[k] = (m, se)
            print(f"  {state:6s} {arm:9s} | {agg['joint_ce'][0]:6.2f}±{agg['joint_ce'][1]:.2f}"
                  f" {agg['joint_pp'][0]:6.2f}±{agg['joint_pp'][1]:.2f} |"
                  f" {agg['cov_ce'][0]:5.1f}  {agg['cov_pp'][0]:5.1f} |"
                  f" {agg['sel_ce'][0]:5.1f}  {agg['sel_pp'][0]:5.1f} |"
                  f" {agg['acc_ce'][0]:5.1f}  {agg['acc_pixel'][0]:5.1f}")
            w_out.append(dict(state=state, arm=arm, n_cells=len(cell),
                              **{f"{k}_mean": agg[k][0] for k in agg},
                              **{f"{k}_se": agg[k][1] for k in agg}))
    with open(os.path.join(PAPER, "results_phase8R_weighting_agg.csv"), "w", newline="") as f:
        if w_out:
            wr = csv.DictWriter(f, fieldnames=list(w_out[0].keys())); wr.writeheader(); wr.writerows(w_out)

    # ---- S3.5 per-class (scope=perclass) ----
    pc = [r for r in rows if r.get("scope") == "perclass"]
    c_out = []
    print("\n===== S3.5 per-class (mean +/- two-way SE over runs) =====")
    print(f"  {'state':6s} {'class':13s} | {'IoU':>10s} {'F1':>10s} | {'cls_joint':>10s} {'cls_sel':>10s} {'cls_cov':>8s}")
    for state in ("clean", "L2A"):
        for cls in ("clear", "thick cloud", "thin cloud", "cloud shadow"):
            # per-class metrics are evaluated at the naive operating point (clean-calibrated threshold)
            cell = [r for r in pc if r["state"] == state and r["cls"] == cls and r["arm"] == "naive"]
            if not cell:
                continue
            agg = {}
            for k in ("iou", "f1", "pa", "ua", "cls_joint", "cls_sel", "cls_cov"):
                m, se = two_way_se([(int(r["seed"]), int(r["split"]), num(r, k)) for r in cell])
                agg[k] = (m, se)
            print(f"  {state:6s} {cls:13s} | {agg['iou'][0]:6.1f}±{agg['iou'][1]:.1f}"
                  f" {agg['f1'][0]:6.1f}±{agg['f1'][1]:.1f} |"
                  f" {agg['cls_joint'][0]:6.1f}±{agg['cls_joint'][1]:.1f}"
                  f" {agg['cls_sel'][0]:6.1f}±{agg['cls_sel'][1]:.1f}"
                  f" {agg['cls_cov'][0]:6.1f}")
            c_out.append(dict(state=state, cls=cls, n_cells=len(cell),
                              **{f"{k}_mean": agg[k][0] for k in agg},
                              **{f"{k}_se": agg[k][1] for k in agg}))
    with open(os.path.join(PAPER, "results_phase8R_perclass_agg.csv"), "w", newline="") as f:
        if c_out:
            wr = csv.DictWriter(f, fieldnames=list(c_out[0].keys())); wr.writeheader(); wr.writerows(c_out)
    print("\nwrote results_phase8R_weighting_agg.csv + results_phase8R_perclass_agg.csv")


if __name__ == "__main__":
    main()
