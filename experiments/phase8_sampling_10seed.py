#!/usr/bin/env python
"""Merge two phase8_sampling raw CSVs into a single N-seed paired-contrast summary.

Motivation: `phase8_sampling.py --seeds 0 1 2 3 4` and a second GPU running
`--seeds 5 6 7 8 9 --out-tag _s5-9` produce two independent raw curves. This script concatenates
their per-seed rows and recomputes the paired contrasts over ALL seeds, reusing
`phase8_sampling.paired()` verbatim so the 95% t-interval methodology is byte-identical to the
single-run output. It writes `results_phase8_sampling_10seed.csv` plus the concatenated raw file.

Faithfulness anchor stays on the ORIGINAL seed block only (the seeds the committed
`results_phase8_cloudsen12_curve.csv` Proposed curve was computed on): a wider seed set would
compare a different sample against that fixed 5-seed mean and inflate |Δ| by seed variance, not by
any coupling bug. So `--anchor-seeds` (default 0 1 2 3 4) selects which seeds the drop arm is
anchored on; the extra seeds only tighten the contrast intervals.

Usage:
  phase8_sampling_10seed.py --raw results_phase8_sampling_raw.csv \
                            --raw results_phase8_sampling_raw_s5-9.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_sampling as PS          # noqa: E402  (paired/anchor_check/ARMS/ANCHOR_TOL reused verbatim)
from bandsim.provenance import stamp  # noqa: E402


def _read_raw(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["kind"], int(r["seed"]), int(r["missing_groups"]), float(r["miou"])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", action="append", required=True,
                    help="a results_phase8_sampling_raw*.csv; pass --raw twice to merge two blocks")
    ap.add_argument("--anchor-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="seeds the drop arm is anchored on (the committed-curve seeds)")
    ap.add_argument("--out-tag", default="_10seed")
    args = ap.parse_args()

    P = lambda n: Path(__file__).resolve().parents[1] / "paper" / n  # noqa: E731
    rows = []
    for r in args.raw:
        p = r if Path(r).is_absolute() else P(Path(r).name)
        rows += _read_raw(p)
    seeds = sorted({s for _k, s, _m, _v in rows})
    max_missing = max(m for _k, _s, m, _v in rows)
    n = len(seeds)
    print(f"  merged raw: seeds={seeds} (n={n}), max_missing={max_missing}, arms={list(PS.ARMS)}")

    # sanity: every arm must have every (seed, missing) cell, else paired() would KeyError or bias
    want = {(arm, s, mm) for arm in PS.ARMS for s in seeds for mm in range(max_missing + 1)}
    have = {(k, s, mm) for k, s, mm, _v in rows}
    missing = want - have
    if missing:
        raise SystemExit(f"incomplete merge: {len(missing)} missing (arm,seed,missing) cells, "
                         f"e.g. {sorted(missing)[:5]} — do NOT trust a partial-seed contrast")

    # faithfulness anchor on the original block only (drop arm vs committed Proposed curve)
    drop_by_seed = {s: {mm: float(np.mean([v for k, ss, m, v in rows
                                           if k == "drop" and ss == s and m == mm]))
                        for mm in range(max_missing + 1)} for s in args.anchor_seeds}
    worst, detail = PS.anchor_check(drop_by_seed, args.anchor_seeds, max_missing)
    if isinstance(worst, float):
        verdict = ("bit-for-bit (<=1e-4)" if worst < 1e-4 else
                   f"faithful within CUDA noise (< {PS.ANCHOR_TOL})" if worst < PS.ANCHOR_TOL else
                   f"MISMATCH >= {PS.ANCHOR_TOL} — coupling bug")
        print(f"  anchor (drop, seeds {args.anchor_seeds}): worst |Δ|={worst:.4f} → {verdict}")
    else:
        print(f"  anchor: {detail}")

    contrasts = {"hcs_minus_drop": ("hcs", "drop"),
                 "dcs_minus_drop": ("dcs", "drop"),
                 "dcs_minus_hcs":  ("dcs", "hcs")}
    summary = {name: PS.paired(rows, a, b, max_missing) for name, (a, b) in contrasts.items()}

    # write merged raw + summary (same schemas as phase8_sampling)
    raw_out = P(f"results_phase8_sampling_raw{args.out_tag}.csv")
    with raw_out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["kind", "seed", "missing_groups", "miou"])
        for r in sorted(rows, key=lambda t: (t[0], t[1], t[2])):
            w.writerow(r)
    sum_out = P(f"results_phase8_sampling{args.out_tag}.csv")
    with sum_out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["contrast", "missing_groups", "delta_miou", "ci_lo", "ci_hi"])
        for name, vals in summary.items():
            for mm, d, lo, hi in vals:
                w.writerow([name, mm, f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    print(f"\n  paired contrasts, n={n} seeds (* = 95% interval excludes zero)\n")
    print(f"  {'contrast':16s} " + " ".join(f"m={m}".rjust(8) for m in range(max_missing + 1)))
    for name, vals in summary.items():
        cells = [f"{d:+7.2f}" + ("*" if (lo > 0 or hi < 0) else " ") for _m, d, lo, hi in vals]
        print(f"  {name:16s} " + " ".join(c.rjust(8) for c in cells))

    import collections
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for arm, seed, mm, v in rows:
        by[arm][mm].append(v)
    print("\n  absolute mIoU by arm (mean over seeds):")
    print(f"  {'arm':6s} " + " ".join(f"m={m}".rjust(7) for m in range(max_missing + 1)))
    for arm in PS.ARMS:
        print(f"  {arm:6s} " + " ".join(f"{np.mean(by[arm][mm]):6.2f}".rjust(7)
                                        for mm in range(max_missing + 1)))

    stamp(sum_out, args, extra={"merged_from": [Path(r).name for r in args.raw],
                                "seeds": seeds, "anchor_seeds": args.anchor_seeds,
                                "anchor_worst_delta": (float(worst) if isinstance(worst, float) else None),
                                "dataset": "cloudsen12", "arms": list(PS.ARMS)})
    stamp(raw_out, args, extra={"n_rows": len(rows), "merged_from": [Path(r).name for r in args.raw]})
    print(f"\n  wrote {sum_out.name} + {raw_out.name}")


if __name__ == "__main__":
    main()
