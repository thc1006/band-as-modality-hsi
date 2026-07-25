#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S9 — generate paper-ready tables from the produced result CSVs (reproducible; no re-training).

Reads paper/results_phase2_curve.csv (the phase-2 degradation curve for baselines B1/B2/B3/B4/B6 and
Proposed -- there is no B5) and emits:
  paper/tables/baselines_table.tex   - LaTeX tabular fragment; the caller must \\usepackage{booktabs}
  paper/tables/baselines_table.md    - the same table as markdown
The endpoint column is labelled from the largest validated `missing_groups`, not hardcoded.

FAIL-CLOSED ON ITS INPUT. This script turns a CSV into a table a reviewer reads as a result, so
every way the CSV can be wrong has to stop it rather than change the numbers quietly. The one that
motivated the rest: `bandsim.metrics.audc` sorts by x internally, while `clean` and `max-miss` used
to be read positionally as the first and last ROW. On a CSV whose rows had been reordered -- by a
spreadsheet round-trip, a manual edit, a merge -- AUDC stayed correct and everything beside it
silently did not. Measured on x=[2,0,1], y=[60,80,70]: the table said clean 60.0, max-miss 70.0,
retention 116.7%, where the truth is 80.0, 60.0 and 75.0%, and only the >100% retention hinted at it.

The current producer emits a sorted, complete, gapless curve, so none of these guards fire today.
They exist because this file's output outlives the run that produced it.

RETENTION HERE IS A RATIO OF MEANS: mean(max-miss) / mean(clean), NOT the mean over seeds of each
seed's own retention. The two differ, and can differ a lot -- for seeds (clean 100, final 50) and
(clean 10, final 9) the mean of ratios is 70.0% and the ratio of means is 53.6%. The per-seed values
are not in this CSV, which carries only means and stds, so the mean-of-ratios cannot be computed
here and the column says which one it is.

Run: python experiments/make_paper_tables.py [--input CSV] [--output-dir DIR]
"""
import os
import re
import csv
import sys
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.metrics import audc, retention                                    # noqa: E402
from bandsim.provenance import file_sha256                                     # noqa: E402

PAPER = os.path.normpath(os.path.join(_HERE, "..", "paper"))

METHODS = [("b1", "B1 MLP + zero-fill (no defense)"),
           ("b2", "B2 MLP + band-group dropout"),
           ("b3", "B3 MLP + spectral interpolation"),
           ("b4", "B4 ChannelViT-style (learned emb + HCS)"),
           ("b6", "B6 SatMAE-style (learned emb + group MAE)"),
           ("proposed", "Proposed (wavelength PE + SGMAE + attn)")]

MIOU_RANGE = (0.0, 100.0)      # this pipeline reports mIoU in percent
DECIMALS = 1                   # the precision the table PRINTS; the winner rule must match it


def latex_escape(s):
    """Escape the characters that would break, or silently re-shape, a LaTeX table cell.

    The current method names contain none of them. They are escaped anyway because the names are the
    part of this table a future edit is most likely to touch, and an unescaped `&` adds a column
    while an unescaped `%` comments out the rest of the line -- both of which produce a
    plausible-looking table rather than an error."""
    out = re.sub(r"([&%$#_{}])", r"\\\1", str(s))
    return out.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def load_curve(path):
    """Read the degradation curve, or raise. Returns (xs sorted, {method: curve aligned to xs}).

    Every check below guards a failure that would otherwise reach the paper as a number."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields:
        raise ValueError(f"{path}: no header row")
    if not rows:
        raise ValueError(f"{path}: header only, no data rows -- nothing to tabulate")

    required = {"missing_groups"} | {f"{k}_mean" for k, _ in METHODS}
    absent = sorted(required - set(fields))
    if absent:
        # Used to be `if col in rows[0]`, which produced a table of whatever HAPPENED to be present:
        # a CSV holding only b1 yielded a complete-looking one-method table, and a renamed or
        # dropped baseline vanished from the paper without a word.
        raise ValueError(f"{path}: missing required columns {absent}. This table names a fixed set "
                         f"of methods; a partial CSV must not produce a partial table.")
    extra = sorted(c for c in fields if c.endswith("_mean")
                   and c not in {f"{k}_mean" for k, _ in METHODS})
    if extra:
        raise ValueError(f"{path}: CSV carries method columns {extra} that this table does not "
                         f"know. Add them to METHODS, or the paper table silently omits a method "
                         f"the experiment ran.")

    try:
        xs = np.array([int(r["missing_groups"]) for r in rows])
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: non-integer missing_groups ({e})")
    if len(np.unique(xs)) != len(xs):
        dup = sorted({int(v) for v in xs if int((xs == v).sum()) > 1})
        raise ValueError(f"{path}: duplicate missing_groups {dup} -- the curve has no unique value "
                         f"there, so neither the endpoints nor the area are defined.")
    if 0 not in xs:
        raise ValueError(f"{path}: no missing_groups=0 row. The clean column and the retention "
                         f"denominator ARE that point; without it the first level would be "
                         f"reported as 'clean' (grid was {sorted(xs.tolist())}).")

    order = np.argsort(xs)                    # ONE sort, shared by the endpoints and by audc()
    xs = xs[order]
    curves = {}
    for k, _ in METHODS:
        v = np.array([float(r[f"{k}_mean"]) for r in rows], dtype=float)[order]
        if not np.isfinite(v).all():
            raise ValueError(f"{path}: {k}_mean has {int((~np.isfinite(v)).sum())} non-finite "
                             f"values; one of them poisons AUDC into nan for that method.")
        lo, hi = MIOU_RANGE
        if v.min() < lo or v.max() > hi:
            raise ValueError(f"{path}: {k}_mean spans [{v.min():.3f}, {v.max():.3f}], outside the "
                             f"expected mIoU percent range {MIOU_RANGE}.")
        # The range check ALONE does not catch the mix-up it was written for: a fraction-scale
        # curve (0.0-1.0) sits comfortably inside [0, 100] and would print a 100x-wrong table that
        # still looks like a table. A whole curve at or below 1.0 is a scale error in every
        # realistic case -- a model that genuinely scored under 1 mIoU has nothing to tabulate.
        if v.max() <= 1.0:
            raise ValueError(f"{path}: {k}_mean never exceeds 1.0 (max {v.max():.4f}). This "
                             f"pipeline reports mIoU in PERCENT; a fraction-scale curve passes the "
                             f"range check above and prints a 100x-wrong table.")
        curves[k] = v
    return xs, curves


def summarise(xs, curves):
    """One (name, clean, final, audc, retention%) tuple per method, endpoints taken BY VALUE."""
    i0 = int(np.flatnonzero(xs == 0)[0])
    i1 = int(np.argmax(xs))                   # by value, not by row position
    out = []
    for k, name in METHODS:
        c = curves[k]
        out.append((name, float(c[i0]), float(c[i1]), audc(xs, c),
                    retention(float(c[i0]), float(c[i1])) * 100.0))
    return out


def render(xs, rows, src, digest):
    """Both tables as strings, rendered BEFORE anything is written, so a failure in the second
    cannot leave a new markdown table beside a stale LaTeX one."""
    end = int(xs.max())
    # The winner is decided on the value the reader SEES. Deciding on the raw float printed 80.04
    # and 80.03 as two identical "80.0" cells with only one of them bold.
    shown = [round(r[3], DECIMALS) for r in rows]
    best = max(shown)
    note = f"source: {os.path.basename(src)} sha256={digest[:16]} grid={xs.tolist()}"

    md = [f"<!-- generated by make_paper_tables.py -- {note} -->",
          f"| Method | clean mIoU | {end}-miss mIoU | AUDC | Retention of means (%) |",
          "|---|---|---|---|---|"]
    for (name, clean, last, a, ret), s in zip(rows, shown):
        cell = f"**{a:.{DECIMALS}f}**" if s == best else f"{a:.{DECIMALS}f}"
        md.append(f"| {name.replace('|', chr(92) + '|')} | {clean:.1f} | {last:.1f} | {cell} "
                  f"| {ret:.1f} |")

    tex = [f"% generated by make_paper_tables.py -- {note}",
           r"% requires \usepackage{booktabs}; this is a tabular fragment, not a float",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Method & Clean mIoU & %d-miss mIoU & AUDC & Retention of means (\%%) \\" % end,
           r"\midrule"]
    for (name, clean, last, a, ret), s in zip(rows, shown):
        cell = (r"\textbf{%.*f}" % (DECIMALS, a)) if s == best else ("%.*f" % (DECIMALS, a))
        tex.append(f"{latex_escape(name)} & {clean:.1f} & {last:.1f} & {cell} & {ret:.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n", best


def write_atomically(pairs):
    """Write every (path, text) or none of them.

    The two tables carry the same numbers and are published together; writing them in sequence meant
    a failure between them left a new markdown table beside a stale LaTeX one -- worse than either
    being missing, because both still look valid."""
    staged = []
    try:
        for path, text in pairs:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            staged.append((tmp, path))
        for tmp, path in staged:
            os.replace(tmp, path)             # atomic rename on POSIX
    finally:
        for tmp, _ in staged:
            if os.path.exists(tmp):
                os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(PAPER, "results_phase2_curve.csv"))
    ap.add_argument("--output-dir", default=os.path.join(PAPER, "tables"))
    args = ap.parse_args()
    # Not at import time: this module is imported by its tests, and importing it should not create
    # directories under paper/.
    os.makedirs(args.output_dir, exist_ok=True)

    xs, curves = load_curve(args.input)
    rows = summarise(xs, curves)
    digest = file_sha256(args.input) or "unavailable"
    md, tex, best = render(xs, rows, args.input, digest)
    write_atomically([(os.path.join(args.output_dir, "baselines_table.md"), md),
                      (os.path.join(args.output_dir, "baselines_table.tex"), tex)])

    print(f"Baselines table (from {args.input}, sha256={digest[:16]}):")
    print(md)
    gaps = sorted(set(range(int(xs.min()), int(xs.max()) + 1)) - set(xs.tolist()))
    if gaps:
        print(f"NOTE: the sweep skips {gaps}; AUDC normalises over the full x-range, so it is the "
              f"mean curve height over that range, not a mean over the levels actually tested.")
    print(f"wrote {args.output_dir}/baselines_table.{{tex,md}}  (best AUDC = {best:.{DECIMALS}f}; "
          f"ties at the printed precision share the bold)")


if __name__ == "__main__":
    main()
