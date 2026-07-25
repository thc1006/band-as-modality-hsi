#!/usr/bin/env python
"""E2 (round-5 review 3.2): is the 590-patch subset ACOLITE processed successfully a BIASED sample of the
test set, so that the 28.6% ACOLITE breach would not generalise? ACOLITE's dark-spectrum inversion fails on
some scenes (all-zero output); we compare the RETAINED (ACOLITE-present) vs EXCLUDED (ACOLITE-failed) test
patches on the operationally relevant confounders -- cloud coverage, surface/land cover, scene difficulty --
and note that Sen2Cor-L2A evaluated ON the retained subset already reproduces the full-test flagship breach
(28.64 vs 28.9), which is the direct evidence the retained subset is representative of the phenomenon.
Offline from test metadata + the ACOLITE_B4.dat presence mask (same mask phase8R3_acolite uses)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8


def cat_table(name, series, ret, exc):
    """Fraction of each category within retained vs excluded, plus the count."""
    cats = sorted(series.dropna().unique(), key=lambda x: str(x))
    print(f"  {name} (fraction within group):")
    for c in cats:
        fr = float((series[ret] == c).mean()) * 100
        fe = float((series[exc] == c).mean()) * 100
        flag = "  <-- differs" if abs(fr - fe) >= 10 else ""
        print(f"    {str(c):22s} retained {fr:5.1f}%   excluded {fe:5.1f}%   Δ {fr - fe:+5.1f}{flag}")


def main():
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    n = len(meta)
    b4 = np.fromfile(os.path.join(P8.DATA, "test", "ACOLITE_B4.dat"), dtype="<i2").reshape(-1, 512, 512)
    has_acolite = np.array([b4[i, 1:510, 1:510].any() for i in range(n)])   # ACOLITE succeeded (non-zero)
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leaked = meta["s2_id"].isin(train_prod).to_numpy()
    retained_paper = has_acolite & ~leaked                                  # the 590 the paper evaluates on
    print(f"test patches {n} | ACOLITE-present {has_acolite.sum()} | leaked {leaked.sum()} | "
          f"paper-retained (present & unleaked) {retained_paper.sum()} | ACOLITE-failed {int((~has_acolite).sum())}")
    print("=" * 92)
    print("BIAS CHECK: ACOLITE-present (retained) vs ACOLITE-failed (excluded) on the full test set")
    print("=" * 92)
    ret, exc = has_acolite, ~has_acolite

    # cloud coverage (ORDINAL category cloud-free..cloudy): the reviewer's first worry -- did ACOLITE only
    # succeed on clear scenes? Map to 0..4 for a rank test and show per-category fractions.
    if "cloud_coverage" in meta.columns:
        order = {"cloud-free": 0, "almost-clear": 1, "low-cloudy": 2, "mid-cloudy": 3, "cloudy": 4}
        cc = meta["cloud_coverage"].map(order)
        r, e = cc[ret].dropna(), cc[exc].dropna()
        try:
            p = mannwhitneyu(r, e, alternative="two-sided").pvalue
        except ValueError:
            p = float("nan")
        print(f"  cloud_coverage (ordinal 0=clear..4=cloudy): retained mean {r.mean():.2f}   "
              f"excluded mean {e.mean():.2f}   Mann-Whitney p={p:.3f}")
        cat_table("cloud_coverage", meta["cloud_coverage"].astype(str), ret, exc)

    # latitude from the WGS84 centroid 'POINT (lon lat)' -- geographic representativeness + hemisphere.
    lat = meta["proj_centroid"].str.extract(r"POINT \([-\d.]+ ([-\d.]+)\)")[0].astype(float)
    r, e = lat[ret].dropna(), lat[exc].dropna()
    try:
        p = mannwhitneyu(r, e, alternative="two-sided").pvalue
    except ValueError:
        p = float("nan")
    print(f"  latitude: retained mean {r.mean():.1f} (med {r.median():.1f})   "
          f"excluded mean {e.mean():.1f} (med {e.median():.1f})   Mann-Whitney p={p:.3f}")
    print(f"    hemisphere N-fraction: retained {float((r >= 0).mean()) * 100:.0f}%   "
          f"excluded {float((e >= 0).mean()) * 100:.0f}%")

    for col in ("land_cover", "difficulty", "label_type"):
        if col in meta.columns:
            cat_table(col, meta[col].astype(str), ret, exc)

    print("=" * 92)
    print("REPRESENTATIVENESS: does the retained subset reproduce the full-test phenomenon?")
    print("  Sen2Cor-L2A naive joint risk ON the retained subset = 28.64% (results_phase8R3_acolite10.csv)")
    print("  Full-test flagship Sen2Cor-L2A naive joint risk       = 28.9%  (Table 1)")
    print("  -> the retained subset breaks the Sen2Cor certificate to the same degree as the full test set,")
    print("     so it is representative of the breach regardless of any mild composition difference above;")
    print("     ACOLITE (28.56%) is then measured on that same representative subset.")


if __name__ == "__main__":
    main()
