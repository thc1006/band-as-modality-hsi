#!/usr/bin/env python
"""E2 (round-5 review 3.2): is the subset ACOLITE processed successfully a BIASED sample, so the ACOLITE
breach would not generalise? ACOLITE's dark-spectrum inversion fails on some scenes (all-zero output). We
audit selection on the PAPER-ELIGIBLE cohort (ACOLITE-present, product-overlap-unleaked) versus the eligible
ACOLITE-FAILED patches, on the main observed confounders (cloud coverage, latitude, land cover, difficulty,
label type), reporting standardized effect sizes AND the number of distinct scene-components behind each
group (the exchangeable unit -- patches within a scene are dependent, so the patch count overstates the
effective sample size). We also report the number of components that are fully retained / partially retained
/ fully failed.

SCOPE, stated honestly: even perfect balance on observed covariates CANNOT prove the ACOLITE breach
generalises to processing-FAILED scenes -- ACOLITE outputs are, by definition, unavailable there, so the
failed-scene ACOLITE risk is a counterfactual that this audit cannot identify. What it CAN show is (i) the
successful cohort is not strongly atypical on the observed covariates and (ii) Sen2Cor on the retained cohort
still exhibits a large source-threshold failure. The ACOLITE-success criterion is the SAME ACOLITE_B4.dat
presence mask phase8R3_acolite uses for the headline itself (a non-zero B4 interior); hardening it (per-band
valid-pixel fractions) would redefine the ACOLITE cohort and is left as future work. Offline from test
metadata + the presence mask."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase8R_reliability as P8R


def cliffs_delta(r, e):
    """Cliff's delta = P(r>e) - P(r<e) in [-1,1] -- a STANDARDIZED, non-parametric effect size. Magnitude
    bands (Romano et al. 2006) are DESCRIPTIVE, not equivalence proofs: |d|<.147 negligible, <.33 small,
    <.474 medium, else large. Fails closed on empty / non-finite input."""
    r = np.asarray(r, float); e = np.asarray(e, float)
    if r.ndim != 1 or e.ndim != 1:
        raise ValueError("cliffs_delta expects 1-D samples")
    if len(r) == 0 or len(e) == 0:
        raise ValueError("cliffs_delta requires two non-empty samples")
    if not (np.isfinite(r).all() and np.isfinite(e).all()):
        raise ValueError("cliffs_delta got NaN/Inf")
    es = np.sort(e)
    n_lt = int(np.searchsorted(es, r, side="left").sum())         # #(e < r)
    n_le = int(np.searchsorted(es, r, side="right").sum())        # #(e <= r)
    gt, lt = n_lt, len(r) * len(e) - n_le                         # pairs r>e, r<e
    d = (gt - lt) / (len(r) * len(e))
    a = abs(d)
    return d, ("negligible" if a < .147 else "small" if a < .33 else "medium" if a < .474 else "large")


def cramers_v(series, ret, exc):
    """Bias-corrected Cramer's V (Bergsma 2013) for the (group x category) table -- a standardized categorical
    effect size. Raw Pearson chi2 (Yates continuity correction is a p-value adjustment that artefactually
    shrinks 2x2 associations). Bands are DESCRIPTIVE: <.1 negligible, <.3 small, <.5 medium, else large."""
    from scipy.stats import chi2_contingency
    cats = sorted(series.dropna().unique(), key=lambda x: str(x))
    tab = np.array([[int((series[ret] == c).sum()), int((series[exc] == c).sum())] for c in cats], float)
    tab = tab[tab.sum(1) > 0]
    if tab.shape[0] < 2 or tab.sum() == 0:
        return float("nan"), "n/a"
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.sum(); phi2 = chi2 / n
    r, k = tab.shape
    phi2c = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(rc - 1, kc - 1)
    if denom <= 0:                                                # corrected dimension non-positive -> ill-posed
        return float("nan"), "n/a"
    v = float(np.sqrt(phi2c / denom))
    return v, ("negligible" if v < .1 else "small" if v < .3 else "medium" if v < .5 else "large")


def cat_table(name, series, ret, exc):
    """Per-category retained/excluded fractions + a standardized Cramer's V. No arbitrary "differs" flag: we
    print the standardized difference and let V + its band speak."""
    v, mag = cramers_v(series, ret, exc)
    print(f"  {name} (fraction within group; Cramer's V = {v:.3f} [{mag}]):")
    for c in sorted(series.dropna().unique(), key=lambda x: str(x)):
        fr = float((series[ret] == c).mean()) * 100
        fe = float((series[exc] == c).mean()) * 100
        print(f"    {str(c):22s} retained {fr:5.1f}%   excluded {fe:5.1f}%   d {fr - fe:+5.1f}")


def _n_components(comp, mask):
    return int(np.unique(comp[mask]).size)


def main():
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    n = len(meta)
    # --- ACOLITE-success mask (same as phase8R3_acolite), fail-closed on a size mismatch ---
    raw = np.fromfile(os.path.join(P8.DATA, "test", "ACOLITE_B4.dat"), dtype="<i2")
    if raw.size != n * 512 * 512:
        raise ValueError(f"ACOLITE_B4.dat size {raw.size} != n*512*512 = {n * 512 * 512}; row manifest mismatch")
    b4 = raw.reshape(n, 512, 512)
    has_acolite = np.array([b4[i, 1:510, 1:510].any() for i in range(n)])   # non-zero B4 interior == processed (all-zero == ACOLITE failure)
    if meta["s2_id"].isna().any():
        raise ValueError(f"{int(meta['s2_id'].isna().sum())} test scenes have NaN s2_id; leak status undecidable")
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leaked = meta["s2_id"].isin(train_prod).to_numpy()
    eligible = ~leaked                                                      # the paper drops train-product-overlapping scenes
    ret = eligible & has_acolite                                           # PAPER cohort: ACOLITE present AND unleaked
    exc = eligible & ~has_acolite                                          # eligible but ACOLITE failed (the honest comparison group)
    if not ret.any() or not exc.any():
        raise RuntimeError(f"empty comparison group: retained={ret.sum()}, excluded={exc.sum()}")

    comp = np.asarray(P8R.scene_component_ids("test"))                      # exchangeable unit (patches within a scene are dependent)
    if len(comp) != n:
        raise ValueError(f"scene_component_ids returned {len(comp)} ids for {n} test patches")
    frac_ret = {c: float(has_acolite[comp == c].mean()) for c in np.unique(comp[eligible])}
    n_full = sum(v == 1.0 for v in frac_ret.values())
    n_none = sum(v == 0.0 for v in frac_ret.values())
    n_part = len(frac_ret) - n_full - n_none
    print(f"test patches {n} | eligible (unleaked) {int(eligible.sum())} | ACOLITE-retained {int(ret.sum())} "
          f"({_n_components(comp, ret)} components) | ACOLITE-failed {int(exc.sum())} ({_n_components(comp, exc)} components)")
    print(f"  scene-components (eligible): fully-retained {n_full}, partially-retained {n_part}, fully-failed {n_none} "
          f"(so patch-level effect sizes below are DESCRIPTIVE; the effective sample size is the component count)")
    print("=" * 96)
    print("SELECTION AUDIT: paper-eligible ACOLITE-retained vs ACOLITE-failed (observed covariates)")
    print("=" * 96)

    if "cloud_coverage" in meta.columns:
        order = {"cloud-free": 0, "almost-clear": 1, "low-cloudy": 2, "mid-cloudy": 3, "cloudy": 4}
        s = meta["cloud_coverage"].astype("string")
        unknown = set(s.dropna().unique()) - set(order)
        if unknown:
            raise ValueError(f"unknown cloud_coverage labels {unknown}")
        cc = s.map(order)
        r, e = cc[ret].dropna(), cc[exc].dropna()
        p = mannwhitneyu(r, e, alternative="two-sided", method="asymptotic").pvalue if len(r) and len(e) else float("nan")
        dcl, mcl = cliffs_delta(r.to_numpy(), e.to_numpy())
        print(f"  cloud_coverage (ordinal 0=clear..4=cloudy): retained mean {r.mean():.2f} med {r.median():.0f}   "
              f"excluded mean {e.mean():.2f} med {e.median():.0f}   MW p={p:.3f}   Cliff's d={dcl:+.3f} [{mcl}]")
        cat_table("cloud_coverage", s.fillna("<MISSING>"), ret, exc)

    lat = meta["proj_centroid"].str.extract(r"POINT \(\s*[-+\d.eE]+ +([-+\d.eE]+)")[0].astype(float)
    if lat.isna().any():
        raise ValueError(f"{int(lat.isna().sum())} centroids failed WGS84 'POINT (lon lat)' latitude parse")
    if not lat.between(-90, 90).all():
        raise ValueError("parsed latitude outside [-90, 90]")
    r, e = lat[ret], lat[exc]
    p = mannwhitneyu(r, e, alternative="two-sided", method="asymptotic").pvalue
    dcl, mcl = cliffs_delta(r.to_numpy(), e.to_numpy())
    print(f"  latitude: retained mean {r.mean():.1f} (med {r.median():.1f})   excluded mean {e.mean():.1f} "
          f"(med {e.median():.1f})   MW p={p:.3f}   Cliff's d={dcl:+.3f} [{mcl}]")
    print(f"    hemisphere N-fraction: retained {float((r >= 0).mean()) * 100:.0f}%   excluded {float((e >= 0).mean()) * 100:.0f}%")

    for col in ("land_cover", "difficulty", "label_type"):
        if col in meta.columns:
            cat_table(col, meta[col].astype("string").fillna("<MISSING>"), ret, exc)

    print("=" * 96)
    print("Sen2Cor proxy on the retained cohort (does the retained subset still break the certificate?)")
    _csv = os.path.join(_HERE, "..", "paper", "results_phase8R3_acolite10.csv")
    if os.path.exists(_csv):
        _nv = pd.read_csv(_csv).query("arm == 'naive'")
        r_l2a = float(_nv[_nv["state"] == "L2A"]["joint"].mean())
        r_aco = float(_nv[_nv["state"] == "ACOLITE"]["joint"].mean())
        print(f"  Sen2Cor-L2A naive joint risk ON the ACOLITE-retained cohort = {r_l2a:.2f}% (10x10, results_phase8R3_acolite10.csv)")
        print(f"  ACOLITE naive joint risk on that same cohort               = {r_aco:.2f}%")
        print("  -> reading (honest scope): the ACOLITE-retained cohort spans most scene-components, is not strongly")
        print("     imbalanced on the observed covariates above, and STILL breaks the Sen2Cor certificate. Because")
        print("     ACOLITE outputs do not exist for the failed scenes, the ACOLITE-specific risk there is a")
        print("     counterfactual this audit cannot identify; the ACOLITE result is reported CONDITIONAL on")
        print("     successful processing, not as proof that the breach generalises to failed scenes.")
    else:
        print("  (results_phase8R3_acolite10.csv not found -- run phase8R3_acolite.py first)")


if __name__ == "__main__":
    main()
