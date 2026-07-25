#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the EMIT C-f generality-vs-NDVI trend from results_phase8F_multi.csv.

Panel A: per-PIXEL Spearman(EMIT uncertainty, held-out recon error) vs scene NDVI -- shows the
         spatial reliability<->uncertainty link is strong for low-NDVI (bare/arid) scenes and
         inverts for dense vegetation (the brightness<->spectral-complexity confound).
Panel B: per-BAND (spectral) Spearman vs NDVI -- shows the spectral link is universally positive
         across biomes (the robust, biome-independent grounding of the reliability signal).

Output: paper/fig_emit_ndvi_trend.png (+ .pdf)
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(_HERE, "..", "paper"))
CSV = os.path.join(PAPER, "results_phase8F_multi.csv")


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    g = lambda k: np.array([float(r[k]) for r in rows])
    names = [r["granule"] for r in rows]
    return names, g


def main():
    names, g = load()
    ndvi = g("median_ndvi")
    sgpix = g("sgmae_perpixel"); sgstd = g("sgmae_perpixel_std")
    pcpix = g("pca_perpixel"); part = g("sgmae_partial_brightness")
    sgband = g("sgmae_perband_raw")                              # raw reflectance-unit per-band (physical)
    n = len(names)
    tr_pix = spearmanr(ndvi, sgpix).correlation
    tr_part = spearmanr(ndvi, part).correlation
    tr_band = spearmanr(ndvi, sgband).correlation

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel A: per-pixel vs NDVI ----
    axA.axhline(0, color="0.6", lw=1, ls="--", zorder=1)
    axA.errorbar(ndvi, sgpix, yerr=sgstd, fmt="o", ms=8, color="#1f77b4", capsize=3,
                 label=f"SGMAE per-pixel (ρ vs NDVI={tr_pix:+.2f})", zorder=3)
    axA.scatter(ndvi, pcpix, marker="s", s=45, facecolors="none", edgecolors="#7f7f7f",
                label="linear PCA proxy", zorder=2)
    axA.scatter(ndvi, part, marker="^", s=45, color="#ff7f0e",
                label=f"partial (ctrl brightness, ρ={tr_part:+.2f})", zorder=2)
    # linear trend line
    if n >= 3:
        b, a = np.polyfit(ndvi, sgpix, 1)
        xs = np.linspace(ndvi.min(), ndvi.max(), 50)
        axA.plot(xs, a + b * xs, color="#1f77b4", lw=1.2, alpha=0.5, zorder=1)
    for x, y, nm in zip(ndvi, sgpix, names):
        axA.annotate(nm, (x, y), fontsize=6.5, xytext=(3, 4), textcoords="offset points", color="0.3")
    axA.set_xlabel("scene median NDVI (bare/arid  →  dense vegetation)")
    axA.set_ylabel("Spearman(EMIT uncertainty, recon error)")
    axA.set_title(f"(A) per-PIXEL uncertainty–reconstruction correlation vs scene NDVI (n={n}, trend rho={tr_pix:+.2f})")
    axA.legend(fontsize=8, loc="lower left"); axA.grid(alpha=0.25)

    # ---- Panel B: per-band vs NDVI ----
    axB.axhline(0, color="0.6", lw=1, ls="--")
    axB.scatter(ndvi, sgband, marker="o", s=55, color="#2ca02c", zorder=3)
    for x, y, nm in zip(ndvi, sgband, names):
        axB.annotate(nm, (x, y), fontsize=6.5, xytext=(3, 4), textcoords="offset points", color="0.3")
    axB.set_ylim(min(-0.1, sgband.min() - 0.1), max(0.7, sgband.max() + 0.1))
    axB.set_xlabel("scene median NDVI")
    axB.set_ylabel("per-band Spearman (full spectrum)")
    axB.set_title(f"(B) per-BAND (raw) correlation vs scene NDVI (trend rho={tr_band:+.2f}, mean={sgband.mean():+.2f})")
    axB.grid(alpha=0.25)

    fig.suptitle("EMIT real retrieval-uncertainty vs self-supervised reconstruction reliability, across "
                 f"{n} granules spanning NDVI {ndvi.min():+.2f}..{ndvi.max():+.2f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # figs/, matching every other figure in the paper. This wrote to paper/ root, so the EMIT trend
    # would not have been picked up by anything globbing paper/figs/*.pdf (the harness backup does).
    out = os.path.join(PAPER, "figs", "fig_emit_ndvi_trend.png")
    fig.savefig(out, dpi=150); fig.savefig(out.replace(".png", ".pdf"))
    print(f"per-pixel trend Spearman(NDVI, corr) = {tr_pix:+.3f}  (partial: {tr_part:+.3f})")
    print(f"per-band  trend Spearman(NDVI, corr) = {tr_band:+.3f}  (mean per-band {sgband.mean():+.3f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
