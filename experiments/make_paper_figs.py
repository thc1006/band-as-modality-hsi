#!/usr/bin/env python3
"""Publication figures for the robustness != reliability paper, from the COMMITTED result CSVs.
Fig 2 = flagship atmospheric shift (naive/control/mondrian joint risk + coverage per state).
Fig 3 = the domain-gap axes (surface, and geography once its run lands): naive vs mondrian on the
unseen target domain. Reads only committed CSVs; no re-run. Writes paper/figs/*.pdf + *.png."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, "..", "paper"))
FIGS = os.path.join(PAPER, "figs")
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})
NAIVE, CTRL, MOND = "#c0392b", "#8e6a00", "#1f6f3a"


def _rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


TCRIT = 2.262  # t(min(G1,G2)-1) = t(9): the balanced 10x10 crossed design's small-cluster critical
               # value (Cameron-Gelbach-Miller 2011). Error bars are the t(9) 95% CI = TCRIT * SE.


def fig2_flagship():
    rows = [r for r in _rows(os.path.join(PAPER, "results_phase8R_reliability_10seed.csv"))
            if r["method"] == "proposed"]
    if not rows:
        print("  [fig2] no 10-seed flagship CSV -- skip"); return
    states = ["clean", "dropB10", "dropB1B9B10", "L2A_real"]
    labels = ["clean\n(L1C)", "$-$B10", "$-$B1/B9/B10", "L2A\n(real shift)"]
    arms = [("naive", "Naive (clean-calibrated)", NAIVE),
            ("naiveThr_freshT", "Naive thr + state temp (control)", CTRL),
            ("mondrian", "Degradation-aware (Mondrian)", MOND)]
    by = {(r["state"], r["arm"]): r for r in rows}
    x = np.arange(len(states)); w = 0.26
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for i, (arm, lab, col) in enumerate(arms):
        risk = [float(by[(s, arm)]["mean_heldout_joint_loss_pct"]) for s in states]
        se = [float(by[(s, arm)]["se_heldout_joint_loss_pct"]) for s in states]
        cov = [float(by[(s, arm)]["mean_coverage_pct"]) for s in states]
        sel = [float(by[(s, arm)]["mean_selection_stat_pct"]) for s in states]  # CRC calibration statistic
        xb = x + (i - 1) * w
        ax.bar(xb, risk, w, yerr=[TCRIT * e for e in se], color=col, label=lab, capsize=2,
               error_kw={"lw": 0.8})
        # the calibration statistic the certificate MINIMISES -- stays <= alpha even on L2A, which is
        # exactly why the failure is silent (dashed line sits at target while the realised risk bar soars)
        ax.plot(xb, sel, ls="--", lw=1.0, color=col, marker="o", ms=3.5, mfc="white", mew=1.0)
        axc.bar(xb, cov, w, color=col)
    ax.axhline(10, ls=":", color="k", lw=1, label="target $\\alpha=10\\%$")
    ax.plot([], [], ls="--", color="0.35", marker="o", ms=3.5, mfc="white",
            label="CRC calibration statistic")
    ax.set_ylabel("empirical joint risk\n$P(\\mathrm{acc}\\wedge\\mathrm{wrong})$  %", fontsize=8)
    axc.set_ylabel("coverage  %", fontsize=8)
    ax.set_title("(a) joint risk (source threshold)", fontsize=9, pad=6)
    axc.set_title("(b) coverage (abstention cost)", fontsize=9, pad=6)
    for a in (ax, axc):
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=7)
        a.margins(y=0.14)                                   # headroom: bars/markers never touch the frame top
    # legend OUTSIDE the axes, below both panels, so it can NEVER overlap the plotted data (fixes the
    # in-plot legend that crowded panel (a)); short labels keep it to two tidy rows.
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.3, handlelength=1.8)
    fig.suptitle("Naive conformal exceeds its nominal 10% target under the L1C$\\to$L2A shift while "
                 "Mondrian restores control (proposed, 100 runs, error bars $t_9$ 95% CI)", fontsize=7.5)
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig2_flagship_reliability.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  [fig2] wrote fig2_flagship_reliability.{pdf,png} (10-seed, t9 CI, calibration statistic)")


def fig3_domain_gaps():
    axes_data = []
    for stem, title in (("landcover", "surface gap\n(deploy unseen bright)"),
                        ("geography", "geographic gap\n(train N, deploy S)")):
        rows = [r for r in _rows(os.path.join(PAPER, f"results_phase8R2_{stem}_10seed.csv"))
                if r["method"] == "proposed"]
        if rows:
            axes_data.append((title, {r["arm"]: r for r in rows}))
    if len(axes_data) < 2:
        print(f"  [fig3] only {len(axes_data)}/2 axes (10-seed) present -- skip until geography lands")
        return
    arms = [("naive", "Naive (source-calibrated)", NAIVE),
            ("mondrian", "Mondrian (target-calibrated)", MOND)]
    x = np.arange(len(axes_data)); w = 0.35
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for j, (arm, lab, col) in enumerate(arms):
        v = [float(d[arm]["joint_risk_mean"]) for _, d in axes_data]
        se = [float(d[arm]["joint_risk_se"]) for _, d in axes_data]
        cov = [float(d[arm]["coverage_mean"]) for _, d in axes_data]
        ax.bar(x + (j - 0.5) * w, v, w, yerr=[TCRIT * e for e in se], color=col, label=lab, capsize=3)
        axc.bar(x + (j - 0.5) * w, cov, w, color=col)
    ax.axhline(10, ls=":", color="k", lw=1, label="target $\\alpha=10\\%$")
    for a, ttl, yl in ((ax, "(a) joint risk on the unseen domain", "joint risk  %"),
                       (axc, "(b) coverage (the abstention cost)", "coverage  %")):
        a.set_xticks(x); a.set_xticklabels([t for t, _ in axes_data], fontsize=7)
        a.set_ylabel(yl); a.set_title(ttl, fontsize=8); a.margins(y=0.14)
    handles, lbls = ax.get_legend_handles_labels()      # legend OUTSIDE the axes (below), never over data
    fig.legend(handles, lbls, loc="lower center", ncol=3, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Two operational deployment shifts on a fixed input "
                 "(proposed, 100 runs, error bars $t_9$ 95% CI)", fontsize=8)
    fig.tight_layout(rect=[0, 0.09, 1, 0.92])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig3_domain_gaps.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig3] wrote fig3_domain_gaps.{{pdf,png}} ({len(axes_data)} axes, 10-seed, t9 CI, +coverage)")


if __name__ == "__main__":
    fig2_flagship()
    fig3_domain_gaps()
