#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 9 (E2) — 6S-transmittance-weighted synthetic group dropout: does WHICH groups go missing
change the reliability picture, holding HOW MANY fixed?

Robustness studies drop spectral bands uniformly at random. Operationally, loss is not uniform: the
atmosphere attenuates the water-vapour windows, Sen2Cor removes B10 from every L2A product, a failed
detector takes a contiguous stripe. This asks whether a non-uniform SELECTION POLICY changes the
reliability conclusion at a fixed number of dropped groups.

NAMING, DELIBERATELY NARROW. The two arms are `uniform` and `transmittance_weighted`. They are NOT
"random versus physical" and this file no longer says "the bands the atmosphere takes", for two
reasons:
  * no radiative transfer is simulated. A whole group is DELETED; transmittance only sets the
    probability of selecting it. Attenuation, path radiance, aerosol scattering, adjacency, detector
    behaviour and the retrieval itself are all absent. phase3/phase5 are where attenuation is applied.
  * the Indian Pines CORRECTED cube has already had the deepest water-absorption bands removed
    (indices 104-108, 150-163 and 220 of the 220-band acquisition; 200 remain). Measured on the axis
    this file uses, only 3 bands survive in 1350-1420 nm and 3 in 1800-1950 nm. What the weighting
    emphasises is therefore the SHOULDERS of those windows, not their cores.
This is a physics-INFORMED sampling policy on a nominal wavelength axis, not a physical forward model.

THE POLICY. Group g is selected with probability proportional to (1 - T_g)^tau, where T_g is the mean
6S gaseous transmittance over the group's bands at the stated column water vapour. tau = 0 recovers
the uniform policy EXACTLY, so both arms are one sampler at two settings of one knob rather than two
code paths that could differ for unrelated reasons. It must be a distribution and not the
deterministic worst-m set: a fixed drop set gives every calibration row the same mask, one draw in
the mask dimension, and CRC has nothing to calibrate against.

Reported as a 2x2 over (calibration policy, deployment policy). NO outcome is asserted here in
advance; an earlier version of this docstring pre-labelled the cells "holds" and "BREACHES" before
any run, which is not a hypothesis, it is a conclusion written ahead of its evidence.

THE ESTIMAND. CRC (Angelopoulos et al. 2022) Theorem 2.1 bounds a MARGINAL expectation E[L_{n+1}]
over the joint draw of calibration and test under exchangeability. The primary statistic here is
therefore the MEAN held-out joint loss over independent SEEDS, with a standard error. The fraction of
runs whose empirical loss exceeded alpha is reported too, but only as a DESCRIPTIVE column: bounding
a mean implies nothing directly about an exceedance probability, which needs a high-probability
(RCPS/PAC) result instead. Drop counts m are repeated measures within a seed -- one scene, one
trained model, one block split -- so they are averaged within seed before the SE is taken across
seeds, never pooled as if each cell were an independent replication.

INDEPENDENT MASK PER BLOCK. Each spatial block draws its OWN mask. An earlier version drew one mask
per trial and applied it to every row, so all blocks took the same corruption shock inside a trial:
they were dependent conditional on the draw, and the mask dimension contributed `trials` independent
draws while the bound was computed from the block count.

THE MODEL'S TRAINING IS UNIFORM. `finetune_proposed` applies band-group dropout with a uniform group
choice, so under the weighted policy the model is off-distribution in what it was trained to expect
as well as in what it is evaluated on. That is the operational situation, and it means a gap between
the arms cannot be attributed to the evaluation policy alone.

EXCHANGEABILITY HERE IS APPROXIMATE. Calibration and evaluation blocks are disjoint, but adjacent
blocks of one scene stay spatially correlated. phase8R holds the rigorous ROI-level version on real
CloudSEN12. What this file supports is a COMPARISON between policies computed identically on both
arms.

Output (../paper/):
  results_phase9_physics_vs_random{sfx}.csv    per (seed, m, calib_policy, deploy_policy), with the
                                               selected threshold and temperature for audit
  results_phase9_summary{sfx}.csv              seed-clustered mean loss ± SE per 2x2 cell
  figs/fig_phase9_physics_vs_random{sfx}.pdf
"""
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from bandsim import hw, parallel                                        # noqa: E402
from bandsim.io import AVIRIS_WL_NM, disjoint_block_split                # noqa: E402
from bandsim.grouping import contiguous_groups, group_center_wavelengths  # noqa: E402
from bandsim.model import GroupedCrossBandAttention                     # noqa: E402
from bandsim.reliability import (conformal_risk_control, confidence_msp,  # noqa: E402
                                 fit_temperature)
from bandsim.provenance import stamp, file_sha256                       # noqa: E402
import phase2_degradation as P2                                         # noqa: E402

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(os.path.join(PAPER_DIR, "figs"), exist_ok=True)


def P(rel):
    return os.path.join(PAPER_DIR, rel)


LUT = os.path.join(os.path.dirname(_HERE), "data", "srf_cache", "T_6s_grid.npz")
TARGET_RISK = 0.10          # alpha: the certificate promises E[P(accepted AND wrong)] <= this
POLICIES = ("uniform", "transmittance_weighted")


# ------------------------------------------------------------------ the drop-selection policy
def group_transmittance(groups, cwv_key):
    """Mean 6S gaseous transmittance per band group, on the cube's own wavelength axis.

    Reads the table through its own identity checks rather than np.load: the axis must match
    AVIRIS_WL_NM exactly and the table must be `direct_6s`. An interpolated table reproduces the axis
    and its hash and is still not what 6S returns at absorption edges -- which is the whole content of
    the word "physical" in this experiment's title."""
    from bandsim.atmosphere import load_cached_transmittance
    T = load_cached_transmittance(LUT, cwv_key, n_bands=len(AVIRIS_WL_NM),
                                  expected_wavelengths_nm=AVIRIS_WL_NM,
                                  require_generation_mode="direct_6s")
    return np.array([float(np.mean(T[np.asarray(ix)])) for ix in groups])


def drop_weights(t_group, tau):
    """P(group is dropped) ∝ (1 - T_g)^tau, normalised. tau=0 is EXACTLY uniform.

    Raises rather than renormalising if every group is fully transmitting: that would silently hand
    back the uniform policy under the name "physical" and the 2x2 would compare a thing with itself."""
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if tau == 0:
        return np.full(len(t_group), 1.0 / len(t_group))
    a = np.clip(1.0 - np.asarray(t_group, float), 0.0, None) ** float(tau)
    s = a.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(
            f"physical drop weights are all zero (tau={tau}, transmittances={np.round(t_group,4)}). "
            f"With nothing absorbing, 'physical' would fall back to uniform and the comparison would "
            f"be a policy against itself.")
    return a / s


def sample_drop(rng, G, m, weights):
    if m <= 0:
        return []
    return sorted(rng.choice(G, size=m, replace=False, p=weights).tolist())


# ------------------------------------------------------------------------------- logits pooling
def pooled_logits(model, X, y, gid, groups, m, trials, rng, weights, record=None):
    """Pool logits over `trials` draws, with an INDEPENDENT mask per exchangeable unit.

    This used to draw ONE mask per trial and apply it to every row. Every spatial block then took the
    same corruption shock inside a trial, so blocks were dependent conditional on the draw and the
    mask dimension contributed `trials` independent draws -- not one per block. Handing CRC the block
    id as its group fixes within-block correlation and does nothing about that common shock, so the
    effective number of independent draws was ~`trials` (8) while the bound was computed as if it
    were the block count (~26). Each block now draws its own mask.

    `gid` is tiled with the repeats: a pixel reappears once per trial, and counting those repeats as
    independent calibration units would shrink CRC's B/(n+1) term by a factor of `trials`."""
    import torch
    G = len(groups)
    L, Y, U = [], [], []
    dev = next(model.parameters()).device
    units = np.unique(gid)
    for t in range(trials if m > 0 else 1):
        pm = np.ones((X.shape[0], G), bool)
        for u in units:
            sel = gid == u
            drop = sample_drop(rng, G, m, weights)
            row = np.ones(G, bool)
            row[drop] = False
            pm[sel] = row
            if record is not None:
                record.append({"trial": t, "unit": int(u), "dropped": tuple(drop)})
        with torch.no_grad():
            lg = model(torch.from_numpy(X).to(dev), torch.from_numpy(pm).to(dev)).cpu().numpy()
        L.append(lg); Y.append(y); U.append(gid)
    return np.concatenate(L), np.concatenate(Y), np.concatenate(U)


# ---------------------------------------------------------------------------------- one seed
def run_seed(seed, cube, gt, n_groups, max_missing, trials, epochs, tau, cwv_key):
    # NO hw.setup() here. bandsim.parallel already configures each worker's thread budget and
    # determinism (parallel.py: "configure ... exactly like the serial main would via hw.setup()"),
    # and calling it again WITHOUT threads= resets the intra-op count to the machine default. That
    # measured as 32 torch threads on an 8-core box while the flagship was running: the smoke burned
    # 4620 s of CPU in 22 minutes on work that takes about a minute at one thread. Per-seed
    # randomness for TRAINING does not need it -- pretrain_sgmae and finetune_proposed reseed torch
    # themselves (seed+7 / seed+11) -- but the model's WEIGHT INITIALIZATION at construction does: without
    # seeding first it is drawn from the ambient RNG state, which depends on worker count and execution order.
    # We therefore call hw.seed_model(seed) immediately before the constructor below (adversarial review P0-A).
    # hw.seed_model only reseeds RNGs (NOT the thread budget), so it is safe in a worker unlike hw.setup().
    Xtr, ytr, Xte, yte = P2.prep(cube, gt, block=10, offset=seed)
    groups = contiguous_groups(Xtr.shape[1], n_groups)
    t_group = group_transmittance(groups, cwv_key)
    W = {"uniform": drop_weights(t_group, 0.0),
         "transmittance_weighted": drop_weights(t_group, tau)}
    wl = np.asarray(AVIRIS_WL_NM, float)
    cwl = group_center_wavelengths(wl, groups)

    hw.seed_model(seed)   # seed BEFORE the constructor: deterministic weight init per seed, order-independent
    model = GroupedCrossBandAttention(groups, cwl, P2.NUM_CLASSES)
    P2.pretrain_sgmae(model, Xtr, groups, seed, epochs=max(1, epochs // 2))
    P2.finetune_proposed(model, Xtr, ytr, groups, seed, epochs=epochs)

    # BLOCK-disjoint three-way split. The unit CRC is handed must be the unit that is split on:
    # calibrating on pixels drawn from the same blocks as eval is the defect phase8R's ROI overlap
    # made concrete (92% shared), and it makes the bound describe a population that was never held out.
    # Block ids are rebuilt exactly as phase4R does: prep() splits on the same checkerboard and
    # cube[te_mask] enumerates in C order, so argwhere(te_mask) lists (row,col) in Xte's own order.
    _, te_mask = disjoint_block_split(gt, block=10, guard=1, offset=seed)
    rcs = np.argwhere(te_mask)
    if len(rcs) != len(Xte):
        raise ValueError(f"block-id order does not match Xte ({len(rcs)} vs {len(Xte)}) -- "
                         f"prep()/disjoint_block_split drifted")
    n_bj = (gt.shape[1] + seed) // 10 + 1
    blk = ((rcs[:, 0] + seed) // 10) * n_bj + ((rcs[:, 1] + seed) // 10)
    ublk = np.unique(blk)
    perm = np.random.default_rng(90000 + seed).permutation(ublk)
    c1, c2 = len(perm) // 3, 2 * (len(perm) // 3)
    temp_b, calib_b, eval_b = perm[:c1], perm[c1:c2], perm[c2:]
    if set(calib_b.tolist()) & set(eval_b.tolist()):
        raise ValueError("calibration and evaluation blocks overlap -- the split is not disjoint")
    ti = np.where(np.isin(blk, temp_b))[0]
    ci = np.where(np.isin(blk, calib_b))[0]
    ei = np.where(np.isin(blk, eval_b))[0]

    rows = []
    for m in range(1, max_missing + 1):
        for cal_pol in POLICIES:
            rng_t = np.random.default_rng(1000 * seed + 7 * m)
            lt, yt, _ = pooled_logits(model, Xte[ti], yte[ti], blk[ti], groups, m, trials,
                                      rng_t, W[cal_pol])
            T = fit_temperature(lt, yt)          # tuned under the CALIBRATION policy, as a user would
            rng_c = np.random.default_rng(2000 * seed + 7 * m)
            lc, yc, uc = pooled_logits(model, Xte[ci], yte[ci], blk[ci], groups, m, trials,
                                       rng_c, W[cal_pol])
            corr_c = (lc.argmax(1) == yc).astype(int)
            conf_c = confidence_msp(lc / T)
            for dep_pol in POLICIES:
                rng_e = np.random.default_rng(3000 * seed + 7 * m + 1)
                le, ye, ue = pooled_logits(model, Xte[ei], yte[ei], blk[ei], groups, m, trials,
                                           rng_e, W[dep_pol])
                corr_e = (le.argmax(1) == ye).astype(int)
                conf_e = confidence_msp(le / T)
                crc = conformal_risk_control(corr_c, conf_c, corr_e, conf_e, alpha=TARGET_RISK,
                                             calib_group=uc, eval_group=ue)
                realized = crc["eval_group_joint_risk"]
                rows.append({
                    "seed": seed, "missing_groups": m,
                    "calib_policy": cal_pol, "deploy_policy": dep_pol,
                    "acc": round(float(corr_e.mean()) * 100, 3),
                    "crc_feasible": int(crc["feasible"]),
                    "crc_calibration_selection_stat": round(crc["calib_crc_bound"] * 100, 6),
                    "crc_heldout_empirical_joint_loss": round(realized * 100, 6),
                    "crc_threshold": float(crc["threshold"]),
                    "temperature": float(T),
                    "crc_heldout_coverage": round(crc["eval_group_coverage"] * 100, 6),
                    "crc_heldout_selective_risk": round(crc["eval_group_selective_risk"] * 100, 6),
                    # DESCRIPTIVE, not a theorem verdict. CRC Theorem 2.1 controls E[L] over the
                    # joint draw of calibration and test; P(one batch's empirical loss > alpha) is a
                    # different quantity that needs a high-probability (RCPS/PAC) result, not this
                    # one. Only meaningful where a threshold was feasible: an infeasible run
                    # abstains on everything, and its 0.0 loss would otherwise count as a pass.
                    "exceeded_alpha": int(bool(crc["feasible"]) and realized > TARGET_RISK),
                    "n_calib_units": crc["n_calib_units"], "n_eval_units": crc["n_eval_units"],
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--max-missing", type=int, default=4)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="sharpness of the transmittance weighting; 0 IS the uniform policy")
    ap.add_argument("--cwv", default="cwv2.0", help="column water vapour key in the 6S table")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    ap.add_argument("--nondeterministic", action="store_true")
    args = ap.parse_args()
    if args.groups < 2:
        raise ValueError(f"--groups must be >= 2, got {args.groups}")
    if not (1 <= args.max_missing < args.groups):
        raise ValueError(f"--max-missing must satisfy 1 <= m < groups={args.groups}, "
                         f"got {args.max_missing}")
    if args.trials < 1:
        raise ValueError(f"--trials must be >= 1, got {args.trials}")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if not np.isfinite(args.tau) or args.tau < 0:
        raise ValueError(f"--tau must be finite and >= 0, got {args.tau}")
    if not args.seeds:
        raise ValueError("need at least one seed")

    sfx = ""
    if args.smoke:
        args.seeds, args.epochs, args.trials, args.max_missing = [0], 8, 3, 2
        sfx = "_smoke"
        print("[smoke] reduced settings; writing *_smoke artefacts, NOT the real deliverables")

    hw.setup(deterministic=not args.nondeterministic, prefer=args.device)
    print("HW:", hw.info())
    cube, gt = P2.load_data()

    out = parallel.run_jobs(
        run_seed, args.seeds,
        shared=dict(cube=cube, gt=gt, n_groups=args.groups, max_missing=args.max_missing,
                    trials=args.trials, epochs=args.epochs, tau=args.tau, cwv_key=args.cwv),
        prefer=args.device, jobs=args.jobs, deterministic=not args.nondeterministic,
        label="phase9/seed")
    rows = [r for sub in out for r in sub]

    fields = ["seed", "missing_groups", "calib_policy", "deploy_policy", "acc", "crc_feasible",
              "temperature", "crc_threshold", "crc_calibration_selection_stat",
              "crc_heldout_empirical_joint_loss", "crc_heldout_coverage",
              "crc_heldout_selective_risk", "exceeded_alpha", "n_calib_units", "n_eval_units"]
    with open(P(f"results_phase9_physics_vs_random{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    # ---- the 2x2 ---------------------------------------------------------------------------------
    # PRIMARY statistic: the MEAN held-out joint loss over independent SEEDS, which is the quantity
    # CRC Theorem 2.1 bounds. The previous primary was a "breach rate" -- the fraction of cells whose
    # empirical loss exceeded alpha -- and that is a different estimand: controlling E[L] implies
    # nothing directly about P(empirical L > alpha), which needs a high-probability (RCPS/PAC)
    # result. The exceedance rate is kept as a DESCRIPTIVE column, clearly named.
    #
    # Clustering: seeds are the independent replicate. The drop counts m share one scene, one trained
    # model and one block split within a seed, so pooling seed x m cells and dividing by their count
    # would treat repeated measures as independent replications. Each seed contributes its mean over
    # m, and the SE is taken across seeds.
    summary = []
    seeds_all = sorted({r["seed"] for r in rows})
    for cal in POLICIES:
        for dep in POLICIES:
            sel = [r for r in rows if r["calib_policy"] == cal and r["deploy_policy"] == dep]
            feas = [r for r in sel if r["crc_feasible"]]
            per_seed = []
            for sd in seeds_all:
                v = [r["crc_heldout_empirical_joint_loss"] for r in feas if r["seed"] == sd]
                if v:
                    per_seed.append(float(np.mean(v)))
            arr = np.asarray(per_seed, float)
            se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
            summary.append({
                "calib_policy": cal, "deploy_policy": dep,
                "n_cells": len(sel), "n_feasible": len(feas), "n_seeds_clustered": int(arr.size),
                "mean_heldout_joint_loss_pct": round(float(arr.mean()), 4) if arr.size else float("nan"),
                "se_across_seeds_pct": round(se, 4) if arr.size > 1 else float("nan"),
                "mean_coverage_pct": round(float(np.mean([r["crc_heldout_coverage"] for r in feas])), 4) if feas else float("nan"),
                # accuracy depends ONLY on the deployment policy, so it must not be conditioned on
                # CRC feasibility -- doing so made the same classifier look different across
                # calibration policies purely because their feasible subsets differed.
                "mean_acc_pct_all_runs": round(float(np.mean([r["acc"] for r in sel])), 4) if sel else float("nan"),
                "descriptive_exceedance_rate_pct": round(100 * sum(r["exceeded_alpha"] for r in feas) / len(feas), 2) if feas else float("nan"),
            })
    with open(P(f"results_phase9_summary{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ms = sorted({r["missing_groups"] for r in rows})
    for cal in POLICIES:
        for dep in POLICIES:
            ys = [float(np.mean([r["crc_heldout_empirical_joint_loss"] for r in rows
                                 if r["missing_groups"] == m and r["calib_policy"] == cal
                                 and r["deploy_policy"] == dep and r["crc_feasible"]] or [np.nan]))
                  for m in ms]
            ax[0].plot(ms, ys, marker="o", label=f"calib {cal} / deploy {dep}")
    ax[0].axhline(TARGET_RISK * 100, ls="--", c="k", lw=1, label=f"alpha = {TARGET_RISK*100:.0f}%")
    ax[0].set_xlabel("groups missing"); ax[0].set_ylabel("realized joint risk (%)")
    ax[0].set_title("P(accepted AND wrong) at the certified threshold")
    ax[0].legend(fontsize=6)
    cells = [f"{s['calib_policy'][:4]}/{s['deploy_policy'][:4]}" for s in summary]
    ax[1].bar(cells, [s["mean_heldout_joint_loss_pct"] for s in summary],
          yerr=[s["se_across_seeds_pct"] for s in summary], capsize=3)
    ax[1].axhline(TARGET_RISK * 100, ls="--", c="k", lw=1)
    ax[1].set_ylabel("mean held-out joint loss (%)")
    ax[1].set_title("mean over seeds ± SE (calib/deploy)")
    fig.tight_layout(); fig.savefig(P(f"figs/fig_phase9_physics_vs_random{sfx}.pdf")); plt.close(fig)

    prov = {"target_risk": TARGET_RISK, "policies": list(POLICIES),
            "weight_rule": "p(group) proportional to (1 - mean 6S transmittance)^tau; tau=0 is uniform",
            "not_a_forward_model": "transmittance selects WHICH group is deleted; no radiative "
                                   "transfer is applied to the signal",
            "axis": "nominal reconstructed AVIRIS axis; the corrected cube already has the "
                    "deepest water bands removed, so the weighting emphasises the shoulders",
            "lut": os.path.relpath(LUT, os.path.dirname(_HERE)), "lut_sha256": file_sha256(LUT),
            "exchangeable_unit": "spatial checkerboard block",
            "calib_eval_split": "by block (disjoint)",
            "estimand": "JOINT P(accepted AND wrong); quote only with coverage",
            "exchangeability": "APPROXIMATE on Indian Pines (see module docstring); phase8R holds "
                               "the rigorous ROI-level certificate"}
    for _n in ("physics_vs_random", "summary"):
        stamp(P(f"results_phase9_{_n}{sfx}.csv"), args, extra=prov)

    print("\n=== 2x2: MEAN held-out joint loss over independent seeds (the CRC estimand) ===")
    for r in summary:
        print(f"  calib {r['calib_policy']:8s} -> deploy {r['deploy_policy']:8s} : "
              f"loss {r['mean_heldout_joint_loss_pct']:6.2f} ± {r['se_across_seeds_pct']:5.2f} %  "
              f"coverage {r['mean_coverage_pct']:6.2f}%  acc {r['mean_acc_pct_all_runs']:5.2f}%  "
              f"(feasible {r['n_feasible']}/{r['n_cells']}, {r['n_seeds_clustered']} seeds)  "
              f"[descriptive exceedance {r['descriptive_exceedance_rate_pct']:5.1f}%]")
    print("  NOTE exceedance is DESCRIPTIVE. CRC bounds E[L]; P(one batch's empirical L > alpha) is a")
    print("       different quantity requiring a high-probability (RCPS/PAC) result, not this one.")
    print(f"\nwrote: {P(f'results_phase9_physics_vs_random{sfx}.csv')}")
    print(f"       {P(f'results_phase9_summary{sfx}.csv')}")
    print(f"       {P(f'figs/fig_phase9_physics_vs_random{sfx}.pdf')}")


if __name__ == "__main__":
    main()
