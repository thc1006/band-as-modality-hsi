#!/usr/bin/env python3
"""HSI 6S reliability (3rd modality) -- does a conformal certificate calibrated under a DRY atmosphere
silently fail when the same scene is imaged under a HUMID atmosphere? A real, physically-grounded 6S
gaseous-transmittance covariate shift on AVIRIS hyperspectral (Indian Pines, 200 bands), complementing
CloudSEN12 (Sentinel-2 multispectral, L1C->L2A) and EMIT (imaging spectrometer, radiance->reflectance).

  source  rho * T_6S(CWV=0.5)   dry  atmosphere  (train + calibrate here)
  target  rho * T_6S(CWV=4.0)   humid atmosphere (deploy here; the water-vapour bands are attenuated)

The two products share the scene/geometry, so this isolates the ATMOSPHERIC covariate shift, exactly like
the flagship isolates L1C->L2A. Split is spatially DISJOINT (checkerboard blocks + a 1px guard band) so
train/calib/eval never touch across a block seam. We report the confidently-wrong JOINT risk and the
certificate's own calibration statistic (which stays <= alpha even as the deployed risk rises = SILENT).

Honest scope: single small scene (~10k labelled px), pixel-level CRC (no exchangeable scene unit), and the
6S shift only moves the ~20 water-vapour bands of 200, so the breach is expected to be milder and noisier
than the product-normalization shifts -- an honest third data point on shift SEVERITY, not a headline.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from scipy.special import softmax

import phase1_indian_pines as P1
import phase2_degradation as P2
from bandsim import hw
from bandsim.io import disjoint_block_split
from bandsim.grouping import contiguous_groups
from bandsim.reliability import conformal_risk_control, fit_temperature
from phase8G_emit_shift import logits_of, joint         # reuse the exact inference + joint-risk helpers

ALPHA = 0.10
LUT = os.path.join(os.path.dirname(_HERE), "data", "srf_cache", "T_6s_grid.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--src-cwv", default="cwv0.5")
    ap.add_argument("--tgt-cwv", default="cwv4.0")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8H_hsi6s_shift"))
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer="auto")
    dev = hw.device()
    print("HW:", hw.info(), flush=True)

    cube, gt = P1.load_indian_pines()                                   # (145,145,200) reflectance, gt 1..16
    z = np.load(LUT)
    Ts = np.asarray(z[args.src_cwv], float); Tt = np.asarray(z[args.tgt_cwv], float)
    if Ts.shape[0] != cube.shape[-1] or Tt.shape[0] != cube.shape[-1]:
        raise ValueError(f"6S LUT has {Ts.shape[0]} bands but cube has {cube.shape[-1]} -- axis mismatch")
    src_cube = cube * Ts                                                 # dry-atmosphere product
    tgt_cube = cube * Tt                                                 # humid-atmosphere product
    dw = float(np.mean(np.abs(Ts - Tt)))                                # mean per-band transmittance gap
    print(f"6S shift {args.src_cwv}->{args.tgt_cwv}: mean |dT| {dw:.3f}, "
          f"max |dT| {float(np.max(np.abs(Ts-Tt))):.3f} over {len(Ts)} bands", flush=True)
    groups = contiguous_groups(cube.shape[-1], 12)

    rows = []
    for seed in args.seeds:
        hw.seed_model(seed)
        tr_mask, te_mask = disjoint_block_split(gt, block=10, guard=1, offset=seed % 2)
        yx = gt - 1                                                      # 1..16 -> 0..15
        tri = np.argwhere(tr_mask); tei = np.argwhere(te_mask)
        gv = lambda C, ij: C[ij[:, 0], ij[:, 1]]
        # calib/eval = a pixel split of the spatially-disjoint TEST region (same product, exchangeable)
        rng = np.random.default_rng(seed + 13)
        perm = rng.permutation(len(tei)); cut = len(tei) // 2
        cai, evi = tei[perm[:cut]], tei[perm[cut:]]
        Xtr_s, ytr = gv(src_cube, tri), gv(yx, tri)
        mu = Xtr_s.mean(0); sd = Xtr_s.std(0) + 1e-6
        norm = lambda X: ((X - mu) / sd).astype(np.float32)
        model = P2.train_mlp(norm(Xtr_s), ytr, groups, seed, group_dropout=False,
                             epochs=args.epochs, hidden=args.hidden, num_classes=16)

        # SOURCE calibration on the (dry) calib pixels: temperature (first half) + CRC threshold (second half)
        lc = logits_of(model, norm(gv(src_cube, cai)), dev); yc = gv(yx, cai); h = len(lc) // 2
        Tc = fit_temperature(lc[:h], yc[:h])
        pc = softmax(lc[h:] / Tc, axis=1); corr = pc.argmax(1) == yc[h:]
        thr = float(conformal_risk_control(corr, pc.max(1), corr, pc.max(1), alpha=ALPHA)["threshold"])
        cstat = float(np.mean((~corr) & (pc.max(1) >= thr))) * 100

        yev = gv(yx, evi)
        pe_src = softmax(logits_of(model, norm(gv(src_cube, evi)), dev) / Tc, axis=1)   # dry -> should hold
        pe_tgt = softmax(logits_of(model, norm(gv(tgt_cube, evi)), dev) / Tc, axis=1)   # humid, STALE dry stats
        muT = gv(tgt_cube, cai).mean(0); sdT = gv(tgt_cube, cai).std(0) + 1e-6          # target stats, eval-disjoint
        pe_rn = softmax(logits_of(model, ((gv(tgt_cube, evi) - muT) / sdT).astype(np.float32), dev) / Tc, axis=1)
        cs, ct, cr = pe_src.argmax(1) == yev, pe_tgt.argmax(1) == yev, pe_rn.argmax(1) == yev
        # Mondrian: recalibrate the threshold on the humid product (calib pixels, eval-disjoint)
        pca = softmax(logits_of(model, norm(gv(tgt_cube, cai)), dev) / Tc, axis=1); cca = pca.argmax(1) == gv(yx, cai)
        thrM = float(conformal_risk_control(cca, pca.max(1), cca, pca.max(1), alpha=ALPHA)["threshold"])

        r = dict(seed=seed, n_eval=int(len(evi)), calib_stat=cstat,
                 src_acc=float(cs.mean()) * 100, tgt_acc=float(ct.mean()) * 100)
        r["clean_src"] = joint(cs, pe_src.max(1), thr)
        r["naive_tgt"] = joint(ct, pe_tgt.max(1), thr)
        r["mondrian_tgt"] = joint(ct, pe_tgt.max(1), thrM)
        r["renorm_tgt"] = joint(cr, pe_rn.max(1), thr)
        rows.append(r)
        print(f"  seed {seed}: {len(tri)} train / {len(evi)} eval px; acc dry {r['src_acc']:.0f}->humid "
              f"{r['tgt_acc']:.0f}; calib-stat {cstat:.1f}%  || joint%@cov: clean {r['clean_src'][0]:.1f}@"
              f"{r['clean_src'][1]:.0f}  NAIVE {r['naive_tgt'][0]:.1f}@{r['naive_tgt'][1]:.0f}  "
              f"Mondrian {r['mondrian_tgt'][0]:.1f}@{r['mondrian_tgt'][1]:.0f}  "
              f"re-norm {r['renorm_tgt'][0]:.1f}@{r['renorm_tgt'][1]:.0f}", flush=True)

    def agg(key, i):
        v = np.array([r[key][i] for r in rows], float)
        return float(np.nanmean(v)), (float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))
    print(f"\n=== HSI 6S reliability (Indian Pines, {args.src_cwv}->{args.tgt_cwv}, {len(rows)} seeds, "
          f"alpha={ALPHA*100:.0f}%) ===")
    for k, l in [("clean_src", "clean (dry source)"), ("naive_tgt", "NAIVE (humid, stale)"),
                 ("mondrian_tgt", "Mondrian (humid recalib)"), ("renorm_tgt", "re-norm (humid stats)")]:
        jm, js = agg(k, 0); cm, _ = agg(k, 1)
        print(f"  {l:26s} joint {jm:5.1f} +/- {js:4.1f} % @ cov {cm:3.0f}%")
    print(f"  calibration statistic (source, stays <= alpha): {np.mean([r['calib_stat'] for r in rows]):.1f}%")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out + ".json", "w"), indent=1)
    print(f"\nwrote {args.out}.json")


if __name__ == "__main__":
    main()
