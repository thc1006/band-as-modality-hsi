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
import phase6_second_dataset as P6
from bandsim import hw
from bandsim.io import disjoint_block_split
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention
from bandsim.reliability import conformal_risk_control, fit_temperature
from phase8G_emit_shift import logits_of, logits_band, joint   # reuse the exact inference + joint-risk helpers

ALPHA = 0.10
SRF = os.path.join(os.path.dirname(_HERE), "data", "srf_cache")


def _load_dataset(name):
    """(cube (H,W,B) reflectance, gt (H,W) labels 1..K, K, LUT path). Indian Pines keeps its original
    loader + the shipped 200-band LUT; salinas/pavia use the phase6 registry + their own generated LUT
    (run experiments/precompute_6s_dataset.py --dataset <name> first)."""
    if name == "indian_pines":
        cube, gt = P1.load_indian_pines()
        return cube, gt, 16, os.path.join(SRF, "T_6s_grid.npz")
    cube, gt, _wl, K, _ = P6.load_dataset(name)
    return cube, gt, int(K), os.path.join(SRF, f"T_6s_grid_{name}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="indian_pines", choices=["indian_pines", "salinas", "pavia"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--model", choices=["mlp", "band"], default="mlp",
                    help="mlp = plain MLP (default); band = GroupedCrossBandAttention + SGMAE (arch-independence check)")
    ap.add_argument("--bs", type=int, default=0, help="band training batch size; 0=2048 (launch-bound fix)")
    ap.add_argument("--src-cwv", default="cwv0.5")
    ap.add_argument("--tgt-cwv", default="cwv4.0")
    ap.add_argument("--spatial-cwv", action="store_true",
                    help="target atmosphere varies PER PIXEL (a smooth CWV gradient src->tgt down the scene) so "
                         "the shift is NOT a global per-band multiply -- global per-band re-norm then only "
                         "PARTIALLY realigns it (the realistic heterogeneous-atmosphere case). The uniform "
                         "default is a per-band-multiply POSITIVE CONTROL that re-norm inverts by construction.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        stem = "results_phase8H_hsi6s_shift" if args.dataset == "indian_pines" else f"results_phase8H_hsi6s_{args.dataset}"
        if args.spatial_cwv:
            stem += "_spatial"
        if args.model == "band":
            stem += "_band"
        args.out = os.path.join(_HERE, "..", "paper", stem)
    hw.setup(deterministic=True, prefer="auto")
    dev = hw.device()
    print("HW:", hw.info(), flush=True)

    cube, gt, K, lut = _load_dataset(args.dataset)                      # (H,W,B) reflectance, gt 1..K
    z = np.load(lut)
    Ts = np.asarray(z[args.src_cwv], float); Tt = np.asarray(z[args.tgt_cwv], float)
    if Ts.shape[0] != cube.shape[-1] or Tt.shape[0] != cube.shape[-1]:
        raise ValueError(f"6S LUT has {Ts.shape[0]} bands but cube has {cube.shape[-1]} -- axis mismatch")
    src_cube = cube * Ts                                                 # dry (uniform) source product
    dw = float(np.mean(np.abs(Ts - Tt)))                                # mean per-band transmittance gap
    if args.spatial_cwv:
        # per-PIXEL atmosphere: a smooth CWV gradient src->tgt down the scene, T linearly interpolated over the
        # LUT's CWV grid. tgt = cube * T(cwv[i,j]) is NOT a global per-band multiply, so per-band re-norm (which
        # is EXACTLY invariant to a global per-band multiply -- the uniform case is a positive control) can only
        # PARTIALLY realign it. This is the realistic heterogeneous-atmosphere case, matching CloudSEN12/EMIT.
        grid = np.array([0.5, 2.0, 4.0])
        Tg = np.stack([np.asarray(z[f"cwv{c}"], float) for c in grid])   # (3, nbands)
        sv = float(args.src_cwv.replace("cwv", "")); tv = float(args.tgt_cwv.replace("cwv", ""))
        H, W = gt.shape
        cwv_map = (sv + (tv - sv) * np.linspace(0, 1, H)[:, None] * np.ones((1, W))).reshape(-1)
        Tt_map = np.stack([np.interp(cwv_map, grid, Tg[:, b]) for b in range(Tg.shape[1])], axis=1)
        tgt_cube = (cube.reshape(-1, cube.shape[-1]) * Tt_map).reshape(cube.shape)
        print(f"6S shift {args.src_cwv}->{args.tgt_cwv} SPATIAL (per-pixel CWV {sv}->{tv} gradient): uniform-gap "
              f"mean |dT| {dw:.3f} over {len(Ts)} bands -- re-norm expected PARTIAL", flush=True)
    else:
        tgt_cube = cube * Tt                                             # humid (uniform) target product
        print(f"6S shift {args.src_cwv}->{args.tgt_cwv} UNIFORM: mean |dT| {dw:.3f}, "
              f"max |dT| {float(np.max(np.abs(Ts-Tt))):.3f} over {len(Ts)} bands -- per-band multiply, "
              f"re-norm inverts by construction (POSITIVE CONTROL)", flush=True)
    groups = contiguous_groups(cube.shape[-1], 12)
    cwl = group_center_wavelengths(np.asarray(z["wl_nm"], float), groups) if args.model == "band" else None

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
        if args.model == "band":
            _bs = args.bs if args.bs > 0 else 2048
            model = GroupedCrossBandAttention(groups, cwl, K)
            P2.pretrain_sgmae(model, norm(Xtr_s), groups, seed, epochs=max(1, args.epochs // 2), bs=_bs)
            P2.finetune_proposed(model, norm(Xtr_s), ytr, groups, seed, epochs=args.epochs, bs=_bs,
                                 group_dropout=False)
            Lg = lambda X: logits_band(model, X, groups, dev)
        else:
            model = P2.train_mlp(norm(Xtr_s), ytr, groups, seed, group_dropout=False,
                                 epochs=args.epochs, hidden=args.hidden, num_classes=K)
            Lg = lambda X: logits_of(model, X, dev)

        # SOURCE calibration on the (dry) calib pixels: temperature (first half) + CRC threshold (second half)
        lc = Lg(norm(gv(src_cube, cai))); yc = gv(yx, cai); h = len(lc) // 2
        Tc = fit_temperature(lc[:h], yc[:h])
        pc = softmax(lc[h:] / Tc, axis=1); corr = pc.argmax(1) == yc[h:]
        thr = float(conformal_risk_control(corr, pc.max(1), corr, pc.max(1), alpha=ALPHA)["threshold"])
        cstat = float(np.mean((~corr) & (pc.max(1) >= thr))) * 100

        yev = gv(yx, evi)
        pe_src = softmax(Lg(norm(gv(src_cube, evi))) / Tc, axis=1)   # dry -> should hold
        pe_tgt = softmax(Lg(norm(gv(tgt_cube, evi))) / Tc, axis=1)   # humid, STALE dry stats
        muT = gv(tgt_cube, cai).mean(0); sdT = gv(tgt_cube, cai).std(0) + 1e-6          # target stats, eval-disjoint
        pe_rn = softmax(Lg(((gv(tgt_cube, evi) - muT) / sdT).astype(np.float32)) / Tc, axis=1)
        cs, ct, cr = pe_src.argmax(1) == yev, pe_tgt.argmax(1) == yev, pe_rn.argmax(1) == yev
        # Mondrian: recalibrate the threshold on the humid product (calib pixels, eval-disjoint)
        pca = softmax(Lg(norm(gv(tgt_cube, cai))) / Tc, axis=1); cca = pca.argmax(1) == gv(yx, cai)
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
    print(f"\n=== HSI 6S reliability ({args.dataset}/{args.model}, {args.src_cwv}->{args.tgt_cwv}"
          f"{' SPATIAL/per-pixel' if args.spatial_cwv else ' UNIFORM'}, {len(rows)} seeds, "
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
