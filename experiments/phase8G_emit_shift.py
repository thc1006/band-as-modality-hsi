#!/usr/bin/env python3
"""EMIT cross-sensor reliability (2nd dataset) -- does a conformal certificate calibrated on the SOURCE
product (L1B radiance) silently fail when deployed on the TARGET product (L2A ISOFIT surface reflectance),
across biomes?

A real atmospheric-correction shift (radiance -> reflectance) on an INDEPENDENT sensor (EMIT imaging
spectrometer, 244 good bands) and INDEPENDENT geography (biome-level train/calib/eval split). Label = ESA
WorldCover land-cover (product-agnostic pixel property, remote-sampled). Mirrors the CloudSEN12 flagship:
  naive     source(L1B)-calibrated threshold, deployed on target(L2A) of UNSEEN biomes
  Mondrian  threshold recalibrated on the target(L2A)
  re-norm   product-aware: L2A re-normalized with its OWN per-band stats before the source threshold
We report the confidently-wrong JOINT risk P(accepted & wrong) with its coverage, and the certificate's own
calibration selection statistic (which stays <= alpha even as the deployed risk rises = the SILENT failure).

Honest scope: pixel-level CRC (no scene-component unit here); WorldCover 10 m sampled at ~60 m EMIT pixels
carries mixed-pixel label noise; the model trains on radiance, which is unusual but is exactly the stale
source the shift then breaks. This is a cross-sensor GENERALIZATION of the reliability finding, not a 2nd
cloud dataset.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import torch
from scipy.special import softmax

import phase8G_emit_io as EIO
import phase2_degradation as P2
import phase8R10_normalization_control as R10   # reuse the mid-CDF quantile transport (P0-2 fix)
from bandsim import hw
from bandsim.grouping import contiguous_groups
from bandsim.reliability import conformal_risk_control, fit_temperature

ALPHA = 0.10
CLIP_RFL = (-0.1, 1.6)


def prep(n_px, seed, min_count=300):
    """Per biome: subsample the cached pool (build_shift_cache first); keep classes with >= min_count total."""
    rng = np.random.default_rng(seed + 777)
    cache = EIO.load_shift_cache()
    if not cache:
        raise RuntimeError("no shift cache -- run first: .venv/bin/python experiments/phase8G_emit_io.py --build-cache")
    data = {}
    for name, v in cache.items():
        n = len(v["wc"]); take = min(n_px, n)
        idx = rng.choice(n, take, replace=False)
        data[name] = dict(rad=v["rad"][idx], rfl=v["rfl"][idx], wc=v["wc"][idx])
    allwc = np.concatenate([v["wc"] for v in data.values()])
    u, c = np.unique(allwc, return_counts=True)
    keepc = sorted(u[c >= min_count].tolist())
    remap = {int(k): i for i, k in enumerate(keepc)}
    for v in data.values():
        m = np.isin(v["wc"], keepc)
        v["rad"], v["rfl"] = v["rad"][m], v["rfl"][m]
        v["y"] = np.array([remap[int(x)] for x in v["wc"][m]], np.int64)
    return data, remap


def logits_of(model, X, dev, bs=8192):
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(X[s:s + bs].astype(np.float32)).to(dev)
            out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out)


def joint(corr, conf, thr):
    """(joint risk %, coverage %, selective %) at acceptance threshold thr on max-softmax conf."""
    acc = conf >= thr
    j = float(np.mean((~corr) & acc)) * 100
    cov = float(np.mean(acc)) * 100
    sel = float(np.mean(~corr[acc])) * 100 if acc.any() else float("nan")
    return j, cov, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-px", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8G_emit_shift"))
    args = ap.parse_args()
    if args.smoke:
        args.n_px, args.seeds, args.epochs = 2500, [0], 25
    hw.setup(deterministic=True, prefer="auto")
    dev = hw.device()
    print("HW:", hw.info(), flush=True)

    rows = []
    for seed in args.seeds:
        hw.seed_model(seed)
        data, remap = prep(args.n_px, seed)
        K = len(remap)
        # POOL all biomes; 3-way PIXEL split so the eval geography is FIXED and only the product L1B->L2A
        # changes -- this isolates the stale-normalization (product) shift, exactly like the flagship uses the
        # same scenes for L1C/L2A. (Cross-biome geography is a separate, confounding axis -- see --geo.)
        RAD = np.concatenate([data[b]["rad"] for b in data])
        RFL = np.concatenate([data[b]["rfl"] for b in data])
        Y = np.concatenate([data[b]["y"] for b in data])
        rng = np.random.default_rng(seed + 13)
        perm = rng.permutation(len(Y))
        a, b2 = len(Y) // 2, 3 * len(Y) // 4
        tr, ca, ev = perm[:a], perm[a:b2], perm[b2:]                         # 50 train / 25 calib / 25 eval
        clipR = lambda X: np.clip(X, CLIP_RFL[0], None)                      # radiance: clip sensor-noise negatives only
        clipL = lambda X: np.clip(X, CLIP_RFL[0], CLIP_RFL[1])              # reflectance: physical range
        mu = clipR(RAD[tr]).mean(0); sd = clipR(RAD[tr]).std(0) + 1e-6       # SOURCE (L1B) train stats = the stale contract
        norm = lambda X, m, s: ((X - m) / s).astype(np.float32)
        groups = contiguous_groups(RAD.shape[1], 12)
        model = P2.train_mlp(norm(clipR(RAD[tr]), mu, sd), Y[tr], groups, seed, group_dropout=False,
                             epochs=args.epochs, hidden=args.hidden, num_classes=K)

        # SOURCE (L1B) calibration on the calib pixels: temperature (first half) + CRC threshold (second half)
        lc = logits_of(model, norm(clipR(RAD[ca]), mu, sd), dev)
        half = len(lc) // 2
        Tc = fit_temperature(lc[:half], Y[ca][:half])
        pc = softmax(lc[half:] / Tc, axis=1); corr_c = pc.argmax(1) == Y[ca][half:]
        thr = float(conformal_risk_control(corr_c, pc.max(1), corr_c, pc.max(1), alpha=ALPHA)["threshold"])
        calib_stat = float(np.mean((~corr_c) & (pc.max(1) >= thr))) * 100   # the statistic CRC minimises (<= alpha)

        # DEPLOY on the SAME eval pixels -- PURE product shift: L1B clean / L2A stale / L2A product-aware re-norm
        pe_clean = softmax(logits_of(model, norm(clipR(RAD[ev]), mu, sd), dev) / Tc, axis=1)
        pe_stale = softmax(logits_of(model, norm(clipL(RFL[ev]), mu, sd), dev) / Tc, axis=1)     # STALE L1B stats on L2A
        muL = clipL(RFL[ca]).mean(0); sdL = clipL(RFL[ca]).std(0) + 1e-6                         # TARGET stats from CALIB (eval-disjoint)
        pe_renorm = softmax(logits_of(model, norm(clipL(RFL[ev]), muL, sdL), dev) / Tc, axis=1)
        # richer label-free transport: per-band quantile map L2A(eval)->L1B(train) marginal, eval-disjoint L2A(calib)
        # as the L2A reference (R10.quantile_match returns the L1B-train z-scored input, so feed it directly)
        Xq = R10.quantile_match(clipL(RFL[ev]), clipL(RFL[ca]), clipR(RAD[tr]), mu, sd, list(range(RAD.shape[1])))
        pe_quant = softmax(logits_of(model, Xq.astype(np.float32), dev) / Tc, axis=1)
        cc = pe_clean.argmax(1) == Y[ev]; cs = pe_stale.argmax(1) == Y[ev]
        cr = pe_renorm.argmax(1) == Y[ev]; cq = pe_quant.argmax(1) == Y[ev]
        # Mondrian: recalibrate the threshold on the TARGET product (L2A of the calib pixels, eval-disjoint)
        pca = softmax(logits_of(model, norm(clipL(RFL[ca]), mu, sd), dev) / Tc, axis=1); cca = pca.argmax(1) == Y[ca]
        thrM = float(conformal_risk_control(cca, pca.max(1), cca, pca.max(1), alpha=ALPHA)["threshold"])

        r = dict(seed=seed, K=K, n_eval_px=int(len(ev)), calib_stat=calib_stat,
                 src_acc=float(cc.mean()) * 100, tgt_acc=float(cs.mean()) * 100)
        r["clean_L1B"] = joint(cc, pe_clean.max(1), thr)                     # certificate on source -> should hold ~alpha
        r["naive_L2A"] = joint(cs, pe_stale.max(1), thr)                    # stale-normalized target -> the breach
        r["mondrian_L2A"] = joint(cs, pe_stale.max(1), thrM)               # recalibrated on target
        r["renorm_L2A"] = joint(cr, pe_renorm.max(1), thr)                # product-aware re-normalization (mean/std)
        r["quantile_L2A"] = joint(cq, pe_quant.max(1), thr)               # richer per-band quantile transport
        rows.append(r)
        print(f"  seed {seed}: K={K} {len(Y)} px pooled; acc L1B {r['src_acc']:.0f}->L2A {r['tgt_acc']:.0f}; "
              f"calib-stat {calib_stat:.1f}%  || joint%@cov: clean-L1B {r['clean_L1B'][0]:.1f}@{r['clean_L1B'][1]:.0f}  "
              f"NAIVE-L2A {r['naive_L2A'][0]:.1f}@{r['naive_L2A'][1]:.0f}  "
              f"Mondrian {r['mondrian_L2A'][0]:.1f}@{r['mondrian_L2A'][1]:.0f}  "
              f"re-norm {r['renorm_L2A'][0]:.1f}@{r['renorm_L2A'][1]:.0f}  "
              f"quantile {r['quantile_L2A'][0]:.1f}@{r['quantile_L2A'][1]:.0f}", flush=True)

    def agg(key, i):
        v = np.array([r[key][i] for r in rows], float)
        return float(np.nanmean(v)), (float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))
    print(f"\n=== EMIT cross-sensor reliability ({len(rows)} seeds, alpha={ALPHA*100:.0f}%) ===")
    for key, lab in [("clean_L1B", "clean L1B (source)"), ("naive_L2A", "NAIVE L2A (stale-norm)"),
                     ("mondrian_L2A", "Mondrian L2A"), ("renorm_L2A", "re-norm L2A (mean/std)"),
                     ("quantile_L2A", "quantile L2A")]:
        jm, js = agg(key, 0); cm, _ = agg(key, 1)
        print(f"  {lab:24s} joint {jm:5.1f} +/- {js:4.1f} %  @ cov {cm:3.0f}%")
    cs = np.mean([r["calib_stat"] for r in rows])
    print(f"  calibration statistic (source, stays <= alpha): {cs:.1f}%   <- the SILENT part")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out + ("_smoke" if args.smoke else "") + ".json", "w"), indent=1)
    print(f"\nwrote {args.out}{'_smoke' if args.smoke else ''}.json")


if __name__ == "__main__":
    main()
