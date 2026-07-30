#!/usr/bin/env python3
"""EMIT geographic (cross-biome) axis -- ISOLATE the geographic shift from the product shift.

Train + calibrate on the SOURCE product (L1B radiance) of some biomes, then deploy on the SOURCE product
(L1B) of UNSEEN biomes. No product change -> a PURE cross-biome geographic deployment shift. Does a
source-calibrated conformal certificate hold when the deployment geography (biome) is unseen? Complements
phase8G_emit_shift.py (which isolates the product shift on fixed geography). Reuses its cache + helpers.
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

import phase2_degradation as P2
from phase8G_emit_shift import prep, logits_of, joint, ALPHA, CLIP_RFL
from bandsim import hw
from bandsim.grouping import contiguous_groups
from bandsim.reliability import conformal_risk_control, fit_temperature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-px", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8G_emit_geo"))
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer="auto")
    dev = hw.device()
    print("HW:", hw.info(), flush=True)
    rows = []
    for seed in args.seeds:
        hw.seed_model(seed)
        data, remap = prep(args.n_px, seed)
        K = len(remap)
        order = list(np.random.default_rng(seed).permutation(sorted(data)))
        nt = max(2, int(round(len(order) * 0.55))); nc = max(1, int(round(len(order) * 0.20)))
        tb, cb, eb = order[:nt], order[nt:nt + nc], order[nt + nc:]
        cat = lambda bs, k: np.concatenate([data[b][k] for b in bs])
        clipR = lambda X: np.clip(X, CLIP_RFL[0], None)
        RADt = clipR(cat(tb, "rad")); yt = cat(tb, "y")
        mu = RADt.mean(0); sd = RADt.std(0) + 1e-6
        norm = lambda X: ((clipR(X) - mu) / sd).astype(np.float32)
        groups = contiguous_groups(RADt.shape[1], 12)
        model = P2.train_mlp(((RADt - mu) / sd).astype(np.float32), yt, groups, seed, group_dropout=False,
                             epochs=args.epochs, hidden=args.hidden, num_classes=K)
        lc = logits_of(model, norm(cat(cb, "rad")), dev); yc = cat(cb, "y"); h = len(lc) // 2
        Tc = fit_temperature(lc[:h], yc[:h])
        pc = softmax(lc[h:] / Tc, axis=1); corr = pc.argmax(1) == yc[h:]
        thr = float(conformal_risk_control(corr, pc.max(1), corr, pc.max(1), alpha=ALPHA)["threshold"])
        cstat = float(np.mean((~corr) & (pc.max(1) >= thr))) * 100
        # deploy on UNSEEN biomes, SOURCE L1B -> pure geographic shift
        pe = softmax(logits_of(model, norm(cat(eb, "rad")), dev) / Tc, axis=1); ye = cat(eb, "y")
        ce = pe.argmax(1) == ye
        thrM = float(conformal_risk_control(ce, pe.max(1), ce, pe.max(1), alpha=ALPHA)["threshold"])
        r = dict(seed=seed, K=K, n_eval_b=len(eb), eval_biomes=eb, calib_stat=cstat, acc=float(ce.mean()) * 100)
        r["naive_geo"] = joint(ce, pe.max(1), thr)
        r["mondrian_geo"] = joint(ce, pe.max(1), thrM)
        rows.append(r)
        print(f"  seed {seed}: eval biomes {eb}; calib-stat {cstat:.1f}%; acc {r['acc']:.0f}%  || "
              f"NAIVE-geo {r['naive_geo'][0]:.1f}@{r['naive_geo'][1]:.0f}  "
              f"Mondrian {r['mondrian_geo'][0]:.1f}@{r['mondrian_geo'][1]:.0f}", flush=True)

    def agg(key, i):
        v = np.array([r[key][i] for r in rows], float)
        return float(np.nanmean(v)), (float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))
    print(f"\n=== EMIT geographic (cross-biome) axis -- SOURCE L1B, {len(rows)} seeds, alpha={ALPHA*100:.0f}% ===")
    for k, l in [("naive_geo", "NAIVE (source-calib, unseen biomes)"), ("mondrian_geo", "Mondrian (biome-recalib)")]:
        jm, js = agg(k, 0); cm, _ = agg(k, 1)
        print(f"  {l:38s} joint {jm:5.1f} +/- {js:4.1f} % @ cov {cm:3.0f}%")
    print(f"  calibration statistic: {np.mean([r['calib_stat'] for r in rows]):.1f}%  "
          f"(holds => geography alone is safe, product shift is the culprit; breaches => geography also breaks it)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out + ".json", "w"), indent=1)
    print(f"\nwrote {args.out}.json")


if __name__ == "__main__":
    main()
