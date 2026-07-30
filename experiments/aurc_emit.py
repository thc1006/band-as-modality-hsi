#!/usr/bin/env python3
"""AURC + matched-coverage on the EMIT product shift (radiance -> stale-normalized reflectance), MLP vs band.

Parallels aurc_matched_coverage.py (HSI 6S) for the EMIT cross-sensor case. Question: is the band model's
LOWER joint-risk on EMIT (NAIVE 57.5 @ 83% cov vs MLP 66.6 @ 95%) genuine robustness or an operating-point
artifact, and is EITHER confidence informative under this EXTREME shift? AURC is invariant to temperature
scaling (monotone -> ranking unchanged), so no calibration is needed; confidence = max-softmax. Pixel-level
(EMIT has no scene-component unit here).
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
from scipy.integrate import trapezoid

import phase8G_emit_shift as S
import phase2_degradation as P2
from bandsim import hw
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention


def aurc(conf, corr):
    o = np.argsort(-conf); c = corr[o].astype(float); k = np.arange(1, len(c) + 1)
    return float(trapezoid(1.0 - np.cumsum(c) / k, k / len(c))) * 100


def sel_risk_at(conf, corr, cov):
    o = np.argsort(-conf); k = max(1, int(cov * len(conf)))
    return float(1.0 - corr[o][:k].mean()) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs-mlp", type=int, default=40)
    ap.add_argument("--epochs-band", type=int, default=60)
    ap.add_argument("--covs", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_aurc_emit.json"))
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer="auto"); dev = hw.device()
    clipR = lambda X: np.clip(X, -0.1, None); clipL = lambda X: np.clip(X, -0.1, 1.6)

    def run(seed, mk):
        hw.seed_model(seed); data, remap = S.prep(4000, seed); K = len(remap)
        RAD = np.concatenate([data[b]["rad"] for b in data]); RFL = np.concatenate([data[b]["rfl"] for b in data])
        Y = np.concatenate([data[b]["y"] for b in data])
        rng = np.random.default_rng(seed + 13); perm = rng.permutation(len(Y)); a = len(Y) // 2
        tr, ev = perm[:a], perm[3 * len(Y) // 4:]
        mu = clipR(RAD[tr]).mean(0); sd = clipR(RAD[tr]).std(0) + 1e-6; norm = lambda X, m, s: ((X - m) / s).astype(np.float32)
        gr = contiguous_groups(RAD.shape[1], 12); Xtr = norm(clipR(RAD[tr]), mu, sd)
        if mk == "band":
            m = GroupedCrossBandAttention(gr, group_center_wavelengths(S._first_wl(), gr), K)
            P2.pretrain_sgmae(m, Xtr, gr, seed, epochs=max(1, args.epochs_band // 2), bs=2048)
            P2.finetune_proposed(m, Xtr, Y[tr], gr, seed, epochs=args.epochs_band, bs=2048, group_dropout=False)
            L = lambda X: S.logits_band(m, X, gr, dev)
        else:
            m = P2.train_mlp(Xtr, Y[tr], gr, seed, group_dropout=False, epochs=args.epochs_mlp, hidden=256, num_classes=K)
            L = lambda X: S.logits_of(m, X, dev)
        pc = softmax(L(norm(clipR(RAD[ev]), mu, sd)), axis=1); cc = pc.argmax(1) == Y[ev]     # clean L1B
        pn = softmax(L(norm(clipL(RFL[ev]), mu, sd)), axis=1); cn = pn.argmax(1) == Y[ev]     # naive L2A (stale)
        return dict(seed=seed, aurc_clean=aurc(pc.max(1), cc), aurc_naive=aurc(pn.max(1), cn),
                    err_clean=float((~cc).mean()) * 100, err_naive=float((~cn).mean()) * 100,
                    sr_clean={f"{c}": sel_risk_at(pc.max(1), cc, c) for c in args.covs},
                    sr_naive={f"{c}": sel_risk_at(pn.max(1), cn, c) for c in args.covs})

    out = {}
    print(f"EMIT product shift (radiance->stale reflectance) AURC, {len(args.seeds)} seeds, pixel-level:", flush=True)
    for mk in ["mlp", "band"]:
        rows = [run(s, mk) for s in args.seeds]; out[mk] = rows
        m = lambda k: float(np.mean([r[k] for r in rows]))
        an, en = m("aurc_naive"), m("err_naive")
        tag = "INFORMATIVE" if en - an > 5 else "DEAD (AURC ~ error)"
        print(f"  {mk:4s}: AURC clean {m('aurc_clean'):4.1f} -> naive {an:4.1f} (naive err {en:.0f}) => {tag}", flush=True)
        for c in args.covs:
            scc = np.mean([r["sr_clean"][f"{c}"] for r in rows]); scn = np.mean([r["sr_naive"][f"{c}"] for r in rows])
            print(f"        matched-cov {int(c*100)}%: selective-risk clean {scc:4.1f}  naive {scn:4.1f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nVERDICT: band naive AURC vs MLP naive AURC + matched-cov selective-risk -> is band's lower JOINT\n"
          f"risk genuine ranking or just operating point? wrote {args.out}")


if __name__ == "__main__":
    main()
