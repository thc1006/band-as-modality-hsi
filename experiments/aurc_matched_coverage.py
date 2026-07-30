#!/usr/bin/env python3
"""AURC + matched-coverage analysis: is the band model's LOWER joint-risk under the 6S shift a genuine
robustness (better confidence RANKING) or merely a lower operating point (it just abstains more)?

Joint risk P(accepted & wrong) conflates accuracy with coverage, so a model that happens to accept fewer
pixels looks "safer". Two coverage-honest metrics disentangle that:
  * AURC (area under the risk-coverage curve): integrate selective risk over ALL coverage levels -- purely a
    measure of whether confidence RANKS errors below correct predictions. Coverage-independent. Lower = better.
  * selective risk at MATCHED coverage: error among the top-c% most-confident, at the SAME c for both models.

Finding (Indian Pines HSI 6S dry->humid): under shift the band-as-modality model is WORSE, not better -- its
AURC rises above its own full-coverage error (its most-confident predictions become MORE wrong than average:
anti-informative confidence), and at every matched coverage its selective risk exceeds the MLP's. So the
band's lower joint-risk is an operating-point artifact (coverage collapse), NOT graceful self-protection.
Corrects an earlier over-generous "band self-protects" reading.
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

import phase1_indian_pines as P1
import phase2_degradation as P2
from bandsim import hw
from bandsim.io import disjoint_block_split
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention
from bandsim.reliability import fit_temperature
from phase8G_emit_shift import logits_of, logits_band


def aurc(conf, corr):
    """Area under the risk-coverage curve (%). trapezoid of selective risk vs coverage; lower = confidence
    ranks errors better. If it exceeds the full-coverage error, the most-confident are MORE wrong than
    average -- anti-informative confidence."""
    o = np.argsort(-conf); c = corr[o].astype(float); k = np.arange(1, len(c) + 1)
    sel = 1.0 - np.cumsum(c) / k
    from scipy.integrate import trapezoid
    return float(trapezoid(sel, k / len(c))) * 100


def sel_risk_at(conf, corr, cov):
    o = np.argsort(-conf); k = max(1, int(cov * len(conf)))
    return float(1.0 - corr[o][:k].mean()) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--covs", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_aurc_matched_coverage.json"))
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer="auto"); dev = hw.device()
    cube, gt = P1.load_indian_pines(); z = np.load(os.path.join(_HERE, "..", "data", "srf_cache", "T_6s_grid.npz"))
    src = cube * np.asarray(z["cwv0.5"], float); tgt = cube * np.asarray(z["cwv4.0"], float)
    groups = contiguous_groups(cube.shape[-1], 12)
    cwl = group_center_wavelengths(np.asarray(z["wl_nm"], float), groups)

    def run(seed, mk):
        hw.seed_model(seed)
        tr, te = disjoint_block_split(gt, block=10, guard=1, offset=seed % 2); yx = gt - 1
        tri = np.argwhere(tr); tei = np.argwhere(te); gv = lambda C, ij: C[ij[:, 0], ij[:, 1]]
        rng = np.random.default_rng(seed + 13); pm = rng.permutation(len(tei)); cut = len(tei) // 2
        cai, evi = tei[pm[:cut]], tei[pm[cut:]]
        Xtr = gv(src, tri); ytr = gv(yx, tri); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        norm = lambda X: ((X - mu) / sd).astype(np.float32)
        if mk == "band":
            m = GroupedCrossBandAttention(groups, cwl, 16)
            P2.pretrain_sgmae(m, norm(Xtr), groups, seed, epochs=max(1, args.epochs // 2), bs=2048)
            P2.finetune_proposed(m, norm(Xtr), ytr, groups, seed, epochs=args.epochs, bs=2048, group_dropout=False)
            L = lambda X: logits_band(m, X, groups, dev)
        else:
            m = P2.train_mlp(norm(Xtr), ytr, groups, seed, group_dropout=False, epochs=args.epochs, hidden=256, num_classes=16)
            L = lambda X: logits_of(m, X, dev)
        lc = L(norm(gv(src, cai))); yc = gv(yx, cai); h = len(lc) // 2; Tc = fit_temperature(lc[:h], yc[:h])
        yev = gv(yx, evi)
        ps = softmax(L(norm(gv(src, evi))) / Tc, axis=1); cs = ps.argmax(1) == yev
        pn = softmax(L(norm(gv(tgt, evi))) / Tc, axis=1); cn = pn.argmax(1) == yev
        return dict(seed=seed, aurc_clean=aurc(ps.max(1), cs), aurc_naive=aurc(pn.max(1), cn),
                    err_clean=float((~cs).mean()) * 100, err_naive=float((~cn).mean()) * 100,
                    sr_clean={f"{c}": sel_risk_at(ps.max(1), cs, c) for c in args.covs},
                    sr_naive={f"{c}": sel_risk_at(pn.max(1), cn, c) for c in args.covs})

    out = {}
    print(f"HSI-6S Indian Pines AURC/matched-coverage, {len(args.seeds)} seeds (lower AURC = better ranking):", flush=True)
    for mk in ["mlp", "band"]:
        rows = [run(s, mk) for s in args.seeds]
        out[mk] = rows
        ac = np.mean([r["aurc_clean"] for r in rows]); an = np.mean([r["aurc_naive"] for r in rows])
        en = np.mean([r["err_naive"] for r in rows])
        anti = " (ANTI-informative: AURC > full error!)" if an > en else ""
        print(f"  {mk:4s}: AURC clean {ac:4.1f} -> naive {an:4.1f} (full-cov err {en:.0f}){anti}", flush=True)
        for c in args.covs:
            scc = np.mean([r["sr_clean"][f"{c}"] for r in rows]); scn = np.mean([r["sr_naive"][f"{c}"] for r in rows])
            print(f"        matched-cov {int(c*100)}%: selective-risk clean {scc:4.1f}  naive {scn:4.1f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nVERDICT: if band AURC_naive >= MLP AURC_naive and band matched-cov selective-risk >= MLP, the band's\n"
          f"lower JOINT risk is an operating-point (coverage) artifact, NOT genuine robustness. wrote {args.out}")


if __name__ == "__main__":
    main()
