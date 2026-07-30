#!/usr/bin/env python3
"""AURC + matched-coverage: is a model's LOWER joint-risk under the 6S shift genuine robustness (better
confidence RANKING) or merely a lower operating point (it just abstains more)?

Joint risk P(accepted & wrong) conflates accuracy with coverage, so a model that accepts fewer pixels looks
"safer". Two coverage-honest metrics disentangle it:
  * AURC (area under the risk-coverage curve): selective risk integrated over ALL coverage. Coverage-
    independent; lower = confidence ranks errors better. If it exceeds the full-coverage error, the most-
    confident predictions are MORE wrong than average (anti-informative confidence).
  * selective risk at MATCHED coverage: error among the top-c% most-confident, at the SAME c for both models.

Generalized over --dataset {indian_pines,salinas,pavia} and --spatial-cwv (uniform vs per-pixel atmosphere).
AURC/matched-coverage are only APPROXIMATELY invariant to temperature scaling: for 2 classes max-softmax is
monotone in T (exactly invariant), but for K>2 a single scalar T can slightly reorder max-softmax across
samples. A direct check (fit T on the source calib, T=1.2-2.4 here) shifts AURC by <=2 pp and flips NO verdict,
so no calibration is applied. Pixel-level.

Finding so far (Indian Pines HSI 6S uniform): under shift the band-as-modality model is WORSE, not better --
higher AURC and higher matched-coverage selective risk than the MLP -- so its lower joint-risk is an
operating-point artifact (coverage collapse), NOT graceful self-protection.
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

import phase2_degradation as P2
from bandsim import hw
from bandsim.io import disjoint_block_split
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.model import GroupedCrossBandAttention
from phase8G_emit_shift import logits_of, logits_band
from phase8H_hsi6s_shift import _load_dataset


def aurc(conf, corr):
    o = np.argsort(-conf); c = corr[o].astype(float); k = np.arange(1, len(c) + 1)
    return float(trapezoid(1.0 - np.cumsum(c) / k, k / len(c))) * 100


def sel_risk_at(conf, corr, cov):
    o = np.argsort(-conf); k = max(1, int(cov * len(conf)))
    return float(1.0 - corr[o][:k].mean()) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="indian_pines", choices=["indian_pines", "salinas", "pavia"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--covs", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    ap.add_argument("--spatial-cwv", action="store_true")
    ap.add_argument("--src-cwv", default="cwv0.5")
    ap.add_argument("--tgt-cwv", default="cwv4.0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        stem = "results_aurc_matched_coverage" if args.dataset == "indian_pines" else f"results_aurc_{args.dataset}"
        if args.spatial_cwv:
            stem += "_spatial"
        args.out = os.path.join(_HERE, "..", "paper", stem + ".json")
    hw.setup(deterministic=True, prefer="auto"); dev = hw.device()

    cube, gt, K, lut = _load_dataset(args.dataset); z = np.load(lut)
    src = cube * np.asarray(z[args.src_cwv], float)
    if args.spatial_cwv:                                    # per-pixel CWV gradient (matches phase8H)
        grid = np.array([0.5, 2.0, 4.0]); Tg = np.stack([np.asarray(z[f"cwv{c}"], float) for c in grid])
        sv = float(args.src_cwv.replace("cwv", "")); tv = float(args.tgt_cwv.replace("cwv", "")); H, W = gt.shape
        cwv_map = (sv + (tv - sv) * np.linspace(0, 1, H)[:, None] * np.ones((1, W))).reshape(-1)
        Tt_map = np.stack([np.interp(cwv_map, grid, Tg[:, b]) for b in range(Tg.shape[1])], axis=1)
        tgt = (cube.reshape(-1, cube.shape[-1]) * Tt_map).reshape(cube.shape)
    else:
        tgt = cube * np.asarray(z[args.tgt_cwv], float)
    groups = contiguous_groups(cube.shape[-1], 12)
    cwl = group_center_wavelengths(np.asarray(z["wl_nm"], float), groups)

    def run(seed, mk):
        hw.seed_model(seed)
        tr, te = disjoint_block_split(gt, block=10, guard=1, offset=seed % 2); yx = gt - 1
        tri = np.argwhere(tr); tei = np.argwhere(te); gv = lambda C, ij: C[ij[:, 0], ij[:, 1]]
        rng = np.random.default_rng(seed + 13); pm = rng.permutation(len(tei)); cut = len(tei) // 2
        evi = tei[pm[cut:]]
        Xtr = gv(src, tri); ytr = gv(yx, tri); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        norm = lambda X: ((X - mu) / sd).astype(np.float32)
        if mk == "band":
            m = GroupedCrossBandAttention(groups, cwl, K)
            P2.pretrain_sgmae(m, norm(Xtr), groups, seed, epochs=max(1, args.epochs // 2), bs=2048)
            P2.finetune_proposed(m, norm(Xtr), ytr, groups, seed, epochs=args.epochs, bs=2048, group_dropout=False)
            L = lambda X: logits_band(m, X, groups, dev)
        else:
            m = P2.train_mlp(norm(Xtr), ytr, groups, seed, group_dropout=False, epochs=args.epochs, hidden=256, num_classes=K)
            L = lambda X: logits_of(m, X, dev)
        yev = gv(yx, evi)
        ps = softmax(L(norm(gv(src, evi))), axis=1); cs = ps.argmax(1) == yev
        pn = softmax(L(norm(gv(tgt, evi))), axis=1); cn = pn.argmax(1) == yev
        return dict(seed=seed, aurc_clean=aurc(ps.max(1), cs), aurc_naive=aurc(pn.max(1), cn),
                    err_clean=float((~cs).mean()) * 100, err_naive=float((~cn).mean()) * 100,
                    sr_clean={f"{c}": sel_risk_at(ps.max(1), cs, c) for c in args.covs},
                    sr_naive={f"{c}": sel_risk_at(pn.max(1), cn, c) for c in args.covs})

    out = {}
    tag = "SPATIAL/per-pixel" if args.spatial_cwv else "UNIFORM"
    print(f"{args.dataset} HSI-6S {tag} AURC/matched-coverage, {len(args.seeds)} seeds (lower AURC = better ranking):", flush=True)
    for mk in ["mlp", "band"]:
        rows = [run(s, mk) for s in args.seeds]; out[mk] = rows
        m = lambda k: float(np.mean([r[k] for r in rows]))
        an, en = m("aurc_naive"), m("err_naive")
        note = " (ANTI-informative: AURC > full error!)" if an > en else (" DEAD (AURC ~ error)" if en - an < 5 else "")
        print(f"  {mk:4s}: AURC clean {m('aurc_clean'):4.1f} -> naive {an:4.1f} (full-cov err {en:.0f}){note}", flush=True)
        for c in args.covs:
            scc = np.mean([r["sr_clean"][f"{c}"] for r in rows]); scn = np.mean([r["sr_naive"][f"{c}"] for r in rows])
            print(f"        matched-cov {int(c*100)}%: selective-risk clean {scc:4.1f}  naive {scn:4.1f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
