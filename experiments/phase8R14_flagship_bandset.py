#!/usr/bin/env python
"""R6-E5 (round-6 review 3.7 / concern-6): the -B9 / matched-9-band ablation on the ACTUAL band-as-modality
FLAGSHIP (not the plain-MLP proxy of phase8R9). Two questions the reviewer raises:
  (1) Is the flagship's 28.9% L2A joint breach a B9 (water-vapour, the single most L1C->L2A-volatile band)
      artefact? -> retrain the flagship WITHOUT B9; if the breach stays far above 10% it is not a B9 artefact.
  (2) At DOFA's exact 9 bands, does the per-pixel flagship still breach while spatial DOFA (9.95%) does not?
      The band set is then CONTROLLED, so a residual breach isolates the per-pixel architecture (not the band
      set) as DOFA's differentiator.
We retrain the real flagship (grouped-band tokenisation + wavelength-conditioned PE + spectral-group MAE
pretraining + band-group dropout) on three band sets -- all 13, 13 minus B9, DOFA's 9 -- with the IDENTICAL
scene-component CRC / temperature / threshold protocol as the flagship.

SANITY GATE: the 13-band arm must reproduce the flagship's ~28.9% -- it validates the restricted-group
harness before the -B9 / 9-band arms are trusted (the 13-band restriction is the identity, so this arm IS
the flagship). CAVEAT (reported honestly): restricting the flagship's bands also re-partitions its spectral
groups, so this ablation confounds the band SET with the group STRUCTURE; the CLEAN, same-architecture
band-set isolation is the plain MLP of phase8R9 (48.4 -> 29.2 -> 23.8). This arm answers the narrower
'is the flagship headline B9-driven?' question, which phase8R9 (a different model) cannot.

Run: CUDA_VISIBLE_DEVICES=1 python phase8R14_flagship_bandset.py --seeds 0 1 2 3 4
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R
from phase8R3_acolite import overall_metrics
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.model import GroupedCrossBandAttention
from bandsim.reliability import fit_temperature, conformal_risk_control
from bandsim import hw

ALPHA = 0.10
L1C = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
DOFA9 = ["B4", "B3", "B2", "B5", "B6", "B7", "B8", "B11", "B12"]


def restrict_groups(groups, wl, keep):
    """Remap the flagship's spectral groups + per-group centre wavelengths onto the kept original band
    indices (`keep`, sorted). A partially-kept group survives with its centre recomputed from its kept
    bands; a fully-dropped group vanishes. For keep=range(13) this is the identity (sanity: == flagship)."""
    pos = {orig: new for new, orig in enumerate(keep)}
    ng, nw = [], []
    for g in groups:
        kept = [i for i in g if i in pos]
        if kept:
            ng.append(sorted(pos[i] for i in kept))
            nw.append(float(np.mean([wl[i] for i in kept])))
    return ng, np.array(nw, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)

    groups0 = P8.s2_physical_groups()
    wl = np.array(P8.S2_WL_NM, float)
    B9, B10 = L1C.index("B9"), P8.B10_IDX
    sets = {"13-band(all)": list(range(13)),
            "-B9(12-band)": [i for i in range(13) if i != B9],
            "9-band(DOFA)": sorted(L1C.index(b) for b in DOFA9)}

    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    ids = np.flatnonzero(~meta["s2_id"].isin(train_prod).to_numpy())    # full leak-guarded test

    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=(prd == "L1C"))
    X_l1c, y_te, pid = load("L1C")
    X_l2a, _ = load("L2A")                                              # same seed -> pixel-aligned with L1C
    comp_all = P8R.scene_component_ids("test")[pid]
    Xtr_l1c, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    ytr, y_te = ytr.astype(np.int64), y_te.astype(np.int64)
    clip = lambda A: np.clip(A, -0.1, 1.6)

    tcrit = {3: 4.303, 5: 2.776, 10: 2.262}.get(len(args.seeds), 2.776)
    print(f"flagship band-set ablation: {len(y_te)} test px / {len(np.unique(comp_all))} components; "
          f"{len(args.seeds)} seeds x {len(args.split_seeds)} splits; alpha {ALPHA}", flush=True)

    for name, keep in sets.items():
        groups, cwl = restrict_groups(groups0, wl, keep)
        has_b10 = B10 in keep
        # B10 (cirrus) is zero/absent in L2A -> drop its singleton group at L2A inference, as the flagship does.
        band2grp = {i: gi for gi, g in enumerate(groups) for i in g}
        drop_l2a = [band2grp[keep.index(B10)]] if has_b10 else []
        Xtr_k = clip(Xtr_l1c)[:, keep]
        mu, sd = Xtr_k.mean(0), Xtr_k.std(0) + 1e-8
        nrm = lambda A: ((clip(A)[:, keep] - mu) / sd).astype(np.float32)
        Xtr_n, Xc_clean, Xc_l2a = ((Xtr_k - mu) / sd).astype(np.float32), nrm(X_l1c), nrm(X_l2a)
        bs = P2.auto_bs(Xtr_n.shape[0])
        rows, ca, la = [], 0.0, 0.0
        for seed in args.seeds:
            random.seed(seed + 101); np.random.seed(seed + 101); torch.manual_seed(seed + 101)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + 101)               # seed BEFORE the constructor (P0-1)
            m = GroupedCrossBandAttention(groups, cwl, 4)
            P2.pretrain_sgmae(m, Xtr_n, groups, seed, epochs=max(1, args.epochs // 2), bs=bs)
            P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)
            lc = P8R.logits_at("proposed", m, Xc_clean, groups, [])
            ll = P8R.logits_at("proposed", m, Xc_l2a, groups, drop_l2a)
            ca += (lc.argmax(1) == y_te).mean() * 100 / len(args.seeds)
            la += (ll.argmax(1) == y_te).mean() * 100 / len(args.seeds)
            for ss in args.split_seeds:
                mt, mc, me = P8R.split_test_rois(comp_all, ss)
                Tc = fit_temperature(lc[mt], y_te[mt])
                pc = softmax(lc[mc] / Tc, axis=1)
                corr_c = pc.argmax(1) == y_te[mc]
                thr = conformal_risk_control(corr_c, pc.max(1), corr_c, pc.max(1), alpha=ALPHA,
                                             calib_group=comp_all[mc], eval_group=comp_all[mc])["threshold"]
                pe = softmax(ll[me] / Tc, axis=1)
                j, sel, cov = overall_metrics(pe.argmax(1) == y_te[me], pe.max(1), comp_all[me], thr)
                rows.append((seed, ss, j))
        mn, se = two_way_se(rows)
        lo, hi = mn - tcrit * se, mn + tcrit * se
        print(f"  {name:14s} ({len(keep):2d}b) clean acc {ca:4.1f}  L2A acc {la:4.1f}  ->  naive L2A joint "
              f"{mn:5.2f} +/- {se:.2f}  [{lo:.1f},{hi:.1f}]  {'clear-excess' if lo > 10 else ('at/below-target' if hi <= 10 else 'inconclusive')}", flush=True)

    print("\n  reference: flagship 13-band 28.9 (10x10);  DOFA-9 spatial 9.95")
    print("  -> 13-band arm reproducing ~28.9 validates the restricted-group harness; -B9 staying >>10 shows")
    print("     the headline breach is NOT a B9 artefact; 9-band staying >>10 (while spatial DOFA-9 = 9.95)")
    print("     isolates the per-pixel architecture, not the band set, as DOFA's differentiator.")
    print("  NOTE: restricting bands also re-partitions the spectral groups (band-set/group-structure confound);")
    print("        the clean same-architecture band-set isolation is the plain MLP of phase8R9.")


if __name__ == "__main__":
    main()
