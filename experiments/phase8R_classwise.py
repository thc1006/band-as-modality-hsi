#!/usr/bin/env python
"""Class-wise segmentation + reliability metrics for the flagship (reviewer S3.7).

Pixel accuracy alone can hide a per-class collapse on the imbalanced 4-class cloud task. Here we
report, for the proposed model on the clean (L1C) and shifted (L2A) states:
  * class-wise IoU, F1, producer accuracy (recall) and user accuracy (precision), plus mIoU / macro-F1;
  * class-wise CRC operating-point metrics at the SAME clean-calibrated threshold used by the flagship:
    joint risk P(accepted and wrong | class), selective risk P(wrong | accepted, class), coverage.
This is a descriptive per-class breakdown, so we use a single representative training seed and one
scene-component-disjoint calibration/temperature/evaluation split (leak-guarded), not the 100-run
design of the headline (the headline number is unchanged; this only decomposes it by class).
"""
import os, sys, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                          # experiments/ on path (mirror phase8R flat imports)
sys.path.insert(0, os.path.dirname(_HERE))         # repo root for the bandsim package
import phase8_cloudsen12 as P8
import phase2_degradation as P2
import phase8R_reliability as P8R
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import group_center_wavelengths
from bandsim.reliability import fit_temperature, conformal_risk_control
from bandsim import hw

CLASSES = ["clear", "thick cloud", "thin cloud", "cloud shadow"]
NC = 4


def perclass_seg(y, pred):
    """IoU / F1 / PA(recall) / UA(precision) / support per class from a confusion count."""
    out = {}
    for c in range(NC):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        pa = tp / (tp + fn) if (tp + fn) else float("nan")          # recall / producer acc
        ua = tp / (tp + fp) if (tp + fp) else float("nan")          # precision / user acc
        f1 = 2 * pa * ua / (pa + ua) if (pa + ua) else float("nan")
        out[c] = dict(iou=iou, f1=f1, pa=pa, ua=ua, support=int(np.sum(y == c)))
    return out


def perclass_risk(y, pred, conf, thr):
    """At acceptance threshold `thr`: per-class joint risk P(acc&wrong|class), selective risk
    P(wrong|acc,class), coverage P(acc|class)."""
    acc = conf >= thr
    wrong = pred != y
    out = {}
    for c in range(NC):
        m = y == c
        n = int(m.sum())
        if n == 0:
            out[c] = dict(joint=float("nan"), selective=float("nan"), coverage=float("nan"))
            continue
        na = int((acc & m).sum())
        joint = float((acc & wrong & m).sum()) / n
        sel = (float((acc & wrong & m).sum()) / na) if na else float("nan")
        out[c] = dict(joint=joint, selective=sel, coverage=na / n)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)

    groups = P8.s2_physical_groups()
    cwl = group_center_wavelengths(np.array(P8.S2_WL_NM, float), groups)
    NUM_CLASSES = NC

    # train (source L1C) -- same subsample rule as phase8R
    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)

    # test L1C + L2A, leak-guarded (mirror phase8R), same pixel sampling
    import pandas as pd
    meta = pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv"))
    train_prod = set(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv"))["s2_id"].dropna())
    leaked = meta.index[meta["s2_id"].isin(train_prod)].to_numpy()
    ids = np.setdiff1d(np.arange(len(meta)), leaked)
    def load(prd):
        return P8.load_split("test", prd, pixels_per_patch=400, patch_ids=ids, seed=54321,
                             return_patch_id=(prd == "L1C"))
    X_l1c, y_te, pid = load("L1C")
    X_l2a, _ = load("L2A")
    comp = P8R.scene_component_ids("test")[pid]                     # exchangeable unit
    Xtr = norm(Xtr); Xc = {"clean": norm(X_l1c), "L2A": norm(X_l2a)}

    # train proposed
    bs = P2.auto_bs(Xtr.shape[0])
    hw.seed_model(args.seed)                                        # P0-2: seed before the constructor
    m = GroupedCrossBandAttention(groups, cwl, NUM_CLASSES)
    P2.pretrain_sgmae(m, Xtr, groups, args.seed, epochs=max(1, args.epochs // 2), bs=bs)
    P2.finetune_proposed(m, Xtr, ytr, groups, args.seed, epochs=args.epochs, bs=bs)

    # scene-component-disjoint temp / calib / eval split (reuse phase8R's splitter)
    mt, mc, me = P8R.split_test_rois(comp, args.split_seed)
    g_b10 = P8._assert_singleton(groups, P8.B10_IDX, "B10")
    drop = {"clean": [], "L2A": [g_b10]}                            # L2A: B10 absent

    def logits(state, mask):
        return P8R.logits_at("proposed", m, Xc[state][mask], groups, drop[state])

    # temperature on the clean temp split; clean-calibrated CRC threshold on the calib split
    from scipy.special import softmax
    lt = logits("clean", mt); T = fit_temperature(lt, y_te[mt])
    def conf_pred(state, mask):
        lg = logits(state, mask) / T
        p = softmax(lg, axis=1)
        return p.argmax(1), p.max(1)
    pc, cc = conf_pred("clean", mc)
    thr = conformal_risk_control((pc == y_te[mc]), cc, (pc == y_te[mc]), cc, alpha=args.alpha,
                                 calib_group=comp[mc], eval_group=comp[mc])["threshold"]

    rows = []
    print(f"\n===== class-wise (proposed, seed {args.seed}, split {args.split_seed}, alpha {args.alpha:.0%}) =====")
    print(f"clean-calibrated CRC threshold = {thr:.4f}")
    for state in ("clean", "L2A"):
        pred, conf = conf_pred(state, me)
        yy = y_te[me]
        seg = perclass_seg(yy, pred)
        rk = perclass_risk(yy, pred, conf, thr)
        miou = float(np.nanmean([seg[c]["iou"] for c in range(NC)]))
        mf1 = float(np.nanmean([seg[c]["f1"] for c in range(NC)]))
        oa = float(np.mean(pred == yy))
        print(f"\n[{state}]  OA={oa*100:.1f}  mIoU={miou*100:.1f}  macroF1={mf1*100:.1f}")
        print(f"  {'class':13s} {'IoU':>5s} {'F1':>5s} {'PA':>5s} {'UA':>5s} | {'joint':>6s} {'selec':>6s} {'cov':>6s}  support")
        for c in range(NC):
            s, r = seg[c], rk[c]
            print(f"  {CLASSES[c]:13s} {s['iou']*100:5.1f} {s['f1']*100:5.1f} {s['pa']*100:5.1f} "
                  f"{s['ua']*100:5.1f} | {r['joint']*100:6.2f} {r['selective']*100:6.2f} "
                  f"{r['coverage']*100:6.1f}  {s['support']}")
            rows.append(dict(state=state, cls=CLASSES[c], iou=s["iou"]*100, f1=s["f1"]*100,
                             pa=s["pa"]*100, ua=s["ua"]*100, joint=r["joint"]*100,
                             selective=r["selective"]*100, coverage=r["coverage"]*100,
                             support=s["support"], oa=oa*100, miou=miou*100, macrof1=mf1*100))
    out = P8R.P("results_phase8R_classwise.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
