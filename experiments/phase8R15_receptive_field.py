#!/usr/bin/env python
"""R7-C4 (round-7 review): CLEANLY isolate SPATIAL CONTEXT as the causal driver of breach attenuation.

The round-6 U-Net-vs-MLP contrast (phase8R10_unet_spatial) shrinks many confounds into one harness, but a
reviewer correctly notes it still varies architecture (conv vs linear), parameter count, convolutional inductive
bias, TRAINING REGIME (full-patch bs=12 vs sampled-pixel bs=4096), optimisation schedule and BatchNorm layout
ALL AT ONCE -- so the per-pixel->U-Net gap cannot be read as the pure effect of spatial context.

This experiment removes every one of those confounds. It trains ONE convolutional family in which the RECEPTIVE
FIELD (kernel size k) is the ONLY structural knob. Depth, width, output head, optimiser, learning rate, epochs,
batch size, training patches, normalisation and BatchNorm2d layout are IDENTICAL across arms; every arm trains on
the SAME full 224x224 L1C patches with the SAME bs=12 schedule (so the full-patch-vs-sampled-pixel training-regime
confound is gone -- the k=1 arm is a per-pixel model trained in the same regime as k=3), and every arm is
evaluated through the IDENTICAL scene-component conformal-risk-control on the SAME sampled test pixels.

  * k=1  : receptive field 1x1 -- NO spatial context (a 1x1 CNN == a per-pixel MLP applied convolutionally).
  * k=3,5,7 : receptive field 9x9 / 17x17 / 25x25 for a depth-4 stack -- spatial context grows monotonically.
  * k=1, w=96 ("1x1_wide") : a PARAMETER-MATCHED control to the k=3 arm with NO receptive field, so a breach
    change from k=1(w=32) -> k=3 that the wide-1x1 does NOT reproduce is attributable to the receptive field,
    not to capacity.

The L2A stale-normalisation breach as a function of k (at fixed depth/width/optimiser/epochs/batch/samples) is the
clean causal estimate of how much spatial context attenuates the silent failure. Parameter counts, training
samples, optimiser steps and batch size are all reported so the remaining (kernel-intrinsic) capacity difference
is transparent. No cross-experiment comparison and no claim beyond the matched family.

Run: CUDA_VISIBLE_DEVICES=0 python phase8R15_receptive_field.py --model-seeds 0 1 2 3 4
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8E_dofa as P8E                                  # spatial loader / IMG / _subsample reused
import phase8_cloudsen12 as P8
import phase8R_reliability as P8R
from phase8R10_unet_spatial import logits_sampled, norm_of  # identical eval path + normalisation as the U-Net arm
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.reliability import confidence_msp, fit_temperature, conformal_risk_control
from bandsim import hw

IMG = P8E.IMG                                               # 224
ALPHA = 0.10
STATES = [("clean", "L1C", "src"), ("L2A_src", "L2A", "src"), ("L2A_prod", "L2A", "prod")]
# (label, kernel, width): the receptive-field sweep at fixed width (k1..k7), PLUS a parameter-matched RF=1
# (wide 1x1) control for EACH kernel, so the receptive-field effect is capacity-controlled at every field
# size (not only at k3). Widths chosen so k1_wK params ~= kK params (reported exactly at run time).
ARMS = [("k1", 1, 32), ("k3", 3, 32), ("k5", 5, 32), ("k7", 7, 32),
        ("k1_w3", 1, 96), ("k1_w5", 1, 164), ("k1_w7", 1, 230)]


class ConvNet(nn.Module):
    """Depth/width/optimiser/epoch/batch-matched conv stack; kernel size k is the ONLY structural knob.
    padding=k//2 keeps 224x224 (NO pooling, so the receptive field is controlled purely by k and depth, not by
    down/up-sampling). A depth-d stack of kxk convs has receptive field 1 + d*(k-1). k=1 -> per-pixel."""
    def __init__(self, cin=9, cout=4, w=32, k=3, depth=4):
        super().__init__()
        layers, c = [], cin
        for _ in range(depth):
            layers += [nn.Conv2d(c, w, k, padding=k // 2), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        layers += [nn.Conv2d(w, cout, 1)]                  # identical 1x1 classification head across all arms
        self.net = nn.Sequential(*layers)
        self.rf = 1 + depth * (k - 1)                      # theoretical receptive field (pixels, one side)

    def forward(self, x):
        return self.net(x)


def train_convnet(X, Y, cin, k, w, seed, dev, epochs, bs, lr=1e-3):
    """IDENTICAL recipe for every arm; only (k, w) differ. Full-patch training for ALL arms (incl. k=1), so the
    training regime is held fixed and cannot confound the receptive-field contrast."""
    hw.seed_model(seed)                                            # P0-2: full seed (+101) before constructor
    m = ConvNet(cin, w=w, k=k).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr)
    lf = nn.CrossEntropyLoss()
    N, steps = len(X), 0
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs].numpy()
            xb = torch.from_numpy(X[idx]).float().to(dev)
            yb = torch.from_numpy(Y[idx]).long().to(dev)
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
            steps += 1
    return m.eval(), steps


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--patches-train", type=int, default=800)
    ap.add_argument("--patches-test", type=int, default=400)
    ap.add_argument("--px-per-patch", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "paper", "results_phase8R15_receptive_field"))
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)
    dev = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    import json
    import pandas as pd
    rng = np.random.default_rng(12345)
    ntr = len(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv")))
    nte = len(pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv")))
    train_ids = np.sort(rng.choice(ntr, size=min(args.patches_train, ntr), replace=False))
    comp = P8R.scene_component_ids("test")
    test_ids = np.sort(np.random.default_rng(70000).choice(nte, size=min(args.patches_test, nte), replace=False))
    unit_patch = comp[test_ids]

    # ---- train data: L1C spatial, normalized with L1C-TRAIN stats (identical to the U-Net arm) ----
    Xtr, Ytr = P8E.load_spatial("train", "L1C", train_ids)
    mu1 = Xtr.mean(axis=(0, 2, 3)); sd1 = Xtr.std(axis=(0, 2, 3)) + 1e-6      # stale source (L1C-train) stats
    Xtr_n = norm_of(Xtr, mu1, sd1)

    # ---- test patches per state + a fixed pixel subsample (shared across states/seeds/arms) ----
    Xl1c, yfull = P8E.load_spatial("test", "L1C", test_ids)
    Xl2a = P8E.load_spatial("test", "L2A", test_ids)[0]
    mu2 = Xl2a.mean(axis=(0, 2, 3)); sd2 = Xl2a.std(axis=(0, 2, 3)) + 1e-6     # product-aware (L2A own) stats
    Xn = {"clean": norm_of(Xl1c, mu1, sd1),
          "L2A_src": norm_of(Xl2a, mu1, sd1),
          "L2A_prod": norm_of(Xl2a, mu2, sd2)}
    rs = np.random.default_rng(54321)
    keep = [P8E._subsample(rs, IMG * IMG, args.px_per_patch) for _ in range(len(test_ids))]
    y_px = np.concatenate([yfull[p].reshape(-1)[keep[p]] for p in range(len(test_ids))])
    unit_px = np.concatenate([np.full(len(keep[p]), unit_patch[p]) for p in range(len(test_ids))])

    steps_per_epoch = int(np.ceil(len(train_ids) / args.bs))
    print(f"receptive-field isolation (ONE matched conv family): train {len(train_ids)} full patches, bs {args.bs}, "
          f"{args.epochs} epochs = {steps_per_epoch * args.epochs} optimiser steps/arm (IDENTICAL across arms); "
          f"test {len(test_ids)} patches / {np.unique(unit_px).size} scene-components / {len(y_px)} eval px; dev {dev}",
          flush=True)
    # report the (kernel-intrinsic) capacity of each arm ONCE, transparently
    meta = {}
    for label, k, w in ARMS:
        mm = ConvNet(Xtr_n.shape[1], w=w, k=k)
        meta[label] = {"kernel": k, "width": w, "receptive_field": mm.rf, "params": n_params(mm)}
        print(f"    arm {label:8s}: kernel {k}x{k}, width {w}, receptive field {mm.rf}x{mm.rf}px, "
              f"params {meta[label]['params']:,}", flush=True)
    print(f"    -> capacity-matched RF=1 controls: "
          + "; ".join(f"k1_w{kk} {meta[f'k1_w{kk}']['params']:,} vs k{kk} {meta[f'k{kk}']['params']:,} "
                      f"({meta[f'k1_w{kk}']['params'] / meta[f'k{kk}']['params']:.2f}x)" for kk in (3, 5, 7))
          + f"; k1 {meta['k1']['params']:,} (no-context baseline). Training samples/steps/batch identical across ALL arms.",
          flush=True)

    labels = [a[0] for a in ARMS]
    rows = {lb: {s[0]: [] for s in STATES} for lb in labels}
    covs = {lb: {s[0]: [] for s in STATES} for lb in labels}
    for ms in args.model_seeds:
        lg = {}
        for label, k, w in ARMS:
            m, steps = train_convnet(Xtr_n, Ytr, Xtr_n.shape[1], k, w, ms, dev, args.epochs, args.bs)
            lg[label] = {name: logits_sampled(m, Xn[name], dev, keep) for name, _p, _n in STATES}
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
        print(f"  model seed {ms}: " + "  ".join(
            f"{lb} clean {(lg[lb]['clean'].argmax(1) == y_px).mean() * 100:.1f}" for lb in labels), flush=True)
        for ss in args.split_seeds:
            mt, mc, me = P8R.split_test_rois(unit_px, ss)
            for lb in labels:
                T = fit_temperature(lg[lb]["clean"][mt], y_px[mt])                  # temp on disjoint clean split
                corr_cc = (lg[lb]["clean"][mc].argmax(1) == y_px[mc]).astype(int)
                conf_cc = confidence_msp(lg[lb]["clean"][mc] / T)
                for name, _p, _n in STATES:
                    le = lg[lb][name][me]
                    corr_e = (le.argmax(1) == y_px[me]).astype(int)
                    conf_e = confidence_msp(le / T)                                 # naive: clean temperature
                    crc = conformal_risk_control(corr_cc, conf_cc, corr_e, conf_e, alpha=ALPHA,
                                                 calib_group=unit_px[mc], eval_group=unit_px[me])
                    rows[lb][name].append((ms, ss, crc["eval_group_joint_risk"] * 100))
                    covs[lb][name].append(crc["eval_group_coverage"] * 100)

    df = min(len(set(args.model_seeds)), len(set(args.split_seeds))) - 1
    if df < 1:
        raise ValueError("need >= 2 model seeds AND >= 2 split seeds for two-way inference")
    from scipy.stats import t as student_t
    tcrit = float(student_t.ppf(0.975, df))
    agg = {lb: {} for lb in labels}
    print(f"\n  joint risk by receptive field (naive stale-norm certificate), two-way SE, t df={df}:")
    for lb in labels:
        for name, _p, _n in STATES:
            expected = {(s, r) for s in args.model_seeds for r in args.split_seeds}
            if {(s, r) for s, r, _ in rows[lb][name]} != expected:
                raise ValueError(f"arm {lb}/{name}: unbalanced grid ({len(rows[lb][name])}/{len(expected)})")
            mm, se = two_way_se(rows[lb][name]); agg[lb][name] = (mm, se)
        b = agg[lb]["L2A_src"]
        print(f"    {lb:8s} rf {meta[lb]['receptive_field']:2d}px  clean {agg[lb]['clean'][0]:5.1f}  "
              f"L2A_src {b[0]:5.1f} +/- {b[1]:.2f} [{b[0] - tcrit * b[1]:.1f},{b[0] + tcrit * b[1]:.1f}]  "
              f"L2A_prod {agg[lb]['L2A_prod'][0]:5.1f}  cov {np.mean(covs[lb]['L2A_src']):.0f}%", flush=True)

    # --- causal reads within the matched family (paired over the shared seed x split grid) ---
    def paired(a, b, state):
        da = {(s, r): v for s, r, v in rows[a][state]}
        db = {(s, r): v for s, r, v in rows[b][state]}
        d = [(s, r, da[(s, r)] - db[(s, r)]) for s in args.model_seeds for r in args.split_seeds]
        m, se = two_way_se(d)
        return m, se, m - tcrit * se, m + tcrit * se

    print("\n  PAIRED contrasts on the L2A stale-norm breach (mean +/- two-way SE [95% t CI]):")
    for a, b, lab in [("k1_w3", "k3", "CAPACITY-MATCHED (~30k): receptive field 1->9px"),
                      ("k1_w5", "k5", "CAPACITY-MATCHED (~85k): receptive field 1->17px"),
                      ("k1_w7", "k7", "CAPACITY-MATCHED (~165k): receptive field 1->25px"),
                      ("k1", "k1_w3", "CAPACITY-ONLY (RF fixed 1): 4k->30k params"),
                      ("k1", "k1_w7", "CAPACITY-ONLY (RF fixed 1): 4k->165k params"),
                      ("k1", "k7", "RF 1->25px WITH capacity co-growing (NOT matched)")]:
        md, se, lo, hi = paired(a, b, "L2A_src")
        z0 = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"    d[{a:6s} - {b:5s}] breach = {md:+6.2f} +/- {se:.2f} [{lo:+6.2f},{hi:+6.2f}] ({z0})  {lab}", flush=True)

    # --- persist ---
    per_cell = []
    for lb in labels:
        for name, _p, _n in STATES:
            for (s, r, j), cov in zip(rows[lb][name], covs[lb][name]):
                per_cell.append(dict(arm=lb, kernel=meta[lb]["kernel"], width=meta[lb]["width"],
                                     receptive_field=meta[lb]["receptive_field"], params=meta[lb]["params"],
                                     state=name, model_seed=int(s), split_seed=int(r), joint=j, coverage=cov))
    pd.DataFrame(per_cell).to_csv(args.out + "_percell.csv", index=False)
    summary = {"alpha": ALPHA, "df": df, "tcrit": tcrit, "meta": meta,
               "train_patches": int(len(train_ids)), "epochs": args.epochs, "batch": args.bs,
               "steps_per_arm": steps_per_epoch * args.epochs,
               "arms": {lb: {name: {"joint_mean": agg[lb][name][0], "joint_se": agg[lb][name][1],
                                    "coverage": float(np.mean(covs[lb][name]))} for name, _p, _n in STATES}
                        for lb in labels}}
    with open(args.out + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    bb = lambda a: agg[a]["L2A_src"][0]
    print(f"\n  SUMMARY (data, not an auto-verdict): L2A stale-norm breach by receptive field "
          f"k1 {bb('k1'):.1f} -> k3 {bb('k3'):.1f} -> k5 {bb('k5'):.1f} -> k7 {bb('k7'):.1f}; "
          f"the CAPACITY-MATCHED RF=1 controls are k1_w3 {bb('k1_w3'):.1f} (~30k), k1_w5 {bb('k1_w5'):.1f} (~85k), "
          f"k1_w7 {bb('k1_w7'):.1f} (~165k). The receptive-field effect is CAUSALLY identified only where a "
          "CAPACITY-MATCHED contrast (k1_wK minus kK, same params, only the kernel differs) excludes zero; the "
          "CAPACITY-ONLY contrasts (k1 minus k1_wK) test whether parameters alone attenuate. Attribute attenuation "
          "to spatial context ONLY for the capacity-matched fields whose CI excludes zero -- not from the raw "
          "k1->k7 sweep (whose capacity co-grows) -- and claim no fixed point count across architectures.")
    print(f"  wrote {args.out}_percell.csv + {args.out}_summary.json")


if __name__ == "__main__":
    main()
