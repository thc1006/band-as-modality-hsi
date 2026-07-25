#!/usr/bin/env python
"""DOFA factor isolation: does SPATIAL CONTEXT alone explain why the DOFA foundation model escapes the
L1C->L2A certificate failure? DOFA differs from the fragile per-pixel model in four confounded ways
(capacity, pretraining, channel-adaptivity, spatial context). We isolate spatial context: on the SAME 9
DOFA bands and the SAME from-scratch training, we train (a) a per-pixel MLP -- no spatial context, like the
flagship model -- and (b) a small CNN with a 7x7 receptive field -- spatial context, no pretraining, no
foundation-scale capacity -- and run the identical CRC. If the CNN's naive L2A joint risk drops toward the
target while the MLP's stays high, spatial context is the driver; if both breach, spatial context alone
does not explain DOFA's robustness (pointing at pretraining/scale). Reuses phase8E's verified spatial
loader (load_spatial: 509->224 resized patches) so only the encoder differs.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase8E_dofa as P8E
from phase8R5_secondbench import split3, crc_thr, ce_joint
from bandsim.reliability import fit_temperature
from bandsim import hw


class SmallCNN(nn.Module):
    """3 conv blocks -> 7x7 receptive field: spatial context, from scratch, modest width."""
    def __init__(self, cin=9, ncls=4, w=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.Conv2d(w, ncls, 1))

    def forward(self, x):
        return self.net(x)


class PixelMLP(nn.Module):
    """Same 9 bands but a 1x1 receptive field: no spatial context, comparable capacity."""
    def __init__(self, cin=9, ncls=4, w=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, w), nn.ReLU(inplace=True),
                                 nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, ncls))

    def forward(self, x):
        return self.net(x)


def norm_fn(mu, sd):
    return lambda X: (X - mu[None, :, None, None]) / sd[None, :, None, None]


def train_cnn(Xtr, Ytr, seed, epochs, dev, bs=16):
    torch.manual_seed(seed)
    m = SmallCNN().to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    P = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(P)
        for i in range(0, P, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(Xtr[idx]).float().to(dev)
            yb = torch.from_numpy(Ytr[idx]).long().to(dev)
            opt.zero_grad()
            lossf(m(xb), yb).backward()
            opt.step()
    return m.eval()


def train_mlp(Xpix, Ypix, seed, epochs, dev, bs=8192):
    torch.manual_seed(seed)
    m = PixelMLP().to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    N = Xpix.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(Xpix[idx]).float().to(dev)
            yb = torch.from_numpy(Ypix[idx]).long().to(dev)
            opt.zero_grad()
            lossf(m(xb), yb).backward()
            opt.step()
    return m.eval()


@torch.no_grad()
def cnn_logits(m, X, dev, bs=16):
    out = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i + bs]).float().to(dev)
        out.append(m(xb).permute(0, 2, 3, 1).reshape(-1, 4).cpu().numpy())  # [P*H*W, 4]
    return np.concatenate(out)


@torch.no_grad()
def mlp_logits(m, Xpix, dev, bs=65536):
    out = []
    for i in range(0, Xpix.shape[0], bs):
        out.append(m(torch.from_numpy(Xpix[i:i + bs]).float().to(dev)).cpu().numpy())
    return np.concatenate(out)


def crc_report(lc, ll, y, comp, tag):
    """Naive (clean-cal) vs Mondrian (L2A-cal) component-equal joint risk over 3 splits."""
    nj, mj = [], []
    for ss in range(3):
        mt, mc, me = split3(comp, ss)
        Tc = fit_temperature(lc[mt], y[mt]); Tl = fit_temperature(ll[mt], y[mt])
        thr_n = crc_thr(lc, Tc, y, mc, comp)
        thr_m = crc_thr(ll, Tl, y, mc, comp)
        nj.append(ce_joint(ll, Tc, y, me, comp, thr_n))
        mj.append(ce_joint(ll, Tl, y, me, comp, thr_m))
    print(f"  {tag:18s} clean acc {(lc.argmax(1) == y).mean() * 100:4.1f}  L2A acc "
          f"{(ll.argmax(1) == y).mean() * 100:4.1f}  ->  naive L2A joint {np.mean(nj):5.2f}  "
          f"Mondrian {np.mean(mj):5.2f}", flush=True)
    return np.mean(nj), np.mean(mj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-train", type=int, default=300)
    ap.add_argument("--patches-test", type=int, default=200)
    ap.add_argument("--px-per-patch", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.patches_train, args.patches_test, args.px_per_patch, args.epochs, args.seeds = 40, 40, 150, 4, [0]
    dev = hw.setup(deterministic=True, prefer="auto")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    n_tr = len(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv")))
    n_te = len(pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv")))
    tr_ids = np.sort(np.random.default_rng(12345).choice(n_tr, min(args.patches_train, n_tr), replace=False))
    te_ids = np.sort(np.random.default_rng(70000).choice(n_te, min(args.patches_test, n_te), replace=False))
    comp_patch = np.unique(P8.scene_component_ids("test")[te_ids], return_inverse=True)[1].astype(np.int32)

    Xc_tr, Y_tr = P8E.load_spatial("train", "L1C", tr_ids)          # [P,9,224,224], [P,224,224]
    Y_tr = np.where((Y_tr >= 0) & (Y_tr < 4), Y_tr, -1).astype(np.int64)   # -1 -> ignore_index in the CNN loss
    Xc_te, Y_te = P8E.load_spatial("test", "L1C", te_ids)
    Xl_te, _ = P8E.load_spatial("test", "L2A", te_ids)
    mu = Xc_tr.mean((0, 2, 3)); sd = Xc_tr.std((0, 2, 3)) + 1e-6
    nrm = norm_fn(mu, sd)
    Xc_tr_n, Xc_te_n, Xl_te_n = nrm(Xc_tr), nrm(Xc_te), nrm(Xl_te)
    IMG = Xc_tr.shape[-1]
    print(f"loaded train {Xc_tr.shape[0]} / test {Xc_te.shape[0]} patches ({IMG}x{IMG}, 9 bands), "
          f"{len(np.unique(comp_patch))} scene-components; device {dev}", flush=True)

    # per-pixel views for the MLP (train pixels; test sampled pixels shared with the CNN scoring)
    Ytr_lab = Y_tr.reshape(-1)
    Xtr_pix = Xc_tr_n.transpose(0, 2, 3, 1).reshape(-1, 9)
    keep = (Ytr_lab >= 0) & (Ytr_lab < 4)
    Xtr_pix, Ytr_pix = Xtr_pix[keep], Ytr_lab[keep]
    if len(Ytr_pix) > 500_000:                                       # comparable to the flagship's ~900k train px
        s = np.random.default_rng(7).choice(len(Ytr_pix), 500_000, replace=False)
        Xtr_pix, Ytr_pix = Xtr_pix[s], Ytr_pix[s]

    rs = np.random.default_rng(999)
    sel = np.concatenate([P8E._subsample(rs, IMG * IMG, args.px_per_patch) + p * IMG * IMG
                          for p in range(Xc_te.shape[0])])
    y_te = Y_te.reshape(-1)[sel].astype(np.int64)
    comp = np.repeat(comp_patch, args.px_per_patch)
    valid = (y_te >= 0) & (y_te < 4)
    y_te, comp, sel = y_te[valid], comp[valid], sel[valid]

    naive_cnn, naive_mlp = [], []
    for seed in args.seeds:
        cnn = train_cnn(Xc_tr_n, Y_tr, seed, args.epochs, dev)
        lc_cnn = cnn_logits(cnn, Xc_te_n, dev)[sel]
        ll_cnn = cnn_logits(cnn, Xl_te_n, dev)[sel]
        n1, _ = crc_report(lc_cnn, ll_cnn, y_te, comp, f"seed{seed} CNN(spatial)")
        naive_cnn.append(n1)

        mlp = train_mlp(Xtr_pix, Ytr_pix, seed, args.epochs, dev)
        lc_mlp = mlp_logits(mlp, Xc_te_n.transpose(0, 2, 3, 1).reshape(-1, 9)[sel], dev)
        ll_mlp = mlp_logits(mlp, Xl_te_n.transpose(0, 2, 3, 1).reshape(-1, 9)[sel], dev)
        n2, _ = crc_report(lc_mlp, ll_mlp, y_te, comp, f"seed{seed} MLP(per-pixel)")
        naive_mlp.append(n2)

    print(f"\n=== spatial-context isolation (same 9 bands, from scratch, L2A naive joint) ===")
    print(f"  per-pixel MLP  : {np.mean(naive_mlp):5.2f}%  (no spatial context)")
    print(f"  spatial CNN    : {np.mean(naive_cnn):5.2f}%  (7x7 receptive field)")
    print(f"  DOFA (frozen, pretrained, foundation-scale): 9.95% for reference")
    verdict = ("spatial context ALONE largely restores control" if np.mean(naive_cnn) < 15
               else "spatial context alone does NOT explain it -- points at pretraining/scale"
               if np.mean(naive_cnn) > 22 else "spatial context PARTIALLY helps")
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
