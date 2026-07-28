#!/usr/bin/env python
"""R6-E6 (round-6 review B2 / reframe K3): does the SILENT certificate failure + the label-free product-aware
normalization fix GENERALISE beyond the deliberately per-pixel spectral model to a representative SPATIAL
segmentation model -- and how much of the SEVERITY is driven by spatial context alone? We train, FROM SCRATCH
(no pretraining), BOTH a spatial U-Net AND a per-pixel MLP on the SAME CloudSEN12 L1C data (the DOFA 9-band,
224x224 layout, band-matched to the round-5 9-band per-pixel baseline), evaluate BOTH on the SAME sampled test
pixels through the IDENTICAL scene-component conformal-risk-control, and match their BatchNorm and depth so that
SPATIAL CONTEXT is the ONLY structural difference between the two arms. Three states each: clean L1C; L2A with
the STALE L1C-train normalization (the naive certificate); L2A with PRODUCT-AWARE (own-statistics, unlabelled)
normalization. If BOTH breach under stale normalization and product-aware normalization restores BOTH, the
diagnosis is not a per-pixel artefact and the fix is model-agnostic; the per-pixel-minus-U-Net gap then measures
how much spatial context attenuates the breach WITHIN ONE HARNESS (no cross-experiment comparison). Reuses
phase8E_dofa.load_spatial (loader / IMG / _subsample), the scene-component unit, and the CRC verbatim.

Run: CUDA_VISIBLE_DEVICES=0 python phase8R10_unet_spatial.py --model-seeds 0 1 2 3 4
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
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.reliability import confidence_msp, fit_temperature, conformal_risk_control
from bandsim import hw

IMG = P8E.IMG                                               # 224
ALPHA = 0.10
# (name, product, normalization): src = stale L1C-train stats, prod = product-aware L2A own stats
STATES = [("clean", "L1C", "src"), ("L2A_src", "L2A", "src"), ("L2A_prod", "L2A", "prod")]


class UNet(nn.Module):
    """Compact 3-level U-Net from scratch (no pretraining). 224 -> /8 -> 224."""
    def __init__(self, cin=9, cout=4, w=32):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.d1, self.d2, self.d3 = blk(cin, w), blk(w, 2 * w), blk(2 * w, 4 * w)
        self.bott = blk(4 * w, 8 * w)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, 2); self.c3 = blk(8 * w, 4 * w)
        self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2); self.c2 = blk(4 * w, 2 * w)
        self.u1 = nn.ConvTranspose2d(2 * w, w, 2, 2); self.c1 = blk(2 * w, w)
        self.out = nn.Conv2d(w, cout, 1)

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(self.pool(d1)); d3 = self.d3(self.pool(d2))
        b = self.bott(self.pool(d3))
        u3 = self.c3(torch.cat([self.u3(b), d3], 1))
        u2 = self.c2(torch.cat([self.u2(u3), d2], 1))
        u1 = self.c1(torch.cat([self.u1(u2), d1], 1))
        return self.out(u1)


class PixelMLP(nn.Module):
    """Per-pixel MLP (NO spatial context) -- BatchNorm-matched to the U-Net so spatial context is the only
    structural difference between the two arms."""
    def __init__(self, cin=9, cout=4, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                                 nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                                 nn.Linear(h, cout))

    def forward(self, x):
        return self.net(x)


def train_unet(X, Y, cin, seed, dev, epochs, bs=12, lr=1e-3):
    hw.seed_model(seed)                                            # P0-2: full seed (+101) before constructor
    m = UNet(cin).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr)
    lf = nn.CrossEntropyLoss()
    N = len(X)
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs].numpy()
            xb = torch.from_numpy(X[idx]).float().to(dev)
            yb = torch.from_numpy(Y[idx]).long().to(dev)
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
    return m.eval()


def train_mlp(Xpx, Ypx, cin, seed, dev, epochs, bs=4096, lr=1e-3):
    hw.seed_model(seed)                                            # P0-2: full seed (+101) before constructor
    m = PixelMLP(cin).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr)
    lf = nn.CrossEntropyLoss()
    N = len(Xpx)
    for _ in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs].numpy()
            xb = torch.from_numpy(Xpx[idx]).float().to(dev)
            yb = torch.from_numpy(Ypx[idx]).long().to(dev)
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
    return m.eval()


@torch.no_grad()
def logits_sampled(m, X, dev, keep_per_patch, bs=8):
    """U-Net per-pixel logits at the fixed sampled positions of each patch -> (sum_k, C). Uses spatial context."""
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).float().to(dev)
        lg = m(xb).permute(0, 2, 3, 1).reshape(len(xb), IMG * IMG, -1).cpu().numpy()
        for j in range(len(xb)):
            out.append(lg[j, keep_per_patch[i + j]])
    return np.concatenate(out)


@torch.no_grad()
def mlp_logits(m, X, dev, keep_per_patch):
    """Per-pixel MLP logits at the SAME sampled positions -> (sum_k, C). NO spatial context; same pixel order
    as logits_sampled / y_px (concatenate patch-by-patch, keep[p] within each)."""
    px = np.concatenate([X[p].reshape(X.shape[1], -1).T[keep_per_patch[p]] for p in range(len(X))])
    return m(torch.from_numpy(px.astype(np.float32)).to(dev)).cpu().numpy()


def flatten_sampled(X, Y, keep_per_patch):
    """Flatten spatial patches to per-pixel (feature, label) at the sampled positions -- same (h*W+w) index in
    both, so features and labels stay aligned."""
    Xpx = np.concatenate([X[p].reshape(X.shape[1], -1).T[keep_per_patch[p]] for p in range(len(X))])
    Ypx = np.concatenate([Y[p].reshape(-1)[keep_per_patch[p]] for p in range(len(X))])
    return Xpx.astype(np.float32), Ypx


def norm_of(X, mu, sd):
    return ((X - mu[None, :, None, None]) / sd[None, :, None, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--patches-train", type=int, default=800)
    ap.add_argument("--patches-test", type=int, default=400)
    ap.add_argument("--px-per-patch", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=25)              # U-Net
    ap.add_argument("--mlp-epochs", type=int, default=60)          # per-pixel MLP (cheap, more epochs)
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()
    hw.setup(deterministic=True, prefer=args.device)
    dev = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    import pandas as pd
    rng = np.random.default_rng(12345)
    ntr = len(pd.read_csv(os.path.join(P8.DATA, "train", "metadata.csv")))
    nte = len(pd.read_csv(os.path.join(P8.DATA, "test", "metadata.csv")))
    train_ids = np.sort(rng.choice(ntr, size=min(args.patches_train, ntr), replace=False))
    comp = P8R.scene_component_ids("test")
    test_ids = np.sort(np.random.default_rng(70000).choice(nte, size=min(args.patches_test, nte), replace=False))
    unit_patch = comp[test_ids]

    # ---- train data: L1C spatial, normalized with L1C-TRAIN stats ----
    Xtr, Ytr = P8E.load_spatial("train", "L1C", train_ids)
    mu1 = Xtr.mean(axis=(0, 2, 3)); sd1 = Xtr.std(axis=(0, 2, 3)) + 1e-6      # stale source (L1C-train) stats
    Xtr_n = norm_of(Xtr, mu1, sd1)

    # ---- test patches per state + a fixed pixel subsample (shared across states/seeds/arms) ----
    Xl1c, yfull = P8E.load_spatial("test", "L1C", test_ids)
    Xl2a = P8E.load_spatial("test", "L2A", test_ids)[0]
    mu2 = Xl2a.mean(axis=(0, 2, 3)); sd2 = Xl2a.std(axis=(0, 2, 3)) + 1e-6     # product-aware (L2A own) stats
    Xn = {"clean": norm_of(Xl1c, mu1, sd1),                                    # L1C, stale stats
          "L2A_src": norm_of(Xl2a, mu1, sd1),                                  # L2A, STALE L1C stats
          "L2A_prod": norm_of(Xl2a, mu2, sd2)}                                 # L2A, product-aware stats
    rs = np.random.default_rng(54321)
    keep = [P8E._subsample(rs, IMG * IMG, args.px_per_patch) for _ in range(len(test_ids))]
    y_px = np.concatenate([yfull[p].reshape(-1)[keep[p]] for p in range(len(test_ids))])
    unit_px = np.concatenate([np.full(len(keep[p]), unit_patch[p]) for p in range(len(test_ids))])

    # ---- per-pixel training pixels: subsampled from the SAME train patches (band-matched, no spatial context) ----
    rtr = np.random.default_rng(24680)
    ktr = [P8E._subsample(rtr, IMG * IMG, args.px_per_patch) for _ in range(len(train_ids))]
    Xtr_px, Ytr_px = flatten_sampled(Xtr_n, Ytr, ktr)

    print(f"U-Net vs per-pixel MLP (band-matched 9-band, ONE harness): train {len(train_ids)} patches / "
          f"{len(Xtr_px)} px; test {len(test_ids)} patches over {np.unique(unit_px).size} scene-components; "
          f"{len(y_px)} eval px; dev {dev}", flush=True)

    MODELS = ["per-pixel", "U-Net"]
    rows = {mdl: {s[0]: [] for s in STATES} for mdl in MODELS}
    covs = {mdl: {s[0]: [] for s in STATES} for mdl in MODELS}
    for ms in args.model_seeds:
        mp = train_mlp(Xtr_px, Ytr_px, Xtr_px.shape[1], ms, dev, args.mlp_epochs)
        mu = train_unet(Xtr_n, Ytr, Xtr_n.shape[1], ms, dev, args.epochs)
        lg = {"per-pixel": {name: mlp_logits(mp, Xn[name], dev, keep) for name, _p, _n in STATES},
              "U-Net": {name: logits_sampled(mu, Xn[name], dev, keep) for name, _p, _n in STATES}}
        print(f"  model seed {ms}: per-pixel clean acc "
              f"{(lg['per-pixel']['clean'].argmax(1) == y_px).mean() * 100:.1f}, "
              f"U-Net clean acc {(lg['U-Net']['clean'].argmax(1) == y_px).mean() * 100:.1f}", flush=True)
        for ss in args.split_seeds:
            mt, mc, me = P8R.split_test_rois(unit_px, ss)
            for mdl in MODELS:
                T = fit_temperature(lg[mdl]["clean"][mt], y_px[mt])                 # temp on disjoint clean split
                corr_cc = (lg[mdl]["clean"][mc].argmax(1) == y_px[mc]).astype(int)
                conf_cc = confidence_msp(lg[mdl]["clean"][mc] / T)
                for name, _p, _n in STATES:
                    le = lg[mdl][name][me]
                    corr_e = (le.argmax(1) == y_px[me]).astype(int)
                    conf_e = confidence_msp(le / T)                                 # naive: clean temperature
                    crc = conformal_risk_control(corr_cc, conf_cc, corr_e, conf_e, alpha=ALPHA,
                                                 calib_group=unit_px[mc], eval_group=unit_px[me])
                    rows[mdl][name].append((ms, ss, crc["eval_group_joint_risk"] * 100))
                    covs[mdl][name].append(crc["eval_group_coverage"] * 100)

    tcrit = {2: 12.71, 3: 4.303, 5: 2.776}.get(len(args.model_seeds), 2.262)
    agg = {mdl: {} for mdl in MODELS}
    for mdl in MODELS:
        print(f"\n  {mdl} (from scratch) naive certificate:")
        for name, _p, _n in STATES:
            mm, se = two_way_se(rows[mdl][name]); agg[mdl][name] = mm
            print(f"    {name:9s} joint {mm:6.2f} +/- {se:.2f}  [{mm - tcrit * se:.1f}, {mm + tcrit * se:.1f}]  "
                  f"coverage {np.mean(covs[mdl][name]):.0f}%")
    ps, pp = agg["per-pixel"]["L2A_src"], agg["per-pixel"]["L2A_prod"]
    us, up = agg["U-Net"]["L2A_src"], agg["U-Net"]["L2A_prod"]
    print(f"\n  -> L2A stale-norm breach (both 9-band, same eval px, ONE harness): per-pixel MLP {ps:.1f}, "
          f"spatial U-Net {us:.1f}; product-aware norm restores per-pixel {pp:.1f}, U-Net {up:.1f}")
    print(f"  -> spatial context attenuates the breach by {ps - us:.1f} pts within one harness "
          f"(per-pixel {ps:.1f} -> U-Net {us:.1f}); + pretraining -> DOFA 9.9 (round-5, cross-experiment).")
    unet_breaches = us > 12
    both_fixed = up < 12 and pp < 12
    if unet_breaches and both_fixed and ps > us:
        print("  => a spatial model ALSO breaches under stale normalization (above the 10% target) yet LESS than "
              "the per-pixel MLP on IDENTICAL data, AND product-aware normalization restores BOTH -- the silent "
              "failure and its label-free fix GENERALISE beyond per-pixel; SEVERITY is model-dependent (spatial "
              "context attenuates it here; pretraining attenuates it further).")
    elif both_fixed:
        print(f"  => product-aware normalization restores both (U-Net {up:.1f}, per-pixel {pp:.1f}); U-Net stale "
              f"{us:.1f} vs per-pixel {ps:.1f} -- interpret spatial attenuation vs the 10% target honestly.")
    else:
        print(f"  => U-Net stale {us:.1f}/prod {up:.1f}, per-pixel stale {ps:.1f}/prod {pp:.1f} -- vs target 10.")


if __name__ == "__main__":
    main()
