#!/usr/bin/env python
"""E6 (round-5 review 3.8): CLEAN band-set isolation for the DOFA robustness question. Per-pixel, full-
resolution (load_split -- NO 509->224 resize confound), a from-scratch MLP trained on three band sets:
(a) all 13 L1C bands (the flagship input), (b) 13 minus the water-vapour band B9 (the single most volatile
band under L1C->L2A), (c) DOFA's exact 9 bands. Each is scored for the naive L2A joint breach with the
identical CRC. If the breach falls toward the 10% target as B9 / the volatile bands are removed, the BAND
SET drives DOFA's robustness (which the 9-vs-13 comparison confounds); if it stays high, the band set is
not the driver and DOFA's escape is spatial/pretraining. This is the intervention the reviewer asks for."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8_cloudsen12 as P8
import phase8R_reliability as P8R
from phase8R5_secondbench import crc_thr, ce_joint
from phase8R_perclass_weighting_agg import two_way_se
from bandsim.reliability import fit_temperature
from bandsim import hw

NSEED, NSPLIT = 5, 5
TCRIT = 2.776   # Student-t 0.975 at df = min(NSEED,NSPLIT)-1 = 4. (Was wrongly 2.262 = t9, which only holds
                # for a 10x10 design; this ablation is smaller, so its interval must use the smaller df.)
L1C = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
DOFA9 = ["B4", "B3", "B2", "B5", "B6", "B7", "B8", "B11", "B12"]


class MLP(nn.Module):
    def __init__(self, cin, w=256):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(cin, w), nn.ReLU(inplace=True),
                               nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, 4))

    def forward(self, x):
        return self.n(x)


def train(X, y, cin, seed, dev, epochs=30, bs=8192):
    torch.manual_seed(seed)
    m = MLP(cin).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    lf = nn.CrossEntropyLoss()
    N = len(X)
    for _ in range(epochs):
        pm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = pm[i:i + bs]
            xb = torch.from_numpy(X[idx]).float().to(dev)
            yb = torch.from_numpy(y[idx]).long().to(dev)
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
    return m.eval()


@torch.no_grad()
def logits(m, X, dev, bs=65536):
    return np.concatenate([m(torch.from_numpy(X[i:i + bs]).float().to(dev)).cpu().numpy()
                           for i in range(0, len(X), bs)])


def main():
    hw.setup(deterministic=True, prefer="auto")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {"13-band(all)": list(range(13)),
            "-B9(12-band)": [i for i, b in enumerate(L1C) if b != "B9"],
            "9-band(DOFA)": [L1C.index(b) for b in DOFA9]}
    Xtr, ytr = P8.load_split("train", "L1C", pixels_per_patch=300, n_patches=3000, seed=12345)
    Xc, y, pid = P8.load_split("test", "L1C", pixels_per_patch=400, seed=54321, return_patch_id=True)
    Xl, _ = P8.load_split("test", "L2A", pixels_per_patch=400, seed=54321)   # L2A pid not needed (== L1C's)
    comp = np.unique(P8R.scene_component_ids("test")[pid], return_inverse=True)[1].astype(np.int32)
    ytr, y = ytr.astype(np.int64), y.astype(np.int64)
    print(f"loaded train {len(Xtr)} / test {len(Xc)} px, {len(np.unique(comp))} components; dev {dev}", flush=True)

    for name, ix in sets.items():
        mu, sd = Xtr[:, ix].mean(0), Xtr[:, ix].std(0) + 1e-8
        nrm = lambda A: ((A[:, ix] - mu) / sd).astype(np.float32)
        Xtr_n, Xc_n, Xl_n = nrm(Xtr), nrm(Xc), nrm(Xl)
        rn, ca, la = [], 0.0, 0.0
        for seed in range(NSEED):
            m = train(Xtr_n, ytr, len(ix), seed, dev)
            lc, ll = logits(m, Xc_n, dev), logits(m, Xl_n, dev)
            ca += (lc.argmax(1) == y).mean() * 100 / NSEED
            la += (ll.argmax(1) == y).mean() * 100 / NSEED
            for ss in range(NSPLIT):
                mt, mc, me = P8R.split_test_rois(comp, ss)
                Tc = fit_temperature(lc[mt], y[mt])
                rn.append((seed, ss, ce_joint(ll, Tc, y, me, comp, crc_thr(lc, Tc, y, mc, comp))))
        mn, sen = two_way_se(rn)
        print(f"  {name:14s} ({len(ix):2d}b) clean acc {ca:4.1f} L2A acc {la:4.1f}  ->  naive L2A joint "
              f"{mn:5.2f} +/- {sen:.2f}  [{mn - TCRIT * sen:.1f},{mn + TCRIT * sen:.1f}]  "
              f"{'BREACH' if mn - TCRIT * sen > 10 else 'controlled'}", flush=True)
    print("\n  reference: proposed 13-band flagship 28.9;  DOFA 9-band 9.95")
    print("  -> breach falls toward 10 as B9/volatile bands go => the BAND SET drives DOFA's robustness")
    print("     (so a fair model-class comparison must control it); breach stays high => it does not.")


if __name__ == "__main__":
    main()
