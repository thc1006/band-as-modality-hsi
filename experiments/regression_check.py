#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-consistency smoke test for two shared-module fixes (model.py, metrics.py).

HONEST SCOPE: this does NOT diff against saved golden outputs from a fixed checkpoint, so it
is NOT a true regression/golden test and does NOT by itself prove that prior published numbers
stand (a real golden test would compare stored logits/metrics at a stated tolerance). BOTH
branches below call the CURRENT model internals, so a regression inside encode()/classifier()/
confusion() would be invisible here. What it DOES check is INTERNAL SELF-CONSISTENCY — that the
two fixes reduce to the simpler pre-fix expression in the regime the experiments actually use:

  (1) GroupedCrossBandAttention.__call__ (full forward, incl. the new fully-missing-row guard)
      agrees to < 1e-6 with a manual reconstruction from the SAME encode()+pool+classifier()
      WITHOUT the guard, for any row with >=1 present group. I.e. the guard is a no-op whenever
      >=1 group is present (fully-missing rows never occur at max_missing < G).
  (2) average_accuracy agrees to < 1e-9 with a manual mean-per-class-recall built from the SAME
      confusion() when every class is present in y_true (16/16 in the disjoint split).

A failure flags a self-inconsistency introduced by a future edit; a pass does NOT certify the
published numbers — use a checkpoint-based golden test for that.

Run: python experiments/regression_check.py
"""
import os, sys, numpy as np
import torch; torch.set_num_threads(8)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.io import load_mat_cube, disjoint_block_split, AVIRIS_WL_NM
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.metrics import average_accuracy, confusion
from bandsim.model import GroupedCrossBandAttention
import phase2_degradation as P2

def reconstructed_forward(model, x, present_mask):
    """The full forward path rebuilt from the SAME current internals but WITHOUT the fully-missing
    -row guard: encode(present_mask) -> masked mean-pool -> classifier. Comparing model(x, pm)
    against this checks the guard is a NO-OP when >=1 group is present. NOTE: it reuses today's
    encode()/classifier(), so it is a self-consistency reference, NOT a pre-fix/golden output."""
    h = model.encode(x, present_mask, use_padding_mask=True)
    w = present_mask.float()[:, :, None]
    pooled = (h * w).sum(1) / w.sum(1).clamp(min=1e-6)
    return model.classifier(pooled)


def manual_aa(yt, yp, K):
    """Mean per-class recall from the SAME current confusion(); average_accuracy should reduce to
    this when every class is present. Self-consistency reference, NOT a golden value."""
    m = confusion(yt, yp, K)
    pc = np.divide(np.diag(m), m.sum(1), out=np.zeros(K), where=m.sum(1) > 0)
    return float(np.mean(pc)) * 100


def main():
    ok = True

    # ---- (1) missing-row guard is a no-op for rows with >=1 present group ---------------
    groups = contiguous_groups(200, 10)
    # The RUNTIME axis, not a fabricated gapless one. Building the wavelength PE from
    # linspace(400,2500,200) here would regression-check the model against an axis production never
    # uses -- and a gapless 200-point stand-in for the real gapped AVIRIS axis is exactly the defect
    # that silently mis-assigned every 6S transmittance value until it was caught.
    cwl = group_center_wavelengths(np.asarray(AVIRIS_WL_NM, float), groups)
    torch.manual_seed(0)
    model = GroupedCrossBandAttention(groups, cwl, 16); model.eval()

    rng = np.random.default_rng(0)
    maxdiff = 0.0
    with torch.no_grad():
        for _ in range(200):
            x = torch.randn(16, 200)
            m = int(rng.integers(0, 7))                       # 0..6 missing (experiment regime)
            pm = np.ones((16, 10), bool)
            for r in range(16):
                drop = rng.choice(10, size=m, replace=False)
                pm[r, drop] = False                           # each row keeps >=4 groups
            pm_t = torch.from_numpy(pm)
            new = model(x, pm_t); rec = reconstructed_forward(model, x, pm_t)
            maxdiff = max(maxdiff, (new - rec).abs().max().item())
    print(f"(1) forward vs guard-free reconstruction, 200 masked batches (0..6 missing): "
          f"max|diff|={maxdiff:.2e}")
    r1 = maxdiff < 1e-6
    msg1 = ("PASS self-consistent (<1e-6): missing-row guard is a no-op when >=1 group present"
            if r1 else "FAIL self-inconsistency — the guard changed in-regime outputs!")
    print(f"    => {msg1}")
    ok &= r1

    # ---- (2) average_accuracy == mean-per-class-recall when all classes present ---------
    cube = load_mat_cube(os.path.join(os.path.dirname(_HERE), "data/indian_pines/Indian_pines_corrected.mat"),
                         key="indian_pines_corrected").astype(float)
    gt = load_mat_cube(os.path.join(os.path.dirname(_HERE), "data/indian_pines/Indian_pines_gt.mat"),
                       key="indian_pines_gt").astype(int)
    # guard=0 (default) DELIBERATELY: this identity test needs an all-16-classes-present split.
    # The paper's phase2 pipeline uses guard=1 (leakage-hardened), which drops the scene's
    # smallest class from test — that would make the "all classes present" reduction untestable.
    tr, te = disjoint_block_split(gt, block=10)
    yte = gt[te].astype(int) - 1
    rng = np.random.default_rng(1)
    maxaa = 0.0
    for _ in range(50):
        yp = np.where(rng.uniform(size=yte.size) < 0.8, yte, rng.integers(0, 16, yte.size))  # ~80% correct
        maxaa = max(maxaa, abs(average_accuracy(yte, yp, 16) - manual_aa(yte, yp, 16)))
    allpresent = len(np.unique(yte)) == 16
    print(f"(2) average_accuracy vs mean-per-class-recall on real test labels "
          f"(all 16 classes present={allpresent}): max|diff|={maxaa:.2e}")
    r2 = (maxaa < 1e-9) and allpresent
    msg2 = ("PASS self-consistent (<1e-9) with all 16 classes present"
            if r2 else "FAIL self-inconsistency or a class missing from y_true!")
    print(f"    => {msg2}")
    ok &= r2

    print("\n" + "=" * 60)
    verdict = ("both fixes reduce to the pre-fix expression in the experiment regime "
               "(self-consistency only — NOT a golden/regression proof of published numbers)"
               if ok else "SELF-INCONSISTENCY DETECTED")
    print(f"SELF-CONSISTENCY: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
