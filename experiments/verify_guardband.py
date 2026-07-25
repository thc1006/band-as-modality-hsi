#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial verification of the seam-leakage concern (H1c).

A plain block-checkerboard split still puts train & test pixels adjacent along block seams
(~34% of test pixels touch a train edge). This script re-runs the Phase-2 degradation
comparison with a GUARD BAND (guard=2: drop test pixels within 2 px of any train pixel),
which removes that adjacency, and checks whether the qualitative conclusion — Proposed has
the highest AUDC — SURVIVES under the 2 model seeds and single fixed block split evaluated
here. A surviving ranking is EVIDENCE AGAINST seam adjacency driving the result, not a proof
that rules it out (only 2 seeds + one spatial split).

Run: python experiments/verify_guardband.py
"""
import os, sys, numpy as np
from scipy.ndimage import binary_dilation
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch; torch.set_num_threads(8)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.io import load_mat_cube, disjoint_block_split, AVIRIS_WL_NM
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim.metrics import audc
from bandsim.model import GroupedCrossBandAttention
import phase2_degradation as P2

DATA = os.path.join(os.path.dirname(_HERE), "data", "indian_pines")
cube = load_mat_cube(os.path.join(DATA, "Indian_pines_corrected.mat"), key="indian_pines_corrected").astype(float)
gt = load_mat_cube(os.path.join(DATA, "Indian_pines_gt.mat"), key="indian_pines_gt").astype(int)


def seam_leak(te, tr):
    # 8-neighbour adjacency of the TRAIN mask with a ZERO-PADDED border (scipy pads with
    # border_value=0), so opposite image edges are NOT treated as neighbours. The old np.roll
    # used a PERIODIC boundary and falsely reported the top row adjacent to the bottom row (and
    # the left column to the right column). te and tr are disjoint, so dilating tr (including its
    # own cells) is safe: a test pixel is counted iff one of its 8 neighbours is a train pixel.
    adj = binary_dilation(tr, structure=np.ones((3, 3), bool))
    return int((te & adj).sum()), int(te.sum())


def _assert_non_wrapping():
    """RIGOR: the fixed adjacency must NOT mark opposite edges adjacent (np.roll did)."""
    tr_top = np.zeros((3, 3), bool); tr_top[0, 1] = True          # train pixel on the TOP edge
    te_bottom = np.zeros((3, 3), bool); te_bottom[2, 1] = True    # test pixel on the BOTTOM edge
    assert seam_leak(te_bottom, tr_top)[0] == 0, "adjacency wraps top<->bottom (periodic-boundary leak)"
    tr_right = np.zeros((3, 3), bool); tr_right[1, 2] = True      # train pixel on the RIGHT edge
    te_left = np.zeros((3, 3), bool); te_left[1, 0] = True        # test pixel on the LEFT edge
    assert seam_leak(te_left, tr_right)[0] == 0, "adjacency wraps left<->right (periodic-boundary leak)"
    te_adj = np.zeros((3, 3), bool); te_adj[1, 1] = True          # a genuine 8-neighbour IS detected
    assert seam_leak(te_adj, tr_top)[0] == 1, "genuine adjacency missed"


def prep_guarded(guard):
    tr, te = disjoint_block_split(gt, block=10, guard=guard)
    Xtr = cube[tr]; ytr = gt[tr].astype(int) - 1
    Xte = cube[te]; yte = gt[te].astype(int) - 1
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    return ((Xtr - mu) / sd).astype(np.float32), ytr, ((Xte - mu) / sd).astype(np.float32), yte, tr, te


def run(guard, seeds=(0, 1), epochs=50, max_missing=6, trials=8):
    groups = contiguous_groups(200, 10); cwl = group_center_wavelengths(AVIRIS_WL_NM, groups)
    xs = np.arange(0, max_missing + 1)
    audcs = {"b1": [], "b2": [], "proposed": []}
    for sd in seeds:
        Xtr, ytr, Xte, yte, tr, te = prep_guarded(guard)
        m_b1 = P2.train_mlp(Xtr, ytr, groups, sd, group_dropout=False, epochs=epochs)
        m_b2 = P2.train_mlp(Xtr, ytr, groups, sd, group_dropout=True, epochs=epochs)
        m_p = GroupedCrossBandAttention(groups, cwl, 16)
        P2.pretrain_sgmae(m_p, Xtr, groups, sd, epochs=epochs // 2)
        P2.finetune_proposed(m_p, Xtr, ytr, groups, sd, epochs=epochs)
        # need Xte/yte inside degradation_curve; monkey-patch via closure
        for name, model, kind in [("b1", m_b1, "b1"), ("b2", m_b2, "b2"), ("proposed", m_p, "proposed")]:
            cur = P2.degradation_curve(kind, model, Xte, yte, groups, AVIRIS_WL_NM,
                                       max_missing, trials, np.random.default_rng(sd + 999))
            audcs[name].append(audc(xs, cur))
    return {k: float(np.mean(v)) for k, v in audcs.items()}


# --- report seam leakage before/after guard ---
_assert_non_wrapping()   # fail fast if the adjacency ever regresses to a wrap-around
tr0, te0 = disjoint_block_split(gt, block=10, guard=0)
tr2, te2 = disjoint_block_split(gt, block=10, guard=2)
s0, n0 = seam_leak(te0, tr0); s2, n2 = seam_leak(te2, tr2)
print(f"guard=0: {s0}/{n0} test px touch train edge ({s0/n0*100:.1f}%)")
print(f"guard=2: {s2}/{n2} test px touch train edge ({s2/n2*100:.1f}%)  [test set {n2} px]")

SEEDS = (0, 1)
print(f"\nRe-running degradation with GUARD BAND ({len(SEEDS)} seeds, epochs=50)...")
a2 = run(guard=2, seeds=SEEDS)
print(f"guarded AUDC: b1={a2['b1']:.1f}  b2={a2['b2']:.1f}  proposed={a2['proposed']:.1f}")
best = max(a2, key=a2.get); held = best == "proposed"
print(f"\n{'ranking HELD' if held else 'ranking FLIPPED'}: under this guard-band split, highest AUDC = {best}")
if held:
    print(f"=> under these {len(SEEDS)} model seeds and this single fixed block split, the guard band did")
    print("   NOT flip the ranking. Evidence AGAINST seam adjacency driving the result here — a limited")
    print("   check (2 seeds, one spatial split), NOT a general causal proof.")
else:
    print("=> ranking changes under the guard band — investigate seam adjacency as a cause.")
