"""Regression guard for the vectorized group-dropout/HCS/SGMAE mask generation (speed fix).

The Phase-2 training loops replaced per-sample Python loops (rng.integers + rng.choice) with a
vectorized `_vec_group_subset`. These tests verify the vectorized version has the SAME
DISTRIBUTION as the old loop (uniform subset size, symmetric group selection), which is what
guarantees accuracy is statistically unchanged — the whole justification for the swap.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import phase2_degradation as P2  # noqa: E402


def test_subset_size_uniform_in_range():
    rng = np.random.default_rng(0)
    b, G, lo, hi = 40000, 7, 0, 4
    M = P2._vec_group_subset(rng, b, G, lo, hi)
    assert M.shape == (b, G) and M.dtype == bool
    sizes = M.sum(1)
    assert sizes.min() >= lo and sizes.max() < hi           # sizes strictly in [lo, hi)
    assert abs(sizes.mean() - 1.5) < 0.05                   # uniform {0,1,2,3} -> mean 1.5
    for k in range(lo, hi):                                 # each size ~ equally likely
        assert abs((sizes == k).mean() - 1.0 / (hi - lo)) < 0.02


def test_group_selection_symmetric():
    rng = np.random.default_rng(1)
    G = 7
    M = P2._vec_group_subset(rng, 40000, G, 1, 4)           # mask 1..3 groups (SGMAE)
    p = M.mean(0)                                           # per-group marginal selection prob
    assert p.std() < 0.01                                   # no group favoured (symmetry)
    assert abs(p.mean() - 2.0 / G) < 0.01                   # E[size]=2 (uniform 1..3) spread over G


def test_hcs_keeps_1_to_G():
    rng = np.random.default_rng(2)
    G = 7
    M = P2._vec_group_subset(rng, 40000, G, 1, G + 1)       # HCS: keep 1..G groups
    sizes = M.sum(1)
    assert sizes.min() >= 1 and sizes.max() <= G
    assert abs(sizes.mean() - (G + 1) / 2.0) < 0.1          # uniform 1..G -> mean (G+1)/2


def test_band_mask_matches_group_drop():
    """The (b,G) group-drop @ (G,C) membership must zero EXACTLY the dropped groups' bands."""
    from bandsim.grouping import build_group_matrix
    groups = [np.array(g) for g in ([0], [1, 2, 3], [4, 5, 6], [7, 8], [9], [10], [11, 12])]
    C = 13
    Mgb = build_group_matrix(C, groups).astype(np.float32)
    rng = np.random.default_rng(3)
    drop = P2._vec_group_subset(rng, 100, len(groups), 0, 4).astype(np.float32)
    bmask = 1.0 - drop @ Mgb
    assert set(np.unique(bmask)).issubset({0.0, 1.0})       # groups disjoint -> clean 0/1 mask
    # spot-check row 0: bands of dropped groups are 0, others 1
    for r in range(5):
        for gi, idx in enumerate(groups):
            expected = 0.0 if drop[r, gi] else 1.0
            assert np.all(bmask[r, idx] == expected)
