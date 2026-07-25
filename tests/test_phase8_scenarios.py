"""Regression guard for the Phase-8 CloudSEN12 band-exact missing-band scenarios.

Locks the fix for the original BLOCKER: with a naive 6-group split, "drop B10" silently
dropped B9 too (B9 and B10 shared one group). Phase 8 now groups along the physical S2 design
and ISOLATES B1/B9/B10 as singleton groups, guarded at runtime by `_assert_singleton`. These
tests fail loudly if that guarantee ever regresses.
"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(_HERE, ".."))

import phase8_cloudsen12 as P8          # noqa: E402
import phase2_degradation as P2         # noqa: E402
from bandsim.grouping import contiguous_groups  # noqa: E402


def test_atmospheric_bands_are_singletons():
    g = P8.s2_physical_groups()
    for idx, name in [(P8.B1_IDX, "B1"), (P8.B9_IDX, "B9"), (P8.B10_IDX, "B10")]:
        gid = P8._assert_singleton(g, idx, name)     # must not raise
        assert list(g[gid]) == [idx]


def test_dropB10_removes_exactly_band10():
    g = P8.s2_physical_groups()
    gid = P8._group_of_band(g, P8.B10_IDX)
    X = np.ones((5, 13), np.float32)
    Xc = P2.zero_missing(X, g, [gid])
    zeroed = np.where(~Xc.any(0))[0].tolist()        # columns that became all-zero
    assert zeroed == [P8.B10_IDX], f"drop B10 zeroed {zeroed}, expected [{P8.B10_IDX}] only"
    assert Xc[:, P8.B9_IDX].all(), "B9 must be untouched (the original B9+B10 blocker)"


def test_dropB1B9B10_removes_exactly_three_bands():
    g = P8.s2_physical_groups()
    ids = [P8._group_of_band(g, i) for i in (P8.B1_IDX, P8.B9_IDX, P8.B10_IDX)]
    X = np.ones((5, 13), np.float32)
    Xc = P2.zero_missing(X, g, ids)
    zeroed = sorted(np.where(~Xc.any(0))[0].tolist())
    assert zeroed == sorted([P8.B1_IDX, P8.B9_IDX, P8.B10_IDX]), \
        f"atmos drop zeroed {zeroed}, expected exactly B1,B9,B10 (not 5 bands)"


def test_present_mask_drops_only_target_token():
    g = P8.s2_physical_groups()
    gid = P8._group_of_band(g, P8.B10_IDX)
    pm = P2.group_present_mask(4, g, [gid])
    assert not pm[:, gid].any()                      # B10 token absent
    for other in range(len(g)):
        if other != gid:
            assert pm[:, other].all(), "only the B10 token may be dropped"


def test_naive_6group_split_would_fail_the_guard():
    """Documents the original bug: naive contiguous_groups(13,6) puts B9,B10 in one group, so the
    singleton guard MUST reject it — which is exactly why phase8 uses the physical grouping.

    Expects ValueError, not AssertionError: the guard was deliberately moved off `assert` because
    `python -O` strips assert statements, and a guard that disappears under an optimisation flag
    would let `dropB10` silently drop B9 alongside it again — the exact defect it exists to catch."""
    naive = contiguous_groups(13, 6)
    with pytest.raises(ValueError, match="SCENARIO GUARD FAILED"):
        P8._assert_singleton(naive, P8.B10_IDX, "B10")
