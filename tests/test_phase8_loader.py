"""Adversarial regression tests for phase8 load_split's patch-level selection (Phase C enabler).

Guards the correctness of the `patch_ids` / `return_patch_id` additions used to build DISJOINT
calibration/evaluation patch sets for conformal (patch-level, not pixel-level, split). Data-gated:
skips cleanly when the CloudSEN12 files are absent (so CI without data still passes).
"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "experiments"))
sys.path.insert(0, os.path.join(_HERE, ".."))
import phase8_cloudsen12 as P8  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(P8.DATA, "test", "L1C_B2.dat")),
    reason="CloudSEN12 data not present")


def test_patch_ids_selects_exactly_those_patches():
    ids = np.array([0, 5, 10])
    X, y, pid = P8.load_split("test", "L1C", pixels_per_patch=50, patch_ids=ids, return_patch_id=True)
    assert set(np.unique(pid).tolist()) == set(ids.tolist())    # exactly those patches, no others
    assert X.shape[0] == y.shape[0] == pid.shape[0]


def test_return_patch_id_backward_compat():
    out = P8.load_split("test", "L1C", pixels_per_patch=20, n_patches=3, seed=1)
    assert len(out) == 2                                        # (X, y) when return_patch_id=False


def test_l1c_l2a_pixel_correspondence_with_patch_ids():
    ids = np.array([1, 2, 3, 4])
    Xl, yl, pl = P8.load_split("test", "L1C", pixels_per_patch=40, patch_ids=ids, seed=7, return_patch_id=True)
    Xa, ya, pa = P8.load_split("test", "L2A", pixels_per_patch=40, patch_ids=ids, seed=7, return_patch_id=True)
    assert np.array_equal(yl, ya)                              # same pixels -> identical labels (1:1)
    assert np.array_equal(pl, pa)                              # same source patches
    assert np.abs(Xa[:, P8.B10_IDX]).max() == 0.0             # L2A has no B10


def test_calib_eval_patch_disjoint():
    calib, ev = np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])
    _, _, pc = P8.load_split("test", "L1C", pixels_per_patch=30, patch_ids=calib, return_patch_id=True)
    _, _, pe = P8.load_split("test", "L1C", pixels_per_patch=30, patch_ids=ev, return_patch_id=True)
    assert set(np.unique(pc).tolist()).isdisjoint(set(np.unique(pe).tolist()))  # no patch (⇒ no pixel) shared
