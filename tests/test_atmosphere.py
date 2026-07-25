"""Design B (6S atmosphere) physics-sanity unit tests (roadmap docs/guide/03 §3). pytest tests/ -v

Guards: transmittance in [0,1]; water-vapour bands' transmittance decreases monotonically
with column water vapour (CWV); hard-mask zeroes exactly the strong absorption cores.
"""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandsim.atmosphere import (hard_mask_absorption_cores, apply_atmosphere,
                                STRONG_ABSORPTION_CORES_NM)
from bandsim.io import AVIRIS_WL_NM

_TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "srf_cache", "T_6s_grid.npz")


def test_hard_mask_drops_exactly_the_absorption_cores():
    wl = AVIRIS_WL_NM
    keep = hard_mask_absorption_cores(wl)
    assert keep.dtype == bool and keep.shape == wl.shape
    for lo, hi in STRONG_ABSORPTION_CORES_NM:
        inside = (wl >= lo) & (wl <= hi)
        assert inside.any()                 # the core lies within the axis
        assert not keep[inside].any()       # every core band is dropped
    # bands well outside any core are kept
    assert keep[np.argmin(np.abs(wl - 550))]      # green is kept
    assert 0 < keep.sum() < wl.size


def test_apply_atmosphere_multiplies_and_masks():
    wl = AVIRIS_WL_NM
    cube = np.ones((4, 5, wl.size))
    T = np.full(wl.size, 0.7)
    out = apply_atmosphere(cube, T)
    assert np.allclose(out, 0.7)                    # rho' = rho * T
    keep = hard_mask_absorption_cores(wl)
    out2 = apply_atmosphere(cube, T, keep_mask=keep)
    assert np.allclose(out2[..., ~keep], 0.0)       # cores zeroed
    assert np.allclose(out2[..., keep], 0.7)        # others untouched


@pytest.mark.skipif(not os.path.exists(_TABLE), reason="6S table not precomputed")
def test_6s_transmittance_unit_range_and_cwv_monotonic():
    d = np.load(_TABLE)
    for k in d.files:
        if k.startswith("cwv"):
            T = d[k]
            assert (T >= 0).all() and (T <= 1.0 + 1e-6).all()   # physical transmittance
    # higher CWV -> more water-vapour absorption -> lower mean transmittance
    m05 = d["cwv0.5"].mean()     # table is keyed by CWV only (AOD is inert for gaseous T)
    m2 = d["cwv2.0"].mean()
    m4 = d["cwv4.0"].mean()
    assert m05 > m2 > m4
