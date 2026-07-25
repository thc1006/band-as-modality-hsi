"""Design C (thin cirrus) physics-sanity unit tests (roadmap docs/guide/03 §3). pytest tests/ -v

Guards: tau=0 is the identity; SWIR reflectance attenuates monotonically as tau grows;
the extinction shape has a bump near the 1.375 um cirrus band.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandsim.cirrus import apply_cirrus, extinction_shape
from bandsim.io import AVIRIS_WL_NM


def test_tau_zero_is_identity():
    wl = AVIRIS_WL_NM
    cube = np.random.default_rng(0).random((3, 4, wl.size))
    out = apply_cirrus(cube, wl, tau=0.0)
    assert np.array_equal(out, cube)          # tau=0 -> unchanged


def test_swir_attenuation_increases_with_tau():
    wl = AVIRIS_WL_NM
    cube = np.ones((8, wl.size))
    swir = int(np.argmin(np.abs(wl - 2200)))   # SWIR band, no additive VNIR path here
    vals = [apply_cirrus(cube, wl, tau)[:, swir].mean() for tau in [0.0, 0.1, 0.3, 0.6, 1.0]]
    # Beer-Lambert exp(-tau*k) -> reflectance strictly decreasing in tau
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    assert vals[-1] < vals[0]


def test_extinction_bump_near_cirrus_band():
    wl = AVIRIS_WL_NM
    k = extinction_shape(wl)
    b1375 = int(np.argmin(np.abs(wl - 1375)))
    b600 = int(np.argmin(np.abs(wl - 600)))
    assert k[b1375] > k[b600]                  # extinction rises near 1.375 um


def test_vnir_gets_additive_haze():
    wl = AVIRIS_WL_NM
    cube = np.zeros((5, wl.size))              # zero surface -> only the additive path shows
    out = apply_cirrus(cube, wl, tau=1.0)
    vnir = wl < 1000
    assert out[:, vnir].mean() > 0             # VNIR hazed/brightened
    assert np.allclose(out[:, ~vnir], 0.0)     # no additive path outside VNIR (zero surface)
