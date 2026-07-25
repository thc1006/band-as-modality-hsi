"""Tests for the native-sensor atmosphere primitives (Phase B physics engine).

sensor_band_transmittance integrates a fine-grid 6S transmittance onto sensor bands via their
SRF, so a 6S atmosphere can degrade an already-multispectral product (Sentinel-2 / CloudSEN12)
directly. Verifies boundedness, the flat-T identity, and that an absorption dip suppresses the
overlapping band while leaving a window band intact.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bandsim.srf import gaussian_srf
from bandsim.atmosphere import sensor_band_transmittance, apply_sensor_atmosphere


def test_flat_transmittance_maps_to_itself():
    wl = np.linspace(400, 2500, 200)
    srf = gaussian_srf(wl, [490, 842, 1610], fwhm_nm=30.0)
    Tb, names = sensor_band_transmittance(np.full(200, 0.9), wl, srf)
    assert len(names) == 3
    assert np.allclose(Tb, 0.9, atol=1e-6)          # each row sums to 1 -> convex avg of 0.9


def test_transmittance_stays_bounded():
    wl = np.linspace(400, 2500, 200)
    srf = gaussian_srf(wl, [490, 945, 2190], fwhm_nm=25.0)
    rng = np.random.default_rng(0)
    Tb, _ = sensor_band_transmittance(rng.random(200), wl, srf)
    assert np.all(Tb >= 0.0) and np.all(Tb <= 1.0)


def test_absorption_dip_suppresses_overlapping_band_only():
    wl = np.linspace(400, 2500, 200)
    srf = gaussian_srf(wl, [490, 945], fwhm_nm=20.0)  # window band vs water-vapour band
    T = np.ones(200)
    T[np.abs(wl - 945) < 40] = 0.1                     # carve a dip at 945 nm
    Tb, _ = sensor_band_transmittance(T, wl, srf)
    assert Tb[0] > 0.9, "490 nm window band should be ~unaffected"
    assert Tb[1] < 0.6, "945 nm band overlapping the dip should be suppressed"


def test_shape_mismatch_raises():
    wl = np.linspace(400, 2500, 200)
    srf = gaussian_srf(wl, [490], fwhm_nm=20.0)
    try:
        sensor_band_transmittance(np.ones(199), wl, srf)   # wrong length
    except ValueError:
        return
    raise AssertionError("expected ValueError on transmittance/grid length mismatch")


def test_apply_sensor_atmosphere():
    r = np.ones((10, 3))
    out = apply_sensor_atmosphere(r, np.array([0.9, 0.8, 0.7]))
    assert np.allclose(out, [0.9, 0.8, 0.7])
    try:
        apply_sensor_atmosphere(np.ones((5, 4)), np.array([0.9, 0.8, 0.7]))  # band mismatch
    except ValueError:
        return
    raise AssertionError("expected ValueError on reflectance/band mismatch")
