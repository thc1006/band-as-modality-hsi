"""Unit tests for Design A (SRF). These guard PHYSICAL sanity — run: pytest tests/ -v

They are fully runnable with only numpy installed.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandsim.srf import (
    gaussian_srf, build_resampling_matrix, apply_srf,
    SENTINEL2_MSI_CENTERS_NM, LANDSAT8_OLI_CENTERS_NM,
)


def test_resampling_rows_sum_to_one():
    wl = np.linspace(400, 2500, 200)
    srf = gaussian_srf(wl, [490, 665, 842, 1610], fwhm_nm=30.0)
    W, names = build_resampling_matrix(wl, srf)
    assert W.shape == (4, 200)
    assert np.allclose(W.sum(axis=1), 1.0), "each SRF row must integrate to 1"


def test_apply_srf_shapes():
    wl = np.linspace(400, 2500, 200)
    srf = gaussian_srf(wl, [490, 665, 842], fwhm_nm=30.0)
    W, _ = build_resampling_matrix(wl, srf)
    cube = np.random.default_rng(0).random((5, 6, 200))
    out = apply_srf(cube, W)
    assert out.shape == (5, 6, 3)
    # pixel form
    assert apply_srf(cube.reshape(-1, 200), W).shape == (30, 3)


def test_constant_spectrum_is_preserved():
    # A flat spectrum of value v -> every synthesized band should equal v (row-sum=1).
    wl = np.linspace(400, 2500, 200)
    srf = gaussian_srf(wl, [490, 665, 1610], fwhm_nm=40.0)
    W, _ = build_resampling_matrix(wl, srf)
    flat = np.full((10, 200), 0.37)
    out = apply_srf(flat, W)
    assert np.allclose(out, 0.37, atol=1e-6), "flat spectrum must map to same value"


def test_oli_has_fewer_bands_than_s2():
    # Physical premise of Design A: OLI lacks S2 red-edge / narrow water-vapour bands.
    assert len(LANDSAT8_OLI_CENTERS_NM) < len(SENTINEL2_MSI_CENTERS_NM)
    for redge in ("B5", "B6", "B7"):  # S2 red-edge centres absent from OLI set
        assert SENTINEL2_MSI_CENTERS_NM[redge] not in LANDSAT8_OLI_CENTERS_NM.values()


def test_band_center_lands_in_expected_band():
    # A near-delta spectrum at 665 nm should excite the band whose SRF centres on 665.
    wl = np.linspace(400, 2500, 200)
    srf = gaussian_srf(wl, [490, 665, 842], fwhm_nm=20.0)
    W, names = build_resampling_matrix(wl, srf)
    spec = np.exp(-0.5 * ((wl - 665) / 5) ** 2).reshape(1, -1)
    out = apply_srf(spec, W).ravel()
    assert names[int(np.argmax(out))] == "B2", "665 nm energy should peak in the 665 nm band"


def test_pyspectral_real_srf_builds_valid_bands():
    # real measured RSR (pyspectral) -> row-normalised resampling matrix; thermal bands dropped.
    import pytest
    try:
        from bandsim.srf import pyspectral_srf
        srf = pyspectral_srf(np.linspace(400, 2500, 200), "Landsat-8", "oli")
    except Exception as e:
        pytest.skip(f"pyspectral RSR unavailable: {e}")
    W, names = build_resampling_matrix(np.linspace(400, 2500, 200), srf)
    assert 5 <= len(names) <= 11                      # reflective OLI bands (thermal dropped)
    assert np.allclose(W.sum(axis=1), 1.0)            # each SRF integrates to 1


# ===================== wavelength-axis ordering guard (Design A) =====================
import pytest  # noqa: E402
from bandsim.srf import _trapezoid_weights  # noqa: E402


def test_descending_axis_is_rejected_not_absolute_valued():
    """A reversed axis is a data-ordering bug and must raise, not be rescued by abs().

    The quadrature weights used to be returned as np.abs(q), so a descending axis produced
    positive-looking weights and a plausible WRONG number instead of an error: with the cube left in
    its original order, a 1000 nm band's value moved from 0.2857 to 0.7143 -- a 2.5x error, silently.
    """
    wl = np.linspace(400.0, 2500.0, 200)
    with pytest.raises(ValueError, match="strictly increasing"):
        _trapezoid_weights(wl[::-1].copy())
    # and through the public entry point, which is where a caller would actually hit it
    srf = gaussian_srf(wl, [1000.0], fwhm_nm=30.0, names=["X"])
    with pytest.raises(ValueError, match="strictly increasing"):
        build_resampling_matrix(wl[::-1].copy(), {"X": srf["X"][::-1].copy()})


def test_non_monotonic_and_duplicate_axis_are_rejected():
    for bad in ([400.0, 700.0, 500.0, 900.0],        # shuffled
                [400.0, 500.0, 500.0, 900.0]):       # a repeated wavelength -> zero-width interval
        with pytest.raises(ValueError, match="strictly increasing"):
            _trapezoid_weights(np.array(bad))


def test_non_finite_or_degenerate_axis_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        _trapezoid_weights(np.array([400.0, np.nan, 600.0]))
    with pytest.raises(ValueError, match="non-finite"):
        _trapezoid_weights(np.array([400.0, np.inf, 600.0]))
    with pytest.raises(ValueError, match=">= 2 samples"):
        _trapezoid_weights(np.array([500.0]))          # one sample cannot define dlambda
    with pytest.raises(ValueError, match="1-D"):
        _trapezoid_weights(np.zeros((3, 4)))


def test_ascending_axis_still_produces_positive_weights():
    # the guard must not change the physics on a legitimate (gapped, non-uniform) axis
    from bandsim.io import AVIRIS_WL_NM
    q = _trapezoid_weights(np.asarray(AVIRIS_WL_NM, float))
    assert (q > 0).all() and np.isfinite(q).all()
    assert np.isclose(q.sum(), AVIRIS_WL_NM[-1] - AVIRIS_WL_NM[0], rtol=1e-9)   # weights tile the span


def test_pyspectral_detector_policy_is_explicit():
    from bandsim.srf import pyspectral_srf
    # the policy name itself is validated, so a typo cannot fall through to an undeclared default
    with pytest.raises(ValueError, match="unknown detector_policy"):
        pyspectral_srf(np.linspace(400, 2500, 200), "Landsat-8", "oli", detector_policy="firts")
    try:
        srf, dets = pyspectral_srf(np.linspace(400, 2500, 200), "Landsat-8", "oli",
                                   return_detectors=True)
    except (ImportError, OSError, KeyError) as e:
        pytest.skip(f"pyspectral RSR unavailable: {e}")
    assert set(dets) == set(srf)                       # every returned band records its detector
    assert all(isinstance(d, str) and d for d in dets.values())
    # OLI publishes one detector per band, so 'first' is unambiguous and must equal 'sorted_first'
    alt = pyspectral_srf(np.linspace(400, 2500, 200), "Landsat-8", "oli",
                         detector_policy="sorted_first")
    assert set(alt) == set(srf)
    for b in srf:
        assert np.array_equal(srf[b], alt[b]), "detector policy changed a single-detector sensor"
