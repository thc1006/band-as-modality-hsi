"""Wavelength-IDENTITY guard on the cached 6S transmittance table (regression, not a physics test).

The bug these guard against: data/srf_cache/T_6s_grid.npz was computed while bandsim.io.AVIRIS_WL_NM
was a gapless linspace(400,2500,200). AVIRIS_WL_NM was later corrected to the true GAPPED axis (220
nominal bands minus the 20 removed water/low-SNR bands). Both are length 200, so every length-based
check still passed while each transmittance value was applied to the WRONG channel -- 183/200 bands
displaced by >5 nm, worst 98.3 nm. A same-length-but-different axis MUST now be rejected.

Every fixture is built synthetically in tmp_path: these tests exercise the guard itself and must run
(and fail on a regression) whether or not the real 6S table is present on the machine.
"""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandsim.atmosphere import load_cached_transmittance
from bandsim.io import AVIRIS_WL_NM
from bandsim.pipeline import simulate
from bandsim.srf import gaussian_srf, SENTINEL2_MSI_CENTERS_NM

# The exact axis the broken table shipped with: same length, uniform spacing, no water-vapour gaps.
LEGACY_UNIFORM_WL = np.linspace(400.0, 2500.0, 200)


def _write_table(tmp_path, wl_nm, name="T.npz", key="cwv2.0", n=None):
    """Minimal stand-in for the 6S table: a physical T in [0,1] plus (optionally) its axis."""
    n = len(AVIRIS_WL_NM) if n is None else n
    T = 0.5 + 0.4 * np.cos(np.linspace(0, 6.0, n))          # smooth, strictly inside [0,1]
    payload = {key: T}
    if wl_nm is not None:
        payload["wl_nm"] = np.asarray(wl_nm, float)
    path = os.path.join(str(tmp_path), name)
    np.savez_compressed(path, **payload)
    return path, T


def test_correct_axis_loads(tmp_path):
    path, T = _write_table(tmp_path, AVIRIS_WL_NM)
    got = load_cached_transmittance(path, "cwv2.0", n_bands=len(AVIRIS_WL_NM),
                                    expected_wavelengths_nm=AVIRIS_WL_NM)
    assert np.array_equal(got, T)


def test_same_length_different_axis_raises(tmp_path):
    # THE original defect: uniform grid vs gapped AVIRIS axis, both 200 long.
    path, _ = _write_table(tmp_path, LEGACY_UNIFORM_WL)
    assert len(LEGACY_UNIFORM_WL) == len(AVIRIS_WL_NM)      # length check cannot catch this
    with pytest.raises(ValueError, match="wavelength-axis MISMATCH"):
        load_cached_transmittance(path, "cwv2.0", n_bands=len(AVIRIS_WL_NM),
                                  expected_wavelengths_nm=AVIRIS_WL_NM)


def test_mismatch_message_names_both_axes_and_max_delta(tmp_path):
    path, _ = _write_table(tmp_path, LEGACY_UNIFORM_WL)
    with pytest.raises(ValueError) as exc:
        load_cached_transmittance(path, "cwv2.0", expected_wavelengths_nm=AVIRIS_WL_NM)
    msg = str(exc.value)
    # Endpoints re-pinned at the 2026-07-20 axis correction (224-band base): the requested axis
    # now ends at 2490.5830 nm, and the worst displacement of the legacy uniform table against it
    # is 78.1756 nm.
    assert "2500.0000" in msg and "2490.5830" in msg        # table endpoint and requested endpoint
    assert "78.1756" in msg                                 # the real worst-band displacement (nm)


def test_reversed_axis_raises(tmp_path):
    # Same values, same length, same min/max -- only the ORDER differs, so any summary-statistic
    # check (count/range/mean) passes while every band is mapped to its mirror image.
    path, _ = _write_table(tmp_path, AVIRIS_WL_NM[::-1])
    with pytest.raises(ValueError, match="wavelength-axis MISMATCH"):
        load_cached_transmittance(path, "cwv2.0", n_bands=len(AVIRIS_WL_NM),
                                  expected_wavelengths_nm=AVIRIS_WL_NM)


def test_wrong_length_axis_raises(tmp_path):
    short = AVIRIS_WL_NM[:100]
    path, _ = _write_table(tmp_path, short, n=100)
    # existing behaviour: caught by the n_bands check
    with pytest.raises(ValueError, match="bands but the cube has"):
        load_cached_transmittance(path, "cwv2.0", n_bands=len(AVIRIS_WL_NM))
    # and by the axis check even when n_bands is not supplied
    with pytest.raises(ValueError, match="LENGTH mismatch"):
        load_cached_transmittance(path, "cwv2.0", expected_wavelengths_nm=AVIRIS_WL_NM)


def test_missing_wl_nm_raises(tmp_path):
    # An unlabelled table cannot be proven to sit on the requested axis -> refuse it.
    path, _ = _write_table(tmp_path, None)
    with pytest.raises(ValueError, match="no 'wl_nm' axis"):
        load_cached_transmittance(path, "cwv2.0", n_bands=len(AVIRIS_WL_NM),
                                  expected_wavelengths_nm=AVIRIS_WL_NM)
    # ...but a caller with genuinely no axis to assert still gets the legacy behaviour.
    assert load_cached_transmittance(path, "cwv2.0", n_bands=len(AVIRIS_WL_NM)).shape == (200,)


def test_tolerance_is_tight_not_nominal(tmp_path):
    # A sub-nanometre shift is still a different axis: the guard must not wave through "close enough".
    path, _ = _write_table(tmp_path, AVIRIS_WL_NM + 1e-3)
    with pytest.raises(ValueError, match="wavelength-axis MISMATCH"):
        load_cached_transmittance(path, "cwv2.0", expected_wavelengths_nm=AVIRIS_WL_NM)


def test_pipeline_threads_the_axis_into_the_guard(tmp_path):
    """The end-to-end regression: simulate() must reject a wrong-axis table, not silently apply it.

    Before the fix pipeline.simulate passed only n_bands, so this exact configuration ran to
    completion and returned a corrupted cube."""
    srf = gaussian_srf(AVIRIS_WL_NM, list(SENTINEL2_MSI_CENTERS_NM.values()), fwhm_nm=30.0)
    cube = np.ones((3, 3, len(AVIRIS_WL_NM)))

    bad, _ = _write_table(tmp_path, LEGACY_UNIFORM_WL, name="bad.npz")
    cfg_bad = {"seed": 0, "A": {"enable": True, "srf": srf},
               "B": {"enable": True, "hard_mask_cores": True,
                     "cache_npz": bad, "cache_key": "cwv2.0"}}
    with pytest.raises(ValueError, match="wavelength-axis MISMATCH"):
        simulate(cube, AVIRIS_WL_NM, cfg_bad)

    good, _ = _write_table(tmp_path, AVIRIS_WL_NM, name="good.npz")
    cfg_good = dict(cfg_bad, B=dict(cfg_bad["B"], cache_npz=good))
    out, _ = simulate(cube, AVIRIS_WL_NM, cfg_good)
    assert out.shape == (3, 3, 12) and np.isfinite(out).all()


def test_shipped_table_is_on_the_aviris_axis():
    """The repaired data/srf_cache/T_6s_grid.npz must load under the guard it now ships with.

    Skipped only when the table is genuinely absent (it is gitignored); a PRESENT table that fails
    the axis check is a hard failure, never a skip."""
    table = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "srf_cache", "T_6s_grid.npz")
    if not os.path.exists(table):
        pytest.skip("6S table not present on this machine")
    d = np.load(table, allow_pickle=True)
    assert "wl_nm" in d.files and "wl_sha256" in d.files
    from bandsim.io import axis_sha256
    assert str(d["wl_sha256"]) == axis_sha256(AVIRIS_WL_NM)
    for k in [k for k in d.files if k.startswith("cwv")]:
        T = load_cached_transmittance(table, k, n_bands=len(AVIRIS_WL_NM),
                                      expected_wavelengths_nm=AVIRIS_WL_NM)
        assert T.shape == (200,) and (T >= 0).all() and (T <= 1.0 + 1e-6).all()
