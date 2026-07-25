"""The band-set contract and the grid-sampling guards behind the Phase 2 cross-sensor panel.

Every test here is a counterexample to a defect that produced a NUMBER rather than an error:

  * `--srf pyspectral` and `--srf gaussian` silently ran on DIFFERENT band sets (13/9 vs 12/7), so
    they differed in input dimension and parameter count, not just band shape.
  * Sentinel-2 B10 / OLI cirrus (~1373 nm) sit on the water-vapour gap the Indian Pines 'corrected'
    cube removes, so they were synthesized from a surviving tail and renormalised back to row-sum 1
    -- a finite, plausible-looking channel that measured ~1362 nm and called itself cirrus.
  * The first version of the guard for that used a `coverage >= 0.95` threshold, which would have
    rejected Sentinel-2's 15-nm red-edge bands (the very bands the panel is about) as a side effect
    of the AVIRIS grid being 9.6 nm coarse. That near-miss has its own regression test below.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bandsim.io import AVIRIS_WL_NM
from bandsim.srf import (gaussian_srf, pyspectral_srf, grid_sampling_diagnostics,
                         check_grid_sampling, select_canonical_bandset, sensor_bandset,
                         SENSOR_SPECS, SENTINEL2_MSI_CENTERS_NM, LANDSAT8_OLI_CENTERS_NM)

MED_DL = float(np.median(np.diff(AVIRIS_WL_NM)))       # ~9.59 nm nominal AVIRIS spacing


# ------------------------------------------------------------------ canonical band-set contract
def test_canonical_bandset_matches_by_centre_and_ignores_misleading_names():
    """Band NAMES are not comparable across SRF sources: the pyspectral store and this repo's centre
    tables both spell a Landsat-8 OLI band 'B6' and mean 1373 nm (cirrus) vs 1609 nm (SWIR-1). The
    selector must therefore key on wavelength. Here the source deliberately names the 1609 nm band
    'B9' and the 655 nm band 'B2' -- a name-based mapping would scramble them."""
    wl = AVIRIS_WL_NM
    # 442.9 nm is edge-truncated on the corrected axis (effective centre drifts ~7 nm), so B1 is
    # excluded here exactly as at the production call sites; the name-scrambling point is carried
    # by the six interior bands.
    srf = gaussian_srf(wl, [482.0, 561.3, 655.0, 864.6, 1609.0, 2200.7],
                       fwhm_nm=40.0, names=["B2", "qq", "B1", "B7", "B9", "B3"])
    out, report = select_canonical_bandset(wl, srf, LANDSAT8_OLI_CENTERS_NM,
                                           exclude=("B1",), exclude_reason="edge-truncated on this axis")
    assert list(out) == [n for n in LANDSAT8_OLI_CENTERS_NM if n != "B1"], \
        "canonical names, in the table's order (B1 excluded)"
    assert report["matched"]["B6"]["source_band"] == "B9", "B6 is SWIR-1 (1609 nm), whatever it is called"
    assert report["matched"]["B4"]["source_band"] == "B1", "B4 is 655 nm, whatever it is called"
    assert all(m["delta_nm"] < 5.0 for m in report["matched"].values())


def test_canonical_bandset_drops_exactly_the_non_product_bands():
    """The store's extra bands (S2 B10 cirrus; OLI pan + cirrus) must be dropped, and the drop must be
    REPORTED rather than silent -- 'which bands were these' is the question the panel's numbers turned
    out to depend on most."""
    wl = AVIRIS_WL_NM
    centres = dict(SENTINEL2_MSI_CENTERS_NM)
    srf = gaussian_srf(wl, list(centres.values()) + [1375.0], fwhm_nm=40.0,
                       names=list(centres) + ["B10"])
    out, report = select_canonical_bandset(wl, srf, SENTINEL2_MSI_CENTERS_NM)
    assert len(out) == 12 and "B10" not in out
    assert "B10" in report["dropped"] and 1300 < report["dropped"]["B10"] < 1450


def test_canonical_bandset_raises_when_the_axis_cannot_express_the_product():
    """A canonical band with no match must RAISE, not be quietly omitted: a quietly shorter band list
    changes the input dimension and the parameter count of everything downstream, which is exactly
    how the two --srf modes came to be different experiments."""
    wl = np.linspace(400.0, 900.0, 60)                 # no SWIR: S2 B11/B12 cannot exist here
    srf = gaussian_srf(wl, [443, 490, 560, 665], fwhm_nm=30.0, names=["B1", "B2", "B3", "B4"])
    with pytest.raises(ValueError, match="no source band within"):
        select_canonical_bandset(wl, srf, SENTINEL2_MSI_CENTERS_NM)


def test_canonical_bandset_rejects_a_non_injective_mapping():
    """Two canonical bands must never collapse onto one source band. S2's B8 (833 nm) and B8A
    (865 nm) are only ~32 nm apart, so a tolerance loose enough to absorb catalogue-vs-measured drift
    is also loose enough to mispair them; only requiring a one-to-one assignment rules that out."""
    wl = AVIRIS_WL_NM
    srf = gaussian_srf(wl, [443, 490, 560, 665, 705, 740, 783, 850, 945, 1610, 2190],
                       fwhm_nm=30.0,
                       names=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "only850", "B9", "B11", "B12"])
    with pytest.raises(ValueError, match="one-to-one"):
        select_canonical_bandset(wl, srf, SENTINEL2_MSI_CENTERS_NM, tol_nm=40.0)


# ------------------------------------------------------------------- grid-sampling diagnostics
def test_a_band_straddling_the_aviris_water_gap_is_caught():
    """THE counterexample, re-sited at the 2026-07-20 axis correction. On the corrected axis the
    water-vapour axis gap spans 1398-1455 nm (it sat at ~1378-1436 on the stretched axis), so the
    probe band straddles it at 1430 nm. A band there still has positive response, so nothing
    downstream objects; the trapezoid weight at the gap edge simply grows (~25 nm) and those samples
    speak for wavelengths that were never measured. Note what this move revealed: the REAL 1375 nm
    cirrus band no longer straddles anything on the corrected axis -- the celebrated 'cirrus
    straddles the AVIRIS gap' phenomenon was partly an artifact of the wrong axis (see the next
    test)."""
    srf = gaussian_srf(AVIRIS_WL_NM, [1430.0], fwhm_nm=75.0, names=["B10"])
    d = grid_sampling_diagnostics(AVIRIS_WL_NM, srf)["B10"]
    assert d["grid_dlambda_ratio"] > 2.0, f"gap-straddling band not detected: {d}"
    assert d["grid_dlambda_nm"] > 2 * MED_DL
    with pytest.raises(ValueError, match="ACROSS a gap"):
        check_grid_sampling(grid_sampling_diagnostics(AVIRIS_WL_NM, srf))


def test_the_real_cirrus_band_is_invisible_to_every_check_except_the_dlambda_ratio():
    """REWRITTEN at the 2026-07-20 axis correction, because the fact it pinned turned out to be
    partly an artifact of the wrong axis. On the stretched axis the real S2 B10 straddled the water
    gap (ratio 2.74) and only the dlambda gate saw it. On the corrected axis the gap moved to
    1398-1455 nm and the REAL cirrus band at ~1373 nm is sampled cleanly -- measured ratio ~1.0 --
    so it is no longer a sampling defect at all; it stays out of the experiment purely by the
    PRODUCT contract. What this test pins now is both halves of that: the real B10 samples cleanly,
    and a band that genuinely straddles the corrected gap is caught by the ratio gate with a
    centre that still looks innocent (the centroid is pulled back into place by the inflated edge
    weights, so a centre sanity-check sees nothing)."""
    pytest.importorskip("pyspectral")
    try:
        srf, meta = pyspectral_srf(AVIRIS_WL_NM, "Sentinel-2A", "msi", return_meta=True)
    except Exception as e:
        pytest.skip(f"pyspectral RSR store unavailable: {e}")
    d = grid_sampling_diagnostics(AVIRIS_WL_NM, srf, native=meta["native"])["B10"]
    assert d["grid_dlambda_ratio"] < 2.0, \
        f"the real cirrus band samples cleanly on the corrected axis; got {d}"
    probe = gaussian_srf(AVIRIS_WL_NM, [1430.0], fwhm_nm=75.0, names=["probe"])
    dp = grid_sampling_diagnostics(AVIRIS_WL_NM, probe)["probe"]
    assert dp["grid_dlambda_ratio"] > 2.0, f"a true straddler of the 1398-1455 gap must be caught: {dp}"
    assert abs(dp["center_nm"] - 1430.0) < 20.0, \
        "the straddler's centre still looks innocent -- which is why only the ratio gate can see it"
    with pytest.raises(ValueError, match="ACROSS a gap"):
        check_grid_sampling({"probe": dp})


def test_a_narrow_but_well_sampled_band_is_not_caught():
    """The negative control for the test above. S2's red-edge B5 is only ~15 nm wide on a 9.6 nm grid
    -- genuinely coarsely sampled, but sampled UNIFORMLY, with no gap. It must pass: a guard that
    rejects the red-edge would delete the panel's entire subject."""
    srf = gaussian_srf(AVIRIS_WL_NM, [705.0], fwhm_nm=15.0, names=["B5"])
    d = grid_sampling_diagnostics(AVIRIS_WL_NM, srf)["B5"]
    assert d["grid_dlambda_ratio"] == pytest.approx(1.0, abs=1e-9), \
        f"a uniformly sampled band must have ratio exactly 1.0, got {d['grid_dlambda_ratio']}"
    check_grid_sampling(grid_sampling_diagnostics(AVIRIS_WL_NM, srf))       # must not raise


def test_coverage_is_reported_but_never_gated_by_default():
    """REGRESSION LOCK on a guard that was nearly shipped wrong, in BOTH directions.

    The first version of this check gated on `coverage >= 0.95`. On a 9.6 nm axis a near-RECTANGULAR
    band lands only one or two samples inside itself, so its Riemann sum can be a fraction of the true
    integral through no fault of the data: measured on the real store, the canonical bands span 0.93
    (S2 B1) to 1.35 (S2 B6), so the gate would have failed the default Sentinel-2 run outright. And it
    would not even have bought anything: the gap-straddling cirrus bands the gate was meant to catch
    score 1.65 and 2.07 and would have sailed through it (see the test above). Coverage measures
    quadrature quality, not validity."""
    wl = AVIRIS_WL_NM
    # Centre re-picked at the 2026-07-20 axis correction (grid phase shifted): 605.5 nm puts a
    # single sample inside the 15 nm rect on the corrected axis, giving coverage ~0.63.
    band = ((wl >= 598.0) & (wl <= 613.0)).astype(float)        # 15 nm hard-edged rectangle
    native = {"rect": {"integral_nm": 15.0, "in_range_nm": 15.0}}
    d = grid_sampling_diagnostics(wl, {"rect": band}, native=native)["rect"]
    assert d["coverage"] < 0.8, f"expected a poor quadrature ratio for a 15 nm rect, got {d}"
    assert d["grid_dlambda_ratio"] == pytest.approx(1.0, abs=1e-9)
    check_grid_sampling({"rect": d})                              # default: MUST NOT raise
    with pytest.raises(ValueError, match="coverage"):             # opt-in only
        check_grid_sampling({"rect": d}, min_coverage=0.95)


def test_a_band_truncated_by_the_end_of_the_axis_is_caught():
    """The third failure mode, which neither of the other two sees: a band running off the END of the
    axis. Every surviving sample has a perfectly normal quadrature weight (so the dlambda ratio is
    1.0) and it is not a gap -- only comparing against the band's native extent finds it."""
    wl = AVIRIS_WL_NM[AVIRIS_WL_NM <= 2200.0]
    with pytest.raises(ValueError, match="OUTSIDE this wavelength axis"):
        sensor_bandset(wl, "sentinel2", source="gaussian", fwhm_nm=30.0)


def test_out_of_range_fraction_is_near_zero_for_a_fully_contained_band():
    """Negative control for the test above: the same machinery must not flag a band the axis covers.

    Not asserted as exactly zero, because it genuinely is not: S2's B1 sits at 443 nm and the AVIRIS
    axis starts at 400, i.e. only 3.4 sigma away for a 30 nm Gaussian, so ~4e-4 of that band really
    does fall off the end. The measurement is right and the gate (5%) has two orders of magnitude of
    headroom over it -- which is the property worth pinning, rather than a zero that would only hold
    for bands nowhere near an edge."""
    # B1 excluded (2026-07-20 axis): on the corrected axis its truncation is 34%, i.e. the very
    # positive case the guard exists for -- the negative control is every band that REMAINS.
    info = sensor_bandset(AVIRIS_WL_NM, "sentinel2", source="gaussian", fwhm_nm=30.0,
                          exclude=("B1",), exclude_reason="axis starts above B1's blue edge")
    for name, d in info["diagnostics"].items():
        assert d["out_of_range_fraction"] < 0.005, f"{name} reported as truncated: {d}"
        assert d["grid_dlambda_ratio"] < 2.0, f"{name} reported as gap-straddling: {d}"


# ------------------------------------------------------------------------- the sensor contract
@pytest.mark.parametrize("sensor,expected", [
    ("sentinel2", ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]),
    ("landsat_oli", ["B2", "B3", "B4", "B5", "B6", "B7"]),
])
def test_gaussian_sensor_bandset_is_exactly_the_surface_reflectance_product(sensor, expected):
    """11 and 6 bands ON THE AVIRIS-BASED AXIS, in order, by name (2026-07-20: the corrected axis
    starts at 437.7 nm, so B1's 443 nm response cannot be fully sampled and both call sites exclude
    it by decision -- see synth_sensor / phase5.sensor_srf). The module docstring, the figure axis
    labels and every parameter count in Phase 2 are downstream of these two lists."""
    info = sensor_bandset(AVIRIS_WL_NM, sensor, source="gaussian", fwhm_nm=30.0, exclude=("B1",), exclude_reason="axis starts above B1's blue edge")
    assert info["names"] == expected
    assert info["W"].shape == (len(expected), AVIRIS_WL_NM.size)
    assert np.allclose(info["W"].sum(axis=1), 1.0), "each row is a normalised bandpass integral"
    assert len(info["centers_nm"]) == len(expected)
    assert info["centers_nm"] == sorted(info["centers_nm"]), "bands must be in ascending wavelength"


def test_sensor_bandset_rejects_unknown_sensors_and_sources():
    """The old `if source == 'pyspectral': ... else: gaussian` silently treated any typo as a request
    for the synthetic SRF -- a different experiment, chosen by a misspelling."""
    with pytest.raises(ValueError, match="unknown sensor"):
        sensor_bandset(AVIRIS_WL_NM, "sentinel3", source="gaussian")
    with pytest.raises(ValueError, match="unknown srf source"):
        sensor_bandset(AVIRIS_WL_NM, "sentinel2", source="pyspectrl")


def test_both_srf_sources_produce_the_identical_band_contract():
    """THE headline regression test. `--srf` is documented as choosing measured vs synthetic band
    SHAPE. Before the contract it also changed the band SET (13/9 vs 12/7), hence the input dimension
    and the parameter count, so the two modes were never comparable. They must now agree on the band
    names, their order and their centres; only the response shapes may differ.

    Skips only if the pyspectral RSR store is unavailable. A ValueError is NOT skipped -- that is the
    contract itself failing, which is what this test exists to see."""
    pytest.importorskip("pyspectral")
    try:
        meas = {s: sensor_bandset(AVIRIS_WL_NM, s, source="pyspectral", exclude=("B1",), exclude_reason="axis starts above B1's blue edge")
                for s in SENSOR_SPECS}
    except ValueError:
        raise                                     # a contract violation must fail, never skip
    except Exception as e:                        # store not downloaded in this environment
        pytest.skip(f"pyspectral RSR store unavailable: {e}")
    for sensor, info in meas.items():
        synth = sensor_bandset(AVIRIS_WL_NM, sensor, source="gaussian", fwhm_nm=30.0, exclude=("B1",), exclude_reason="axis starts above B1's blue edge")
        assert info["names"] == synth["names"], f"{sensor}: the two SRF sources disagree on the band set"
        assert len(info["names"]) == len(SENSOR_SPECS[sensor]["centers_nm"]) - 1  # B1 excluded on this axis
        for a, b, n in zip(info["centers_nm"], synth["centers_nm"], info["names"]):
            assert abs(a - b) < 25.0, f"{sensor} {n}: centres {a:.1f} vs {b:.1f} nm are not the same band"
        # and the store's extra bands must have been dropped by name-independent selection
        assert info["selection"]["n_source_bands"] > len(info["names"]), \
            f"{sensor}: nothing was dropped, so the store list was already the product list?"


def test_measured_srf_never_keeps_a_band_the_axis_cannot_resolve():
    """The cirrus band must not merely be absent from the contract -- had it been kept, the sampling
    check would have to catch it. Assert the surviving measured bands all pass both gates."""
    pytest.importorskip("pyspectral")
    try:
        # B1 excluded as at every production call site on this axis; the refusal path B1 would
        # exercise is pinned separately by the truncated-band tests in this file.
        info = sensor_bandset(AVIRIS_WL_NM, "sentinel2", source="pyspectral",
                              exclude=("B1",), exclude_reason="axis starts above B1's blue edge")
    except ValueError:
        raise
    except Exception as e:
        pytest.skip(f"pyspectral RSR store unavailable: {e}")
    for name, d in info["diagnostics"].items():
        assert d["grid_dlambda_ratio"] < 2.0, f"{name}: {d}"
        assert d["out_of_range_fraction"] < 0.05, f"{name}: {d}"
    # dropped now carries TWO legitimate entries: the ~1375 nm cirrus (out of product) and the
    # source B01 whose canonical target was excluded by the axis decision (its 444 nm centre then
    # matches nothing). Anything else dropping is still an error.
    assert all((1300 < c < 1450) or (438 < c < 452)
               for c in info["selection"]["dropped"].values()), \
        f"the only dropped S2 band should be the ~1375 nm cirrus: {info['selection']['dropped']}"


# --------------------------------------------------------------------------- red-edge ablation
def test_red_edge_ablation_removes_exactly_b5_b6_b7():
    """The ablation exists so the red-edge claim stops being an inference from the S2-OLI gap. It must
    remove three bands and nothing else, and the surviving bands must be BIT-IDENTICAL to the full
    arm's -- otherwise the two conditions differ by more than the red-edge."""
    import phase2_cross_sensor as CS
    wl = AVIRIS_WL_NM
    # exclude B1 exactly as both production call sites do on the AVIRIS-based axis (the corrected
    # axis starts at 437.7 nm and cannot sample B1's full response) -- this test is about the
    # red-edge trio, and without the exclusion the contract correctly refuses the whole call.
    info = sensor_bandset(wl, "sentinel2", source="gaussian", fwhm_nm=30.0,
                          exclude=("B1",), exclude_reason="axis starts above B1's blue edge")
    rng = np.random.default_rng(0)
    cube = rng.random((4, 5, wl.size))
    from bandsim.srf import apply_srf
    full = apply_srf(cube, info["W"])
    sub, sub_info, dropped = CS.drop_bands(full, info, *CS.RED_EDGE_NM)
    assert sub_info["names"] == ["B2", "B3", "B4", "B8", "B8A", "B9", "B11", "B12"]  # B1 excluded on this axis
    assert sorted(d.split("@")[0] for d in dropped) == ["B5", "B6", "B7"]
    keep = [i for i, n in enumerate(info["names"]) if n in set(sub_info["names"])]
    assert np.array_equal(sub, full[..., keep]), "surviving bands must be untouched"
    assert set(sub_info["diagnostics"]) == set(sub_info["names"]), \
        "the ablated arm's provenance must not still describe the bands it does not have"


def test_red_edge_window_does_not_touch_the_neighbouring_bands():
    """B4 (665 nm) and B8 (833 nm) bracket the red-edge; the window must clear both with margin, or
    the 'red-edge contribution' would silently include a red or NIR band."""
    import phase2_cross_sensor as CS
    lo, hi = CS.RED_EDGE_NM
    assert lo - SENTINEL2_MSI_CENTERS_NM["B4"] > 20.0
    assert SENTINEL2_MSI_CENTERS_NM["B8"] - hi > 20.0
    for b in ("B5", "B6", "B7"):
        assert lo < SENTINEL2_MSI_CENTERS_NM[b] < hi
