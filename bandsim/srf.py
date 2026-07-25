"""Design A — SRF convolution: synthesize a target sensor's bands from a hyperspectral cube.

Physics (NASA HLS bandpass operator):

    rho_b(x) = sum_i rho(x, lam_i) * R_b(lam_i) / sum_i R_b(lam_i)

which is a linear operator  rho_sensor = rho_hsi @ W.T  with the resampling matrix

    W[b, i] = R_b(lam_i) / sum_j R_b(lam_j)          # each row sums to 1

"Missing bands" then arise physically because different sensors have different band
sets (e.g. Landsat-8/9 OLI lacks Sentinel-2's red-edge B5/B6/B7 and narrow water-vapour
B9). See docs/guide/03_physical_simulation.md.

This module is fully functional given (a) the cube's wavelength axis and (b) a dict of
band SRFs sampled on that axis.

TWO SRF SOURCES, TWO LEVELS OF CLAIM. The bandpass OPERATOR above is exact for any SRF, but the
SRF it is handed decides what may be claimed:
  * `pyspectral_srf` -> REAL measured per-band relative spectral response (ESA S2-SRF / USGS OLI
    RSR via the pyspectral store). Results built on this are sensor-specific.
  * `gaussian_srf`   -> a synthetic Gaussian bandpass. Callers here pass a SINGLE fwhm_nm for every
    band (30 nm in the configs and demos), which is a FIRST-ORDER approximation and NOT a measured
    sensor response: against the pyspectral store the real FWHM spans 13 nm (S2 B5/B6) to 173 nm
    (S2 B12), so a flat 30 nm is ~2.3x too WIDE on the red-edge and ~5.8x too NARROW on B12 (OLI
    SWIR-2 is worse, 30 vs 186 nm). Use it for tests/demos and for band-SET (missing-band) geometry;
    do not describe a Gaussian-SRF result as "sensor-realistic" or as a measured-SRF simulation.
"""
from __future__ import annotations
import numpy as np

# np.trapz was DEPRECATED (not removed) in NumPy 2.0 and renamed np.trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def gaussian_srf(wavelengths_nm, centers_nm, fwhm_nm, names=None):
    """Build a dict {band_name: R_b(lam)} of SYNTHETIC Gaussian SRFs on the cube's wavelength grid.

    Approximation, not a measured response: a real MSI/OLI band is a near-rectangular interference
    filter with per-band width, whereas this is a Gaussian of the width you pass. Passing one scalar
    fwhm_nm for a whole sensor (as the demo configs do) additionally flattens the real 13-173 nm
    spread across bands. Prefer `pyspectral_srf` whenever the claim depends on band SHAPE; this is
    appropriate when the claim depends on which bands EXIST (the missing-band geometry).

    Parameters
    ----------
    wavelengths_nm : (C_hsi,) array   hyperspectral wavelength axis, nm
    centers_nm     : (B,) array       target band centres, nm
    fwhm_nm        : float or (B,) array  full-width-half-max per band, nm
    names          : (B,) sequence of str, optional  REAL band names to preserve (e.g. Sentinel-2
                     B1..B8, B8A, B9, B11, B12). If omitted, auto-names B1..BN -- which SILENTLY
                     corrupts sensors with non-sequential names (B8A/B9 -> B9/B10), so callers with
                     named bands MUST pass names.
    """
    wl = np.asarray(wavelengths_nm, float)
    centers = np.atleast_1d(np.asarray(centers_nm, float))
    fwhm = np.broadcast_to(np.asarray(fwhm_nm, float), centers.shape)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if names is None:
        names = [f"B{k+1}" for k in range(centers.size)]
    elif len(names) != centers.size:
        raise ValueError(f"names ({len(names)}) != centers ({centers.size})")
    return {str(names[k]): np.exp(-0.5 * ((wl - c) / s) ** 2) for k, (c, s) in enumerate(zip(centers, sigma))}


def pyspectral_srf(wavelengths_nm, platform, instrument, exclude=(), detector_policy="first",
                   return_detectors=False, return_meta=False):
    """Build {band: R_b(lam)} from REAL measured RSR via pyspectral (vs synthetic gaussian_srf).

    Interpolates each band's measured relative spectral response onto the cube's wavelength
    grid (nm). Bands whose measured range does not overlap the grid (e.g. OLI thermal B10/B11
    at ~10-12 um) get an all-zero response and are dropped automatically -> the reflective
    band set remains. Requires pyspectral (first use downloads the RSR store).

    DETECTOR POLICY. A pyspectral band maps to one RSR per DETECTOR, and a push-broom instrument
    can publish several (per-module / per-focal-plane) responses. The previous code took
    `list(dets.keys())[0]` -- whichever detector the RSR file happened to enumerate first -- so the
    resulting band was an undeclared function of dict ordering. The policy is now explicit:

      "first"       (default) the sole detector, and a ValueError if the band HAS several, so an
                    ambiguous sensor forces a deliberate choice instead of inheriting file order.
      "sorted_first" lowest detector name in sorted() order: deterministic, still one detector.
      "mean"        unweighted mean response across detectors, i.e. a nominal field-average band.

    "first" reproduces the historical numbers exactly for the sensors used here: Sentinel-2A MSI and
    Landsat-8 OLI both publish exactly ONE detector ('det-1') per band in the pyspectral store, so no
    result changes -- the guard only converts a latent ordering dependence into a loud failure.

    NAMING CAVEAT: pyspectral's band names are the RSR store's, NOT this module's. Sentinel-2 comes
    back zero-padded ('B01'..'B09' plus 'B10' cirrus, 13 bands) while SENTINEL2_MSI_CENTERS_NM uses
    'B1'..'B9' and omits B10; Landsat-8 OLI comes back with the store's ordering where 'B6' is the
    1373 nm cirrus band and 'B7'/'B8' are SWIR-1/SWIR-2, whereas LANDSAT8_OLI_CENTERS_NM follows the
    USGS numbering where 'B6' IS SWIR-1. The two paths therefore share band NAMES that denote
    DIFFERENT wavelengths -- never match bands by name across srf_source; match by centre wavelength.

    THE BAND SET IS THE STORE'S, NOT A PRODUCT'S. What comes back is every band the RSR file holds
    that overlaps the grid — for Sentinel-2 that is 13 bands INCLUDING B10 cirrus, which the L2A
    surface-reflectance product does not contain; for Landsat-8 OLI it is 9 bands INCLUDING the 15 m
    panchromatic B8 and the cirrus band, neither of which is in the L2 surface-reflectance product.
    `gaussian_srf` driven by this module's centre tables returns the 12/7 PRODUCT sets instead. A
    caller that treats the two sources as interchangeable is therefore swapping the BAND SET (and
    the input dimension of anything downstream), not just the band SHAPE. Use
    `select_canonical_bandset` / `sensor_bandset` to pin one contract across both sources.

    Example: pyspectral_srf(AVIRIS_WL_NM, "Sentinel-2A", "msi")  # 13 real MSI bands

    Returns the SRF dict; (srf, detectors_used) when return_detectors=True so a run can record which
    physical detector each band came from; or (srf, meta) when return_meta=True, where
    meta["native"][band] carries {integral_nm, in_range_nm, center_nm} measured on the band's OWN
    native RSR grid — the reference `grid_sampling_diagnostics` needs, and which is unrecoverable
    once the response has been interpolated onto a coarse or gappy target grid.
    """
    if detector_policy not in ("first", "sorted_first", "mean"):
        raise ValueError(f"unknown detector_policy {detector_policy!r} "
                         f"(expected 'first', 'sorted_first' or 'mean')")
    from pyspectral.rsr_reader import RelativeSpectralResponse
    rsr = RelativeSpectralResponse(platform, instrument)
    wl_um = np.asarray(wavelengths_nm, float) / 1000.0
    lo_nm, hi_nm = float(wl_um.min() * 1000.0), float(wl_um.max() * 1000.0)
    srf, used, native, dropped = {}, {}, {}, {}
    for band, dets in rsr.rsr.items():
        if band in exclude:
            continue
        names = list(dets.keys())
        if detector_policy == "first":
            if len(names) > 1:
                raise ValueError(
                    f"{platform}/{instrument} band {band!r} has {len(names)} detectors {sorted(names)}; "
                    f"detector_policy='first' would silently pick whichever the RSR file lists first. "
                    f"Pass detector_policy='sorted_first' or 'mean' to state the aggregation.")
            chosen = [names[0]]
        elif detector_policy == "sorted_first":
            chosen = [sorted(names)[0]]
        else:
            chosen = sorted(names)
        acc = np.zeros(wl_um.shape, float)
        areas, in_range, centres = [], [], []
        for det in chosen:
            w = np.asarray(dets[det]["wavelength"], float)  # um
            r = np.asarray(dets[det]["response"], float)
            order = np.argsort(w)
            w, r = w[order], r[order]
            acc += np.interp(wl_um, w, r, left=0.0, right=0.0)
            # Integral, in-axis-range integral and centroid on the band's OWN native grid -- the only
            # place the band's TRUE extent still exists. After interpolation onto a coarse or GAPPY
            # target axis, whatever fell outside it is unrecoverable, and row-normalisation then
            # rescales the survivors back to 1, so nothing downstream can tell a truncated band from
            # a whole one. (The in-range integral zeroes the response outside the target axis rather
            # than splitting the boundary sample: the native RSR grid is ~1 nm, so the difference is
            # far below the 5% threshold this feeds.)
            w_nm = w * 1000.0
            area = float(_trapezoid(r, w_nm))
            areas.append(area)
            in_range.append(float(_trapezoid(np.where((w_nm >= lo_nm) & (w_nm <= hi_nm), r, 0.0), w_nm)))
            centres.append(float(_trapezoid(r * w_nm, w_nm) / area) if area > 0 else float("nan"))
        resp = acc / len(chosen)
        entry = {"integral_nm": float(np.mean(areas)), "in_range_nm": float(np.mean(in_range)),
                 "center_nm": float(np.mean(centres))}
        if resp.sum() > 0:                              # band overlaps the HSI grid
            srf[band] = resp
            used[band] = chosen[0] if len(chosen) == 1 else f"mean({','.join(chosen)})"
            native[band] = entry
        else:
            dropped[band] = entry                       # zero overlap (e.g. OLI thermal at 10-12 um)
    if return_meta:
        return srf, {"detectors": used, "native": native, "dropped_no_overlap": dropped}
    return (srf, used) if return_detectors else srf


def _trapezoid_weights(wl):
    """Trapezoidal quadrature weights for integrating over a (possibly NON-UNIFORM) wavelength grid.
    Plain R/sum(R) implicitly assumes uniform spacing; a real (gappy) axis must weight by dlambda.

    The axis must be STRICTLY INCREASING and this is enforced, because the quadrature weights alone
    cannot distinguish a legitimately non-uniform axis from a mis-ordered one. An earlier version
    returned np.abs(q), which made a descending axis produce positive-looking weights and silently
    turned a data-ordering error into a plausible wrong answer: feeding a reversed axis (without
    reversing the cube) moved a 1000 nm band's value from 0.2857 to 0.7143 with no warning. A
    reversed or shuffled axis is always a caller bug, so it is rejected here rather than absorbed.
    """
    wl = np.asarray(wl, float)
    if wl.ndim != 1:
        raise ValueError(f"wavelength axis must be 1-D, got shape {wl.shape}")
    if wl.size < 2:
        raise ValueError(f"wavelength axis needs >= 2 samples to define dlambda, got {wl.size}")
    if not np.isfinite(wl).all():
        bad = int((~np.isfinite(wl)).sum())
        raise ValueError(f"wavelength axis has {bad} non-finite value(s) (NaN/Inf); quadrature "
                         f"weights would silently propagate them into every band")
    d = np.diff(wl)
    if not (d > 0).all():
        n_bad = int((d <= 0).sum())
        i = int(np.argmin(d))
        order = "descending" if (d < 0).all() else "non-monotonic"
        raise ValueError(
            f"wavelength axis must be strictly increasing ({order}: {n_bad} of {d.size} steps are "
            f"<= 0, first at index {i}: {wl[i]:.4f} -> {wl[i+1]:.4f} nm). Sort the axis AND reorder "
            f"the cube/SRF the same way -- reordering only the axis mis-assigns every band.")
    q = np.empty_like(wl)
    q[0] = (wl[1] - wl[0]) / 2.0
    q[-1] = (wl[-1] - wl[-2]) / 2.0
    q[1:-1] = (wl[2:] - wl[:-2]) / 2.0
    return q


def build_resampling_matrix(wavelengths_nm, srf):
    """Build the row-normalised resampling matrix W [B, C_hsi] from an SRF dict.

    Each row = SRF * trapezoidal-quadrature-weight(dlambda), then normalised, so apply_srf computes
    the correct bandpass integral  sum(rho R dl) / sum(R dl)  on ANY (uniform or non-uniform)
    wavelength grid. Plain R/sum(R) is only correct on a UNIFORM grid (e.g. wl=[0,1,10], flat R over
    rho=[0,0,10] gives 3.33 vs the physical 4.5).

    Returns
    -------
    W          : (B, C_hsi) ndarray, each row sums to 1
    band_names : list[str]
    """
    wl = np.asarray(wavelengths_nm, float)
    q = _trapezoid_weights(wl)
    band_names = list(srf.keys())
    W = np.zeros((len(band_names), wl.size), float)
    for b, name in enumerate(band_names):
        r = np.asarray(srf[name], float)
        if r.shape != wl.shape:
            raise ValueError(f"SRF '{name}' shape {r.shape} != wavelengths {wl.shape}")
        weighted = r * q                           # incorporate dlambda (trapezoid quadrature)
        denom = weighted.sum()
        if denom <= 0:
            raise ValueError(f"SRF '{name}' has non-positive total response over the grid")
        W[b] = weighted / denom
    return W, band_names


def apply_srf(cube, W):
    """Convolve a hyperspectral cube with a resampling matrix.

    Parameters
    ----------
    cube : (H, W_, C_hsi) or (N, C_hsi) ndarray of reflectance/radiance
    W    : (B, C_hsi) resampling matrix from build_resampling_matrix

    Returns
    -------
    out  : (H, W_, B) or (N, B) ndarray — the synthesized sensor bands
    """
    cube = np.asarray(cube, float)
    C = W.shape[1]
    if cube.shape[-1] != C:
        raise ValueError(f"cube last dim {cube.shape[-1]} != W cols {C}")
    flat = cube.reshape(-1, C)
    out = flat @ W.T
    return out.reshape(*cube.shape[:-1], W.shape[0])


# NOMINAL band centres (nm), rounded to the nearest nm, used to *define* which bands each sensor
# has (hence which are "missing" cross-sensor) — the band SET is the physical content here, not the
# band shape. Centres are nominal per the ESA S2 / USGS OLI band tables and are NOT per-satellite:
# S2A, S2B and S2C differ by a few nm per band, and this dict does not say which unit it is. Pair
# them with pyspectral_srf when a result depends on the measured response (see module docstring).
SENTINEL2_MSI_CENTERS_NM = {   # 12 reflective bands; B10 (1375 nm cirrus) deliberately omitted
    "B1": 443, "B2": 490, "B3": 560, "B4": 665, "B5": 705, "B6": 740,
    "B7": 783, "B8": 842, "B8A": 865, "B9": 945, "B11": 1610, "B12": 2190,
}
LANDSAT8_OLI_CENTERS_NM = {  # USGS numbering: B6 IS SWIR-1. NO red-edge (705/740/783), NO narrow
    # 945 water-vapour, and no pan (B8) / cirrus (B9) — the B1-B7 reflective subset only. pyspectral
    # returns a DIFFERENT name->wavelength map for OLI; see the naming caveat in pyspectral_srf.
    "B1": 443, "B2": 482, "B3": 561, "B4": 655, "B5": 865, "B6": 1609, "B7": 2201,
}


# ------------------------------------------------------------------------------------------------
# ONE canonical band-set contract, shared by both SRF sources
# ------------------------------------------------------------------------------------------------
# The centre tables above define which bands a sensor PRODUCT has. `pyspectral_srf` does not -- it
# returns whatever the RSR store holds. Everything below exists so a caller can ask for "sentinel2"
# and get the SAME ordered, canonically-named band list from either source, because a downstream
# model's input dimension, its parameter count, and the claim built on it all depend on that list.

SENSOR_SPECS = {
    "sentinel2": {
        "platform": "Sentinel-2A", "instrument": "msi", "centers_nm": SENTINEL2_MSI_CENTERS_NM,
        "product": "L2A surface reflectance: 12 bands. B10 (1375 nm cirrus) is acquired by the "
                   "instrument but is NOT in the L2A product, and on a 'corrected' AVIRIS axis it "
                   "lands on the removed water-vapour gap -- excluded on both grounds.",
    },
    "landsat_oli": {
        "platform": "Landsat-8", "instrument": "oli", "centers_nm": LANDSAT8_OLI_CENTERS_NM,
        "product": "L2 surface reflectance: 7 bands. The 15 m panchromatic B8 and the 1373 nm "
                   "cirrus band are acquired but are not surface-reflectance bands; pan is also a "
                   "different ground sample distance, so including it would compare band sets that "
                   "differ in spatial support as well as spectral content.",
    },
}


def effective_band_centers_nm(wavelengths_nm, srf):
    """{band: response-weighted mean wavelength AS SAMPLED ON THIS GRID} (nm).

    This -- not the catalogue centre -- is what a synthesized band actually measures, and it is the
    only band identity comparable ACROSS srf sources. The NAMES are not comparable: the pyspectral
    store and the centre tables above both spell a Landsat-8 OLI band 'B6', and they mean 1373 nm
    (cirrus) and 1609 nm (SWIR-1) respectively. Matching by name across sources silently pairs
    different wavelengths; matching by centre cannot.
    """
    wl = np.asarray(wavelengths_nm, float)
    q = _trapezoid_weights(wl)
    out = {}
    for name, r in srf.items():
        r = np.asarray(r, float)
        if r.shape != wl.shape:
            raise ValueError(f"SRF '{name}' shape {r.shape} != wavelengths {wl.shape}")
        w = r * q
        tot = float(w.sum())
        if tot <= 0:
            raise ValueError(f"SRF '{name}' has non-positive total response over the grid")
        out[name] = float((w @ wl) / tot)
    return out


def grid_sampling_diagnostics(wavelengths_nm, srf, native=None):
    """Per-band evidence that the wavelength axis actually RESOLVES each band -> {band: {...}}.

    The failure being guarded: a band whose response lands where the cube has no data. `resp.sum()>0`
    does not detect it (any sliver of overlap passes) and `build_resampling_matrix` then renormalises
    the survivors back to row-sum 1, so the band comes back looking like every other healthy channel
    while measuring something else. On the Indian Pines 'corrected' axis this is not hypothetical: the
    removed water-vapour bands leave a gap at ~1378-1436 nm and Sentinel-2's B10 / OLI's cirrus band
    sit at ~1373 nm, half inside it.

    `native` (from pyspectral_srf(..., return_meta=True), or built analytically) is
    {band: {integral_nm, in_range_nm}} on the band's own dense RSR grid. Three quantities come out,
    and they are NOT interchangeable -- each detects a different way of being unresolved:

    out_of_range_fraction  1 - in_range_nm/integral_nm: the share of the band's response lying
                           OUTSIDE [min(lam), max(lam)]. A pure range test, immune to quadrature
                           error, and the only one that catches a band hanging off the END of the
                           axis (every surviving sample there has a perfectly normal weight).
    grid_dlambda_ratio     (sum_i R q_i^2 / sum_i R q_i) / median(diff(lam)). A GAP makes the
                           trapezoid weight of the sample at its edge span the whole gap, so one
                           sample stands in for wavelengths never measured. This is the only one of
                           the three that catches an INTERIOR gap, and on this data it is the only
                           signal of ANY kind: MEASURED on AVIRIS + the pyspectral store, the ratio is
                           2.74 for S2 B10 and 2.92 for OLI's cirrus band against exactly 1.00 for
                           every canonical band. Note what does NOT show it -- the truncated cirrus
                           band's effective CENTRE reads 1374.2 nm against a native 1373.5, i.e. it
                           looks precisely where it belongs, because the samples on BOTH sides of the
                           gap are inflated and the centroid is pulled back into place.
    coverage               sum_i R(lam_i) q_i / integral_nm. REPORTED, NEVER A GATE, and the reason is
                           measured rather than argued. On the AVIRIS axis the 19 canonical S2+OLI
                           bands span coverage 0.93 (S2 B1) to 1.35 (S2 B6) purely from quadrature
                           error on a 9.6 nm grid, while the two gap-straddling cirrus bands this
                           guard exists to catch score 1.65 and 2.07 -- ABOVE one, because the samples
                           at both gap edges carry 33.6 nm of weight each and the sum over-counts. So
                           a `coverage >= 0.95` gate (which is what this module first shipped) would
                           have rejected a legitimate band AND admitted both defective ones. Use
                           coverage to judge how well a band is integrated; never to decide whether a
                           band is valid.

    Deliberately NOT a "largest single-sample share" test either, for the same reason coverage fails:
    S2's 20-nm B9 gets ~2 samples on a 9.6-nm grid and would trip any share threshold while being
    perfectly well sampled (its dlambda ratio is exactly 1.0).
    """
    wl = np.asarray(wavelengths_nm, float)
    q = _trapezoid_weights(wl)
    med = float(np.median(np.diff(wl)))
    out = {}
    for name, r in srf.items():
        r = np.asarray(r, float)
        if r.shape != wl.shape:
            raise ValueError(f"SRF '{name}' shape {r.shape} != wavelengths {wl.shape}")
        w = r * q
        tot = float(w.sum())
        if tot <= 0:
            raise ValueError(f"SRF '{name}' has non-positive total response over the grid")
        eff = float((w @ q) / tot)
        d = {"center_nm": float((w @ wl) / tot),
             "n_samples": int((r > 0.01 * float(r.max())).sum()),
             "grid_dlambda_nm": eff,
             "grid_dlambda_ratio": (eff / med) if med > 0 else float("nan")}
        nat = (native or {}).get(name)
        if nat and float(nat.get("integral_nm", 0.0)) > 0:
            total = float(nat["integral_nm"])
            d["coverage"] = tot / total
            d["out_of_range_fraction"] = max(0.0, 1.0 - float(nat.get("in_range_nm", total)) / total)
        out[name] = d
    return out


def check_grid_sampling(diagnostics, max_out_of_range=0.05, max_dlambda_ratio=2.0,
                        min_coverage=None, who="SRF"):
    """Raise unless every band is genuinely resolved by the axis it was sampled on.

    Fails CLOSED rather than warning, because the defect it guards still produces a number: a warning
    scrolls past once while the number goes on into a CSV, a figure and a claim.

    Only the two well-posed tests are gated by default. `min_coverage` defaults to None ON PURPOSE --
    see grid_sampling_diagnostics: coverage cannot separate "unresolved" from "narrower than the grid
    spacing", so gating on it would reject Sentinel-2's red-edge bands on an AVIRIS axis. It is
    available for callers whose grid is fine enough for the distinction to be meaningful.

    Bands with no native reference (e.g. a hand-built SRF dict) are still checked by the dlambda
    ratio, which needs nothing beyond the axis itself.

    LIMIT WORTH KNOWING: every quantity here is computed AGAINST the axis it is handed, so none of
    them can tell that the AXIS ITSELF is wrong. A band's effective centre is the centroid on that
    axis, so it lands on the catalogue centre whether or not the axis is right, and the band-set
    contract passes either way. Measured against the +4 correction proposed for
    `bandsim.io.AVIRIS_WL_NM` (Indian Pines' 220 bands are AVIRIS' 224 minus the first four, so the
    axis should start near 437.7 nm rather than 400): the gap-straddling cirrus band's dlambda ratio
    falls from 2.24 to 1.79 and would stop tripping the 2.0 gate, while `out_of_range_fraction`
    starts firing on the 443 nm coastal band, a third of whose response would then sit below the
    first measured channel. Both are the checks behaving correctly on different axes -- but it means
    the thresholds are calibrated to an axis, and the PRODUCT band-set contract, not this, is what
    keeps a non-product band out of a result.
    """
    bad = []
    for name, d in diagnostics.items():
        oor = d.get("out_of_range_fraction")
        if max_out_of_range is not None and oor is not None and oor > max_out_of_range:
            bad.append(f"{name} (centre {d['center_nm']:.0f} nm): {100 * oor:.0f}% of its response "
                       f"lies OUTSIDE this wavelength axis -- the band is truncated by the edge of "
                       f"the cube, and renormalisation hides that the missing part ever existed")
        if max_dlambda_ratio is not None and d["grid_dlambda_ratio"] > max_dlambda_ratio:
            bad.append(f"{name} (centre {d['center_nm']:.0f} nm): grid_dlambda_ratio "
                       f"{d['grid_dlambda_ratio']:.2f} > {max_dlambda_ratio} -- effective sample width "
                       f"{d['grid_dlambda_nm']:.1f} nm means it is integrated ACROSS a gap in the "
                       f"axis, with samples at the gap edge standing in for unmeasured wavelengths")
        cov = d.get("coverage")
        if min_coverage is not None and cov is not None and not (cov >= min_coverage):
            bad.append(f"{name} (centre {d['center_nm']:.0f} nm): coverage {cov:.3f} < {min_coverage}")
    if bad:
        raise ValueError(
            f"{who}: {len(bad)} band(s) not resolved by this wavelength axis:\n  " + "\n  ".join(bad)
            + "\nRemove the band from the contract, or use a cube whose axis covers it. Note these "
              "bands do NOT crash and do NOT look wrong -- they yield finite, plausible reflectance.")


def select_canonical_bandset(wavelengths_nm, srf, canonical_centers_nm, tol_nm=25.0,
                             exclude=(), exclude_reason=""):
    """Reduce an SRF dict to ONE canonical, ordered, canonically-NAMED band set, matched by CENTRE.

    Returns (srf_canonical, report): a fresh dict in the canonical table's order keyed by canonical
    names, plus a report recording which source band each came from, the centre mismatch, and every
    source band that was dropped.

    This is the contract that makes `--srf pyspectral` and `--srf gaussian` the same EXPERIMENT.
    Without it they are not: pyspectral returns the RSR store's list (Sentinel-2: 13 bands including
    B10 cirrus; Landsat-8 OLI: 9 including the 15 m panchromatic and cirrus), gaussian returns the
    12/7 product bands of the centre tables, so switching sources changed the band SET, the input
    dimension and every parameter count downstream -- not just the band SHAPE the flag documents.

    Matching is nearest-centre with an INJECTIVITY check, not nearest-name and not greedy-by-order:
    S2's B8 (833 nm) and B8A (865 nm) sit ~32 nm apart, so a tolerance loose enough to absorb
    catalogue-vs-measured drift is also loose enough to mispair them, and requiring the assignment to
    be one-to-one is what rules that out. A canonical band with no source band inside `tol_nm` RAISES:
    it means this axis cannot express the requested product -- a fact about the experiment, not a band
    to drop quietly.

    `exclude` names canonical bands that are DELIBERATELY not requested, with `exclude_reason`
    recorded beside them in report["excluded"]. This is the sanctioned way to run a product subset:
    an axis that cannot sample a band's full response (the AVIRIS-based axis starts at 437.7 nm and
    S2/OLI B1 sits at 443, leaving 16-24% of its response unsampled) must not synthesize it, and
    the refusal guard downstream is deliberately fail-closed -- so the subset has to be an explicit
    decision at the call site, never a silent drop here.
    """
    centers = effective_band_centers_nm(wavelengths_nm, srf)
    src = list(centers)
    if not src:
        raise ValueError("no source bands to select from (empty SRF dict)")
    exclude = tuple(exclude)
    unknown_ex = [b for b in exclude if b not in canonical_centers_nm]
    if unknown_ex:
        raise ValueError(f"exclude names {unknown_ex} not in the canonical table "
                         f"{list(canonical_centers_nm)} -- a typo here would silently change "
                         f"nothing while claiming to")
    excluded = {b: (exclude_reason or "excluded by caller") for b in exclude}
    canonical_centers_nm = {k: v for k, v in canonical_centers_nm.items() if k not in exclude}
    available = ", ".join(f"{b}@{c:.0f}nm" for b, c in sorted(centers.items(), key=lambda kv: kv[1]))
    out, claimed, matched = {}, {}, {}
    for name, target in canonical_centers_nm.items():
        k = min(src, key=lambda b: abs(centers[b] - float(target)))
        d = abs(centers[k] - float(target))
        if d > tol_nm:
            raise ValueError(
                f"canonical band {name!r} ({target} nm) has no source band within {tol_nm} nm on this "
                f"wavelength axis; nearest is {k!r} at {centers[k]:.1f} nm ({d:.1f} nm away). "
                f"Available: {available}. This axis cannot express the requested band set.")
        if k in claimed:
            raise ValueError(
                f"canonical bands {claimed[k]!r} and {name!r} both matched source band {k!r} "
                f"({centers[k]:.1f} nm); the mapping must be one-to-one. Tighten tol_nm (currently "
                f"{tol_nm} nm) or fix the centre table. Available: {available}.")
        claimed[k] = name
        out[name] = np.asarray(srf[k], float)
        matched[name] = {"source_band": k, "center_nm": round(centers[k], 2),
                         "canonical_center_nm": float(target), "delta_nm": round(d, 2)}
    return out, {"matched": matched, "excluded": excluded,
                 "dropped": {b: round(centers[b], 2) for b in src if b not in claimed},
                 "n_source_bands": len(src), "n_canonical_bands": len(out), "tol_nm": float(tol_nm)}


def sensor_bandset(wavelengths_nm, sensor, source="pyspectral", fwhm_nm=30.0, tol_nm=25.0,
                   max_out_of_range=0.05, max_dlambda_ratio=2.0, min_coverage=None,
                   detector_policy="first", exclude=(), exclude_reason=""):
    """A sensor name + a wavelength axis -> ONE checked, canonical bandpass operator.

    Returns a dict: W (row-normalised (B, C) resampling matrix), names (canonical, ordered),
    centers_nm (as actually sampled), the sampling diagnostics, the band-selection report, and enough
    identity (source, platform/instrument, detectors, W hash) to stamp straight into provenance.

    Every experiment that synthesizes a sensor should come through here instead of composing
    pyspectral_srf/gaussian_srf + build_resampling_matrix by hand. Two scripts already composed it by
    hand, identically and identically wrong: neither passed an `exclude` to pyspectral_srf, so both
    ran on the store's 13/9-band list while their own docstrings and figures said 12/7.
    """
    if sensor not in SENSOR_SPECS:
        raise ValueError(f"unknown sensor {sensor!r}; expected one of {sorted(SENSOR_SPECS)}")
    if source not in ("pyspectral", "gaussian"):
        raise ValueError(f"unknown srf source {source!r}; expected 'pyspectral' or 'gaussian'")
    spec = SENSOR_SPECS[sensor]
    wl = np.asarray(wavelengths_nm, float)
    if source == "pyspectral":
        srf, meta = pyspectral_srf(wl, spec["platform"], spec["instrument"],
                                   detector_policy=detector_policy, return_meta=True)
        native, detectors = meta["native"], meta["detectors"]
    else:
        import math
        centers = spec["centers_nm"]
        # names= is REQUIRED: auto-numbering renames S2's B8A->B9 and B9->B10 (see gaussian_srf).
        srf = gaussian_srf(wl, list(centers.values()), fwhm_nm=fwhm_nm, names=list(centers.keys()))
        sigma = float(fwhm_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        total = sigma * np.sqrt(2.0 * np.pi)             # exact integral of a unit-peak Gaussian
        lo, hi = float(wl.min()), float(wl.max())
        # The in-range share is the Gaussian CDF between the axis endpoints, so a synthetic band that
        # hangs off the end of the axis is caught by exactly the same test as a measured one.
        native = {n: {"integral_nm": total, "center_nm": float(c),
                      "in_range_nm": total * 0.5 * (math.erf((hi - c) / (sigma * math.sqrt(2.0)))
                                                    - math.erf((lo - c) / (sigma * math.sqrt(2.0))))}
                  for n, c in zip(srf, centers.values())}
        detectors = None
    srf, report = select_canonical_bandset(wl, srf, spec["centers_nm"], tol_nm=tol_nm,
                                           exclude=exclude, exclude_reason=exclude_reason)
    native = {n: native[m["source_band"]] for n, m in report["matched"].items()
              if m["source_band"] in native}
    diag = grid_sampling_diagnostics(wl, srf, native=native)
    check_grid_sampling(diag, max_out_of_range=max_out_of_range,
                        max_dlambda_ratio=max_dlambda_ratio, min_coverage=min_coverage,
                        who=f"{sensor}/{source}")
    W, names = build_resampling_matrix(wl, srf)
    import hashlib
    return {"W": W, "names": names, "centers_nm": [diag[n]["center_nm"] for n in names],
            "diagnostics": diag, "selection": report, "source": source,
            "platform": spec["platform"], "instrument": spec["instrument"],
            "product": spec["product"], "detectors": detectors,
            "fwhm_nm": (float(fwhm_nm) if source == "gaussian" else None),
            "W_sha256": hashlib.sha256(np.ascontiguousarray(W, dtype="<f8").tobytes()).hexdigest()}


if __name__ == "__main__":
    # tiny self-demo: 200-band AVIRIS-like axis -> synthesize S2 vs OLI, show missing bands
    wl = np.linspace(400, 2500, 200)
    for name, centers in [("Sentinel-2", SENTINEL2_MSI_CENTERS_NM),
                          ("Landsat-8 OLI", LANDSAT8_OLI_CENTERS_NM)]:
        srf = gaussian_srf(wl, list(centers.values()), fwhm_nm=30.0)
        W, names = build_resampling_matrix(wl, srf)
        demo = apply_srf(np.random.default_rng(0).random((5, 5, 200)), W)
        print(f"{name}: {len(names)} bands, output {demo.shape}, row-sums~={W.sum(1).round(3)[:3]}")
    print("OLI is missing S2 red-edge/water-vapour bands -> physical 'missing bands'.")
