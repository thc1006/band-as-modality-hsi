"""Design B — 6S GASEOUS-absorption transmittance (band loss via molecular absorption).

Physics:  rho'(lam) = rho(lam) * T_gas(lam; CWV, O3)   then apply SRF (Design A).

We use 6S's `transmittance_global_gas` — the GASEOUS (water-vapour + O2/O3/CO2) transmittance,
which physically suppresses specific bands. It is INDEPENDENT of aerosols, so this is honestly a
COLUMN-WATER-VAPOUR (CWV) stress-test, NOT a full radiative-transfer / AOD ablation: AOD does NOT
change T_gas (an earlier CWV x AOD grid had an inert AOD axis -> AOD removed). Aerosol scattering,
path radiance and adjacency are NOT modelled. Lead the paper's atmosphere story with the REAL
Sen2Cor L1C->L2A transition; use this as a supplementary controlled gaseous-absorption axis.

Water-vapour (~940, 1130 nm), O2-A and CO2 windows physically suppress specific bands;
under high column water vapour, narrow bands (S2 B9, parts of SWIR) lose signal.

IMPLEMENTATION NOTES
- Requires Py6S + the 6S Fortran binary. On Linux/dell:  conda install -c conda-forge sixs py6s
  (Windows is painful — that's a reason to develop on dell). See ENVIRONMENT_SETUP.md §3.
- CACHE transmittance spectra per atmosphere (they are independent of the cube).
- VALIDITY LIMIT: 6S is invalid inside strong absorption cores — treat those bands as a
  HARD MASK rather than trusting 6S magnitudes there (see docs/guide/03_physical_simulation.md).

TODO(Phase 3): implement run_6s_transmittance(); until then load_cached_transmittance()
lets you develop the rest of the pipeline from a precomputed .npz.
"""
from __future__ import annotations
import numpy as np

# Suggested parameter grid (see roadmap): keep small. No AOD axis -- gaseous transmittance is
# aerosol-independent, so an AOD grid would be inert (identical T_gas across AOD values).
DEFAULT_GRID = {"cwv_g_cm2": [0.5, 2.0, 4.0]}

# Strong absorption cores (nm) to hard-mask regardless of 6S output (approx.).
STRONG_ABSORPTION_CORES_NM = [(1350, 1450), (1800, 1950)]  # water-vapour / SWIR cores


def run_6s_transmittance(wavelengths_nm, cwv_g_cm2, aod550, geometry=None,
                         ozone_cm_atm=0.35, aero=None, n_threads=8):
    """Return total gaseous atmospheric transmittance T(lam) in [0,1] via Py6S / 6S RTM.

    Requires Py6S + the 6S Fortran binary (conda-forge sixs). The main venv does NOT have
    these — run this from the `sixs` conda env (see ENVIRONMENT_SETUP.md §3) to PRECOMPUTE a
    transmittance table, then read it back in the main pipeline via load_cached_transmittance.

    We return the total GASEOUS transmittance (water vapour + O2/O3/CO2), which is what
    physically suppresses specific bands. Caller should additionally HARD-MASK the strong
    absorption cores (6S is unreliable there) via hard_mask_absorption_cores().

    NOTE: `aod550`/`aero` are accepted for 6S API compatibility but do NOT affect this gaseous
    output (aerosols change scattering / path radiance, not gaseous transmittance) -- there is no
    AOD ablation axis here; the degradation axis is CWV. See the module docstring.

    Parameters
    ----------
    wavelengths_nm : (C,) wavelength axis (nm), e.g. bandsim.io.AVIRIS_WL_NM
    cwv_g_cm2      : column water vapour (g/cm^2)
    aod550         : aerosol optical depth at 550 nm
    geometry       : optional dict {solar_z, solar_a, view_z, view_a} degrees
    """
    from Py6S import SixS, AtmosProfile, AeroProfile
    from Py6S.SixSHelpers import Wavelengths as _Wl
    import numpy as _np

    s = SixS()  # finds the 6S binary on PATH
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(float(cwv_g_cm2), float(ozone_cm_atm))
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental if aero is None else aero)
    s.aot550 = float(aod550)
    if geometry:
        s.geometry.solar_z = geometry.get("solar_z", 30.0)
        s.geometry.solar_a = geometry.get("solar_a", 0.0)
        s.geometry.view_z = geometry.get("view_z", 0.0)
        s.geometry.view_a = geometry.get("view_a", 0.0)
    wl_um = _np.asarray(wavelengths_nm, float) / 1000.0
    # 6S valid range ~0.35-2.5 um; clamp query wavelengths into range.
    wl_um = _np.clip(wl_um, 0.35, 2.5)
    _, results = _Wl.run_wavelengths(s, wl_um, output_name="transmittance_global_gas",
                                     n=n_threads, verbose=False)
    T = _np.array([r.total if hasattr(r, "total") else r for r in results], float)
    return _np.clip(T, 0.0, 1.0)


def load_cached_transmittance(npz_path, key, n_bands=None, expected_wavelengths_nm=None,
                              require_generation_mode=None):
    """Load a precomputed gaseous-transmittance spectrum (fallback when 6S isn't installed here).

    Validates the cached array is a finite, physical ([0,1]) 1-D transmittance and -- when n_bands
    is given -- that its length matches the current band count, so a stale or wrong-axis LUT fails
    LOUDLY instead of silently multiplying the cube by a mismatched/out-of-range spectrum.

    A LENGTH check alone is NOT enough, and that gap was a real bug: a LUT computed on a gapless
    linspace(400,2500,200) and the true GAPPED 200-band AVIRIS axis have identical length, so the
    table loaded cleanly while every T value landed on the wrong channel (183/200 bands off by
    >5 nm, worst 98 nm) -- silently multiplying near-transparent bands by deep-absorption
    transmittance. Pass `expected_wavelengths_nm` (the axis the cube is actually on) to assert the
    LUT's stored `wl_nm` IS that axis; identity is required exactly (atol=1e-6, rtol=0), and a table
    with no `wl_nm` at all is rejected rather than trusted.

    `require_generation_mode` closes a second identity gap the axis check cannot. A table produced by
    INTERPOLATING an older one onto this axis (the approach in
    archive/superseded_experiments/resample_6s_table.py) has the right length,
    the right `wl_nm` and the right axis hash, so it passes every check above -- while its values near
    narrow absorption edges are not what 6S would return there. Pass "direct_6s" when a result depends
    on the transmittance being computed rather than resampled, and an interpolated (or unlabelled)
    table is refused instead of quietly standing in for one.

    Opened with a context manager and allow_pickle=False: an NpzFile holds a file descriptor until
    closed, and these are read in loops."""
    with np.load(npz_path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"transmittance key {key!r} not in {npz_path} (have: {sorted(data.files)})")
        if require_generation_mode is not None:
            mode = str(data["generation_mode"]) if "generation_mode" in data.files else None
            if mode != require_generation_mode:
                raise ValueError(
                    f"{npz_path} generation_mode is {mode!r} but {require_generation_mode!r} was "
                    f"required. A resampled table reproduces this axis exactly and is still not the "
                    f"same physical quantity near narrow absorption features; an unlabelled table "
                    f"cannot be shown to be either. Regenerate with precompute_6s_table.py.")
        return _check_transmittance(data, npz_path, key, n_bands, expected_wavelengths_nm)


def _check_transmittance(data, npz_path, key, n_bands, expected_wavelengths_nm):
    """Value/axis validation for an already-opened table (see load_cached_transmittance)."""
    T = np.asarray(data[key], float)
    if T.ndim != 1:
        raise ValueError(f"cached T[{key}] must be 1-D, got shape {T.shape}")
    if n_bands is not None and T.shape[0] != n_bands:
        raise ValueError(f"cached T[{key}] has {T.shape[0]} bands but the cube has {n_bands} "
                         f"(wavelength-axis / LUT mismatch -- regenerate the 6S table for this axis)")
    if expected_wavelengths_nm is not None:
        want = np.asarray(expected_wavelengths_nm, float).ravel()
        if "wl_nm" not in data.files:
            raise ValueError(
                f"{npz_path} has no 'wl_nm' axis, so T[{key}] cannot be proven to sit on the "
                f"requested axis (expected {want.size} bands, {want[0]:.4f}-{want[-1]:.4f} nm). "
                f"An unlabelled transmittance table is NOT trustworthy -- regenerate it with its "
                f"wavelength axis stored alongside (experiments/precompute_6s_table.py).")
        got = np.asarray(data["wl_nm"], float).ravel()
        if got.size != want.size:
            raise ValueError(
                f"{npz_path} wavelength-axis LENGTH mismatch for T[{key}]: table has {got.size} "
                f"bands ({got[0]:.4f}-{got[-1]:.4f} nm) but the requested axis has {want.size} "
                f"({want[0]:.4f}-{want[-1]:.4f} nm) -- regenerate the 6S table for this axis.")
        if not np.allclose(got, want, rtol=0, atol=1e-6):
            d = float(np.abs(got - want).max())
            raise ValueError(
                f"{npz_path} wavelength-axis MISMATCH for T[{key}]: table axis "
                f"{got[0]:.4f}..{got[-1]:.4f} nm vs requested axis {want[0]:.4f}..{want[-1]:.4f} nm, "
                f"max |delta| = {d:.4f} nm (tolerance 1e-6). Same length is NOT the same axis -- "
                f"applying this table would multiply each band by another band's transmittance. "
                f"Regenerate the table on this axis with experiments/precompute_6s_table.py, which "
                f"runs 6S on the requested axis. Do NOT resample an existing table onto it: that "
                f"passes every check here and is still not what 6S returns at absorption edges.")
    if not np.isfinite(T).all() or T.min() < -1e-6 or T.max() > 1.0 + 1e-6:
        raise ValueError(f"cached T[{key}] is not a physical transmittance in [0,1] "
                         f"(range [{T.min():.3g},{T.max():.3g}], finite={bool(np.isfinite(T).all())})")
    return T


def hard_mask_absorption_cores(wavelengths_nm, cores_nm=STRONG_ABSORPTION_CORES_NM):
    """Boolean mask (True = keep) zeroing bands inside strong absorption cores.

    WHAT THIS ACTUALLY COSTS depends entirely on the SRF that follows, and the two paths differ by
    37 orders of magnitude -- measure before describing it either way. On the AVIRIS axis the mask
    zeroes 8 of 200 HSI bands (1359-1445 and 1800-1819 nm; the axis has already had its worst
    water-vapour bands removed, which is why so few remain). After Design A:

      * gaussian_srf at fwhm=30 nm (what the YAML configs use): every S2/OLI band centre is >=165 nm
        away from a masked core, so a 30 nm Gaussian puts at most 4e-37 of its weight there. The mask
        is NUMERICALLY INERT -- toggling hard_mask_cores does not change a single output digit. Do
        not present it as an active safeguard in a Gaussian-SRF result.
      * pyspectral measured RSR (what phase5/phase2 use): the cirrus bands sit INSIDE a core, so
        essentially all of their bandpass support is zeroed -- S2 B10 loses 99.98% of its weight and
        OLI's 1376 nm band 100%.

    WHAT THIS DOES NOT DO: it does not DELETE a band. The mask zeroes SOURCE-DOMAIN samples on the
    HSI axis; Design A's SRF then runs unchanged, so the output keeps its band count and its band
    names and the affected band becomes an ALL-ZERO channel. Verified directly: with a cirrus band
    responding only at 1375 nm, `simulate` returns shape (...,2) and band_names ['cirrus','outside']
    both before and after masking -- only the values change, [1,1] -> [0,1]. An earlier version of
    this docstring said the mask "DELETES a band" and described the result as a 12-band S2 /
    8-band OLI product; that is wrong, and it matters, because the model still receives that channel
    and can learn "always exactly zero" as a cue that a band is absent. Describe it as a
    spectral-value mask, and if a genuinely reduced band set is wanted, drop the columns explicitly
    and record which names were removed."""
    wl = np.asarray(wavelengths_nm, float)
    keep = np.ones(wl.shape, bool)
    for lo, hi in cores_nm:
        keep &= ~((wl >= lo) & (wl <= hi))
    return keep


def apply_atmosphere(cube, transmittance, keep_mask=None):
    """rho' = rho * T(lam); optionally hard-mask absorption-core bands to 0."""
    out = np.asarray(cube, float) * np.asarray(transmittance, float)
    if keep_mask is not None:
        out = out * keep_mask.astype(float)
    return out


def sensor_band_transmittance(transmittance_hsi, wavelengths_hsi_nm, srf_dict):
    """Integrate a fine-grid gaseous transmittance T(lambda) onto sensor bands via their SRF.

        T_b = sum_i T(lam_i) R_b(lam_i) / sum_i R_b(lam_i)

    the SAME row-normalised bandpass operator used for reflectance (srf.build_resampling_matrix),
    applied to the transmittance spectrum. This lets a 6S atmosphere computed on a fine HSI grid
    be applied DIRECTLY to an already-multispectral product (e.g. Sentinel-2 / CloudSEN12), whose
    native bands are what we degrade — without needing an HSI cube. Because each SRF row sums to 1
    and T in [0,1], each T_b is a convex combination of T and stays in [0,1].

    Parameters
    ----------
    transmittance_hsi  : (C_hsi,) gaseous transmittance in [0,1] on the HSI grid (e.g. a column
                         of the 6S table via load_cached_transmittance)
    wavelengths_hsi_nm : (C_hsi,) HSI wavelength axis the transmittance is sampled on
    srf_dict           : {band_name: R_b(lam)} on the SAME grid, e.g. from
                         bandsim.srf.pyspectral_srf(wavelengths_hsi_nm, "Sentinel-2A", "msi")

    Returns
    -------
    T_bands    : (B,) per-band transmittance in [0,1]
    band_names : list[str] in srf_dict order
    """
    from .srf import build_resampling_matrix
    W, names = build_resampling_matrix(np.asarray(wavelengths_hsi_nm, float), srf_dict)
    T = np.asarray(transmittance_hsi, float)
    if T.shape != (W.shape[1],):
        raise ValueError(f"transmittance shape {T.shape} != HSI grid ({W.shape[1]},)")
    return np.clip(W @ T, 0.0, 1.0), names


def apply_sensor_atmosphere(reflectance, T_bands):
    """Degrade already-sensor reflectance by per-band gaseous transmittance: rho'_b = rho_b * T_b.

    First-order MULTIPLICATIVE gaseous suppression only — path radiance / adjacency / multiple
    scattering are NOT modelled (this is a band-suppression robustness axis, not radiometric
    correction; state this in the paper). `reflectance` is (..., B); `T_bands` is (B,)."""
    r = np.asarray(reflectance, float)
    T = np.asarray(T_bands, float)
    if r.shape[-1] != T.shape[-1]:
        raise ValueError(f"reflectance last dim {r.shape[-1]} != #bands {T.shape[-1]}")
    return r * T
