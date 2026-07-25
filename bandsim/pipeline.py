"""Compose the physical corruptions per config, returning corrupted bands + intermediates.

Execution order is B (atmosphere) -> C (cirrus) -> A (SRF to sensor bands) -> D (instrument noise):
atmosphere and cirrus act on the hyperspectral axis, THEN SRF maps to the target sensor's (possibly
non-uniform) bands, THEN instrument noise acts on those final bands. The "missing bands" are simply
the sensor bands that exist after SRF (reported via info['band_names'] / info['band_centers_nm']).
Keep everything config-driven and seeded.
"""
from __future__ import annotations
import numpy as np

from . import srf as _srf
from . import cirrus as _cirrus
from . import noise as _noise
from . import atmosphere as _atmos


def simulate(cube, wavelengths_nm, cfg, rng=None):
    """Apply the enabled physical corruptions to a hyperspectral cube.

    Parameters
    ----------
    cube           : (H, W, C_hsi) reflectance/radiance
    wavelengths_nm : (C_hsi,) axis
    cfg            : dict, e.g. {"A": {...}, "B": {...}, "C": {...}, "D": {...}}
    rng            : np.random.Generator (for D); created from cfg['seed'] if None

    Returns
    -------
    out  : corrupted cube (sensor bands if Design A ran, else same #bands)
    info : dict with intermediate artefacts (e.g. band_names, keep_mask)
    """
    if rng is None:
        rng = np.random.default_rng(cfg.get("seed", 0))
    x = np.asarray(cube, float)
    wl_eff = np.asarray(wavelengths_nm, float)                 # effective wavelength of each current channel
    info = {}

    # Carry the config's honesty labels INTO the results. `validation_status` (per design) and
    # `claim_scope` (per spec) previously lived only in YAML comments, so the moment a caller held
    # the returned array the "schematic / not sensor-calibrated" qualifier was gone and the numbers
    # travelled unlabelled. config_runner.build_cfg populates these; hand-built cfgs simply omit them.
    if cfg.get("claim_scope"):
        info["claim_scope"] = cfg["claim_scope"]
    status = {k: cfg[k]["validation_status"] for k in ("A", "B", "C", "D")
              if isinstance(cfg.get(k), dict) and cfg[k].get("validation_status")}
    if status:
        info["validation_status"] = status

    # B: atmosphere (on the hyperspectral axis, before SRF) -- Phase 3
    b = cfg.get("B") or {}
    if b.get("enable"):
        keep = _atmos.hard_mask_absorption_cores(wl_eff) if b.get("hard_mask_cores") else None
        # Pass the ACTUAL axis, not just the band count: a LUT built on a gapless linspace has the
        # same length as the real gapped AVIRIS axis, so a length check alone let a wrong-axis table
        # through and mis-assigned every transmittance value. Design B runs before Design A, so
        # wl_eff here is still the hyperspectral axis the table must have been computed on.
        T = _atmos.load_cached_transmittance(b["cache_npz"], b["cache_key"], n_bands=x.shape[-1],
                                             expected_wavelengths_nm=wl_eff)
        x = _atmos.apply_atmosphere(x, T, keep_mask=keep)
        info["atmos_keep_mask"] = keep

    # C: cirrus (still on hyperspectral axis)
    c = cfg.get("C") or {}
    if c.get("enable"):
        x = _cirrus.apply_cirrus(x, wl_eff, tau=c.get("tau", 0.3))

    # A: SRF convolution to target sensor bands (defines the missing-band set) -- Phase 2
    a = cfg.get("A") or {}
    if a.get("enable", True) and "srf" in a:
        W, names = _srf.build_resampling_matrix(wl_eff, a["srf"])
        x = _srf.apply_srf(x, W)
        wl_eff = W @ wl_eff                                     # SRF-weighted effective band centres (nm)
        info["band_names"] = names
        info["band_centers_nm"] = wl_eff

    # D: instrument noise/striping on the FINAL bands, at their REAL centre wavelengths
    d = cfg.get("D") or {}
    if d.get("enable"):
        snr = _noise.hyperion_like_snr(wl_eff)                 # real per-band centres, NOT a fabricated linspace
        x = _noise.add_band_noise(x, snr, rng)
        # dead columns are Bernoulli per column, so the REALISED count varies per seed (and is 0 in
        # ~23% of seeds at 145 columns / frac=0.01) -- record what actually happened, don't assume.
        x, stripe_info = _noise.add_striping(x, rng, d.get("stripe_eps", 0.02),
                                             d.get("dead_col_frac", 0.01), return_info=True)
        info["striping"] = stripe_info

    return x, info
