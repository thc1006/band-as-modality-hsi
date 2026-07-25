"""Design C — thin-cirrus per-band Beer-Lambert corruption.

⚠️ ILLUSTRATIVE ONLY — NOT PHYSICALLY VALIDATED. `extinction_shape()` below is a HAND-CRAFTED
Gaussian placeholder near 1.375 um, NOT a real ice-cloud (Mie / T-matrix / Baum-Yang) optical
model. It is EXCLUDED from any physics-claiming result (esp. the D5 reliability paper): for real
cirrus use the operational Sentinel-2 L1C B10 (1.375 um) cirrus band and the real L1C->L2A
transition instead. Kept only for the legacy Phase-4 robustness ablation.
See docs/review/PAPER_DESIGN.md §5 (physics hardening).

Physics:  rho_obs(lam) = rho_surface(lam) * exp(-tau * k(lam)) + rho_cirrus_path(lam; tau)

tau = cirrus optical depth; k(lam) = spectral extinction shape. In THIS schematic model the
extinction k is LARGEST near 1.375 um, so that band is ATTENUATED (darker) while a small additive
haze term brightens the VNIR (<1000 nm) -- a band-selective corruption. It does NOT reproduce the
operational L1C B10 "cirrus band lights up" radiance (the docstring previously claimed it did); for
that, use the real Sentinel-2 L1C B10 band. Robustness stress-test only, not radiometric.

Fully implementable in ~30 lines of NumPy (below). Validity: analytic tau*k(lam) is a
first-order empirical model of scattering — adequate for robustness stress-testing, not
for radiometric retrieval (state this in the paper).
"""
from __future__ import annotations
import numpy as np

TAU_GRID = [0.0, 0.1, 0.3, 0.6, 1.0]


def extinction_shape(wavelengths_nm):
    """k(lam): a simple, monotone-ish spectral extinction shape (normalised ~1 in VNIR).

    Placeholder physical shape: cirrus extinction is relatively flat in VNIR and rises
    with a broad feature near 1.375 um. Replace with a literature curve if available.
    """
    wl = np.asarray(wavelengths_nm, float)
    k = 0.8 + 0.4 * np.exp(-0.5 * ((wl - 1375) / 120) ** 2)  # bump near cirrus band
    return k


def apply_cirrus(cube, wavelengths_nm, tau, path_gain=0.02):
    """Apply multiplicative transmittance + additive cirrus path term.

    tau=0 returns the input unchanged (identity) — checked by tests.
    """
    if tau < 0:
        raise ValueError(f"tau (cirrus optical depth) must be >= 0, got {tau} (tau<0 amplifies non-physically)")
    cube = np.asarray(cube, float)
    wl = np.asarray(wavelengths_nm, float)
    if cube.shape[-1] != wl.shape[-1]:
        raise ValueError(f"cube last dim {cube.shape[-1]} != wavelengths {wl.shape[-1]}")
    if tau == 0:
        return cube.copy()
    k = extinction_shape(wl)
    trans = np.exp(-tau * k)                     # multiplicative attenuation
    path = path_gain * tau * (wl < 1000)         # additive VNIR haze/brightening
    return cube * trans + path
