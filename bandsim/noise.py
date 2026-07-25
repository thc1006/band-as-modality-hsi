"""Design D — instrument radiometric degradation: per-band SNR + striping + dead detectors.

⚠️ SCHEMATIC / STANDARD-FORM — NOT CALIBRATED TO ANY REAL SENSOR. `hyperion_like_snr()` is a
hand-drawn linear VNIR~500 -> SWIR~50 falloff (the *shape* is standard, the numbers are not a
measured NEdL curve), and `add_striping()` is the textbook multiplicative-gain + dead-column
model. These are adequate for ROBUSTNESS stress-testing (an extra corruption axis) but are NOT
a validated radiometric noise model — EXCLUDE Design D from any physics-claiming result in the
D5 reliability paper (the physics claims rest on real 6S transmittance + real pyspectral SRF,
Designs B & A). If a calibrated model is needed, substitute a per-band measured NEdL/SNR curve.
See docs/review/PAPER_DESIGN.md §5 and the analogous scoping note in bandsim/cirrus.py.

Physics:  rho_obs[:, c, b] = g_c * rho[:, c, b] + N(0, sigma_b^2),  sigma_b = L_b / SNR_b
with occasional g_c = 0 (dead detector columns). Hyperion SNR falls VNIR ~500 -> SWIR ~50.

Pure NumPy, fully implementable (below). Pairs with the destriping literature for validation.
"""
from __future__ import annotations
import numpy as np


def hyperion_like_snr(wavelengths_nm):
    """A schematic per-band SNR curve: high in VNIR (~500), low in SWIR (~50)."""
    wl = np.asarray(wavelengths_nm, float)
    # linear-ish falloff from 500 @ 400nm to 50 @ 2500nm, clipped
    snr = 500.0 - (500.0 - 50.0) * (wl - 400.0) / (2500.0 - 400.0)
    return np.clip(snr, 50.0, 500.0)


def add_band_noise(cube, snr_per_band, rng):
    """Add per-band Gaussian noise sigma_b = signal_b / SNR_b (signal ~ per-band mean).

    SNR is rejected unless strictly positive and finite: SNR=0 divides to sigma=inf and SNR<0
    flips the noise sign, both of which produce a corrupted cube without raising anything."""
    cube = np.asarray(cube, float)
    snr = np.asarray(snr_per_band, float)
    if not np.isfinite(snr).all() or (snr <= 0).any():
        raise ValueError(f"snr_per_band must be finite and > 0 (got range "
                         f"[{np.nanmin(snr):.6g}, {np.nanmax(snr):.6g}], "
                         f"{int((~np.isfinite(snr)).sum())} non-finite)")
    signal = np.abs(cube).mean(axis=tuple(range(cube.ndim - 1)), keepdims=True)
    sigma = signal / snr
    return cube + rng.normal(0.0, 1.0, size=cube.shape) * sigma


def add_striping(cube, rng, stripe_eps=0.02, dead_col_frac=0.01, col_axis=1, return_info=False,
                 *, dead_col_mode="bernoulli"):
    """Multiplicative per-column gain g_c ~ N(1, eps^2) with a fraction of dead (=0) columns.

    Works for ANY ndim (the broadcast shape is built from cube.ndim, not hardcoded to 3-D, which
    made 2-D/4-D inputs raise a broadcasting error). `col_axis` selects the striped axis (default
    1 = W of an (H, W, B) cube).

    `dead_col_mode` chooses how `dead_col_frac` is turned into actual dead columns. The two modes
    are different estimands, not an implementation detail, so the caller must pick deliberately:

    "bernoulli" (default, unchanged) -- `dead_col_frac` is an EXPECTED fraction: each column is
    killed by an INDEPENDENT Bernoulli(dead_col_frac) draw, so the realised count is
    Binomial(ncols, frac) and is frequently zero on small images. At the Indian Pines width (145
    columns) with the default frac=0.01 the chance of NO dead column at all is (1-0.01)^145 = 23.3%
    (measured: 0.2329 over 20k seeds) -- i.e. roughly one seed in four exercises no dead detector
    whatsoever. Correct when the Bernoulli process itself is what you are averaging over.

    "exact" -- `dead_col_frac` is a TARGET fraction realised to the nearest whole column:
    n_dead = floor(frac*ncols + 0.5) columns (round-half-UP, deliberately not np.round's
    round-half-to-even, so the count is predictable from the docstring), drawn uniformly without
    replacement. Use this when `frac` is a SWEEP AXIS rather than a nuisance parameter. Measured on
    145 columns over 5 seeds, the Bernoulli realisation of a nominal 3% is 4.14% of columns with a
    per-seed range of 1.38%-6.21%, and the nominal-5% range (2.76%-7.59%) OVERLAPS it -- so a curve
    plotted against nominal frac is not even ordered by the severity it actually applied. In "exact"
    mode the realised count is a deterministic function of (frac, ncols), so it is identical across
    seeds and the axis orders the conditions by construction.

    Both modes draw AFTER the gain, so with a fixed seed and a fixed `stripe_eps` the dead sets of
    an increasing sweep of `frac` are NESTED (bernoulli: the same uniforms thresholded lower;
    exact: prefixes of the same permutation). A sweep therefore adds dead columns rather than
    resampling them, which keeps per-seed severity monotone in `frac`.

    A nonzero `frac` that rounds to zero columns in "exact" mode RAISES rather than silently
    applying no corruption: a sweep point that is secretly the clean baseline reads as perfect
    robustness. "bernoulli" mode does not raise -- there a zero draw is the model, and it is
    reported instead.

    The REALISED count, indices and mode are reported in `info` (return_info=True) and threaded
    into pipeline.simulate's info dict, making each run auditable after the fact instead of assumed.

    stripe_eps is the std of the gain and must be finite and >= 0. It used to be passed through
    abs(), which silently reinterpreted a negative value as its positive twin (eps=-0.2 gave results
    bit-identical to eps=+0.2), and NaN/Inf passed straight into rng.normal and returned a
    non-finite cube with no error at all.
    """
    cube = np.asarray(cube, float).copy()
    eps = float(stripe_eps)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError(f"stripe_eps must be finite and >= 0 (it is the std of the column gain), "
                         f"got {stripe_eps!r}")
    frac = float(dead_col_frac)
    if not np.isfinite(frac) or not (0.0 <= frac <= 1.0):
        raise ValueError(f"dead_col_frac must be finite and in [0, 1], got {dead_col_frac!r}")
    if dead_col_mode not in ("bernoulli", "exact"):
        # An unrecognised mode must NOT fall through to the default: silently getting Bernoulli
        # when "exact" was requested (a typo, a renamed constant) is precisely the failure this
        # parameter exists to prevent, and it leaves no trace in the output.
        raise ValueError(f"dead_col_mode must be 'bernoulli' or 'exact', got {dead_col_mode!r}")
    if not (-cube.ndim <= col_axis < cube.ndim):
        raise ValueError(f"col_axis {col_axis} out of range for ndim {cube.ndim}")
    ncols = cube.shape[col_axis]
    g = rng.normal(1.0, eps, size=ncols)
    if dead_col_mode == "bernoulli":
        idx = np.flatnonzero(rng.random(ncols) < frac)
    else:
        n_dead = int(np.floor(frac * ncols + 0.5))          # round-half-UP (see docstring)
        if ncols and frac > 0.0 and n_dead == 0:
            raise ValueError(
                f"dead_col_mode='exact' with dead_col_frac={frac!r} on {ncols} columns rounds to 0 "
                f"dead columns, i.e. a nonzero corruption level that applies no corruption and "
                f"would plot as perfect robustness. One column is {1.0 / ncols:.4g} of this axis, "
                f"so the smallest representable nonzero fraction is {0.5 / ncols:.4g}. Pass at "
                f"least that, or dead_col_mode='bernoulli' if an expected (possibly zero) count is "
                f"genuinely the model you want.")
        # permutation()[:n], NOT rng.choice(replace=False): slicing one shuffle guarantees that the
        # sets for increasing n are nested, so a sweep ADDS dead columns instead of resampling them.
        # choice() uses a partial-shuffle whose size-4 result is not a prefix of its size-7 result.
        idx = np.sort(rng.permutation(ncols)[:n_dead])
    g[idx] = 0.0
    shape = [1] * cube.ndim
    shape[col_axis] = ncols
    out = cube * g.reshape(shape)
    if not return_info:
        return out
    return out, {"n_cols": int(ncols), "dead_col_count": int(idx.size),
                 "dead_col_indices": idx.tolist(),
                 "dead_col_mode": dead_col_mode,
                 # what was ASKED for: an expected fraction under "bernoulli", a target fraction
                 # rounded to the nearest column under "exact". The pair (requested, realised) is
                 # only self-explanatory once the mode is beside it.
                 "dead_col_frac_requested": frac,
                 "dead_col_frac_realised": float(idx.size) / ncols if ncols else 0.0,
                 "stripe_eps": eps, "col_axis": int(col_axis)}
