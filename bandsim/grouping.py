"""Spectral grouping — the 'band-as-modality' unit. Each contiguous band group = one modality.

A missing spectral group = a missing modality (the paper's core analogy). Grouping is the
atomic unit for the missing-band degradation experiment (Phase 2) and SGMAE group masking.
"""
from __future__ import annotations
import numpy as np


def contiguous_groups(n_bands, n_groups):
    """Split n_bands into n_groups contiguous, (near-)equal index groups.

    Returns list[np.ndarray] of band indices; the remainder is distributed to the first groups.
    Requires 1 <= n_groups <= n_bands (else this would ZeroDivision on n_groups=0 or emit empty
    groups with NaN centres for n_groups > n_bands).
    """
    # Reject non-integral counts BEFORE truncating: int(10.9) silently yields 10 groups, so a
    # caller that computed n_groups by division (e.g. n_bands/width) would get a different grouping
    # than it asked for and never find out. bool is excluded too -- contiguous_groups(200, True)
    # used to return a single group, which is the exact degenerate case Phase 2 must not run in.
    for name, v in (("n_bands", n_bands), ("n_groups", n_groups)):
        if isinstance(v, bool) or (v != int(v)):
            raise ValueError(f"{name} must be an integer, got {v!r} "
                             f"(int() would silently truncate it and change the grouping)")
    n_bands = int(n_bands); n_groups = int(n_groups)
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    if not (1 <= n_groups <= n_bands):
        raise ValueError(f"n_groups must be in [1, n_bands={n_bands}], got {n_groups} "
                         f"(n_groups>n_bands -> empty groups / NaN group centres)")
    base = n_bands // n_groups
    rem = n_bands % n_groups
    groups = []
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < rem else 0)
        groups.append(np.arange(start, start + size))
        start += size
    return groups


def group_center_wavelengths(wavelengths_nm, groups):
    """Mean wavelength (nm) per group — used as the group token's positional identity.

    The axis LENGTH is checked against the span of the grouping. A wavelength axis LONGER than the
    grouped cube indexes fine and returns plausible-looking centres that are silently WRONG — the
    exact failure you get by handing the nominal 220-band AVIRIS axis (`bandsim.io._AVIRIS_220`) to
    a grouping built on the 200-band 'corrected' cube, which sits one line away from `AVIRIS_WL_NM`
    in this package. That mistake mislabels every group centre and corrupts the wavelength PE with
    no error anywhere. (A SHORTER axis already raised IndexError, so only the long side was silent.)
    Non-finite wavelengths are rejected for the same reason: a NaN in the axis produces a NaN group
    centre, which propagates into the PE and yields NaN logits with no traceback.
    """
    wl = np.asarray(wavelengths_nm, float)
    if any(np.asarray(idx).size == 0 for idx in groups):
        raise ValueError("empty group has no centre wavelength — check the grouping")
    if not np.isfinite(wl).all():
        raise ValueError(f"wavelengths_nm must be finite (got {int((~np.isfinite(wl)).sum())} "
                         f"non-finite entries) — a NaN centre silently propagates into the group PE")
    spanned = int(max(int(np.asarray(idx).max()) for idx in groups)) + 1
    if wl.size != spanned:
        raise ValueError(
            f"wavelength axis has {wl.size} bands but the grouping spans {spanned} bands — "
            f"a longer axis silently yields WRONG group centres (e.g. the nominal 220-band AVIRIS "
            f"axis used with the 200-band 'corrected' cube). Pass the axis of the cube you grouped.")
    return np.array([wl[np.asarray(idx)].mean() for idx in groups])


def build_group_matrix(n_bands, groups):
    """Boolean (n_groups, n_bands) membership matrix, for fast group<->band ops."""
    M = np.zeros((len(groups), n_bands), bool)
    for g, idx in enumerate(groups):
        M[g, idx] = True
    return M
