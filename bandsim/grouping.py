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


def validate_partition(groups, n_bands, *, require_full_cover=True):
    """Fail closed on an invalid band grouping (code-review r2 §8.1). Each group must be a NON-EMPTY 1-D
    INTEGER index array; indices in [0, n_bands); UNIQUE within a group and DISJOINT across groups; with
    require_full_cover the groups must PARTITION all n_bands. Otherwise numpy silently reads a negative index
    as the last band, and overlaps / duplicates / gaps corrupt the group centres and membership matrix with
    no error anywhere."""
    seen = set()
    for gi, idx in enumerate(groups):
        a = np.asarray(idx)
        if a.ndim != 1 or a.size == 0:
            raise ValueError(f"group {gi}: expected a non-empty 1-D index array (got shape {a.shape})")
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"group {gi}: band indices must be integers (got dtype {a.dtype})")
        lo, hi = int(a.min()), int(a.max())
        if lo < 0 or hi >= n_bands:
            raise ValueError(f"group {gi}: band index out of [0,{n_bands}) (got [{lo},{hi}]) -- "
                             f"a negative index is silently read as the last band")
        s = {int(i) for i in a}
        if len(s) != a.size:
            raise ValueError(f"group {gi}: duplicate band indices within the group")
        if s & seen:
            raise ValueError(f"group {gi}: overlaps an earlier group on band(s) {sorted(s & seen)}")
        seen |= s
    if require_full_cover and seen != set(range(n_bands)):
        raise ValueError(f"grouping does not partition all {n_bands} bands "
                         f"(missing {sorted(set(range(n_bands)) - seen)})")


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
    validate_partition(groups, int(wl.size))          # negative/overlap/duplicate/gap/empty (r2 §8.1)
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
    validate_partition(groups, int(n_bands), require_full_cover=False)   # valid, disjoint indices (r2 §8.1)
    M = np.zeros((len(groups), n_bands), bool)
    for g, idx in enumerate(groups):
        M[g, idx] = True
    return M
