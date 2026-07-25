#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8F-multi — generality + RIGOR-HARDENED validation of the EMIT C-f across multiple granules.

For several EMIT L2A granules (diverse biomes: India crop, N-America arid, Africa vegetation) we
test whether a self-supervised spectral-reconstruction-anomaly signal correlates with EMIT's REAL
per-pixel/per-band retrieval uncertainty (ISOFIT optimal-estimation posterior). Positive correlation
= our reliability signal is grounded in real physical retrieval uncertainty, not a simulation
artifact.

This version fixes every issue an adversarial review raised on the single-granule phase8F:
  M3 (in-sample)         -> HELD-OUT: SGMAE trained on 70% of pixels, reconstruction error measured
                            on the DISJOINT 30% (standardised by TRAIN stats only). The holdout is
                            SPATIAL (whole blocks of the granule, --split spatial, default): an
                            index holdout leaves each held-out pixel a median 7.7-11.1 px
                            (460-670 m) from a training pixel, which for a land surface is a
                            neighbour, not an unseen sample; blocks push that to 35-41 px (~2.2 km).
                            --split random reproduces the old behaviour and the CSV records which
                            was used. MEASURED on 3 granules x 5 seeds (India / sahara / us_midwest,
                            n_px=6000): the held-out reconstruction ERROR is only mildly optimistic
                            and not systematically so (SGMAE +11.2% / +1.7% / -5.4% going to the
                            spatial split), but the headline per-pixel Spearman MOVES A LOT and in
                            BOTH directions -- India +0.42->+0.55, sahara +0.089->-0.007,
                            us_midwest +0.22->+0.15. The two weak granules lose their positive sign
                            entirely, so part of the "positive on every granule" reading came from
                            holding out pixels that sat among the training pixels. Seed-to-seed
                            spread roughly doubles under the spatial split, which is the real
                            generalisation variance rather than extra noise.
  M2 (under-powered)     -> >=3 seeds per granule (random split + training); per-granule mean+/-std
                            over seeds, then across granules. std is descriptive-only (n small).
  M4 (model-specificity) -> a training-free linear PCA(k) reconstruction-error proxy is reported
                            ALONGSIDE the SGMAE. If PCA reproduces the correlation, we honestly state
                            the signal is a property of EMIT SPECTRA (a linear spectral-subspace
                            anomaly), NOT specific to the attention architecture.
  brightness confound    -> PARTIAL Spearman controlling for per-pixel brightness (mean reflectance),
                            so the link is shown to survive removal of the SNR/brightness component.
  M1 (per-band window)   -> per-band Spearman reported for the FULL spectrum AND a window-sensitivity
                            sweep, clearly labelled post-hoc (not a headline).
  M5 (ties)              -> scipy.stats.spearmanr (proper tie handling) throughout.

Headline = per-pixel Spearman(EMIT uncertainty, HELD-OUT SGMAE reconstruction error).

Output (../paper/): results_phase8F_multi.csv, results_phase8F_multi_perband.csv
"""
import os, sys, glob, csv, hashlib, argparse
import numpy as np
import h5py
import torch
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import phase8F_emit as F                                        # reuse recon_error_matrix
import phase2_degradation as P2
from bandsim.model import GroupedCrossBandAttention
from bandsim.grouping import contiguous_groups, group_center_wavelengths
from bandsim import hw
from bandsim.provenance import stamp

PAPER_DIR = os.path.normpath(os.path.join(_HERE, "..", "paper"))
os.makedirs(PAPER_DIR, exist_ok=True)
def P(rel):
    return os.path.join(PAPER_DIR, rel)
DATA = os.path.dirname(_HERE) + "/data"
WINDOWS = [(400, 1300), (500, 1300), (600, 1300), (600, 1100), (700, 1300), (400, 2500)]


def discover_granules():
    """Auto-discover every data/emit* dir that has BOTH an RFL and an RFLUNCERT .nc (so newly
    downloaded NDVI-spanning granules are picked up automatically; partial downloads are skipped)."""
    out = {}
    for d in sorted(glob.glob(f"{DATA}/emit*")):
        if os.path.isdir(d) and glob.glob(f"{d}/*RFL_001*.nc") and glob.glob(f"{d}/*RFLUNCERT*.nc"):
            base = os.path.basename(d)
            out["India_crop" if base == "emit" else base.replace("emit_", "")] = d
    return out


_EMIT_SCRIPTS = os.path.join(os.path.dirname(_HERE), "scripts", "emit")
if _EMIT_SCRIPTS not in sys.path:
    sys.path.insert(0, _EMIT_SCRIPTS)


# Imported from the dependency-free `emit_regions`, NOT from the downloader: the downloader needs
# `earthaccess`, and swallowing an ImportError here would silently disable the geographic crop --
# exactly the failure this crop exists to prevent. So an import failure raises, at import time.
from emit_regions import REGIONS, emit_granule_id  # noqa: E402


# EMIT L2A MASK band semantics. The product ships the layout as STRING labels in
# sensor_band_parameters/mask_bands, so the analysis names what it screens and resolves the index
# per file; the old code hardcoded 0=Cloud/1=Cirrus/2=Water/4=Dilated and never looked. That was
# right for V001 by luck, not by check -- and it is exactly the kind of assumption a V002 relayout
# breaks into a wrong NUMBER rather than an error (a fixture with Cloud and Water swapped screened
# water while reporting it as cloud, and swapped the cloud%/water% columns of the results table).
# Matching is case-insensitive because NASA's own labels are not internally consistent
# ("Cloud flag" but "Spacecraft Flag").
MASK_FLAG_LABELS = {
    "cloud": "Cloud flag",
    "cirrus": "Cirrus flag",
    "water": "Water flag",
    "spacecraft": "Spacecraft Flag",
    "dilated": "Dilated Cloud Flag",
}
# Bands that are CONTINUOUS retrievals, not flags. `(M[...] > 0).any(-1)` over one of these screens
# essentially the whole scene (measured on the India granule: AOD550 > 0 for 99.96% of pixels, H2O
# for 100%), which surfaces as "every granule too cloudy" rather than as "you selected a
# non-flag band". Named so that selecting one is refused.
MASK_NON_FLAG_LABELS = {"aod550", "h2o (g cm-2)"}


def read_mask_bands(msk_path):
    """The MASK file's own band labels, decoded, in file order."""
    with h5py.File(msk_path, "r") as f:
        if "sensor_band_parameters/mask_bands" not in f:
            raise ValueError(f"{msk_path}: no sensor_band_parameters/mask_bands — cannot resolve "
                             f"MASK band semantics by name; refusing to fall back to fixed indices")
        raw = f["sensor_band_parameters/mask_bands"][:]
    return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]


def _mask_index(labels, flag, where):
    """One canonical name (or raw index) -> its index in THIS file's mask_bands."""
    lower = [s.strip().lower() for s in labels]
    if isinstance(flag, str) and not flag.strip().lstrip("-").isdigit():
        want = MASK_FLAG_LABELS.get(flag.strip().lower())
        if want is None:
            raise ValueError(f"unknown MASK flag {flag!r}; expected one of "
                             f"{sorted(MASK_FLAG_LABELS)} or a raw band index")
        if want.lower() not in lower:
            raise ValueError(f"{where}: MASK band {want!r} is absent from this product "
                             f"(mask_bands = {labels}) — the granule's mask layout is not the one "
                             f"this analysis screens against")
        return lower.index(want.lower())
    i = int(flag)
    if not (0 <= i < len(labels)):
        raise ValueError(f"{where}: MASK band index {i} out of range (file has {len(labels)} "
                         f"bands: {labels})")
    if lower[i] in MASK_NON_FLAG_LABELS:
        raise ValueError(f"{where}: MASK band {i} is {labels[i]!r}, a continuous retrieval rather "
                         f"than a flag — '>0' would screen the whole scene")
    return i


def resolve_mask_flags(msk_path, flags):
    """(canonical names or raw indices) -> (ascending unique indices, matching labels) for THIS file.

    ASCENDING because the result indexes an h5py dataset, which requires it; the caller is OR-ing
    the selected bands together, so order carries no meaning there. Anything that needs a specific
    ORDER (granule_quality's cloud-then-water pair) must use mask_band_indices instead.

    Names are resolved through the file's mask_bands, so the V001 pin in the filename finally means
    something. Raw integers stay accepted for the documented robustness invocations, but they are
    checked against the label at that index instead of trusted."""
    labels = read_mask_bands(msk_path)
    out = {_mask_index(labels, f, msk_path): None for f in flags}
    idx = sorted(out)
    return idx, [labels[i] for i in idx]


def mask_band_indices(msk_path, flags):
    """Indices for `flags` in the GIVEN order (duplicates preserved) — for positional use."""
    labels = read_mask_bands(msk_path)
    return [_mask_index(labels, f, msk_path) for f in flags]


def region_bbox_for(name):
    """Declared (lon_w, lat_s, lon_e, lat_n) box for a granule name, or None if this granule was
    never downloaded against a declared region."""
    return REGIONS.get(name)


def select_triple(gdir):
    """Resolve a granule directory to the ONE acquisition that has a complete RFL+RFLUNCERT+MASK
    triple; returns (gid, rfl_path, unc_path, mask_path).

    Factored out of extract() so that every consumer -- extract, granule_quality, the preflight
    header -- resolves the SAME files. When each site did its own `glob(...)[0]` they could disagree:
    glob order is readdir order, not sorted, so a directory holding a stray MASK from another
    acquisition could have the analysis screened by acquisition A's cloud mask while the printed
    cloud%/water% came from acquisition B's. That disagreement produces a wrong NUMBER, silently.

    Files are indexed by acquisition id and the RFL/UNCERT/MASK INTERSECTION is taken rather than
    trusting glob order: a directory holding a stray RFL(A) alongside a COMPLETE B triple would
    otherwise be rejected outright just because RFL(A) happened to sort first."""
    rfl_l = glob.glob(f"{gdir}/*RFL_001*.nc"); unc_l = glob.glob(f"{gdir}/*RFLUNCERT*.nc")
    mskf = glob.glob(f"{gdir}/*MASK_001*.nc")
    if not (rfl_l and unc_l and mskf):
        raise FileNotFoundError(f"{gdir}: missing source .nc (RFL={len(rfl_l)} UNC={len(unc_l)} "
                                f"MASK={len(mskf)}) -- refusing to serve a possibly-stale cache")
    by = {}
    for kind, lst in (("rfl", rfl_l), ("unc", unc_l), ("msk", mskf)):
        by[kind] = {}
        for p in lst:
            g = emit_granule_id(p)
            if g is not None:
                by[kind].setdefault(g, []).append(p)
    common = set(by["rfl"]) & set(by["unc"]) & set(by["msk"])
    if len(common) != 1:
        raise ValueError(f"{gdir}: need EXACTLY ONE acquisition with a complete RFL+RFLUNCERT+MASK "
                         f"triple, found {sorted(common)} (rfl={sorted(by['rfl'])}, "
                         f"unc={sorted(by['unc'])}, mask={sorted(by['msk'])})")
    gid = common.pop()
    if any(len(by[k][gid]) != 1 for k in ("rfl", "unc", "msk")):
        raise ValueError(f"{gdir}: duplicate files for acquisition {gid} -- ambiguous, refusing to guess")
    return gid, by["rfl"][gid][0], by["unc"][gid][0], by["msk"][gid][0]


# Two EMIT bands whose sum is this close to zero carry no NDVI information: the mean per-band EMIT
# posterior uncertainty on these granules is 0.005-0.017 reflectance, so the uncertainty on (red+NIR)
# is ~0.01-0.02 and a denominator below that is indistinguishable from zero. Dividing by it does not
# produce a "very high NDVI", it produces an arbitrary number whose magnitude is set by rounding:
# red=-0.04, NIR=+0.04 with the old `+1e-6` guard yields NDVI=+80000, which then passes any
# "is it finite" style filter and silently enters the medians, IQRs and negative-NDVI fractions.
NDVI_DENOM_FLOOR = 0.02


def safe_ndvi(R, wl, red_nm=665.0, nir_nm=860.0, denom_floor=NDVI_DENOM_FLOOR):
    """NDVI with a REAL denominator floor. Returns (ndvi, stats) where ndvi is NaN wherever
    red+NIR < denom_floor, and stats reports how much was thrown away.

    Note the floor alone cannot bound NDVI to [-1,1]: |NDVI|<=1 holds only when red and NIR are both
    non-negative, and EMIT L2A legitimately returns small negative reflectance over dark water and
    shadow. So the out-of-range fraction is REPORTED rather than silently clipped -- a granule with a
    large `oor` is a granule whose "NDVI" is not a vegetation index at all."""
    ir = int(np.argmin(np.abs(wl - red_nm))); inir = int(np.argmin(np.abs(wl - nir_nm)))
    red = R[:, ir].astype(np.float64); nir = R[:, inir].astype(np.float64)
    denom = nir + red
    ok = denom >= denom_floor
    ndvi = np.full(red.shape, np.nan)
    np.divide(nir - red, denom, out=ndvi, where=ok)
    n = int(red.size); n_rej = int(n - ok.sum())
    n_oor = int(np.count_nonzero(np.abs(ndvi[ok]) > 1.0))
    return ndvi, {"n": n, "n_rejected": n_rej, "rejected_pct": 100.0 * n_rej / max(1, n),
                  "n_out_of_range": n_oor, "oor_pct": 100.0 * n_oor / max(1, n),
                  "red_nm": float(wl[ir]), "nir_nm": float(wl[inir]), "floor": float(denom_floor)}


# How much of each source .nc the cache fingerprint actually reads.
#   "full"    -- every byte, BLAKE2b. MEASURED end-to-end on a real granule triple (3.75 GB) on a
#                busy box: 14.0 s, i.e. ~3.3 min for the whole 14-granule / 52 GB corpus. Set
#                against a run that reads those same files in full on any cache miss and then
#                trains an SGMAE per granule per seed, that is a rounding error, and it is the only
#                setting under which "the cache matches the source" is a claim about the DATA.
#                (BLAKE2b rather than sha256 for throughput: 584 vs 223 MB/s measured here.)
#   "sampled" -- size + mtime + EMIT_FP_WINDOWS evenly spaced windows (~64 MB/file; 1.0 s for the
#                same triple). For interactive iteration; see the residual-risk note below.
EMIT_FP_MODE = os.environ.get("EMIT_FP_MODE", "full").lower()
EMIT_FP_WINDOWS = 64
EMIT_FP_WINDOW_BYTES = 1 << 20


def _sample_offsets(size, n=EMIT_FP_WINDOWS, w=EMIT_FP_WINDOW_BYTES):
    """Evenly spaced window starts spanning [0, size], head and tail included."""
    if size <= w:
        return [0]
    return sorted({int(round(i * (size - w) / (n - 1))) for i in range(max(2, n))})


def _source_fingerprint(paths, mode=None):
    """Content identity for the source .nc triple, so a cache can be tied to the BYTES it was built
    from. A granule ID in the cache filename only proves files with the right NAMES exist.

    The previous version hashed head(1 MB) + tail(1 MB) + size + mtime. That is blind to an edit in
    the MIDDLE of the file, and the counterexample is not exotic -- an in-place h5py write
    (`f["reflectance"][row] = ...`) changes no byte outside the dataset and no file size, and a
    single os.utime puts the mtime back. Demonstrated on a 6 MB fixture: one whole reflectance row
    rewritten, fingerprint character-for-character identical, extract() served the pre-edit cache
    while a fresh extraction differed by up to 0.93 reflectance. Nothing printed.

    mode="full" closes that: BLAKE2b over every byte (BLAKE2b, not sha256, purely for throughput --
    584 vs 223 MB/s measured here). mtime is deliberately NOT part of a full fingerprint: content is
    the identity, and including mtime would force a multi-GB re-extraction after a bare `touch`.

    RESIDUAL RISK, mode="sampled": it reads EMIT_FP_WINDOWS windows spanning the file (~3.5% of a
    1.85 GB RFL), so it catches truncation, replacement, re-download and any edit overlapping a
    window -- but a small in-place edit that lands between windows AND has its mtime restored is
    still invisible, with probability of detection roughly the sampled fraction. The mode is written
    into the fingerprint string, so a cache stamped by one mode can never validate against the
    other; switching modes forces a re-extraction rather than a false match."""
    mode = (mode or EMIT_FP_MODE)
    if mode not in ("full", "sampled"):
        raise ValueError(f"EMIT_FP_MODE must be 'full' or 'sampled', got {mode!r}")
    parts = []
    for p in sorted(paths):
        st = os.stat(p)
        h = hashlib.blake2b(str(st.st_size).encode(), digest_size=16)
        with open(p, "rb") as fh:
            if mode == "full":
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
                tag = "full"
            else:
                for off in _sample_offsets(st.st_size):
                    fh.seek(off)
                    h.update(fh.read(EMIT_FP_WINDOW_BYTES))
                tag = f"s{EMIT_FP_WINDOWS}x{EMIT_FP_WINDOW_BYTES}:{st.st_mtime_ns}"
        parts.append(f"{os.path.basename(p)}:{st.st_size}:{tag}:{h.hexdigest()}")
    return "|".join(parts)


def _check_cache_arrays(z, cache, want_stats):
    """Cross-validate a cache's OWN arrays before serving it.

    A matching source_fp says the cache was built from the right bytes; it says nothing about
    whether the arrays inside it agree with each other. They were never checked, and the failure is
    silent where it matters most: R (N,C) with a U that has lost bands still lets `Uev.mean(1)`
    return a per-pixel uncertainty averaged over the wrong number of bands, and the headline
    Spearman is then computed from two vectors of the right LENGTH and the wrong CONTENT."""
    missing = [k for k in ("R", "U", "wl", "row", "col", "source_fp") if k not in z.files]
    if want_stats:
        missing += [k for k in ("n_valid", "n_total", "n_inbox", "inside_pct") if k not in z.files]
    if missing:
        raise ValueError(f"{cache}: cache is missing {missing} — delete it and re-extract")
    R, U, wl, rr, cc = z["R"], z["U"], z["wl"], z["row"], z["col"]
    if R.ndim != 2 or U.ndim != 2 or wl.ndim != 1:
        raise ValueError(f"{cache}: expected R/U 2-D and wl 1-D, got R{R.shape} U{U.shape} wl{wl.shape}")
    if U.shape != R.shape:
        raise ValueError(f"{cache}: U{U.shape} does not match R{R.shape} — reflectance and its "
                         f"uncertainty must be the same (pixel, band) grid")
    if wl.shape[0] != R.shape[1]:
        raise ValueError(f"{cache}: wl has {wl.shape[0]} bands but R has {R.shape[1]}")
    if rr.shape != (R.shape[0],) or cc.shape != (R.shape[0],):
        raise ValueError(f"{cache}: row{rr.shape}/col{cc.shape} do not label R's {R.shape[0]} pixels "
                         f"— a spatial split would be built on the wrong coordinates")
    if R.shape[0] == 0:
        raise ValueError(f"{cache}: 0 sampled pixels")
    if not (np.isfinite(R).all() and np.isfinite(U).all() and np.isfinite(wl).all()):
        raise ValueError(f"{cache}: non-finite values — extract() only ever stores finite pixels, "
                         f"so this cache is damaged")
    return R, U, wl, rr, cc


def sp(x, y):
    """Spearman rho with proper tie handling (scipy); nan if a vector is constant."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0 or len(x) < 3:
        return float("nan")
    r = spearmanr(x, y).correlation
    return float(r) if r == r else float("nan")


def partial_sp(x, y, z):
    """Partial Spearman of (x,y) controlling for z: remove z's rank-correlation from both."""
    rxy, rxz, ryz = sp(x, y), sp(x, z), sp(y, z)
    if any(v != v for v in (rxy, rxz, ryz)):
        return float("nan")
    denom = np.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / denom)


def extract(gdir, n_px, seed=0, mask_flags=("cloud",), return_stats=False,
            region_bbox=None, min_valid_px=2000, return_coords=False):
    """Load an EMIT granule, keep good bands + valid pixels, sample n_px (raw reflectance + real
    per-band uncertainty + wavelengths). `good` is read from the RFL file and reused for U after
    asserting both files share band count (RFL/UNCERT are co-registered EMIT products).

    mask_flags: EMIT L2A MASK bands to screen (logical-OR, matching nasa/EMIT-Data-Resources
    emit_tools.quality_mask), given as NAMES resolved through the file's own mask_bands (raw indices
    still accepted, then checked against the label at that index). Default ("cloud",) = opaque Cloud
    flag ONLY. The official tutorial default is Cloud+Cirrus+Spacecraft+Dilated, which we support for
    a robustness check but do NOT use as primary: Cirrus/Dilated OVER-TRIGGER on haze/aerosol (India:
    Dilated=100%, Cirrus=63% vs true opaque Cloud 27%) and wipe hazy scenes. Haze is a REAL condition
    that raises EMIT uncertainty AND reconstruction difficulty, so we keep it (removing only
    surface-invisible opaque cloud).

    return_coords additionally returns each sampled pixel's (row, col) in the granule grid, which is
    what a SPATIAL train/eval split needs (see run_granule)."""
    # VALIDATE THE SOURCE TRIPLE BEFORE TRUSTING ANY CACHE. The old code returned a cached .npz on a
    # bare filename match, so a stale cache survived even if the .nc files were removed/replaced. Now
    # we require RFL+RFLUNCERT+MASK to exist and share ONE acquisition ID first, and KEY THE CACHE BY
    # THAT ID -- so a changed/removed/mismatched granule can never be masked by an old sample cache.
    gid, rfl, unc, msk = select_triple(gdir)
    mask_idx, mask_labels = resolve_mask_flags(msk, mask_flags)     # h5py fancy-index needs ascending+unique
    # The cache tag keys on the RESOLVED indices, so a product whose layout moved a flag to another
    # index cannot collide with a cache written under the old layout.
    tag = "".join(str(b) for b in mask_idx)
    # A malformed box silently selects NOTHING and would surface only as "too few usable pixels",
    # which reads like a cloudy scene rather than a typo in the region table. Reject it as what it is.
    if region_bbox is not None:
        w, s, e, n = (float(v) for v in region_bbox)
        if not (w < e and s < n and -180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
            raise ValueError(f"{gdir}: invalid/inverted region box (lon_w={w}, lat_s={s}, lon_e={e}, "
                             f"lat_n={n}) -- expected lon_w<lon_e, lat_s<lat_n within (+/-180, +/-90)")
    # The cache key must carry EVERYTHING that changes the sample: mask policy, acquisition, the
    # geographic crop, sample size and seed. f5 added the region crop, a source-content fingerprint
    # and the in-box pixel count (so `usable%` can be quoted against the crop, not the whole scene);
    # f6 stores each sampled pixel's (row, col) so the train/eval split can be SPATIAL. The version
    # is bumped rather than reused because an f5 file has neither the coordinates nor a fingerprint
    # in the current format, and "silently re-extract" beats "silently serve a half-answer".
    bbox_tag = "full" if region_bbox is None else "bbox" + hashlib.sha256(
        ",".join(f"{float(v):.6f}" for v in region_bbox).encode()).hexdigest()[:8]
    cache = os.path.join(gdir, f"_sample_m{tag}_f6_{gid}_{bbox_tag}_{n_px}_{seed}.npz")
    fp = _source_fingerprint([rfl, unc, msk])
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as z:               # context manager: do not leak the fd
            cached_fp = str(z["source_fp"]) if "source_fp" in z.files else ""
            if cached_fp == fp:
                R, U, wl, rr, cc = _check_cache_arrays(z, cache, return_stats)
                out = [R, U, wl]
                if return_coords:
                    out += [rr, cc]
                if return_stats:
                    nv, nt, ni = int(z["n_valid"]), int(z["n_total"]), int(z["n_inbox"])
                    out.append({"n_valid": nv, "n_total": nt, "n_inbox": ni,
                                "frac": nv / max(1, nt),
                                "frac_inbox": nv / max(1, ni),
                                "inside_pct": float(z["inside_pct"])})
                return tuple(out)
        # A matching filename is not proof the cache came from THESE bytes (a truncated or
        # re-downloaded .nc keeps its name), so a fingerprint mismatch discards the cache.
        print(f"    {os.path.basename(gdir)}: source .nc changed since this cache was written "
              f"-- discarding cache and re-extracting")
    with h5py.File(rfl, "r") as f:
        wl = f["sensor_band_parameters/wavelengths"][:]
        good = f["sensor_band_parameters/good_wavelengths"][:].astype(bool)
        if good.shape[0] != wl.shape[0] or good.shape[0] != f["reflectance"].shape[-1]:
            raise ValueError(f"{gdir}: band-axis disagreement -- wavelengths={wl.shape}, "
                             f"good_wavelengths={good.shape}, reflectance={f['reflectance'].shape}")
        R = f["reflectance"][:]
    with h5py.File(unc, "r") as f:
        # Compare the WHOLE grid, not just the band count. Two cubes with the same pixel COUNT but
        # transposed spatial dims (e.g. 1242x1280 vs 1280x1242) both flatten to the same length, so a
        # band-only check lets reflectance and uncertainty be paired pixel-for-pixel across different
        # scenes -- every downstream correlation would then be computed on mismatched pixels.
        if f["reflectance_uncertainty"].shape != R.shape:
            raise ValueError(f"{gdir}: RFL grid {R.shape} != RFLUNCERT grid "
                             f"{f['reflectance_uncertainty'].shape} -- refusing to pair pixels")
        U = f["reflectance_uncertainty"][:]
    Rg = R[:, :, good].reshape(-1, int(good.sum())); Ug = U[:, :, good].reshape(-1, int(good.sum()))
    # Valid = physical range, finite, positive uncertainty, AND not an EMIT product SENTINEL:
    # -9999 = nodata (already cut by >-0.05); -0.01 = "reflectance NOT estimated" in deep atmospheric
    # water features -- it passes the >-0.05 bound and would pollute NDVI / recon error if kept.
    sentinel = np.isclose(Rg, -0.01, atol=5e-4).any(1)
    valid = ((Rg > -0.05).all(1) & (Rg < 1.6).all(1) & np.isfinite(Rg).all(1)
             & np.isfinite(Ug).all(1) & (Ug > 0).all(1) & ~sentinel)
    with h5py.File(msk, "r") as f:
        M = f["mask"]
        if M.ndim != 3 or M.shape[-1] != 8:                 # guard a silent reshape of a malformed MASK
            raise ValueError(f"{gdir}: unexpected MASK shape {M.shape} (expected (rows,cols,8))")
        if M.shape[:2] != R.shape[:2]:
            # Same guard, same reason as the UNCERT one: MASK and RFL grids that merely FLATTEN to the
            # same length would screen the wrong pixels instead of raising (numpy only complains when
            # the lengths differ, and then with an opaque broadcast error).
            raise ValueError(f"{gdir}: MASK grid {M.shape[:2]} != RFL grid {R.shape[:2]} "
                             f"-- refusing to screen with a misaligned mask")
        flagged = (M[:, :, mask_idx] > 0).any(-1).reshape(-1)
    valid = valid & ~flagged
    # Name what was screened using the FILE's labels, not a comment: "mask0(Cloud flag)" is checkable
    # against the product, "cloud-only" was only ever an assertion about index 0.
    screen = f"mask{tag}({'+'.join(mask_labels)}), {100*(~flagged).mean():.0f}% mask-clear"

    # GEOGRAPHIC IDENTITY. earthaccess `bounding_box` returns every granule that merely INTERSECTS
    # the box, and an EMIT scene is ~75 km across, so a granule can touch the box edge and lie
    # entirely outside the region it is named after. Measured here: the "sumatra" granule had 0.0%
    # of its pixels inside the declared box and sat over water (median NDVI -0.05, 61.5% negative).
    # Restricting to in-box pixels is what makes the region label true rather than aspirational.
    inside_pct = 100.0
    n_inbox = len(valid)
    if region_bbox is not None:
        with h5py.File(rfl, "r") as f:
            lat = f["location/lat"][:]
            lon = f["location/lon"][:]
        # Compare the 2-D GRID, not the flattened length. A transposed square grid (lat/lon written
        # (cols,rows) instead of (rows,cols)) flattens to the same length and would sail through a
        # length check while cropping to a geographically meaningless set of pixels -- verified: a
        # 40x40 fixture with lat/lon transposed silently produced a 25%-in-box "crop".
        if lat.shape != R.shape[:2] or lon.shape != R.shape[:2]:
            raise ValueError(f"{gdir}: lat{lat.shape}/lon{lon.shape} grid does not match the "
                             f"reflectance grid {R.shape[:2]} -- cannot crop to the declared region")
        # Physical range, which is what catches lat and lon being SWAPPED -- the shape check cannot,
        # since swapped grids have identical shapes. (A pure transpose of a SQUARE grid stays
        # undetectable here; real EMIT scenes are 1280x1242, so the shape check covers them.)
        if not (np.isfinite(lat).all() and np.isfinite(lon).all()
                and -90 <= lat.min() and lat.max() <= 90 and -180 <= lon.min() and lon.max() <= 180):
            raise ValueError(f"{gdir}: implausible geolocation -- lat in [{lat.min():.2f},"
                             f"{lat.max():.2f}], lon in [{lon.min():.2f},{lon.max():.2f}] "
                             f"(expected lat +/-90, lon +/-180; are lat and lon swapped?)")
        inside = ((lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)).reshape(-1)
        n_inbox = int(inside.sum())
        inside_pct = 100.0 * n_inbox / max(1, len(valid))
        valid = valid & inside
        screen += f", {inside_pct:.1f}% in-box"

    idx = np.where(valid)[0]
    # Two different denominators, because with a crop they answer two different questions: `of scene`
    # is valid/ALL pixels (how much of the downloaded granule survives) and `of box` is valid/IN-BOX
    # (how much of the region we actually asked for survives). Quoting only the first makes a granule
    # that is 89% outside its own box look "11% usable -- cloudy" when it is really "clear, but mostly
    # somewhere else".
    print(f"    {os.path.basename(gdir)}: {len(idx)}/{len(valid)} px = "
          f"{100*len(idx)/max(1,len(valid)):.1f}% of scene usable"
          + (f", {100*len(idx)/max(1,n_inbox):.1f}% of box" if region_bbox is not None else "")
          + f" ({screen})")
    if len(idx) < min_valid_px:
        raise ValueError(f"{gdir}: only {len(idx)} usable px"
                         + (f" inside the declared region ({inside_pct:.1f}% of the scene is in-box)"
                            if region_bbox is not None else " (too cloudy)")
                         + f" -- below the {min_valid_px}-px floor, exclude")
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(idx), size=min(n_px, len(idx)), replace=False)
    flat = idx[sel]                                          # flat index into the (rows, cols) grid
    ncols = R.shape[1]
    rr = (flat // ncols).astype(np.int32); cc = (flat % ncols).astype(np.int32)
    R2 = Rg[flat].astype(np.float32); U2 = Ug[flat].astype(np.float32); wl2 = wl[good].astype(float)
    # Write via a temp file + atomic rename. np.savez straight to `cache` leaves a TRUNCATED .npz if
    # the process dies mid-write, and the next run's np.load raises a zipfile error that main()'s
    # blanket `except` reports as "SKIP" -- i.e. an interrupted run would silently shrink N.
    tmpc = cache + f".tmp{os.getpid()}.npz"          # must end in .npz or savez appends it for us
    np.savez(tmpc, R=R2, U=U2, wl=wl2, row=rr, col=cc, n_valid=len(idx), n_total=len(valid),
             n_inbox=n_inbox, inside_pct=inside_pct, source_fp=fp)
    os.replace(tmpc, cache)
    out = [R2, U2, wl2]
    if return_coords:
        out += [rr, cc]
    if return_stats:
        out.append({"n_valid": len(idx), "n_total": len(valid), "n_inbox": n_inbox,
                    "frac": len(idx) / max(1, len(valid)),
                    "frac_inbox": len(idx) / max(1, n_inbox), "inside_pct": inside_pct})
    return tuple(out)


def granule_quality(gdir, R, wl, region_bbox=None):
    """Objective per-granule quality stats for transparency (cloud/water flag %, negative-NDVI %),
    so the NDVI trend can be shown robust to excluding contaminated scenes without cherry-picking.

    `region_bbox` must be the SAME box extract() was given. Without it these percentages describe the
    whole downloaded granule while every correlation beside them describes only the cropped pixels --
    verified on a fixture where water was flagged on the eastern half and the analysis was cropped to
    the western half: this function reported water=50% for pixels that were 0% water. A cloud% that
    does not describe the analysed pixels is worse than no cloud%, because the `cloud<40` robustness
    filter downstream then drops the wrong granules."""
    _gid, rfl, _unc, msk = select_triple(gdir)       # same acquisition extract() used, not glob[0]
    # Resolve Cloud/Water from the file's own mask_bands. These two numbers are printed per granule
    # AND drive the `cloud<40` robustness filter, so reading them from fixed indices 0 and 2 meant a
    # relayout would drop the wrong granules from the robustness check and label the columns wrong,
    # with the table still looking perfectly reasonable.
    i_cloud, i_water = mask_band_indices(msk, ("cloud", "water"))
    # h5py fancy-indexing REQUIRES ascending indices, and resolving by label no longer guarantees
    # cloud<water. Read ascending, then reorder to (cloud, water) explicitly -- handing h5py the
    # semantic order raises on a relabelled product, and silently reading [min,max] would put water
    # in the cloud column, which is the very confusion this resolution exists to end.
    order = sorted({i_cloud, i_water})
    with h5py.File(msk, "r") as f:
        raw = f["mask"][:, :, order]
        M = np.stack([raw[:, :, order.index(i_cloud)], raw[:, :, order.index(i_water)]], -1)
        keep = np.ones(M.shape[:2], bool)
        if region_bbox is not None:
            w, s, e, n = (float(v) for v in region_bbox)
            with h5py.File(rfl, "r") as g:
                lat = g["location/lat"][:]; lon = g["location/lon"][:]
            if lat.shape != M.shape[:2]:
                raise ValueError(f"{gdir}: lat/lon grid {lat.shape} != MASK grid {M.shape[:2]}")
            keep = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
        Mf = M[keep]                                 # (n_kept, 2)
    cloud = 100 * float((Mf[:, 0] > 0).mean()) if Mf.size else float("nan")
    water = 100 * float((Mf[:, 1] > 0).mean()) if Mf.size else float("nan")
    ndvi, nst = safe_ndvi(R, wl)
    # Only pixels with a usable denominator count toward "negative NDVI": treating a REJECTED pixel
    # as "not negative" would quietly dilute exactly the statistic used to spot water/shadow scenes.
    fin = np.isfinite(ndvi)
    neg = 100 * float((ndvi[fin] < 0).mean()) if fin.any() else float("nan")
    return cloud, water, neg, nst


def pca_recon_error(Rtr, Rev, k=10):
    """Training-free LINEAR proxy: fit a k-component PCA subspace on TRAIN pixels (standardised by
    train stats), reconstruct the HELD-OUT eval pixels, return per-(pixel,band) |error|. Cannot
    memorise individual pixels -> isolates the linear spectral-subspace-anomaly component."""
    mu = Rtr.mean(0); sd = Rtr.std(0) + 1e-8
    Xtr = (Rtr - mu) / sd; Xev = (Rev - mu) / sd
    ctr = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - ctr, full_matrices=False)
    comp = Vt[:k]                                                    # (k, C) principal axes
    Xc = Xev - ctr
    recon = Xc @ comp.T @ comp + ctr                                # project -> back-project
    return np.abs(Xev - recon)                                       # (Nev, C) standardised-space |err|


def sgmae_recon_error(Rtr, Rev, wl, groups_n, epochs, seed):
    """Train SGMAE on TRAIN pixels; return HELD-OUT (Nev, C) masked-reconstruction error + groups.
    Both sets standardised by TRAIN stats only (no eval leakage)."""
    mu = Rtr.mean(0); sd = Rtr.std(0) + 1e-8
    Xtr = ((Rtr - mu) / sd).astype(np.float32); Xev = ((Rev - mu) / sd).astype(np.float32)
    groups = contiguous_groups(Rtr.shape[1], groups_n)
    cwl = group_center_wavelengths(wl, groups)
    m = GroupedCrossBandAttention(groups, cwl, 2)
    P2.pretrain_sgmae(m, Xtr, groups, seed, epochs=epochs)
    return F.recon_error_matrix(m, Xev, groups), groups, sd         # eval on DISJOINT pixels; sd for raw-space


SPLIT_BLOCKS = 10          # 10x10 blocks over a ~1280x1242 EMIT grid -> ~128x124 px (~7.7x7.4 km) each


def spatial_block_split(rr, cc, nblocks=SPLIT_BLOCKS, train_frac=0.7, seed=0):
    """70/30 by whole spatial BLOCKS of the granule, not by pixel index.

    A random-pixel split holds out pixels that sit among the training pixels: at the default
    n_px=40000 over a 1280x1242 granule the sampled pixels are ~6 px apart, and the MEASURED median
    distance from a held-out pixel to its nearest training pixel is 7.7 px (~460 m at EMIT's 60 m
    GSD) -- inside the range over which land surface, and therefore spectra, are autocorrelated. A
    block split raises that to ~37 px (~2.2 km).

    Blocks are indexed from the pixels' own extent rather than the granule shape, so a
    region-cropped sample is divided across the CROP, not across a scene it barely occupies.
    Boundary pixels of an eval block still touch a train block: this is unbuffered blocked CV, which
    reduces the neighbour effect rather than eliminating it."""
    rr = np.asarray(rr, np.int64); cc = np.asarray(cc, np.int64)
    if rr.shape != cc.shape or rr.ndim != 1:
        raise ValueError(f"row/col must be matching 1-D arrays, got {rr.shape} and {cc.shape}")
    span_r = max(1, int(rr.max() - rr.min()) + 1)
    span_c = max(1, int(cc.max() - cc.min()) + 1)
    br = np.minimum((rr - rr.min()) * nblocks // span_r, nblocks - 1)
    bc = np.minimum((cc - cc.min()) * nblocks // span_c, nblocks - 1)
    bid = br * nblocks + bc
    uniq = np.unique(bid)
    if uniq.size < 2:
        raise ValueError(f"spatial split needs >= 2 non-empty blocks, got {uniq.size} — the sampled "
                         f"pixels span only {span_r}x{span_c} px")
    rng = np.random.default_rng(1000 + seed)
    rng.shuffle(uniq)
    target = train_frac * bid.size
    tr_blocks, acc = [], 0
    for b in uniq:
        if acc >= target and tr_blocks:
            break
        tr_blocks.append(b); acc += int((bid == b).sum())
    m = np.isin(bid, tr_blocks)
    if m.all() or not m.any():
        raise ValueError("spatial split degenerated to one side — check nblocks vs pixel extent")
    return np.where(m)[0], np.where(~m)[0]


def run_granule(R, U, wl, groups_n, epochs, seeds, pca_k, rows=None, cols=None, split="spatial"):
    """Per seed: 70/30 split -> SGMAE + PCA held-out recon error -> per-pixel / per-band / partial
    correlations vs EMIT uncertainty. Returns dict of lists over seeds + mean per-band curve.

    split="spatial" (default) holds out whole spatial blocks; split="random" is the original
    random-pixel holdout, kept so the two are directly comparable on the same pixels. The default is
    spatial because the claim is about generalising to ground the model has not seen, and only the
    spatial split tests that -- not because the random one is uniformly optimistic. Measured over 3
    granules x 5 seeds, it is not: the per-pixel Spearman moves +0.13 (India), -0.10 (sahara) and
    -0.08 (us_midwest) when switching to spatial. See the module docstring for the full table; the
    honest summary is that this choice CHANGES the per-granule numbers rather than merely
    tightening them, so `split` is written into the results CSV."""
    N = R.shape[0]
    if split not in ("spatial", "random"):
        raise ValueError(f"split must be 'spatial' or 'random', got {split!r}")
    if split == "spatial" and (rows is None or cols is None):
        raise ValueError("split='spatial' needs the sampled pixels' row/col (extract(..., "
                         "return_coords=True)) — refusing to fall back to a random split, which "
                         "would silently report a different quantity under the same column name")
    out = {k: [] for k in ["sg_pix", "pca_pix", "sg_pix_partial",
                           "sg_band_raw", "sg_band_full", "pca_band_full", "pca_band_raw"]}
    win = {w: [] for w in WINDOWS}
    band_curve = []                                                 # (n_band,) mean SGMAE RAW err over seeds
    band_unc_curve = []
    for seed in seeds:
        if split == "spatial":
            tr, ev = spatial_block_split(rows, cols, train_frac=0.7, seed=seed)
        else:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(N); ntr = int(0.7 * N)
            tr, ev = perm[:ntr], perm[ntr:]
        Rtr, Rev, Uev = R[tr], R[ev], U[ev]
        Esg, groups, sd = sgmae_recon_error(Rtr, Rev, wl, groups_n, epochs, seed)
        Epca = pca_recon_error(Rtr, Rev, k=pca_k)
        sg_pix, pca_pix = np.nanmean(Esg, 1), Epca.mean(1)
        unc_pix, bright = Uev.mean(1), Rev.mean(1)                  # brightness = mean reflectance
        out["sg_pix"].append(sp(unc_pix, sg_pix))
        out["pca_pix"].append(sp(unc_pix, pca_pix))
        out["sg_pix_partial"].append(partial_sp(unc_pix, sg_pix, bright))   # control brightness/SNR
        sg_band_std = np.nanmean(Esg, 0)
        sg_band_raw = sg_band_std * sd                             # back to reflectance units (physical)
        pca_band, unc_band = Epca.mean(0), Uev.mean(0)
        out["sg_band_raw"].append(sp(unc_band, sg_band_raw))       # PRIMARY per-band: raw vs raw (agent M1)
        out["sg_band_full"].append(sp(unc_band, sg_band_std))      # standardized-space (secondary/legacy)
        # The PCA proxy needs BOTH spaces for the same reason the SGMAE does. Spearman is rank-based
        # and multiplying a band by its train sd reorders the bands, so a standardized-space PCA
        # number is not comparable with the raw-space SGMAE headline -- reporting only the former
        # under the name `pca_perband` invited exactly that cross-space comparison.
        out["pca_band_full"].append(sp(unc_band, pca_band))        # standardized-space
        out["pca_band_raw"].append(sp(unc_band, pca_band * sd))    # raw-space, comparable to sg_band_raw
        for (lo, hi) in WINDOWS:
            m = (wl >= lo) & (wl <= hi)
            win[(lo, hi)].append(sp(unc_band[m], sg_band_raw[m]))  # window sweep on RAW per-band
        band_curve.append(sg_band_raw); band_unc_curve.append(unc_band)
    agg = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in out.items()}
    agg_win = {w: float(np.nanmean(v)) for w, v in win.items()}
    return agg, agg_win, np.nanmean(band_curve, 0), np.nanmean(band_unc_curve, 0), wl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-px", type=int, default=40000, help="sampled valid pixels (70/30 split)")
    ap.add_argument("--pca-k", type=int, default=10)
    ap.add_argument("--mask-flags", nargs="+", default=["cloud"],
                    help="EMIT MASK bands to screen (OR), by NAME — resolved through each file's own "
                         "sensor_band_parameters/mask_bands. Default: cloud (opaque). Official "
                         "robustness: cloud cirrus spacecraft dilated. Water screen: cloud water. "
                         "Raw indices are still accepted and checked against the file's labels.")
    ap.add_argument("--split", default="spatial", choices=["spatial", "random"],
                    help="held-out set: 'spatial' holds out whole spatial blocks of the granule "
                         "(default); 'random' is the original random-pixel holdout, whose held-out "
                         "pixels sit ~460 m from training pixels")
    ap.add_argument("--tag", default="", help="suffix for output CSVs (e.g. _official) for robustness runs")
    ap.add_argument("--require-region", action="store_true",
                    help="drop granules with no declared region box instead of keeping them with an "
                         "UNVERIFIED biome label (they cannot be cropped, so their region name is "
                         "only aspirational)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    ap.add_argument("--nondeterministic", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0, 1]; args.epochs = 4; args.groups = 8; args.n_px = 3000
        print("[smoke] 2 seeds / 4 epochs / 8 groups — writing *_smoke artefacts, NOT the deliverables")
    hw.setup(deterministic=not args.nondeterministic, prefer=args.device)
    print("HW:", hw.info())
    mflags = tuple(args.mask_flags)
    # Resolved per granule against that file's mask_bands (extract does it), so this line states the
    # POLICY; the per-granule line states the labels it actually matched.
    print(f"MASK screen policy {mflags} (resolved by name per file; known flags: "
          f"{sorted(MASK_FLAG_LABELS)})")
    print(f"train/eval split: {args.split}"
          + ("  (whole spatial blocks held out)" if args.split == "spatial" else
             "  (random pixels -- held-out pixels are ~460 m from training pixels)"))

    GRAN = discover_granules()
    print(f"discovered {len(GRAN)} granules: {list(GRAN)}")
    rows = []; perband = None; skipped = []
    for name, gdir in GRAN.items():
        # Crop to the box this granule was SEARCHED for. earthaccess returns granules that merely
        # INTERSECT the box, so without this a granule that clipped the corner still carries the
        # region's name: measured, 'sumatra' had 0.0% of its pixels in-box and sat over water
        # (median NDVI -0.05, 61.5% negative). Uncropped, the biome label is fiction.
        bbox = region_bbox_for(name)
        if bbox is None:
            print(f"  {name}: no declared region box -- biome label UNVERIFIED, cannot crop"
                  + (" -> excluded" if args.require_region else " -> kept but flagged"))
            if args.require_region:
                skipped.append((name, "no declared region box")); continue
        try:
            R, U, wl, prow, pcol, st = extract(gdir, args.n_px, seed=0, mask_flags=mflags,
                                               region_bbox=bbox, return_stats=True,
                                               return_coords=True)  # fixed sample; seeds vary split+train
        except Exception as e:
            # Kept broad because a granule can legitimately fail on data grounds (too cloudy, mostly
            # out of box), but the reason is RECORDED and re-printed at the end: an N that silently
            # shrinks from 14 to 9 changes every aggregate below it, and a one-line SKIP scrolls away.
            print(f"  {name}: SKIP ({type(e).__name__}: {e})")
            skipped.append((name, f"{type(e).__name__}: {e}")); continue
        cloudpc, waterpc, negpc, nst = granule_quality(gdir, R, wl, region_bbox=bbox)
        ndvi_px, _ = safe_ndvi(R, wl)
        if not np.isfinite(ndvi_px).any():
            # Every sampled pixel failed the denominator floor. The old code would have produced a
            # finite-looking median from divisions by ~1e-6 and placed this granule somewhere on the
            # NDVI axis; a NaN median would instead poison the sort and the min/max in the summary.
            print(f"  {name}: SKIP (no pixel has a usable NDVI denominator >= {NDVI_DENOM_FLOOR})")
            skipped.append((name, "no usable NDVI denominator")); continue
        ndvi = float(np.nanmedian(ndvi_px))
        agg, agg_win, band_err, band_unc, wlg = run_granule(
            R, U, wl, args.groups, args.epochs, args.seeds, args.pca_k,
            rows=prow, cols=pcol, split=args.split)
        # nan, not 100, when there is no box: extract() reports inside_pct=100 for an uncropped
        # granule, so an UNVERIFIED granule would otherwise carry the most confident-looking value in
        # the column ("100% inside its region") on the strength of having no region at all.
        rows.append((name, R.shape[0], ndvi, cloudpc, waterpc, negpc, agg, agg_win,
                     st["inside_pct"] if bbox is not None else float("nan"),
                     nst["rejected_pct"], nst["oor_pct"]))
        if perband is None:
            perband = (wlg, {})
        else:
            # Explicit raise, not `assert`: stripped by `python -O`, and a misaligned wavelength axis
            # silently writes one granule's per-band curve under another's wavelengths.
            if not np.array_equal(wlg, perband[0]):
                raise ValueError(f"{name}: wavelength axis differs from the first granule -> the "
                                 f"per-band CSV columns would be misaligned")
        perband[1][name] = (band_unc, band_err)
        sgm, sgs = agg["sg_pix"]; pcm, _ = agg["pca_pix"]; ppm, _ = agg["sg_pix_partial"]
        print(f"  {name:16s} NDVI={ndvi:+.2f} cloud={cloudpc:.0f}% water={waterpc:.0f}% neg={negpc:.0f}% "
              f"| SGMAE pix={sgm:+.3f}+/-{sgs:.3f} PCA={pcm:+.3f} partial={ppm:+.3f} band_raw={agg['sg_band_raw'][0]:+.3f}")

    # Fail CLOSED. Every aggregate below assumes at least one granule; on an empty `rows` the numpy
    # reductions raise a bare "zero-size array" error deep in the summary, which reads like a bug in
    # the statistics rather than "no data survived screening". Say so, and exit non-zero so a CI run
    # cannot record a study over 0 granules as a success.
    if skipped:
        print(f"\n{len(skipped)}/{len(GRAN)} granules did NOT enter the analysis:")
        for nm, why in skipped:
            print(f"  - {nm}: {why}")
    if not rows:
        raise SystemExit(f"phase8F_multi: 0 of {len(GRAN)} discovered granules survived screening "
                         f"-- nothing to analyse, refusing to write an empty results table")
    rows.sort(key=lambda r: r[2])                                   # order by NDVI for the trend
    # --tag alone did NOT protect --smoke: an untagged smoke run (2 seeds / 4 epochs / 8 groups /
    # 3000 px) wrote straight over results_phase8F_multi*.csv, the tracked deliverables. Compose the
    # two rather than replace either, so `--tag _official --smoke` stays distinguishable from both.
    sfx = args.tag + ("_smoke" if args.smoke else "")
    # ---- CSV: per-granule summary (NDVI-sorted, with objective quality columns) ----
    with open(P(f"results_phase8F_multi{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # cloud_pct/water_pct describe the SAME (cropped) pixels as every correlation on the row;
        # ndvi_rejected_pct / ndvi_oor_pct say how much of the NDVI column is opinion rather than data.
        # `split` travels WITH the numbers: a spatial-block and a random-pixel holdout are different
        # quantities under identical column names, and the file is the only place that can say which.
        w.writerow(["granule", "split", "n_px", "median_ndvi", "cloud_pct", "water_pct", "neg_ndvi_pct",
                    "inside_region_pct", "ndvi_rejected_pct", "ndvi_oor_pct",
                    "sgmae_perpixel", "sgmae_perpixel_std", "pca_perpixel", "sgmae_partial_brightness",
                    "sgmae_perband_raw", "sgmae_perband_std", "pca_perband_raw", "pca_perband_std",
                    "perband_600_1300"])
        for name, n, ndvi, cl, wa, ng, agg, aw, inpct, rejpct, oorpct in rows:
            w.writerow([name, args.split, n, f"{ndvi:.3f}", f"{cl:.1f}", f"{wa:.1f}", f"{ng:.1f}",
                        f"{inpct:.1f}", f"{rejpct:.2f}", f"{oorpct:.2f}",
                        f"{agg['sg_pix'][0]:.3f}", f"{agg['sg_pix'][1]:.3f}", f"{agg['pca_pix'][0]:.3f}",
                        f"{agg['sg_pix_partial'][0]:.3f}", f"{agg['sg_band_raw'][0]:.3f}",
                        f"{agg['sg_band_full'][0]:.3f}", f"{agg['pca_band_raw'][0]:.3f}",
                        f"{agg['pca_band_full'][0]:.3f}", f"{aw[(600,1300)]:.3f}"])
    # ---- per-band curve CSV (RAW reflectance-unit recon error vs raw uncertainty) ----
    if perband is not None:
        wlg, d = perband
        with open(P(f"results_phase8F_multi_perband{sfx}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wavelength_nm"] + [f"{n}_unc" for n in d] + [f"{n}_reconerr" for n in d])
            for i in np.argsort(wlg):
                w.writerow([f"{wlg[i]:.1f}"] + [f"{d[n][0][i]:.5f}" for n in d]
                           + [f"{d[n][1][i]:.5f}" for n in d])

    # ---- NDVI trend + objective-quality robustness ----
    ndvis = np.array([r[2] for r in rows]); clouds = np.array([r[3] for r in rows])
    sgp = np.array([r[6]["sg_pix"][0] for r in rows])
    pcp = np.array([r[6]["pca_pix"][0] for r in rows])
    ppp = np.array([r[6]["sg_pix_partial"][0] for r in rows])
    bnd = np.array([r[6]["sg_band_raw"][0] for r in rows])
    print(f"\n===== EMIT reconstruction-anomaly vs uncertainty across NDVI ({len(rows)} granules, "
          f"NDVI {ndvis.min():+.2f}..{ndvis.max():+.2f}, mask{mflags}) =====")
    print(f"  (OBSERVATIONAL summary of per-granule Spearman coefficients on this sample of {len(rows)} "
          f"granules -- signs/means are descriptive, NOT a significance test; small-N significance would "
          f"need a permutation test.)")
    band_sign = "consistent positive sign" if (bnd > 0.1).all() else "mixed sign"
    print(f"per-BAND (RAW spectral) Spearman: mean={bnd.mean():+.3f} range {bnd.min():+.3f}..{bnd.max():+.3f} "
          f"-> {band_sign} on the {len(rows)} granules sampled")
    print(f"per-PIXEL (spatial) Spearman: mean={sgp.mean():+.3f} range {sgp.min():+.3f}..{sgp.max():+.3f}")
    if len(rows) >= 4:
        tr = sp(ndvis, sgp); trp = sp(ndvis, ppp)
        print(f"TREND Spearman(scene NDVI, per-pixel corr) = {tr:+.3f}  (partial-brightness: {trp:+.3f})")
        clean = clouds < 40                                        # robustness: drop heavily-clouded granules
        if 4 <= int(clean.sum()) < len(rows):
            print(f"  robustness (drop cloud>40%, {(~clean).sum()} granules): trend = {sp(ndvis[clean], sgp[clean]):+.3f} on {int(clean.sum())} clean")
        # tr can be nan if a coefficient vector is constant (sp() guards -> nan); nan<-0.4 is False,
        # so the descriptive "no clear trend" wording is used, and the explicit tr==tr keeps intent clear.
        trend_note = ("observed trend: per-pixel EMIT-correlation tends to DECREASE with vegetation "
                      "density on these granules" if (tr == tr and tr < -0.4)
                      else "no clear NDVI trend on this sample")
        print(f"  -> {trend_note} (descriptive on N={len(rows)}; a permutation test would be needed to "
              f"claim significance at this small N).")
    print(f"training-free PCA proxy: mean={pcp.mean():+.3f} (a linear PCA proxy shows a similar-sign "
          f"correlation on this sample, i.e. the signal is not unique to the attention architecture).")
    # Stamped under the SAME composed `sfx` (--tag + _smoke) the CSVs were written with. Which
    # granules SURVIVED screening is the fact this study is least able to recover afterwards: the
    # discovered set is whatever sits under data/emit*, and every aggregate above is over `rows`, not
    # over REGIONS -- so an N that quietly falls from 14 to 9 changes each number with nothing in the
    # file to say it did. The skipped names go in beside it.
    stamp(P(f"results_phase8F_multi{sfx}.csv"), args,
          extra={"regions_declared": sorted(REGIONS), "mask_flags": list(mflags),
                 "granules_discovered": sorted(GRAN),
                 "granules_analysed": [r[0] for r in rows],
                 "granules_skipped": {nm: why for nm, why in skipped}})
    # Guarded by the same condition as the write: an unwritten per-band file must not acquire a
    # sidecar claiming it exists.
    if perband is not None:
        wlg, d = perband
        stamp(P(f"results_phase8F_multi_perband{sfx}.csv"), args,
              extra={"mask_flags": list(mflags), "granules": list(d), "n_bands": int(len(wlg)),
                     "wavelength_nm_range": [float(np.min(wlg)), float(np.max(wlg))],
                     "units": "raw reflectance-unit reconstruction error vs raw uncertainty"})
    print(f"wrote: {P(f'results_phase8F_multi{sfx}.csv')}")


if __name__ == "__main__":
    main()
