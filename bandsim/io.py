"""HSI cube loading + wavelength axes for the benchmark datasets.

Indian Pines / Pavia / Salinas ship as MATLAB .mat files (data cube + ground-truth labels).
Download links & licences: docs/guide/01_datasets.md. Place under data/<dataset>/.
"""
from __future__ import annotations
import numpy as np

# Wavelength axes (nm) for building SRFs / atmosphere / group-centre PE on these sensors.
# Indian Pines/Salinas = AVIRIS 400-2500 nm; Pavia = ROSIS ~430-860 nm.
#
# AVIRIS Indian Pines axis, on the 224-BAND SENSOR BASE (corrected 2026-07-20).
#
# The previous construction stretched 220 bands uniformly across 400-2500 nm and deleted the 20
# water/noise bands from THAT. But AVIRIS is a 224-band instrument, and the "220-band" Indian Pines
# distribution is the same sensor minus its FIRST FOUR channels: the proof is that the Salinas
# removal list (bands 108-112, 154-167, 224 of a 224-band cube) is EXACTLY the Indian Pines list
# (104-108, 150-163, 220) offset by +4 -- one sensor, one list, two zero points. Rebuilt on the 224
# base, Indian Pines' last six group centres land on Salinas' [1280.5 1508.9 1704.3 1998.1 2212.8
# 2401.1] to 0.00 nm; on the stretched axis they missed by 22.3 nm, which two axes of one sensor
# cannot do. Independently derived by three sessions (see shared_layer.md) and reproduced at merge
# time before this edit.
#
# The correction moves every retained centre by +0.2 to +37.7 nm. That is not cosmetic: group
# centres shift 1.8-36.0 nm, sinusoidal_wavelength_pe moves group 0 by 1.44 rad (a quarter cycle of
# the encoding the proposed model reads), B3's interpolation reads it, and the 6S LUT keyed to the
# OLD axis is invalidated -- load_cached_transmittance's axis-identity guard raises until
# experiments/precompute_6s_table.py regenerates it (direct 6S, never resampled).
#
# (Nominal AVIRIS spacing; for exact per-scene calibration, load a wavelength vector shipped with
# the data instead.)
_AVIRIS_224 = np.linspace(400.0, 2500.0, 224)
_IP_REMOVED_BANDS_1BASED = list(range(104, 109)) + list(range(150, 164)) + [220]   # 20 water/noise bands
AVIRIS_WL_NM = np.delete(_AVIRIS_224[4:], [i - 1 for i in _IP_REMOVED_BANDS_1BASED])  # 200 bands, WITH gaps
ROSIS_WL_NM = np.linspace(430, 860, 103)     # Pavia University: 103 bands, ROSIS ~430-860 nm (near-uniform)


def axis_sha256(wavelengths_nm):
    """Content address of a wavelength axis: SHA-256 over its canonical little-endian float64 bytes.

    Hashing the VALUES (not a .npy file, whose header carries version/shape framing) makes the digest
    reproducible from any in-memory array, so a cached artefact can record WHICH axis it was built on
    and a reader can check identity without shipping the axis itself. See data/metadata/ for the
    stored axis artefact and its sidecar."""
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(wavelengths_nm, dtype="<f8").tobytes()).hexdigest()


def load_mat_cube(mat_path, key=None):
    """Load an HSI cube from a .mat file. Requires scipy.

    Returns the array; pass `key` to select the variable if there are several.
    """
    from scipy.io import loadmat
    d = loadmat(mat_path)
    if key is not None:
        return np.asarray(d[key])
    # pick the largest ndarray that isn't a MATLAB metadata field
    cands = {k: v for k, v in d.items() if not k.startswith("__") and hasattr(v, "shape")}
    if not cands:
        raise ValueError(f"No array variables found in {mat_path}")
    best = max(cands, key=lambda k: int(np.prod(cands[k].shape)))
    if len(cands) > 1:
        import warnings
        warnings.warn(f"load_mat_cube auto-picked '{best}' (largest of {list(cands)}) in {mat_path}; "
                      f"pass key= to be explicit — the largest array could be a ground-truth/aux matrix.")
    return np.asarray(cands[best])


def disjoint_region_split(gt, train_frac=0.5, axis=1, ignore_label=0):
    """Spatially DISJOINT train/test split (avoids pixel-adjacency leakage).

    Splits the labelled map along `axis` (default: left/right halves). Returns boolean
    masks (train_mask, test_mask) over labelled pixels only. See docs/guide/01_datasets.md:
    random per-pixel sampling from one scene leaks — reviewers know this.
    """
    gt = np.asarray(gt)
    cut = int(gt.shape[axis] * train_frac)
    idx = [slice(None)] * gt.ndim
    train_region = np.zeros(gt.shape, bool)
    idx[axis] = slice(0, cut)
    train_region[tuple(idx)] = True
    labelled = gt != ignore_label
    return (train_region & labelled), (~train_region & labelled)


def disjoint_block_split(gt, block=10, ignore_label=0, guard=0, offset=0):
    """Spatially disjoint train/test split via a CHECKERBOARD of contiguous blocks.

    The scene is tiled into `block` x `block` cells; cells are assigned to train/test in a
    checkerboard pattern. This keeps train and test in *disjoint spatial regions* while —
    unlike a single left/right half-split — keeping every class present in BOTH splits
    (needed for AA/kappa/mIoU on Indian Pines, whose small classes cluster in one corner).
    See docs/guide/01_datasets.md.

    guard>0 removes TEST pixels within `guard` pixels of any train pixel (a buffer / guard
    band), eliminating the block-seam adjacency leakage: a plain block split still places
    train and test pixels next to each other along block edges. Use guard>=1 for a strict,
    leakage-hardened evaluation (shrinks the test set). guard=0 keeps the original behaviour.

    On Indian Pines (145x145, 16 classes) block=10 yields ~50/50 with all 16 classes in
    each split. Returns boolean (train_mask, test_mask) over labelled pixels only.
    """
    gt = np.asarray(gt)
    H, W = gt.shape[:2]
    # `offset` shifts the checkerboard grid by whole pixels so different seeds get genuinely
    # different (still-disjoint) splits -> reported +/-std then includes data-split variance.
    bi = (np.arange(H)[:, None] + offset) // block
    bj = (np.arange(W)[None, :] + offset) // block
    train_region = ((bi + bj) % 2 == 0)
    labelled = gt != ignore_label
    train_mask = train_region & labelled
    test_mask = (~train_region) & labelled
    if guard > 0:
        from scipy.ndimage import binary_dilation
        train_dilated = binary_dilation(train_region, iterations=int(guard))
        test_mask = test_mask & ~train_dilated       # drop test pixels near any train pixel
    return train_mask, test_mask
