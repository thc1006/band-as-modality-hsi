"""Data-contract regression tests for the three loaders that feed the real-data experiments.

Every test here corresponds to a way one of these loaders accepted bad input and returned a NUMBER
instead of an error. They are deliberately hermetic — each builds its own tiny CloudSEN12 tree or
EMIT .nc triple in tmp_path — so they run on a machine with no downloaded data, and so a fixture can
be corrupted on purpose without touching the real 500 MB/1.8 GB files.

Covered:
  phase8_cloudsen12.load_split  — product enum, patch_ids domain, .dat size, label range, roi_id,
                                  and the fact that `python -O` deletes assert-based guards.
  phase8E_dofa                  — the process-global cuDNN switch is scoped, not leaked.
  phase8F_multi                 — source fingerprint sees middle-of-file edits; cached arrays are
                                  cross-validated; MASK bands are resolved by LABEL; the held-out
                                  split is spatial.
"""
import ast
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "experiments"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "emit"))
sys.path.insert(0, _REPO)

import phase8_cloudsen12 as P8            # noqa: E402
import phase8E_dofa as P8E                # noqa: E402
import phase8F_multi as FM                # noqa: E402

h5py = pytest.importorskip("h5py")
import torch                              # noqa: E402


# ============================================================ CloudSEN12 fixture (tiny, synthetic)
SIDE, VALID, OFF, NPATCH = 8, 6, 1, 6


def make_cloudsen12(root, label_max=3, truncate_band=None, proj_shape=VALID, n=NPATCH):
    """A CloudSEN12 split with SIDE=8 instead of 512 — same layout, ~3 kB instead of 500 MB."""
    d = os.path.join(root, "test")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame({"index": np.arange(n),
                  "roi_id": [f"ROI_{i // 2:04d}" for i in range(n)],   # 2 patches per ROI, like the real split
                  "proj_shape": [proj_shape] * n}).to_csv(os.path.join(d, "metadata.csv"), index=False)
    rng = np.random.default_rng(0)
    rng.integers(0, label_max + 1, size=(n, SIDE, SIDE)).astype(np.uint8).tofile(
        os.path.join(d, "LABEL_manual_hq.dat"))
    for bi, b in enumerate(P8.L1C_BANDS):
        arr = np.zeros((n, SIDE, SIDE), np.int16)
        for p in range(n):
            arr[p] = 1000 * (p + 1) + bi            # value identifies the source patch
        raw = arr.tobytes()
        if truncate_band == b:
            raw = raw[: len(raw) - SIDE * SIDE * 2]  # one patch of bytes missing
        open(os.path.join(d, f"L1C_{b}.dat"), "wb").write(raw)
        if b in P8.L2A_BANDS:
            open(os.path.join(d, f"L2A_{b}.dat"), "wb").write(raw)
    return d


@pytest.fixture
def cs12(tmp_path, monkeypatch):
    """Point the loader at a tiny synthetic split (patched SIDE keeps the fixtures kilobyte-sized)."""
    make_cloudsen12(str(tmp_path))
    monkeypatch.setattr(P8, "DATA", str(tmp_path))
    monkeypatch.setattr(P8, "SIDE", SIDE)
    monkeypatch.setattr(P8, "VALID_SIDE", VALID)
    monkeypatch.setattr(P8, "VALID_OFF", OFF)
    return str(tmp_path)


# ------------------------------------------------------------------- A. load_split input contract
def test_unknown_product_is_named_not_guessed(cs12):
    # `L1C_BANDS if product == "L1C" else L2A_BANDS` sent every non-"L1C" string down the L2A branch
    # and relied on a missing file to stop it. Name the contract.
    with pytest.raises(ValueError, match="unknown product"):
        P8.load_split("test", "l1c", pixels_per_patch=3, patch_ids=[0])
    with pytest.raises(ValueError, match="unknown split"):
        P8.load_split("teste", "L1C", pixels_per_patch=3, patch_ids=[0])


def test_negative_patch_id_is_rejected_not_wrapped(cs12):
    # THE silent one: numpy wrapped -1 to the LAST patch, returned its pixels, and stamped them
    # patch_id=-1 — a caller building a disjoint calib/eval set got the wrong patch, labelled wrongly.
    with pytest.raises(IndexError, match="out of range"):
        P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[-1])
    with pytest.raises(IndexError, match="out of range"):
        P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[0, NPATCH])


def test_duplicate_patch_ids_rejected(cs12):
    # Duplicated patches duplicate their pixels: a conformal calibration set then has an n that
    # counts the same information several times.
    with pytest.raises(ValueError, match="duplicates"):
        P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[2, 2, 3])


def test_empty_patch_ids_named(cs12):
    with pytest.raises(ValueError, match="empty"):
        P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[])


def test_non_integral_patch_ids_rejected(cs12):
    # asarray(...,int) truncated 1.9 -> 1 without a word.
    with pytest.raises(ValueError, match="integers"):
        P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[1.9, 2.9])
    X, y, pid = P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[1.0, 2.0],
                              return_patch_id=True)          # integral floats are fine
    assert set(np.unique(pid).tolist()) == {1, 2}


def test_bad_pixels_per_patch_rejected(cs12):
    with pytest.raises(ValueError, match="pixels_per_patch"):
        P8.load_split("test", "L1C", pixels_per_patch=0, patch_ids=[0])
    with pytest.raises(ValueError, match="n_patches"):
        P8.load_split("test", "L1C", pixels_per_patch=3, n_patches=0)


def test_truncated_dat_names_the_file(tmp_path, monkeypatch):
    # reshape() already refused this, but with "cannot reshape array of size 320 into shape (6,8,8)",
    # which names neither the file nor the shortfall.
    make_cloudsen12(str(tmp_path), truncate_band="B5")
    monkeypatch.setattr(P8, "DATA", str(tmp_path))
    monkeypatch.setattr(P8, "SIDE", SIDE)
    monkeypatch.setattr(P8, "VALID_SIDE", VALID)
    monkeypatch.setattr(P8, "VALID_OFF", OFF)
    with pytest.raises(ValueError) as ei:
        P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[0])
    msg = str(ei.value)
    assert "L1C_B5.dat" in msg and "patches" in msg


def test_labels_outside_class_range_rejected(tmp_path, monkeypatch):
    # A 255 nodata sentinel loads fine and is then counted as "always wrong" by every `pred == y`
    # accuracy/coverage computation in the reliability scripts.
    make_cloudsen12(str(tmp_path), label_max=255)
    monkeypatch.setattr(P8, "DATA", str(tmp_path))
    monkeypatch.setattr(P8, "SIDE", SIDE)
    monkeypatch.setattr(P8, "VALID_SIDE", VALID)
    monkeypatch.setattr(P8, "VALID_OFF", OFF)
    with pytest.raises(ValueError, match=r"labels outside \[0,4\)"):
        P8.load_split("test", "L1C", pixels_per_patch=20, patch_ids=[0, 1])


def test_proj_shape_guard_survives_python_dash_O(tmp_path):
    """`python -O` deletes assert statements, so an assert-based data guard is not a guard.

    Verified against the old code: under -O the proj_shape assert did not fire on a metadata file
    declaring proj_shape=999 and the loader returned pixels cropped on a wrong assumption."""
    make_cloudsen12(str(tmp_path), proj_shape=999)
    prog = (
        f"import sys;sys.path.insert(0,{os.path.join(_REPO, 'experiments')!r});"
        f"sys.path.insert(0,{_REPO!r});import phase8_cloudsen12 as P8;"
        f"P8.DATA={str(tmp_path)!r};P8.SIDE,P8.VALID_SIDE,P8.VALID_OFF={SIDE},{VALID},{OFF};"
        "P8.load_split('test','L1C',pixels_per_patch=3,patch_ids=[0])"
    )
    for flags in ([], ["-O"]):
        r = subprocess.run([sys.executable] + flags + ["-c", prog], capture_output=True, text=True)
        assert r.returncode != 0, f"proj_shape guard did not fire under {flags or 'no flags'}"
        assert "proj_shape" in r.stderr


def test_no_assert_based_data_validation_left_in_the_module():
    """Static lock so a future edit cannot reintroduce an assert as a data guard in this file."""
    for mod in ("phase8_cloudsen12.py", "phase8E_dofa.py", "phase8F_multi.py"):
        tree = ast.parse(open(os.path.join(_REPO, "experiments", mod), encoding="utf-8").read())
        asserts = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
        assert not asserts, f"{mod}: assert at line(s) {asserts} — `python -O` deletes these"


# ---------------------------------------------------------------------- A. roi_id / return shapes
def test_backward_compatible_return_shapes(cs12):
    assert len(P8.load_split("test", "L1C", pixels_per_patch=3, n_patches=2, seed=1)) == 2
    assert len(P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[0], return_patch_id=True)) == 3
    assert len(P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[0], return_roi_id=True)) == 3
    assert len(P8.load_split("test", "L1C", pixels_per_patch=3, patch_ids=[0],
                             return_patch_id=True, return_roi_id=True)) == 4


def test_roi_id_is_returned_per_pixel_and_groups_patches(cs12):
    # The fixture puts 2 patches in each ROI, mirroring CloudSEN12's 5-patches-per-ROI structure.
    X, y, pid, roi = P8.load_split("test", "L1C", pixels_per_patch=4, patch_ids=[0, 1, 2, 3],
                                   return_patch_id=True, return_roi_id=True)
    assert roi.shape == pid.shape == y.shape
    assert set(np.unique(roi).tolist()) == {"ROI_0000", "ROI_0001"}
    for p, r in ((0, "ROI_0000"), (1, "ROI_0000"), (2, "ROI_0001"), (3, "ROI_0001")):
        assert set(np.unique(roi[pid == p]).tolist()) == {r}


def test_patch_roi_ids_exposes_the_location_grouping(cs12):
    roi = P8.patch_roi_ids("test")
    assert len(roi) == NPATCH
    # Patch-disjoint is NOT location-disjoint: these two disjoint patch sets share an ROI.
    assert set(roi[[0, 2]]) & set(roi[[1, 3]])


def test_subsample_frac_domain_is_checked():
    # subsample_frac<=0 collapsed to k=1 via `max(1, ...)`: every method trained on ONE pixel and the
    # run still wrote a full results table.
    kw = dict(Xtr=np.zeros((10, 13), np.float32), ytr=np.zeros(10, int),
              Xte=np.zeros((4, 13), np.float32), yte=np.zeros(4, int), Xte_l2a=None,
              max_missing=1, trials=1, epochs=1)
    for bad in (0, -0.5, 1.5):
        with pytest.raises(ValueError, match="subsample_frac"):
            P8.run_seed(0, subsample_frac=bad, **kw)


# ================================================================= B. phase8E cuDNN global scoping
def test_cudnn_disabled_restores_on_normal_and_error_exit():
    before = torch.backends.cudnn.enabled
    with P8E.cudnn_disabled(True):
        assert torch.backends.cudnn.enabled is False
    assert torch.backends.cudnn.enabled == before
    with pytest.raises(RuntimeError):
        with P8E.cudnn_disabled(True):
            raise RuntimeError("boom")
    assert torch.backends.cudnn.enabled == before
    with P8E.cudnn_disabled(False):                       # cpu path: do not touch the flag at all
        assert torch.backends.cudnn.enabled == before
    assert torch.backends.cudnn.enabled == before


def test_phase8E_load_spatial_shares_the_loader_contract(tmp_path, monkeypatch):
    """phase8E has its OWN loader, which had every hole load_split had: an assert-based proj_shape
    guard, no .dat size check, and unvalidated patch indices where -1 silently returned the last
    patch."""
    make_cloudsen12(str(tmp_path))
    monkeypatch.setattr(P8, "DATA", str(tmp_path))
    monkeypatch.setattr(P8, "SIDE", SIDE)
    monkeypatch.setattr(P8, "VALID_SIDE", VALID)
    monkeypatch.setattr(P8, "VALID_OFF", OFF)
    monkeypatch.setattr(P8E, "SIDE", VALID)          # phase8E copies P8.VALID_SIDE at import time
    X, Y = P8E.load_spatial("test", "L1C", np.array([0, 1]))
    assert X.shape == (2, len(P8E.DOFA_BANDS), P8E.IMG, P8E.IMG) and Y.shape == (2, P8E.IMG, P8E.IMG)
    with pytest.raises(IndexError, match="out of range"):
        P8E.load_spatial("test", "L1C", np.array([-1]))
    with pytest.raises(IndexError, match="out of range"):
        P8E.load_spatial("test", "L1C", np.array([NPATCH]))
    with pytest.raises(ValueError, match="duplicates"):
        P8E.load_spatial("test", "L1C", np.array([1, 1]))
    with pytest.raises(ValueError, match="non-empty"):
        P8E.load_spatial("test", "L1C", np.array([], int))
    with pytest.raises(ValueError, match="unknown product"):
        P8E.load_spatial("test", "L1c", np.array([0]))


def test_phase8E_load_spatial_names_a_truncated_file(tmp_path, monkeypatch):
    make_cloudsen12(str(tmp_path), truncate_band="B4")   # B4 is a DOFA band
    monkeypatch.setattr(P8, "DATA", str(tmp_path))
    monkeypatch.setattr(P8, "SIDE", SIDE)
    monkeypatch.setattr(P8, "VALID_SIDE", VALID)
    monkeypatch.setattr(P8, "VALID_OFF", OFF)
    monkeypatch.setattr(P8E, "SIDE", VALID)
    with pytest.raises(ValueError, match="L1C_B4.dat"):
        P8E.load_spatial("test", "L1C", np.array([0]))


def test_phase8E_calib_eval_split_is_roi_disjoint():
    """phase8E claims the SAME reliability framework as phase8R, but its calib/eval split was by
    patch INDEX while CloudSEN12 gives ~5 patches per location. On the real test split that put a
    same-ROI sibling on both sides for ~51% of calibration patches, making the naive-vs-Mondrian
    comparison easier than the operational situation it stands for."""
    roi = np.repeat([f"ROI_{i:04d}" for i in range(40)], 5)      # 40 locations x 5 dates = 200 patches
    cal, ev = P8E.test_patch_split(200, calib_frac=0.5, seed=0, roi_ids=roi)
    assert len(cal) and len(ev)
    assert not (set(roi[cal]) & set(roi[ev])), "calibration and evaluation share a location"
    assert not (set(cal.tolist()) & set(ev.tolist()))
    with pytest.raises(ValueError, match="200 entries|entries for"):
        P8E.test_patch_split(200, roi_ids=roi[:10])
    # index split without roi_ids stays available (and warns), so existing callers still work
    c2, e2 = P8E.test_patch_split(80, calib_frac=0.5)
    assert len(c2) == 40 and len(e2) == 40


def test_phase8E_never_assigns_a_torch_backend_global_directly():
    """The leak was one line: `torch.backends.cudnn.enabled = False`, never restored, so every conv
    later in the process silently changed backend (measured: bitwise-different conv2d output, max
    |delta| 1.5e-5). Anything that must change a process-global belongs in a restoring scope."""
    tree = ast.parse(open(os.path.join(_REPO, "experiments", "phase8E_dofa.py"),
                          encoding="utf-8").read())
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Attribute) and "torch.backends" in ast.unparse(t):
                # allowed only inside the restoring context manager
                bad.append((node.lineno, ast.unparse(node)))
    scoped = [(f.lineno, max(getattr(n, "lineno", f.lineno) for n in ast.walk(f)))
              for f in ast.walk(tree)
              if isinstance(f, ast.FunctionDef) and f.name == "cudnn_disabled"]
    leaked = [b for b in bad if not any(lo <= b[0] <= hi for lo, hi in scoped)]
    assert not leaked, f"unrestored torch.backends assignment(s): {leaked}"


# ================================================================== EMIT fixture (tiny .nc triple)
GID = "20220810T034103_2222203_001"
V001_MASK_BANDS = [b"Cloud flag", b"Cirrus flag", b"Water flag", b"Spacecraft Flag",
                   b"Dilated Cloud Flag", b"AOD550", b"H2O (g cm-2)", b"Aggregate Flag"]


def make_emit(gdir, rows=120, cols=120, nb=100, mask_bands=None, cloud_rows=None, water_cols=None):
    """Complete EMIT L2A triple. ~6 MB per cube: big enough that a middle-of-file edit falls outside
    the head+tail window the old fingerprint hashed."""
    os.makedirs(gdir, exist_ok=True)
    rng = np.random.default_rng(0)
    wl = np.linspace(381.0, 2493.0, nb)
    good = np.ones(nb, bool); good[:3] = False
    yy, xx = np.mgrid[0:rows, 0:cols]
    base = (0.15 + 0.10 * np.sin(xx / 11.0) + 0.10 * np.cos(yy / 9.0))[..., None]
    spec = 0.5 + 0.5 * np.sin(np.linspace(0, 6, nb))[None, None, :]
    R = (base * spec + 0.01 * rng.standard_normal((rows, cols, nb))).astype(np.float32)
    U = (0.004 + 0.002 * base + 0.0005 * rng.random((rows, cols, nb))).astype(np.float32)
    lat = (21.5 + np.linspace(0, 0.5, rows)[:, None] * np.ones((1, cols)))
    lon = (73.5 + np.ones((rows, 1)) * np.linspace(0, 0.5, cols)[None, :])
    rfl = os.path.join(gdir, f"EMIT_L2A_RFL_001_{GID}.nc")
    with h5py.File(rfl, "w") as f:
        f.create_dataset("reflectance", data=R)
        f.create_dataset("sensor_band_parameters/wavelengths", data=wl)
        f.create_dataset("sensor_band_parameters/good_wavelengths", data=good.astype(np.float32))
        f.create_dataset("location/lat", data=lat)
        f.create_dataset("location/lon", data=lon)
    unc = os.path.join(gdir, f"EMIT_L2A_RFLUNCERT_001_{GID}.nc")
    with h5py.File(unc, "w") as f:
        f.create_dataset("reflectance_uncertainty", data=U)
    M = np.zeros((rows, cols, 8), np.float32)
    if cloud_rows is not None:
        M[cloud_rows, :, 0] = 1.0
    if water_cols is not None:
        M[:, water_cols, 2] = 1.0
    M[:, :, 5] = 0.19                                    # AOD550 — continuous, >0 nearly everywhere
    M[:, :, 6] = 2.7                                     # H2O    — likewise
    msk = os.path.join(gdir, f"EMIT_L2A_MASK_001_{GID}.nc")
    with h5py.File(msk, "w") as f:
        f.create_dataset("mask", data=M)
        f.create_dataset("sensor_band_parameters/mask_bands",
                         data=np.array(mask_bands or V001_MASK_BANDS,
                                       dtype=h5py.special_dtype(vlen=bytes)))
    return rfl, unc, msk


def edit_middle(path, value=0.9):
    """In-place h5py write to the middle of `reflectance`, mtime restored: same name, same size,
    same mtime, different data."""
    st = os.stat(path)
    with h5py.File(path, "r+") as f:
        f["reflectance"][f["reflectance"].shape[0] // 2, :, :] = value
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.path.getsize(path) == st.st_size


# ============================================================== C. source fingerprint + cache guard
def test_fingerprint_sees_a_middle_of_file_edit(tmp_path):
    """The old fingerprint hashed head(1 MB) + tail(1 MB) + size + mtime and was blind to this."""
    g = str(tmp_path / "emit_fix")
    rfl, unc, msk = make_emit(g)
    before = FM._source_fingerprint([rfl, unc, msk])
    edit_middle(rfl)
    assert FM._source_fingerprint([rfl, unc, msk]) != before


def test_fingerprint_modes_never_cross_validate(tmp_path):
    # A cheap sampled fingerprint must not be able to satisfy a cache written under the full one.
    g = str(tmp_path / "emit_fix")
    rfl, unc, msk = make_emit(g)
    full = FM._source_fingerprint([rfl, unc, msk], mode="full")
    sampled = FM._source_fingerprint([rfl, unc, msk], mode="sampled")
    assert full != sampled and "full" in full and "s64" in sampled
    with pytest.raises(ValueError, match="EMIT_FP_MODE"):
        FM._source_fingerprint([rfl, unc, msk], mode="partial")


def test_stale_cache_is_not_served_after_a_middle_edit(tmp_path, capsys):
    """End-to-end: the demonstrated failure was extract() returning the pre-edit sample, silently."""
    g = str(tmp_path / "emit_fix")
    rfl, _u, _m = make_emit(g)
    R0, _, _ = FM.extract(g, 300, seed=0, mask_flags=("cloud",), min_valid_px=50)
    edit_middle(rfl)
    R1, _, _ = FM.extract(g, 300, seed=0, mask_flags=("cloud",), min_valid_px=50)
    assert not np.array_equal(R0, R1), "cache from before the edit was served"
    assert "source .nc changed" in capsys.readouterr().out


def test_cache_arrays_are_cross_validated(tmp_path):
    """source_fp matching says the cache came from the right bytes, not that R/U/wl agree."""
    g = str(tmp_path / "emit_fix")
    make_emit(g)
    FM.extract(g, 300, seed=0, mask_flags=("cloud",), min_valid_px=50)
    cpath = os.path.join(g, [f for f in os.listdir(g) if f.endswith(".npz")][0])
    with np.load(cpath) as z:
        d = {k: z[k] for k in z.files}
    d["U"] = d["U"][:, :5]                       # U loses bands; fingerprint untouched
    np.savez(cpath, **d)
    with pytest.raises(ValueError, match="does not match"):
        FM.extract(g, 300, seed=0, mask_flags=("cloud",), min_valid_px=50)


def test_cache_missing_coordinates_is_rejected(tmp_path):
    g = str(tmp_path / "emit_fix")
    make_emit(g)
    FM.extract(g, 300, seed=0, mask_flags=("cloud",), min_valid_px=50)
    cpath = os.path.join(g, [f for f in os.listdir(g) if f.endswith(".npz")][0])
    with np.load(cpath) as z:
        d = {k: z[k] for k in z.files if k not in ("row", "col")}
    np.savez(cpath, **d)
    with pytest.raises(ValueError, match="missing"):
        FM.extract(g, 300, seed=0, mask_flags=("cloud",), min_valid_px=50)


# ================================================================== D. MASK semantics read by name
def test_mask_bands_are_read_from_the_file(tmp_path):
    g = str(tmp_path / "emit_fix")
    _r, _u, msk = make_emit(g)
    assert FM.read_mask_bands(msk)[:3] == ["Cloud flag", "Cirrus flag", "Water flag"]
    assert FM.resolve_mask_flags(msk, ("cloud", "water"))[0] == [0, 2]


def test_relabelled_product_resolves_to_the_right_index(tmp_path):
    """A V002-style relayout must move the index, not the meaning. With Cloud and Water swapped the
    old fixed indices screened water while calling it cloud — a wrong number, never an error."""
    permuted = [b"Water flag", b"Cirrus flag", b"Cloud flag", b"Spacecraft Flag",
                b"Dilated Cloud Flag", b"AOD550", b"H2O (g cm-2)", b"Aggregate Flag"]
    g = str(tmp_path / "emit_fix")
    _r, _u, msk = make_emit(g, mask_bands=permuted)
    assert FM.mask_band_indices(msk, ("cloud", "water")) == [2, 0]
    idx, labels = FM.resolve_mask_flags(msk, ("cloud",))
    assert idx == [2] and labels == ["Cloud flag"]


def test_missing_expected_label_fails_loudly(tmp_path):
    short = [b"Cloud flag", b"Cirrus flag", b"Spacecraft Flag", b"Dilated Cloud Flag",
             b"AOD550", b"H2O (g cm-2)", b"Aggregate Flag", b"Reserved"]
    g = str(tmp_path / "emit_fix")
    _r, _u, msk = make_emit(g, mask_bands=short)
    with pytest.raises(ValueError, match="absent from this product"):
        FM.resolve_mask_flags(msk, ("water",))
    with pytest.raises(ValueError, match="unknown MASK flag"):
        FM.resolve_mask_flags(msk, ("snow",))


def test_continuous_retrieval_bands_cannot_be_used_as_flags(tmp_path):
    """AOD550/H2O are retrievals, not flags: on the real India granule AOD550 > 0 for 99.96% of
    pixels, so `--mask-flags 5` would screen the scene and read as "too cloudy"."""
    g = str(tmp_path / "emit_fix")
    _r, _u, msk = make_emit(g)
    with pytest.raises(ValueError, match="continuous retrieval"):
        FM.resolve_mask_flags(msk, (5,))
    with pytest.raises(ValueError, match="out of range"):
        FM.resolve_mask_flags(msk, (9,))


def test_granule_quality_columns_follow_the_labels(tmp_path):
    """cloud_pct/water_pct drive the `cloud<40` robustness filter, so a swapped pair drops the wrong
    granules from it."""
    permuted = [b"Water flag", b"Cirrus flag", b"Cloud flag", b"Spacecraft Flag",
                b"Dilated Cloud Flag", b"AOD550", b"H2O (g cm-2)", b"Aggregate Flag"]
    g = str(tmp_path / "emit_fix")
    # index 0 (= "Water flag" here) on 25% of rows; index 2 (= "Cloud flag") on 50% of columns
    make_emit(g, mask_bands=permuted, cloud_rows=slice(0, 30), water_cols=slice(0, 60))
    R, U, wl = FM.extract(g, 300, seed=0, mask_flags=("cloud",), min_valid_px=50)
    cloud, water, _neg, _n = FM.granule_quality(g, R, wl)
    assert round(cloud) == 50 and round(water) == 25


# ========================================================================= E. spatial held-out split
def test_spatial_block_split_holds_out_whole_blocks(tmp_path):
    rng = np.random.default_rng(0)
    rr = rng.integers(0, 1000, 4000); cc = rng.integers(0, 1000, 4000)
    tr, ev = FM.spatial_block_split(rr, cc, nblocks=10, train_frac=0.7, seed=0)
    assert set(tr.tolist()).isdisjoint(ev.tolist())
    assert len(tr) + len(ev) == 4000
    assert 0.55 < len(tr) / 4000 < 0.85                  # roughly 70%, block-quantised
    # the point of the split: held-out pixels are farther from training pixels than a random split
    def median_nn(a, b):
        d = np.hypot(rr[b][:200, None] - rr[a][None, :], cc[b][:200, None] - cc[a][None, :])
        return float(np.median(d.min(1)))
    perm = np.random.default_rng(0).permutation(4000)
    assert median_nn(tr, ev) > 3 * median_nn(perm[:2800], perm[2800:])


def test_spatial_split_refuses_to_silently_fall_back(tmp_path):
    """Without coordinates a 'spatial' request must fail, not quietly produce the random-split
    number under the same column name."""
    R = np.random.default_rng(0).random((50, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="return_coords"):
        FM.run_granule(R, R, np.linspace(400, 2400, 8), 2, 1, [0], 2, split="spatial")
    with pytest.raises(ValueError, match="split must be"):
        FM.run_granule(R, R, np.linspace(400, 2400, 8), 2, 1, [0], 2, split="holdout")


def test_extract_returns_coordinates_that_index_the_granule(tmp_path):
    g = str(tmp_path / "emit_fix")
    make_emit(g, rows=120, cols=120)
    R, U, wl, rr, cc = FM.extract(g, 300, seed=0, mask_flags=("cloud",), min_valid_px=50,
                                  return_coords=True)
    assert rr.shape == cc.shape == (R.shape[0],)
    assert 0 <= rr.min() and rr.max() < 120 and 0 <= cc.min() and cc.max() < 120
    assert len(np.unique(rr * 120 + cc)) == R.shape[0]    # coordinates are unique per sampled pixel
