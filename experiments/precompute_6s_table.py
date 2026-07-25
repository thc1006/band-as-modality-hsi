#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute 6S atmospheric transmittance table for Phase 3 (Design B).

Runs the 6S radiative-transfer model (via Py6S) over the CWV/AOD parameter grid and saves
the per-wavelength total gaseous transmittance to an .npz. The main pipeline (main venv, no
Py6S) then reads this table via bandsim.atmosphere.load_cached_transmittance — this is the
guide's "compute 6S on a capable env, read the table elsewhere" pattern (ENVIRONMENT_SETUP §3).

RUN THIS WITH THE sixs CONDA ENV (which has Py6S + the 6S binary on PATH):
  export PATH=$HOME/miniforge3/envs/sixs/bin:$PATH
  $HOME/miniforge3/envs/sixs/bin/python experiments/precompute_6s_table.py

Output: data/srf_cache/T_6s_grid.npz   keys "cwv{c}" -> T(lam) (200,), plus "wl_nm" and "_provenance".

AOD note: 6S `transmittance_global_gas` is the GASEOUS transmittance and is aerosol-independent, so
AOD is NOT a grid axis here (empirically aod0.1==aod0.4 to machine precision). The table is keyed by
CWV alone; a single nominal AOD is passed to 6S only to satisfy its API.
"""
import os
import sys
import json
import hashlib
import shutil
from datetime import datetime, timezone
import numpy as np


def _version(mod):
    """Installed version of a module, or None. Recorded so a table can be tied to the software that
    made it -- a different Py6S or 6S build is a different physical model, not a detail."""
    try:
        return __import__(mod).__version__
    except Exception:
        try:
            from importlib.metadata import version
            return version(mod)
        except Exception:
            return None


def _sixs_hash():
    """sha256 of the 6S executable actually on PATH, or None. The Fortran binary IS the model: two
    builds with different compiler flags can return different numbers from identical inputs (we hit
    exactly that -- an -O2 build produced NaNs), so the table must record which binary produced it."""
    exe = shutil.which("sixs") or shutil.which("sixsV1.1")
    if not exe:
        return None
    try:
        h = hashlib.sha256()
        with open(exe, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from bandsim.io import AVIRIS_WL_NM, axis_sha256
from bandsim.atmosphere import run_6s_transmittance, DEFAULT_GRID

OUT_DIR = os.path.join(os.path.dirname(_HERE), "data", "srf_cache")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "T_6s_grid.npz")

AOD_NOMINAL = 0.1        # inert for gaseous T; passed only to satisfy the 6S API (see module docstring)
OZONE_CM_ATM = 0.35      # 6S geometry/atmosphere is fixed; recorded in the table's _provenance string


def main():
    wl = AVIRIS_WL_NM
    cwvs = DEFAULT_GRID["cwv_g_cm2"]
    # keys keyed by CWV only (float-normalised so 2 and 2.0 collide, matching config_runner)
    # `wl_nm` + `wl_sha256` are what make the table's axis CHECKABLE rather than merely asserted in
    # the provenance prose: an earlier table's provenance claimed the AVIRIS axis while actually
    # holding a gapless linspace, and nothing could tell, because only the band COUNT was verified.
    # bandsim.atmosphere.load_cached_transmittance now compares wl_nm against the caller's axis.
    # MACHINE-READABLE identity, not prose. Two separate failures motivate every field here.
    #
    # (1) The cache key is `cwv{value}` alone, but 6S gaseous transmittance depends on more than CWV.
    #     AOD genuinely is inert for this quantity (verified: aod0.1 == aod0.4 exactly), which is why
    #     it is not a key axis -- but OZONE absorbs, and GEOMETRY sets the slant path length, so a
    #     table computed at a different solar zenith is a different physical quantity wearing the same
    #     key. Nothing in the old file recorded those, so a table could be silently overwritten by one
    #     computed under different conditions and no check would notice.
    # (2) `resample_6s_table.py` can produce a file with the right length, the right `wl_nm` and the
    #     right axis hash by INTERPOLATING an old table. Every check we had would pass it, yet the
    #     values near narrow absorption edges are not what 6S would return there. `generation_mode`
    #     exists so a consumer can REFUSE a resampled table when it needs a directly computed one.
    meta = {
        "generation_mode": "direct_6s",          # the loader may require this; resampling stamps otherwise
        "schema_version": 2,
        "output_quantity": "transmittance_global_gas",
        "cwv_grid_g_cm2": list(cwvs),
        "ozone_cm_atm": OZONE_CM_ATM,
        "aod550_nominal": AOD_NOMINAL,           # inert for this quantity; recorded, not a key
        "aod_invariance_verified": True,         # aod0.1 == aod0.4 exactly on this axis
        "geometry": "Py6S SixS() defaults (no explicit geometry set)",
        "axis_name": "bandsim.io.AVIRIS_WL_NM",
        "axis_kind": "nominal_reconstruction_not_acquisition_calibration",
        "n_bands": int(wl.size),
        "wl_min_nm": float(wl[0]), "wl_max_nm": float(wl[-1]),
        "wl_sha256": axis_sha256(wl),
        "py6s_version": _version("Py6S"),
        "sixs_binary_sha256": _sixs_hash(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tables = {"wl_nm": wl,
              "wl_sha256": np.array(axis_sha256(wl)),
              "generation_mode": np.array("direct_6s"),
              "_metadata_json": np.array(json.dumps(meta, sort_keys=True)),
              "_provenance": np.array(
                  f"6S gaseous transmittance (transmittance_global_gas) computed DIRECTLY at the "
                  f"wavelengths in wl_nm ({wl.size} bands, {wl[0]:.1f}-{wl[-1]:.1f} nm, nominal AVIRIS "
                  f"axis preserving the documented Indian Pines water-absorption gaps -- NOT an "
                  f"acquisition-specific calibration). No resampling. See _metadata_json for the full "
                  f"machine-readable identity; the prose here is a convenience, not the contract.")}
    import time
    for cwv in cwvs:
        t0 = time.time()
        T = run_6s_transmittance(wl, cwv, AOD_NOMINAL, n_threads=8)
        key = f"cwv{float(cwv)}"
        tables[key] = T
        print(f"{key}: T range [{T.min():.3f}, {T.max():.3f}] mean {T.mean():.3f}  ({time.time()-t0:.1f}s)")
    np.savez_compressed(OUT, **tables)
    print(f"\nwrote {OUT} with {len([k for k in tables if k.startswith('cwv')])} CWV grid points (AOD inert, not a key axis)")


if __name__ == "__main__":
    main()
