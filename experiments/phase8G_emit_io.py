"""EMIT cross-sensor reliability I/O (2nd dataset).

Load a granule's PAIRED products for the real atmospheric-correction shift:
  SOURCE = L1B radiance  (EMIT_L1B_RAD_*.nc, 'radiance')
  TARGET = L2A reflectance (EMIT_L2A_RFL_*.nc, 'reflectance', ISOFIT surface reflectance)
They share the (downtrack x crosstrack x band) sensor grid, so pixels are ALIGNED with no reprojection.
Drops the bad bands (deep water-vapour absorption) via good_wavelengths and drops no-data/fill pixels in
either product. Returns aligned per-pixel radiance + reflectance + lat/lon (for WorldCover labels).

EMIT .nc are HDF5, read with h5py (netCDF4 is not installed in this env).
"""
import glob
import os

import numpy as np
import h5py

FILL = -9999.0


def granule_files(biome_dir):
    rad = sorted(glob.glob(os.path.join(biome_dir, "EMIT_L1B_RAD_*.nc")))
    rfl = sorted(glob.glob(os.path.join(biome_dir, "EMIT_L2A_RFL_*.nc")))
    return (rad[0] if rad else None, rfl[0] if rfl else None)


def load_granule(biome_dir, good_only=True):
    """Aligned valid-pixel radiance (source) + reflectance (target) + lat/lon for one biome granule."""
    rad_f, rfl_f = granule_files(biome_dir)
    if not rad_f or not rfl_f:
        raise FileNotFoundError(f"no paired L1B RAD + L2A RFL in {biome_dir}")
    with h5py.File(rad_f, "r") as h:
        rad = np.asarray(h["radiance"][:], np.float32)                      # (dt, ct, 285)
        wl = np.asarray(h["sensor_band_parameters/wavelengths"][:], np.float32)
    with h5py.File(rfl_f, "r") as h:
        rfl = np.asarray(h["reflectance"][:], np.float32)                   # (dt, ct, 285)
        good = np.asarray(h["sensor_band_parameters/good_wavelengths"][:]).astype(bool)
        lat = np.asarray(h["location/lat"][:], np.float64)
        lon = np.asarray(h["location/lon"][:], np.float64)
    if rad.shape != rfl.shape:
        raise ValueError(f"L1B {rad.shape} != L2A {rfl.shape} in {biome_dir}")
    keep = good if good_only else np.ones(rad.shape[-1], bool)
    B = int(keep.sum())
    radf = rad[:, :, keep].reshape(-1, B)
    rflf = rfl[:, :, keep].reshape(-1, B)
    latf = lat.reshape(-1)
    lonf = lon.reshape(-1)
    valid = (np.isfinite(radf).all(1) & np.isfinite(rflf).all(1)
             & (radf > FILL + 1).all(1) & (rflf > FILL + 1).all(1)
             & np.isfinite(latf) & (latf > -90) & (latf < 90))
    return dict(rad=radf[valid], rfl=rflf[valid], lat=latf[valid], lon=lonf[valid],
                wl=wl[keep], n_valid=int(valid.sum()), grid=rad.shape[:2], n_bands=B)


_WC_URL = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
           "v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
WC_NAMES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
            60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def _wc_tile(lat, lon):
    tla = int(np.floor(lat / 3) * 3)
    tlo = int(np.floor(lon / 3) * 3)
    return f"{'S' if tla < 0 else 'N'}{abs(tla):02d}{'W' if tlo < 0 else 'E'}{abs(tlo):03d}"


def worldcover_labels(lat, lon):
    """ESA WorldCover 2021 (v200) land-cover code for each (lat, lon), remote-sampled from the public S3
    COGs (no download). Independent, product-agnostic pixel label. 0 = nodata/unsampled (ocean, missing tile).
    WorldCover is 10 m point-sampled at ~60 m EMIT pixels, so labels carry mixed-pixel noise (reported)."""
    import os
    import rasterio
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    lat = np.asarray(lat); lon = np.asarray(lon)
    out = np.zeros(len(lat), np.int32)
    groups = {}
    for i in range(len(lat)):
        groups.setdefault(_wc_tile(lat[i], lon[i]), []).append(i)
    for tile, idx in groups.items():
        idx = np.asarray(idx)
        try:
            with rasterio.open(_WC_URL.format(tile=tile)) as src:
                out[idx] = [v[0] for v in src.sample(list(zip(lon[idx], lat[idx])))]
        except Exception:
            pass   # missing tile (ocean / not produced) -> stays 0, dropped downstream
    return out


def list_biomes(data_root=None):
    root = data_root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    out = []
    for d in sorted(glob.glob(os.path.join(root, "emit_*/"))):
        r, a = granule_files(d)
        if r and a:
            out.append((os.path.basename(d.rstrip("/"))[5:], d))   # (biome, dir), strip 'emit_'
    return out


def shift_cache_path(biome_dir):
    return os.path.join(biome_dir, "_shift_cache.npz")


def build_shift_cache(pool_px=8000, seed=12345, verbose=True):
    """Read each granule ONCE, sample a FIXED pool of valid pixels + WorldCover labels, and cache to a
    ~30 MB NPZ per biome, so the multi-seed experiment reads the cache instead of a 3.6 GB granule each
    time. Idempotent: skips biomes whose cache already exists."""
    rng = np.random.default_rng(seed)
    for name, d in list_biomes():
        cp = shift_cache_path(d)
        if os.path.exists(cp):
            if verbose:
                print(f"  cache exists: {name}", flush=True)
            continue
        g = load_granule(d)
        take = min(pool_px, g["n_valid"])
        idx = rng.choice(g["n_valid"], take, replace=False)
        wc = worldcover_labels(g["lat"][idx], g["lon"][idx])
        keep = wc > 0
        np.savez_compressed(cp, rad=g["rad"][idx][keep].astype(np.float32),
                            rfl=g["rfl"][idx][keep].astype(np.float32), wc=wc[keep].astype(np.int32),
                            lat=g["lat"][idx][keep], lon=g["lon"][idx][keep], wl=g["wl"].astype(np.float32))
        if verbose:
            print(f"  cached {name}: {int(keep.sum())} px (wc>0 of {take})", flush=True)


def load_shift_cache():
    """{biome: {rad, rfl, wc, lat, lon}} from the per-biome shift caches (build_shift_cache first)."""
    out = {}
    for name, d in list_biomes():
        cp = shift_cache_path(d)
        if os.path.exists(cp):
            z = np.load(cp)
            out[name] = dict(rad=z["rad"], rfl=z["rfl"], wc=z["wc"], lat=z["lat"], lon=z["lon"])
    return out


if __name__ == "__main__":
    import sys
    if "--build-cache" in sys.argv:
        build_shift_cache()
        raise SystemExit(0)
    bs = list_biomes()
    print(f"{len(bs)} biomes with PAIRED L1B+L2A:")
    for name, d in bs[:3] + bs[-1:]:
        g = load_granule(d)
        print(f"  {name:16s} grid {g['grid']} x {g['n_bands']} good bands; {g['n_valid']:>7d} valid px; "
              f"wl {g['wl'][0]:.0f}-{g['wl'][-1]:.0f}nm; RAD {g['rad'].mean():.2f}"
              f"[{g['rad'].min():.1f},{g['rad'].max():.1f}] vs RFL {g['rfl'].mean():.3f}"
              f"[{g['rfl'].min():.2f},{g['rfl'].max():.2f}]")
    print(f"\n{len(bs)} biomes total (train/deploy cross-biome split feasible)")
