#!/usr/bin/env python
"""ACOLITE 2nd-processor batch for the CloudSEN12 flagship test scenes.

Per unique Sentinel-2 product (s2_id): download the full-tile L1C .SAFE from the GCS public
bucket, run FULL-TILE ACOLITE DSF atmospheric correction (representative AOT -- the whole tile
sees dark targets, unlike a 5 km ROI window), then crop the 509x509 CloudSEN12 window for EVERY
metadata row that uses that product (12 products span two ROIs -> two crops), with ZERO
resampling (the CloudSEN12 patch origin is an integer pixel offset on the native S2 10 m grid).

Output: one <index>.npy per metadata row = (11, 509, 509) float32 surface reflectance in the
ACOLITE band order [443,492,560,665,704,740,783,833,865,1614,2202] nm = S2 B1,B2,B3,B4,B5,B6,B7,
B8,B8A,B11,B12 (ACOLITE emits NO surface reflectance for B09/B10 by design). NaN where ACOLITE
produced no value. Resumable: an existing, valid <index>.npy is skipped.

Run with the verified probe venv (system GDAL 3.8.4, numpy<2). NOT the project .venv.
"""
import os, sys, json, time, glob, shutil, urllib.request, urllib.error, urllib.parse
import numpy as np
import multiprocessing as mp

REPO   = "/home/hctsai1006/cct/band-as-modality-hsi"
ACOLITE = os.path.join(REPO, "data/acolite_stage/acolite")
SCRATCH = "/tmp/claude-38627/-home-hctsai1006-cct-band-as-modality-hsi/5292c0a1-68c9-46da-991a-93739c9c8ad4/scratchpad/acolite_batch"
# WORK holds transient full-tile .SAFE (~346 MB) + full-tile L2R (~5 GB/scene, 11 bands x 10980^2
# float32) -- MUST be on the big /home mount (13 PB), never /tmp. Deleted per scene after cropping.
STAGE   = os.path.join(REPO, "data/acolite_stage")
WORK    = os.path.join(STAGE, "work")        # transient per-scene .SAFE + acolite_out (deleted after crop)
CROPS   = os.path.join(STAGE, "crops")       # persistent output: <index>.npy (~11 MB each, ~11 GB total)
LOGDIR  = os.path.join(SCRATCH, "logs")
for d in (WORK, CROPS, LOGDIR): os.makedirs(d, exist_ok=True)

GCS_OBJ = "https://storage.googleapis.com/gcp-public-data-sentinel-2/o"  # unused; kept for ref
GCS_LIST = "https://storage.googleapis.com/storage/v1/b/gcp-public-data-sentinel-2/o"
GCS_DL   = "https://storage.googleapis.com/gcp-public-data-sentinel-2"

# ACOLITE rhos wavelengths (nm) we expect, in S2 band order (B1..B8A,B11,B12; NO B9/B10).
ACO_WL = [443, 492, 560, 665, 704, 740, 783, 833, 865, 1614, 2202]
N_ACO_BANDS = len(ACO_WL)
CS_SHAPE = 509


def log(scene, msg):
    with open(os.path.join(LOGDIR, "batch.log"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} [{scene}] {msg}\n")


def gcs_get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download_safe(s2_id, dest):
    """Reconstruct the minimal .SAFE tree ACOLITE needs and download it (8-way parallel curl-ish).

    Needs: MTD_MSIL1C.xml (root), GRANULE/<g>/MTD_TL.xml, 13 IMG_DATA/*_B*.jp2, and
    GRANULE/<g>/QI_DATA/*MSK_DETFOO*.gml (default geometry_type=grids_footprint reads them)."""
    tile = s2_id.split("_")[5][1:]                       # T19FDF -> 19FDF
    prefix = f"tiles/{tile[:2]}/{tile[2]}/{tile[3:]}/{s2_id}.SAFE/"
    # enumerate all objects under the .SAFE (paginated)
    objs, tok = [], None
    while True:
        url = f"{GCS_LIST}?prefix={prefix}&maxResults=1000" + (f"&pageToken={tok}" if tok else "")
        d = gcs_get_json(url)
        objs += [it["name"] for it in d.get("items", [])]
        tok = d.get("nextPageToken")
        if not tok: break
    want = []
    for name in objs:
        rel = name[len(prefix):]
        base = os.path.basename(name)
        keep = (base == "MTD_MSIL1C.xml" or base == "MTD_TL.xml"
                or (rel.count("IMG_DATA/") and base.endswith(".jp2") and "_B" in base and "TCI" not in base)
                or ("QI_DATA/" in rel and "MSK_DETFOO" in base))
        if keep: want.append((name, rel))
    if not any(b[1].endswith("MTD_MSIL1C.xml") for b in want):
        raise RuntimeError(f"{s2_id}: MTD_MSIL1C.xml not found among {len(objs)} objects")
    safe_dir = os.path.join(dest, f"{s2_id}.SAFE")
    def fetch(item):
        name, rel = item
        out = os.path.join(safe_dir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        u = f"{GCS_DL}/{urllib.parse.quote(name)}"
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(u, out); return os.path.getsize(out)
            except Exception as e:
                if attempt == 3: raise
                time.sleep(1.5 * (attempt + 1))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(fetch, want))
    return safe_dir


def compute_limit(rows, buffer_deg=0.01):
    """Union of the scene's ROI footprints -> [S, W, N, E] degrees + buffer, so the 509x509 crop is
    fully covered. All rows share the tile's UTM CRS. transform = (px,0,X0,0,-px,Y0)."""
    from pyproj import Transformer
    epsg = rows[0][1]
    minX = min(r[2][2] for r in rows)
    maxY = max(r[2][5] for r in rows)
    maxX = max(r[2][2] + r[2][0] * r[3] for r in rows)     # X0 + px*shape
    minY = min(r[2][5] - r[2][0] * r[3] for r in rows)     # Y0 - px*shape
    tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for X, Y in [(minX, minY), (minX, maxY), (maxX, minY), (maxX, maxY)]:
        lon, lat = tr.transform(X, Y); lons.append(lon); lats.append(lat)
    return [min(lats) - buffer_deg, min(lons) - buffer_deg,
            max(lats) + buffer_deg, max(lons) + buffer_deg]  # [S, W, N, E]


def run_acolite_fixed(safe_dir, out_dir, limit, aot=0.1):
    """ACOLITE with a FIXED climatological AOT550 (bypasses DSF). DSF over-estimates AOT and inflates
    surface reflectance on bright/cloudy scenes (CloudSEN12 is a cloud dataset), so a fixed modest AOT
    gives a stable, physically-plausible, Sen2Cor-independent correction (ACOLITE's own 6SV LUT +
    gas correction + code lineage, different from Sen2Cor's libRadtran). Processing only the ROI
    window is valid because the AOT no longer needs a scene-wide dark target."""
    sys.path.insert(0, ACOLITE)
    import acolite as ac
    settings = {
        "inputfile": safe_dir,
        "output": out_dir,
        "limit": limit,                 # [S, W, N, E] deg -- ROI footprint(s) + buffer
        "dsf_fixed_aot": float(aot),    # FIXED AOT550
        "dsf_fixed_lut": "ACOLITE-LUT-202110-MOD2",
        # CRITICAL settings (root cause traced in acolite_l2r.py): the S2 sensor defaults set
        # dsf_aot_estimate='tiled' + resolved_geometry=True. On MULTI-DETECTOR scenes the tiling block
        # (:467-470) DELETES data_mem['pressure'/'sza'/...] before dsf_fixed_aot switches the estimate
        # to 'fixed' (too late, :645), so the per-pixel/reverse-LUT correction KeyErrors on 'pressure'.
        # Setting dsf_aot_estimate='fixed' explicitly skips the tiling deletion; resolved_geometry=False
        # forces scene-mean geometry via the forward LUT (data_mem['*_mean'], always present). No patch.
        "dsf_aot_estimate": "fixed",
        "resolved_geometry": False,
        "l2r_export_geotiff": True,
        "ancillary_data": False,        # CPU-only: no NASA EARTHDATA/MERRA2 network+auth dependency
        "verbosity": 0,
    }
    ac.acolite.acolite_run(settings)


def crop_rows(out_dir, rows):
    """For each metadata row (index, epsg, transform, shape), window-read the 509x509 crop at the
    row's exact affine from every ACOLITE rhos band, ZERO resampling. Returns {index: (11,509,509)}.
    Bands are ordered by ACO_WL; a missing wl -> all-NaN band (keeps a fixed 11-band tensor)."""
    import rasterio
    from rasterio.windows import Window
    rhos = glob.glob(os.path.join(out_dir, "*_L2R_rhos_*.tif"))
    by_wl = {}
    for p in rhos:
        wl = int(p.split("_rhos_")[1].split(".tif")[0])
        by_wl[wl] = p
    result = {}
    for idx, epsg, tf, shape in rows:
        a, _, X0, _, _, Y0 = tf                          # affine "10,0,X0,0,-10,Y0" (px=a, top-left X0,Y0)
        out = np.full((N_ACO_BANDS, shape, shape), np.nan, np.float32)
        for bi, wl in enumerate(ACO_WL):
            if wl not in by_wl:
                continue
            with rasterio.open(by_wl[wl]) as src:
                gt = src.transform                       # both on native S2 10 m UTM grid -> integer offset
                col_off = (X0 - gt.c) / gt.a
                row_off = (Y0 - gt.f) / gt.e
                # assert integer alignment (zero resampling); tolerate <1e-3 px float noise then round
                if abs(col_off - round(col_off)) > 1e-3 or abs(row_off - round(row_off)) > 1e-3:
                    raise RuntimeError(f"row {idx} wl {wl}: non-integer offset "
                                       f"col={col_off:.4f} row={row_off:.4f} (grid mismatch)")
                win = Window(round(col_off), round(row_off), shape, shape)
                arr = src.read(1, window=win, boundless=True, fill_value=np.nan)  # NaN outside tile
                out[bi] = arr.astype(np.float32)
        result[idx] = out
    return result


def process_scene(args):
    s2_id, rows = args                                    # rows: list of (index, epsg, transform, shape)
    todo = [r for r in rows if not _crop_valid(os.path.join(CROPS, f"{r[0]}.npy"))]
    if not todo:
        return (s2_id, "skip", 0.0)
    t0 = time.time()
    scene_work = os.path.join(WORK, s2_id)
    try:
        safe = download_safe(s2_id, scene_work)
        t_dl = time.time() - t0
        out_dir = os.path.join(scene_work, "aco_out"); os.makedirs(out_dir, exist_ok=True)
        t1 = time.time()
        limit = compute_limit(rows)                       # union of the scene's ROI footprints + buffer
        run_acolite_fixed(safe, out_dir, limit)
        t_ac = time.time() - t1
        crops = crop_rows(out_dir, [(r[0], r[1], r[2], r[3]) for r in todo])
        for idx, arr in crops.items():
            np.save(os.path.join(CROPS, f"{idx}.npy"), arr)
        log(s2_id, f"OK dl={t_dl:.0f}s ac={t_ac:.0f}s rows={len(todo)} nanfrac={np.mean([np.isnan(a).mean() for a in crops.values()]):.3f}")
        return (s2_id, "ok", time.time() - t0)
    except Exception as e:
        log(s2_id, f"FAIL {type(e).__name__}: {e}")
        return (s2_id, f"fail:{type(e).__name__}", time.time() - t0)
    finally:
        shutil.rmtree(scene_work, ignore_errors=True)     # disk mgmt: keep only the small crops


def _crop_valid(path):
    if not os.path.exists(path): return False
    try:
        a = np.load(path, mmap_mode="r")
        return a.shape == (N_ACO_BANDS, CS_SHAPE, CS_SHAPE)
    except Exception:
        return False


def load_rows():
    import csv
    def read(split):
        with open(os.path.join(REPO, f"data/cloudsen12/{split}/metadata.csv")) as f:
            return list(csv.DictReader(f))
    # LEAK-GUARD: mirror phase8R -- drop test rows whose product is also in TRAIN.
    train = set(r["s2_id"] for r in read("train") if r.get("s2_id"))
    by_scene = {}
    for i, r in enumerate(read("test")):
        if r["s2_id"] in train:                            # leak-guarded out
            continue
        tf = tuple(float(x) for x in str(r["proj_transform"]).split(","))
        row = (int(i), int(r["proj_epsg"]), tf, int(r["proj_shape"]))
        by_scene.setdefault(r["s2_id"], []).append(row)
    return by_scene


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-scenes", type=int, default=None, help="process only the first N scenes (smoke)")
    args = ap.parse_args()
    by_scene = load_rows()
    scenes = sorted(by_scene.items())
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    total_rows = sum(len(v) for _, v in scenes)
    print(f"[batch] {len(scenes)} scenes / {total_rows} crop rows (leak-guarded) | {args.workers} workers")
    t0 = time.time()
    done = {"ok": 0, "skip": 0}; fails = []
    with mp.Pool(args.workers) as pool:
        for i, (s2_id, status, dt) in enumerate(pool.imap_unordered(process_scene, scenes), 1):
            if status.startswith("fail"): fails.append((s2_id, status))
            else: done[status] = done.get(status, 0) + 1
            if i % 10 == 0 or i == len(scenes):
                el = time.time() - t0
                print(f"[batch] {i}/{len(scenes)} ok={done['ok']} skip={done['skip']} "
                      f"fail={len(fails)} | {el/60:.1f}min | ~{el/i*(len(scenes)-i)/60:.0f}min left",
                      flush=True)
    print(f"[batch] DONE ok={done['ok']} skip={done['skip']} fail={len(fails)} in {(time.time()-t0)/60:.1f}min")
    if fails:
        print("[batch] failures:", fails[:20])
    n_crops = len(glob.glob(os.path.join(CROPS, "*.npy")))
    print(f"[batch] crops on disk: {n_crops}/{total_rows}")


if __name__ == "__main__":
    main()
