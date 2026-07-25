#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download EMIT L2A granules spanning the NDVI spectrum, for the per-pixel-correlation-vs-NDVI
generality trend. One (cloud-minimal) granule per region: RFL + RFLUNCERT are taken from the SAME
granule's data_links (so they are guaranteed to be the same acquisition). The MASK is fetched
separately by dl_mask_backfill.py, matched by granule ID.

This SORTS candidates clearest-first (by UMM CloudCover) and downloads the clearest one that has
both RFL+UNCERT -- it does NOT hard-reject cloudy granules; final cloud/quality screening happens
downstream (preflight_emit.py + phase8F_multi.extract, which drops any granule with <2000 valid px).
Robust: skips already-present regions, continues past regions with no EMIT coverage."""
import os, sys, glob, math
import earthaccess

DATA = os.environ.get("BANDSIM_DATA", "data")

# Minimum fraction of a candidate granule's own footprint that must lie inside the requested box.
# An EMIT scene is ~75 km across and the boxes here are several degrees, so a genuine hit clears this
# easily; the value exists to reject edge-grazers. Without it the `sumatra` slot was filled by a
# granule with 0.0% of its pixels in-box, sitting over water. Downstream still crops per pixel and
# still enforces its own usable-pixel floor -- this only stops us spending 4 GB on a scene that
# cannot represent the region it is named after.
MIN_INBOX_FRAC = 0.25


# The region table and id parser live in the dependency-free `emit_regions` module so the ANALYSIS
# side (phase8F_multi's geographic crop) can read the declared boxes without importing earthaccess.
# Re-exported here so dl_mask_backfill.py's existing import keeps working from one definition.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emit_regions import REGIONS, emit_granule_id   # noqa: E402,F401


def cloud_cover(g):
    """Best-effort cloud-cover PERCENTAGE (0-100) from UMM; 999 if unknown.

    UNITS ARE NOT GUARANTEED. CMR providers populate CloudCover as either a percentage (20 = 20%) or
    a fraction (0.8 = 80%), and an earlier comment here claimed a unit mismatch could not change the
    RANKING. That was wrong: raw-sorting 0.8 and 20 puts the 80%-cloudy granule first. Values <= 1
    are therefore rescaled to percent.

    The rescaling is itself ambiguous — a genuine 0.8% cloud cover is indistinguishable from the
    fraction 0.8 — but the error is one-directional and harmless here: a truly pristine granule gets
    read as 80% cloudy and loses its place in the ranking, so we pass over a good scene rather than
    choose a bad one. Overlap, not cloud, is the primary sort key anyway."""
    try:
        umm = g["umm"]
        raw = None
        if "CloudCover" in umm and umm["CloudCover"] is not None:
            raw = float(umm["CloudCover"])
        else:
            for a in umm.get("AdditionalAttributes", []):
                if str(a.get("Name", "")).upper().replace(" ", "") in ("CLOUDCOVER", "CLOUD_COVER"):
                    raw = float(a["Values"][0]); break
        if raw is not None and math.isfinite(raw):
            return raw * 100.0 if 0.0 <= raw <= 1.0 else raw    # fraction -> percent (see docstring)
    except Exception:
        pass
    return 999.0


def rfl_unc_urls(g):
    links = g.data_links()
    rfl = [l for l in links if "_RFL_001_" in l and "RFLUNCERT" not in l]
    unc = [l for l in links if "RFLUNCERT" in l]
    return (rfl[:1] + unc[:1]) if (rfl and unc) else []


def _is_readable_nc(path):
    """True only if the file opens as HDF5 and is non-empty. `earthaccess.download()` returning a
    path proves a transfer was attempted, not that it finished: an interrupted or truncated download
    leaves a real path pointing at an unusable file, and judging success on the string alone would
    report [ok] for it."""
    try:
        import h5py
        if os.path.getsize(path) == 0:
            return False
        with h5py.File(path, "r") as f:
            return len(f.keys()) > 0
    except Exception as e:
        print(f"[warn] {os.path.basename(path)}: downloaded but unreadable ({type(e).__name__})")
        return False


def granule_extent(g):
    """(west, south, east, north) of a granule from its UMM spatial metadata, or None.

    Handles both shapes CMR uses: an explicit BoundingRectangle, or a GPolygon whose boundary points
    we reduce to their envelope. The envelope OVERSTATES a real EMIT footprint (the swath is a
    parallelogram, not an axis-aligned box), so any fraction derived from it is APPROXIMATE and its
    direction is not guaranteed: enlarging the footprint grows both the intersection and the area it
    is divided by. It is good enough to separate a scene centred in the box from one grazing its
    edge, which is the decision being made -- but the pixel-level crop downstream, not this, is what
    actually determines which pixels are used."""
    try:
        geom = g["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
    except Exception:
        return None
    for r in geom.get("BoundingRectangles", []) or []:
        try:
            return (float(r["WestBoundingCoordinate"]), float(r["SouthBoundingCoordinate"]),
                    float(r["EastBoundingCoordinate"]), float(r["NorthBoundingCoordinate"]))
        except Exception:
            pass
    for p in geom.get("GPolygons", []) or []:
        try:
            pts = p["Boundary"]["Points"]
            lons = [float(q["Longitude"]) for q in pts]
            lats = [float(q["Latitude"]) for q in pts]
            return (min(lons), min(lats), max(lons), max(lats))
        except Exception:
            pass
    return None


def inbox_fraction(g, bbox):
    """Fraction of the granule's own footprint that falls inside the requested box, or None.

    `earthaccess.search_data(bounding_box=...)` returns every granule that merely INTERSECTS the box
    and does not clip anything, so ranking candidates by cloud cover alone can hand back a scene that
    only grazes the box edge. Measured consequence before this check existed: the granule downloaded
    for `sumatra` had 0.0% of its pixels inside its own declared box (it spans lon 104.02-105.05
    against a box ending at 104) and sat over water -- median NDVI -0.05, 77.7% negative. Selecting
    on overlap turns that from an exclusion at analysis time into usable data at download time."""
    ext = granule_extent(g)
    if ext is None:
        return None
    gw, gs, ge, gn = ext
    w, s, e, n = bbox
    iw, ie = max(gw, w), min(ge, e)
    isth, inth = max(gs, s), min(gn, n)
    if ie <= iw or inth <= isth:
        return 0.0
    g_area = max((ge - gw) * (gn - gs), 1e-12)
    return float(((ie - iw) * (inth - isth)) / g_area)


def main():
    # Prefer an EARTHDATA_TOKEN (bearer JWT) when present, else fall back to ~/.netrc. Token auth lets a
    # session download without a stored password; the env var is never logged.
    earthaccess.login(strategy="environment" if os.environ.get("EARTHDATA_TOKEN") else "netrc")
    summary = []
    for name, bbox in REGIONS.items():
        outdir = f"{DATA}/emit_{name}"
        rfl_present = glob.glob(f"{outdir}/*RFL_001*.nc")
        unc_present = glob.glob(f"{outdir}/*RFLUNCERT*.nc")
        if rfl_present and unc_present:
            # Defense-in-depth: skip ONLY if the present RFL and UNCERT are the SAME acquisition.
            # "some RFL + some UNCERT" is not enough -- a future partial/mixed re-download could leave
            # a mismatched pair that would silently pair unrelated scenes downstream.
            rid, uid = emit_granule_id(rfl_present[0]), emit_granule_id(unc_present[0])
            if rid is not None and rid == uid:
                print(f"[skip] {name}: already present (id {rid})")
                summary.append((name, "present")); continue
            print(f"[warn] {name}: present RFL id {rid} != UNCERT id {uid} -- re-downloading matched pair")
        try:
            res = earthaccess.search_data(short_name="EMITL2ARFL", bounding_box=bbox, count=40)
        except Exception as e:
            print(f"[err ] {name}: search failed {e}"); summary.append((name, "search-fail")); continue
        if not res:
            print(f"[none] {name}: no EMIT granules in bbox"); summary.append((name, "no-coverage")); continue
        # Rank by OVERLAP first, cloud second. Cloud-only ranking is what let a granule that merely
        # grazed the box edge win; overlap decides whether the region name means anything at all,
        # and a cloudy-but-inside scene is still salvageable downstream while an outside one is not.
        scored, unknown = [], 0
        for g in res:
            f = inbox_fraction(g, bbox)
            if f is None:
                # UNKNOWN IS NOT A PASS. Treating "we could not measure the overlap" as acceptable
                # would let exactly the granule this gate exists to reject slip through, and it would
                # do so silently, on the candidates whose metadata we understand least.
                unknown += 1
                continue
            scored.append((f, cloud_cover(g), g))
        if unknown:
            print(f"[warn] {name}: {unknown} candidate(s) had no spatial extent in UMM -- SKIPPED "
                  f"(unmeasurable overlap is not treated as acceptable overlap)")
        qualified = [t for t in scored if t[0] >= MIN_INBOX_FRAC]
        if not qualified:
            best = max((t[0] for t in scored), default=0.0)
            print(f"[fail] {name}: no candidate has >={MIN_INBOX_FRAC:.0%} of its footprint inside the "
                  f"box (best {best:.1%}) -- refusing to download a scene that would not represent "
                  f"this region"); summary.append((name, f"low-overlap({best:.0%})")); continue
        qualified.sort(key=lambda t: (-t[0], t[1]))              # most in-box, then clearest
        got = False
        for frac, cc, g in qualified[:8]:
            urls = rfl_unc_urls(g)
            if not urls:
                continue
            os.makedirs(outdir, exist_ok=True)
            gid = emit_granule_id(urls[0])
            if gid is None:
                # Without a parsed id every later `== gid` comparison degenerates to None == None,
                # which is True — two unparseable filenames would "match" and be accepted as one
                # acquisition. Refuse rather than let that stand in for verification.
                print(f"[warn] {name}: candidate URL {urls[0].split('/')[-1][:50]} is not a "
                      f"recognisable EMIT L2A V001 product name -- skipping")
                continue
            try:
                print(f"[dl  ] {name}: in-box~{frac:.0%} cloud~{cc:.0f}% -> {urls[0].split('/')[-1][:40]}...")
                got_paths = earthaccess.download(urls, local_path=outdir)
                # Judge success on the paths THIS call returned, not by re-globbing a directory that
                # may already hold older files. `download()` skips files that exist and can return
                # nothing at all; the old check then found the pre-existing RFL/UNCERT and printed
                # [ok] for a download that never happened. Require both products, for THIS granule id.
                paths = [str(p) for p in (got_paths or [])]
                # A returned path is a STRING, not evidence of a usable file: a truncated or
                # interrupted transfer still yields a path. Open each one before believing it.
                have_rfl = any("_RFL_001_" in p and "RFLUNCERT" not in p and emit_granule_id(p) == gid
                               and _is_readable_nc(p) for p in paths)
                have_unc = any("RFLUNCERT" in p and emit_granule_id(p) == gid
                               and _is_readable_nc(p) for p in paths)
                if have_rfl and have_unc:
                    print(f"[ok  ] {name}: downloaded id {gid} (in-box~{frac:.0%}, cloud~{cc:.0f}%)")
                    summary.append((name, f"ok in-box~{frac:.0%} cloud~{cc:.0f}")); got = True; break
                print(f"[warn] {name}: download() returned {len(paths)} path(s) but not a matched "
                      f"RFL+RFLUNCERT pair for id {gid} -- trying the next candidate")
            except Exception as e:
                print(f"[retry] {name}: {e}")
                continue
        if not got:
            print(f"[fail] {name}: could not download RFL+UNCERT")
            summary.append((name, "dl-fail"))
    print("\n==== DOWNLOAD SUMMARY ====")
    for n, s in summary:
        print(f"  {n:18s} {s}")
    print(f"total present: {len(glob.glob(f'{DATA}/emit*/'))} granule dirs")


if __name__ == "__main__":
    main()
