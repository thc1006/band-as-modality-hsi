#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backfill the EMIT L2A MASK (cloud/cirrus flags) for granules downloaded with RFL+RFLUNCERT only.
Matches each granule by its ID (timestamp_orbit_scene parsed from the RFL filename) and downloads
ONLY that granule's MASK (no RFL/UNC re-download). Needed for the cloud-screened NDVI trend."""
import os, sys, glob
import earthaccess

DATA = os.environ.get("BANDSIM_DATA", "data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dl_emit_ndvi import REGIONS, emit_granule_id


MASK_BANDS_V001 = 8          # EMIT L2A MASK V001 layout. V002 adds bands, so an unexpected count is
                             # not a corrupt file but a DIFFERENT product, and must be caught here
                             # rather than silently mis-indexed downstream by a fixed band number.


def _mask_is_readable(path):
    """True only if the file opens AND holds a V001-shaped `mask` dataset.

    Existence of a filename proves nothing: an interrupted download leaves a 0-byte or truncated
    .nc that still satisfies a glob, and the old code accepted exactly that as a successful
    backfill. Opening it is the cheapest honest check."""
    try:
        import h5py
        if os.path.getsize(path) == 0:
            return False
        with h5py.File(path, "r") as f:
            m = f["mask"]
            if m.ndim != 3 or m.shape[-1] != MASK_BANDS_V001:
                print(f"[warn] {os.path.basename(path)}: mask shape {m.shape} is not V001 "
                      f"(rows,cols,{MASK_BANDS_V001}) -- treating as unusable")
                return False
            return True
    except Exception as e:
        print(f"[warn] {os.path.basename(path)}: unreadable ({type(e).__name__}: {e})")
        return False


def main():
    earthaccess.login(strategy="netrc")
    for name, bbox in REGIONS.items():
        d = f"{DATA}/emit_{name}"
        if not os.path.isdir(d):
            continue
        # Resolve the target acquisition the SAME way the analysis side does (phase8F_multi.
        # select_triple): index files by acquisition id and take the RFL-and-UNCERT intersection.
        # Refusing outright on "more than one RFL" was inconsistent with it -- the downloader's own
        # mismatch recovery appends a matched pair beside a stray file, producing a directory the
        # analysis accepts and the backfill rejected. Picking rfl[0] blindly is equally wrong: it
        # can parse the id of the stray and backfill a MASK for the wrong acquisition.
        # Index files by id, keeping the LIST per id: collapsing straight to a set would hide the
        # case this must catch -- two distinct files carrying the SAME acquisition id (a partial
        # re-download, a copy with a suffix). A set turns that into one entry and the ambiguity
        # disappears silently.
        def _by_id(pattern):
            out = {}
            for p in glob.glob(f"{d}/{pattern}"):
                g = emit_granule_id(p)
                if g is not None:
                    out.setdefault(g, []).append(p)
            return out
        rfl_by, unc_by = _by_id("*RFL_001*.nc"), _by_id("*RFLUNCERT*.nc")
        if not rfl_by:
            print(f"[skip] {name}: no RFL"); continue
        common = set(rfl_by) & set(unc_by)
        if len(common) != 1:
            print(f"[FAIL] {name}: need exactly ONE acquisition with both RFL and RFLUNCERT, found "
                  f"{sorted(common)} (rfl={sorted(rfl_by)}, unc={sorted(unc_by)}) -- skipping")
            continue
        gid = common.pop()
        dupes = {k: len(v) for k, v in (("RFL", rfl_by[gid]), ("RFLUNCERT", unc_by[gid])) if len(v) > 1}
        if dupes:
            print(f"[FAIL] {name}: {dupes} duplicate file(s) for acquisition {gid} -- ambiguous, "
                  f"refusing to guess which is authoritative"); continue
        # Defense-in-depth: skip ONLY if a MASK for THIS acquisition id is present. A MASK from a
        # different acquisition (e.g. a future mixed re-download) must NOT count as "present".
        # The analysis contract is a COMPLETE triple for ONE acquisition, so a MASK is only useful
        # next to an RFLUNCERT of the same id. Backfilling a MASK beside a mismatched (or missing)
        # uncertainty file produces a directory that looks complete and is not.
        unc = [p for p in glob.glob(f"{d}/*RFLUNCERT*.nc") if emit_granule_id(p) == gid]
        if not unc:
            print(f"[FAIL] {name}: no RFLUNCERT for id {gid} -- a MASK alone does not complete the "
                  f"triple, skipping"); continue
        masks = [m for m in glob.glob(f"{d}/*MASK_001*.nc") if emit_granule_id(m) == gid]
        if masks and _mask_is_readable(masks[0]):
            print(f"[skip] {name}: MASK present and readable (id {gid})"); continue
        if masks:
            # QUARANTINE before re-downloading. earthaccess skips a file that already exists, so
            # leaving the corrupt one in place makes the "re-download" a no-op and the granule stays
            # broken forever while every run cheerfully reports it as being retried. Rename rather
            # than delete: if the retry also fails, the evidence is still on disk.
            for m in masks:
                bad = m + ".corrupt"
                try:
                    os.replace(m, bad)
                    print(f"[warn] {name}: MASK {gid} unreadable -- moved aside to "
                          f"{os.path.basename(bad)} so the re-download is not skipped")
                except OSError as e:
                    print(f"[FAIL] {name}: cannot move aside unreadable MASK {os.path.basename(m)} "
                          f"({e}) -- a re-download would be silently skipped, so skipping")
                    masks = None
                    break
            if masks is None:
                continue
        if masks:
            print(f"[warn] {name}: present MASK id(s) {sorted({str(emit_granule_id(m)) for m in masks})} "
                  f"!= RFL id {gid} -- backfilling matched MASK")
        try:
            # Robust: search the EXACT granule by ID first (the target may fall outside any fixed
            # bbox+count window); fall back to a wider bbox search only if the name search is empty.
            res = earthaccess.search_data(short_name="EMITL2ARFL", granule_name=f"*{gid}*")
            if not res:
                res = earthaccess.search_data(short_name="EMITL2ARFL", bounding_box=bbox, count=100)
        except Exception as e:
            print(f"[err ] {name}: search {e}"); continue
        done = False
        for g in res:
            mask = [l for l in g.data_links() if "_MASK_001_" in l and gid in l]
            if mask:
                try:
                    got = earthaccess.download(mask, local_path=d)
                    # Judge success on the path THIS call returned, for THIS acquisition, and only
                    # after the file actually opens. Re-globbing the directory would accept a stale
                    # MASK from another acquisition that happened to be sitting there, and print
                    # "[ok] MASK <gid>" for a download that never happened.
                    paths = [str(p) for p in (got or []) if emit_granule_id(str(p)) == gid]
                    if paths and _mask_is_readable(paths[0]):
                        print(f"[ok  ] {name}: MASK {gid}"); done = True; break
                    print(f"[warn] {name}: download returned {len(paths)} matching path(s) for {gid}, "
                          f"none readable -- trying next candidate")
                except Exception as e:
                    print(f"[retry] {name}: {e}")
        if not done:
            print(f"[FAIL] {name}: MASK not found for {gid}")
    print("\n==== MASK backfill done ====")
    for d in sorted(glob.glob(f"{DATA}/emit*/")):
        r = len(glob.glob(f"{d}*RFL_001*.nc")); u = len(glob.glob(f"{d}*RFLUNCERT*.nc")); m = len(glob.glob(f"{d}*MASK_001*.nc"))
        print(f"  {os.path.basename(d.rstrip('/')):18s} RFL={r} UNC={u} MASK={m}")


if __name__ == "__main__":
    main()
