#!/usr/bin/env python
"""Fetch EMIT L1B RADIANCE (source product) for each local EMIT biome that already has L2A REFLECTANCE
(target product), to build the real radiance->ISOFIT-reflectance atmospheric-correction shift for the
cross-sensor reliability study (2nd dataset). Matches each local L2A granule to its L1B by the shared
acquisition key <YYYYmmddTHHMMSS>_<orbit>_<scene> and downloads next to the L2A .nc.

  --go        actually download (default: SEARCH-ONLY, prints matches, downloads nothing)
  --biomes    comma list of biome dir stems to restrict (default: all local emit_* with L2A RFL)
"""
import argparse, glob, os, re, sys

BIO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
KEY = re.compile(r"_(\d{8}T\d{6})_(\d+)_(\d+)\.nc$")


def local_l2a():
    out = []
    for d in sorted(glob.glob(os.path.join(BIO_ROOT, "emit_*/"))):
        l2a = sorted(glob.glob(os.path.join(d, "EMIT_L2A_RFL_*.nc")))
        if l2a:
            out.append((os.path.basename(d.rstrip("/")), d, os.path.basename(l2a[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--biomes", default="")
    args = ap.parse_args()
    import earthaccess
    auth = earthaccess.login(strategy="netrc")
    print(f"earthaccess auth: {bool(auth)}", flush=True)
    want = set(b for b in args.biomes.split(",") if b)
    rows = [r for r in local_l2a() if not want or r[0] in want]
    print(f"{len(rows)} local biomes with L2A RFL; mode = {'DOWNLOAD' if args.go else 'SEARCH-ONLY'}\n", flush=True)
    got, miss = 0, 0
    for biome, d, l2a_name in rows:
        m = KEY.search(l2a_name)
        if not m:
            print(f"  {biome:20s} SKIP (unparseable {l2a_name})"); miss += 1; continue
        dt, orbit, scene = m.groups()
        key = f"{dt}_{orbit}_{scene}"
        l1b_target = os.path.join(d, l2a_name.replace("L2A_RFL", "L1B_RAD"))
        if os.path.exists(l1b_target):
            print(f"  {biome:20s} HAVE  {os.path.basename(l1b_target)}"); got += 1; continue
        t = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}T{dt[9:11]}:{dt[11:13]}:{dt[13:15]}"
        try:
            res = earthaccess.search_data(short_name="EMITL1BRAD", temporal=(t, t), count=80)
        except Exception as e:
            print(f"  {biome:20s} SEARCH-ERR {e}"); miss += 1; continue
        match = [g for g in res if key in str(g.get("meta", {}).get("native-id", "")) or key in str(g)]
        if not match:
            print(f"  {biome:20s} NO-MATCH for key {key} ({len(res)} L1B hits in window)"); miss += 1; continue
        print(f"  {biome:20s} {'FOUND' if not args.go else 'DOWNLOADING'} L1B {key}  ({len(match)} match)", flush=True)
        if args.go:
            try:
                earthaccess.download(match[:1], d)
                got += 1
            except Exception as e:
                print(f"    download-err: {e}"); miss += 1
    print(f"\n=== {got} have/fetched, {miss} missing ({'download' if args.go else 'search-only — rerun with --go'}) ===")


if __name__ == "__main__":
    main()
