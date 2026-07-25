#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive pre-flight diagnostic for ALL EMIT granules BEFORE the expensive training run.

For each granule: cloud/cirrus/dilated/water flag %, the REAL scene usable fraction (valid pixels
/ all pixels under the CLOUD-ONLY screen -- band 0 only, because the official Dilated/Cirrus flags
over-trigger on haze; see docs/review/OFFICIAL_ISSUES_TO_REPORT.md), sampled NDVI (median + IQR),
and mean per-band uncertainty. Emits a provenance header (granule IDs + policy + timestamp) so the
printed table is reproducible. Extraction is cached for the run.

NOTE on `usable%`: this is the REAL scene fraction n_valid/n_total (from extract(return_stats=True)),
NOT sampled/40000 -- the latter saturates at 100% for any scene with >=40k valid pixels and is
meaningless. `usable%` is under the CLOUD-ONLY policy, not a full official quality screen.

NOTE on `in%` / `ofbox%`: the preflight extracts through the SAME geographic crop the analysis uses
(phase8F_multi.region_bbox_for). Without that crop this table described a different pixel population
than the study it is a pre-flight for -- e.g. the 'sumatra' granule reported a NDVI median over a
whole scene of which 0.0% lay inside the box it is named after. `in%` is how much of the downloaded
granule is inside its declared box; `ofbox%` is how much of THAT survives the quality screen. A
granule with no declared box is printed with an UNVERIFIED marker: its numbers are real, but its
region name is not checkable, so it must not be read as evidence about that biome.
"""
import os, sys, glob, datetime
import numpy as np
import h5py

# Absolute, __file__-derived: the old relative "experiments" insert silently resolved against the
# CALLER's cwd, so this script only worked when launched from the repo root and otherwise died on an
# ImportError that looks like a missing dependency.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "experiments"))
from phase8F_multi import (extract, discover_granules, region_bbox_for, select_triple,  # noqa: E402
                           safe_ndvi, NDVI_DENOM_FLOOR, mask_band_indices)

RED_NM, NIR_NM = 665.0, 860.0            # NDVI anchors; actual snapped EMIT band centres printed below
# By NAME, not index: the file's own mask_bands is the authority on which band is Cloud, so a
# product that reorders them fails loudly instead of screening the wrong thing. Passed explicitly
# even though it matches the callee's default -- this header prints the policy as fact, and a moved
# default would drift the printed policy away from the screen actually applied.
MASK_FLAGS = ("cloud",)
MASK_POLICY = ("cloud-only (MASK band 0); Dilated/Cirrus excluded. NOTE: 'Cirrus/Dilated over-trigger "
               "on haze' is OUR empirical hypothesis from these granules, not an documented NASA "
               "finding -- report the official-flag screen alongside it, not instead of it.")


def _granule_id(gdir):
    """The acquisition actually used, resolved by the SAME matcher extract() uses. The old version
    took glob(...)[0] and string-replaced the name, so with a stray file in the directory the header
    could advertise a different acquisition than the one the table was computed from."""
    try:
        return select_triple(gdir)[0]
    except Exception as e:
        return f"UNRESOLVED ({type(e).__name__})"


def check_stats(name, R, U, wl, st):
    """Invariants that must hold for the returned sample, checked rather than assumed. These are the
    quantities the printed table divides by and reports as percentages; if any of them is incoherent
    the table is arithmetic on nonsense, and a wrong number here is invisible (it just looks like a
    slightly odd granule) whereas a raised error is not.

    Deliberately NOT written with `assert`: `python -O` strips assert statements entirely, so a
    validation written that way silently evaporates under the one flag someone is most likely to add
    for a long production run -- precisely when nobody is watching the output."""
    nv, nt, ni = st["n_valid"], st["n_total"], st["n_inbox"]
    checks = [
        (0 <= nv <= nt, f"n_valid={nv} outside [0, n_total={nt}]"),
        (0 <= ni <= nt, f"n_inbox={ni} outside [0, n_total={nt}]"),
        (nv <= ni, f"n_valid={nv} exceeds n_inbox={ni} (valid pixels must be in-box)"),
        (bool(np.isfinite(st["frac"])), f"non-finite usable fraction {st['frac']}"),
        (bool(np.isclose(st["frac"], nv / max(1, nt))),
         f"frac={st['frac']} disagrees with n_valid/n_total={nv / max(1, nt)}"),
        (R.shape == U.shape, f"reflectance {R.shape} != uncertainty {U.shape}"),
        (R.shape[1] == len(wl), f"{R.shape[1]} bands but {len(wl)} wavelengths"),
        (R.shape[0] <= nv, f"sampled {R.shape[0]} px from only {nv} valid px"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(f"{name}: {msg}")


def main():
    GRAN = discover_granules()
    # ---- provenance header (reproducibility) ----
    print(f"# EMIT preflight  generated={datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"# screen policy : {MASK_POLICY}")
    print(f"# geographic    : cropped to each granule's DECLARED region box (emit_regions.REGIONS); "
          f"granules with no box are marked UNVERIFIED and reported uncropped")
    print(f"# NDVI          : red~{RED_NM:.0f}nm, NIR~{NIR_NM:.0f}nm, rejected where "
          f"(red+NIR) < {NDVI_DENOM_FLOOR} (actual snapped centres printed per first granule)")
    print(f"# granules ({len(GRAN)}):")
    for name, gdir in GRAN.items():
        bb = region_bbox_for(name)
        print(f"#   {name:17s} id={_granule_id(gdir)}  box={bb if bb else 'NONE (UNVERIFIED)'}")
    # Fail CLOSED on an empty discovery: exiting 0 here told CI that a preflight over ZERO granules
    # had succeeded, which is the one outcome that must never look like a pass.
    if not GRAN:
        print("\nFATAL: discovered 0 EMIT granules (need data/emit*/ with RFL+RFLUNCERT)")
        raise SystemExit(1)
    print(f"{'granule':17s} {'cloud%':>6s} {'cirr%':>6s} {'dil%':>6s} {'water%':>6s} "
          f"{'in%':>6s} {'usable%':>7s} {'ofbox%':>7s} {'nvalid':>8s} {'ntotal':>9s} "
          f"{'NDVImed':>7s} {'NDVIiqr':>13s} {'negNDVI%':>8s} {'ndviRej%':>8s} {'ndviOOR%':>8s} "
          f"{'meanUnc':>7s}  flags")
    issues = []
    printed_wl = False
    for name, gdir in GRAN.items():
        bbox = region_bbox_for(name)
        try:
            mk = select_triple(gdir)[3]
            with h5py.File(mk) as f:
                M = f["mask"]
                if M.ndim != 3 or M.shape[-1] != 8:                       # guard silent reshape of a malformed MASK
                    raise ValueError(f"unexpected MASK shape {M.shape} (expected (rows,cols,8))")
                Mf = M[:].reshape(-1, 8)
            # mask_flags is passed EXPLICITLY even though (0,) is the callee's default: this header
            # prints MASK_POLICY as fact, and if the default ever moves the printed policy, the cache
            # key and the actual screen would drift apart with nothing to notice.
            R, U, wl, st = extract(gdir, 40000, seed=0, return_stats=True,
                                   region_bbox=bbox, mask_flags=MASK_FLAGS)   # cached for the run
            check_stats(name, R, U, wl, st)
            # The flag percentages must describe the SAME pixels as the rest of the row. Computing
            # them over the whole granule while the NDVI comes from the crop puts two different
            # populations on one line, and the reader has no way to tell.
            if bbox is not None:
                w, s, e, n = (float(v) for v in bbox)
                with h5py.File(select_triple(gdir)[1]) as f:
                    lat = f["location/lat"][:]; lon = f["location/lon"][:]
                inbox = ((lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)).reshape(-1)
                Mf = Mf[inbox]
            # Resolve by LABEL, not by fixed index. Hardcoding 0/1/2/4 matched V001 by luck rather
            # than by check: a product whose mask_bands are ordered differently would have this row
            # report water under the "cloud%" heading, with no error anywhere. The same file's own
            # `mask_bands` is the only authority on what band 2 means.
            i_cloud, i_cirr, i_water, i_dil = mask_band_indices(
                mk, ["cloud", "cirrus", "water", "dilated"])
            cloud = 100 * (Mf[:, i_cloud] > 0).mean(); cirr = 100 * (Mf[:, i_cirr] > 0).mean()
            dil = 100 * (Mf[:, i_dil] > 0).mean(); water = 100 * (Mf[:, i_water] > 0).mean()
            ndvi, nst = safe_ndvi(R, wl, RED_NM, NIR_NM)
            if not printed_wl:
                print(f"#   NDVI actual band centres: red={nst['red_nm']:.1f}nm  NIR={nst['nir_nm']:.1f}nm")
                printed_wl = True
            fin = np.isfinite(ndvi)
            if not fin.any():
                raise ValueError(f"every sampled pixel failed the NDVI denominator floor "
                                 f"({NDVI_DENOM_FLOOR}) -- red+NIR is indistinguishable from zero")
            v = ndvi[fin]
            med = np.median(v); q1, q3 = np.percentile(v, [25, 75])
            negfrac = 100 * (v < 0).mean()
            usable = 100 * st["frac"]                                     # REAL scene fraction, not sampled/40000
            ofbox = 100 * st["frac_inbox"]                                # ... and the same against the box
            fl = []
            if bbox is None: fl.append("UNVERIFIED_REGION")                # biome label not checkable
            if st["n_valid"] < 40000: fl.append("FEWVALID")               # fewer valid px than the sample target
            if bbox is not None and st["inside_pct"] < 50: fl.append(f"OUTOFBOX{st['inside_pct']:.0f}")
            if water > 15: fl.append(f"WATER{water:.0f}")
            if negfrac > 25: fl.append(f"NEG{negfrac:.0f}")
            if nst["rejected_pct"] > 1: fl.append(f"NDVIREJ{nst['rejected_pct']:.0f}")
            if nst["oor_pct"] > 1: fl.append(f"NDVIOOR{nst['oor_pct']:.0f}")
            if (q3 - q1) > 0.5: fl.append("WIDE_IQR")                     # wide spread (water+land mix), NOT necessarily bimodal
            # `--` rather than 100.0 when there is no box: extract() reports inside_pct=100 for an
            # uncropped granule, which read as "fully inside its region" for the three granules that
            # have no region at all -- the most confident-looking cell in the table would have been
            # the one backed by nothing.
            inb = f"{st['inside_pct']:6.1f}" if bbox is not None else "    --"
            ofb = f"{ofbox:6.1f}%" if bbox is not None else "     --"
            print(f"{name:17s} {cloud:6.1f} {cirr:6.1f} {dil:6.1f} {water:6.1f} "
                  f"{inb} {usable:6.1f}% {ofb} {st['n_valid']:8d} {st['n_total']:9d} "
                  f"{med:+7.2f} [{q1:+.2f},{q3:+.2f}] {negfrac:7.1f} {nst['rejected_pct']:8.2f} "
                  f"{nst['oor_pct']:8.2f} {U.mean():7.4f}  {','.join(fl)}")
            if fl: issues.append((name, fl, med))
        except Exception as e:
            print(f"{name:17s}  EXCLUDED: {type(e).__name__}: {e}")
            issues.append((name, ["EXCLUDED"], None))

    print("\n==== ISSUES TO DECIDE ====")
    for n, fl, med in issues:
        print(f"  {n}: {fl}" + (f" (NDVI={med:+.2f})" if med is not None else ""))
    n_excl = sum(1 for _, fl, _ in issues if "EXCLUDED" in fl)
    print(f"\nusable granules: {len(GRAN) - n_excl} / {len(GRAN)}")
    # Same fail-closed rule as the empty-discovery case: a preflight where nothing survived is a
    # FAILED preflight, and exiting 0 would let a CI job downstream train on nothing and pass.
    if n_excl == len(GRAN):
        print("FATAL: every discovered granule was excluded -- preflight FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
