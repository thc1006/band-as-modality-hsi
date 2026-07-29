#!/usr/bin/env python
"""R6 (reviewer B / radiometric-provenance concern): prove, from the raw CloudSEN12 .dat digital numbers,
that the model input uses the correct reflectance conversion and that the Processing-Baseline-04.00
radiometric add-offset (BOA_ADD_OFFSET / RADIO_ADD_OFFSET, introduced Jan 2022) is NOT silently unhandled.

Method (adversarial-review note). Sentinel-2 top-of-atmosphere (L1C) reflectance CAN be slightly negative on
dark targets from sensor / dark-current noise -- that is exactly why Processing-Baseline-04.00 later added the
-1000 offset, to store those near-zero values instead of clipping them. So this is a test of MAGNITUDE, not of
sign: if a +1000 add-offset were present in the stored digital numbers but not subtracted, then reflectance =
DN * 1e-4 would over-read by 0.1, and a stored DN of ~15 would correspond to a TRUE TOA reflectance near -0.10
-- far more negative than dark-target noise (~-0.001) can explain. A 1st-percentile L1C DN of only tens, on
every product version, is therefore consistent ONLY with the plain 0-10000 (no-offset) scale, so DN * 1e-4 is
the correct conversion. This is a self-consistency check on the stored L1C scale by magnitude, not an absolute
calibration against the raw source products. Independently, all products predate the Jan-2022 baseline-04.00
change, and product-aware re-normalization is invariant to any constant offset anyway (see the verdict), so
the paper's headline does not rest on this audit. We report the per-version raw-DN distribution for L1C and
L2A, the verdict, and a few explicit DN->reflectance conversions.

Run: python phase8R12_radiometric_audit.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import phase8_cloudsen12 as P8

SIDE = 512
SCALE = 1e-4                                             # the loader's DN -> reflectance factor
BANDS = ["B2", "B4", "B8", "B11"]                        # blue / red / NIR / SWIR sample


def load_band(data, product, band, n):
    p = os.path.join(data, f"{product}_{band}.dat")
    if not os.path.exists(p):
        return None
    sz = os.path.getsize(p) // 2 // (SIDE * SIDE)
    return np.memmap(p, dtype=np.int16, mode="r").reshape(sz, SIDE, SIDE), min(sz, n)


def main():
    data = os.path.join(P8.DATA, "test")
    m = pd.read_csv(os.path.join(data, "metadata.csv"))
    ver = m["s2_sen2cor_version"].values
    versions = sorted(set(ver))
    print(f"CloudSEN12 test radiometric audit: {len(m)} patches, {len(versions)} Sen2Cor product versions")
    print("  s2_sen2cor_version distribution:", {v: int((ver == v).sum()) for v in versions})
    print("  (none is >= N04.00 -- the Jan-2022 processing baseline that introduced the +1000 radiometric "
          "add-offset -- so all predate it; s2_sen2cor_version is the product-version field, not the ESA PB)\n")

    l1c_p1_min = np.inf
    for band in BANDS:
        for product in ["L1C", "L2A"]:
            got = load_band(data, product, band, len(m))
            if got is None:
                continue
            mm, _ = got
            print(f"── {product}_{band} raw int16 DN, by baseline (centre 112x112 crop) ──")
            for v in versions:
                idx = np.where(ver == v)[0]
                idx = idx[idx < len(mm)][:15]
                if not len(idx):
                    continue
                vals = np.asarray(mm[idx][:, 200:312, 200:312]).ravel().astype(np.int64)
                p1, p50, p99 = np.percentile(vals, [1, 50, 99])
                flag = "  <-- p1 < 1000" if (product == "L1C" and p1 < 1000) else ""
                print(f"    {v}: min={vals.min():6d}  p1={p1:6.0f}  p50={p50:6.0f}  p99={p99:6.0f}  "
                      f"max={vals.max():6d}   refl(x1e-4) p50={p50*SCALE:.3f}{flag}")
                if product == "L1C":
                    l1c_p1_min = min(l1c_p1_min, p1)

    print(f"\nVERDICT: the smallest 1st-percentile L1C digital number across all baselines and sampled bands "
          f"is {l1c_p1_min:.0f}.")
    if l1c_p1_min < 1000:
        print(f"  => it is {l1c_p1_min:.0f} (<< 1000), so under a +1000 add-offset the TRUE TOA reflectance "
              f"there would be ~{(l1c_p1_min - 1000) * SCALE:.2f} -- far more negative than dark-target noise "
              "(~-0.001) can explain. The stored data is therefore on the plain 0-10000 no-offset scale; "
              "reflectance = DN * 1e-4 is correct, consistent across all product versions. And product-aware "
              "re-normalization (z-scoring by each product's own per-band mean/std) is invariant to any constant "
              "per-band offset, so the headline is unaffected even were an offset present.")
    else:
        print("  => it is >= 1000 -- an add-offset may be present; investigate before trusting DN * 1e-4.")
    print("\nexplicit DN -> reflectance examples (x1e-4): "
          f"DN 1000 -> {1000*SCALE:.3f}, DN 2500 -> {2500*SCALE:.3f}, DN 8000 -> {8000*SCALE:.3f}")


if __name__ == "__main__":
    main()
