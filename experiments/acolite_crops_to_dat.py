#!/usr/bin/env python
"""Aggregate the per-patch ACOLITE surface-reflectance crops into CloudSEN12 .dat files, so the
existing phase8 loader can read ACOLITE-L2A as a new product with the SAME pixel sampling as L1C/L2A.

Crops (scratchpad/.../crops/<index>.npy) are (11, 509, 509) float32 reflectance in ACOLITE band order
[B1,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12] (ACOLITE emits no surface reflectance for B09/B10). We write one
ACOLITE_<band>.dat per band into data/cloudsen12/test/, matching the on-disk convention of the L1C/L2A
.dat exactly: little-endian int16, (N_patches, 512, 512), reflectance*10000, real 509x509 content at
rows/cols [1:510] (VALID_OFF=1), zero padding elsewhere. Leaked/absent patches (the 4 train/test
overlap products dropped by the leak-guard) are written as all-zero and are dropped again downstream.
"""
import os, glob, csv
import numpy as np

REPO = "/home/hctsai1006/cct/band-as-modality-hsi"
CROPS = os.path.join(REPO, "data/acolite_stage/crops")
OUT = os.path.join(REPO, "data/cloudsen12/test")
ACO_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
SIDE, VALID, OFF = 512, 509, 1


def main():
    meta = list(csv.DictReader(open(os.path.join(REPO, "data/cloudsen12/test/metadata.csv"))))
    n = len(meta)
    have = {int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(CROPS, "*.npy"))}
    print(f"[agg] {n} test patches | {len(have)} crops present")
    # load all present crops once into memory (n x 11 x 509 x 509 float32 ~ 11 GB is too much;
    # instead stream band-by-band: for each band, read that band-plane from every crop).
    for bi, band in enumerate(ACO_BANDS):
        arr = np.zeros((n, SIDE, SIDE), dtype="<i2")
        miss = 0
        for i in range(n):
            p = os.path.join(CROPS, f"{i}.npy")
            if not os.path.exists(p):
                miss += 1
                continue
            plane = np.load(p, mmap_mode="r")[bi]                       # (509, 509) float32 reflectance
            v = np.rint(np.asarray(plane, np.float64) * 1e4)            # reflectance -> *10000, like L2A
            v = np.clip(v, -32768, 32767).astype("<i2")
            arr[i, OFF:OFF + VALID, OFF:OFF + VALID] = v               # real content at [1:510], pad stays 0
        outp = os.path.join(OUT, f"ACOLITE_{band}.dat")
        arr.tofile(outp)
        if bi == 0:
            print(f"[agg] wrote {os.path.basename(outp)} shape=({n},{SIDE},{SIDE}) int16; missing={miss}")
    print(f"[agg] done: {len(ACO_BANDS)} ACOLITE_<band>.dat in {OUT}")


if __name__ == "__main__":
    main()
