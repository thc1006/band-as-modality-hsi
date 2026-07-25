#!/usr/bin/env python
"""Second-benchmark loader: stream the OFFICIAL CloudSEN12 validation split (high-quality expert labels,
535 patches / 107 ROIs, disjoint from the flagship's train and test) with PAIRED L1C and Sen2Cor L2A, and
sample pixels into a compact .npz the flagship reliability run can consume. This is the independent
scene-set replication the reviewers ask for (a second benchmark on the same task/sensor).

Runs in the throwaway venv that has tacoreader<1.0 + rasterio (NOT the main .venv). The .npz it writes is
plain numpy and is loaded by the main-.venv flagship run. Band mapping is VERIFIED, not assumed:
  L1C raster bands 0..12 == L1C_BANDS = [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12]  (descriptions confirm)
  L2A raster bands 0..11 == L2A_BANDS = [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12]      (correlation confirms;
    L2A[10]->B11 r=0.998, so B10 is dropped exactly as Sen2Cor does; bands 12,13 are QA -> ignored)
  label = sub-asset 'target', uint8 {0:clear,1:thick,2:thin,3:shadow}                (matches CLASS_NAMES)
Reflectance is uint16 x10000 in both, the same scale as the .dat train split, so no rescale is needed.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
import tacoreader

# the exact band lists the flagship uses (kept in sync with phase8_cloudsen12.py by construction)
L1C_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
L2A_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
B10_POS = L1C_BANDS.index("B10")                 # 10; the L2A array leaves this position zero (dropped group)
L2A_TO_L1C = [L1C_BANDS.index(b) for b in L2A_BANDS]   # where each L2A band sits in the 13-band layout


def valid_mask(l1c13):
    """Non-padding pixels: the 512x512 storage pads the 509x509 content with all-zero rows/cols; a real
    pixel has non-zero reflectance in at least one band. Robust to WHERE the padding sits."""
    return l1c13.reshape(13, -1).sum(0) > 0


def load_patch(l1c_row, l2a_row, n_px, rng):
    """Read one patch's L1C(13) + L2A(12->13 layout) + label, sample n_px non-padding pixels.
    Returns (x1, x2, y, n_valid) or None. CRITICAL: reflectance is divided by 10000 to the [0,1] scale
    that load_split returns for the train split -- feeding raw x10000 values would be 10^4 off the
    train mean/std used to normalise, silently corrupting every downstream number."""
    p1 = l1c_row.read(0)                          # sub-asset 0 = s2l1c GTiff
    with rasterio.open(p1) as s:
        l1c = s.read().astype(np.float32) / 10000.0   # (13, H, W) -> reflectance [0,1], matches load_split
    lp = l1c_row.read(1)                          # sub-asset 1 = target (label)
    with rasterio.open(lp) as s:
        lab = s.read()[0].astype(np.int16)        # (H, W) {0,1,2,3}
    p2 = l2a_row.read(0)
    with rasterio.open(p2) as s:
        l2a_full = s.read().astype(np.float32) / 10000.0   # (14, H, W) -> [0,1]
    l2a12 = l2a_full[:12]                          # first 12 bands = L2A_BANDS (verified)

    # spatial alignment is REQUIRED: L1C, L2A and label must share the same H x W grid, else the sampled
    # flat indices point at different ground locations in each product -- silent misregistration
    if l2a12.shape[1:] != l1c.shape[1:] or lab.shape != l1c.shape[1:]:
        return None

    flatv = valid_mask(l1c)
    idx = np.flatnonzero(flatv)
    if idx.size < n_px:
        return None
    pick = rng.choice(idx, n_px, replace=False)

    x1 = l1c.reshape(13, -1)[:, pick].T           # (n_px, 13)  clean L1C, native 13-band layout
    x2 = np.zeros((n_px, 13), np.float32)          # L2A in the 13-band layout, B10 position left at 0
    x2[:, L2A_TO_L1C] = l2a12.reshape(12, -1)[:, pick].T
    y = lab.reshape(-1)[pick].astype(np.int16)
    return x1, x2, y, int(idx.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-px", type=int, default=400)      # pixels/patch, matching the flagship test sampling
    ap.add_argument("--limit", type=int, default=0)       # 0 = all 535; small value to smoke-test first
    ap.add_argument("--out", default="data/cloudsen12_val_secondbench.npz")
    ap.add_argument("--seed", type=int, default=54321)
    ap.add_argument("--workers", type=int, default=10)     # concurrent network reads (the work is I/O-bound)
    args = ap.parse_args()

    l1c = tacoreader.load("tacofoundation:cloudsen12-l1c")
    l2a = tacoreader.load("tacofoundation:cloudsen12-l2a")

    m = ((l1c["tortilla:data_split"] == "validation") &
         (l1c["label_type"] == "high") &
         (l1c["real_proj_shape"] == 509))
    sub = l1c[m].reset_index(drop=True)
    tids = list(sub["tortilla:id"])
    rois = list(sub["old_roi_id"])
    s2ids = list(sub["s2_id"])                     # per-patch Sentinel-2 product; ROIs sharing one are merged
    if args.limit:
        tids, rois, s2ids = tids[:args.limit], rois[:args.limit], s2ids[:args.limit]
    print(f"val+high+509: {len(sub)} patches / {sub['old_roi_id'].nunique()} ROIs; loading {len(tids)}", flush=True)

    def fetch(k, t, r, sid):
        """Read + sample one patch. Thread-safe: its OWN rng seeded by k (so sampling is deterministic per
        patch regardless of completion order) and read-only access to the shared tacoreader frames. The
        ENTIRE read+sample is inside the retry/except: /vsicurl TIFF reads transiently fail under
        concurrency, and load_patch (which does the actual rasterio.open) must never be allowed to raise
        out of here -- one uncaught read error previously killed the whole loader after 150 good patches."""
        for _ in range(3):
            rng = np.random.default_rng(args.seed + k)
            try:
                l1r = l1c[l1c["tortilla:id"] == t].read(0)
                l2r = l2a[l2a["tortilla:id"] == t].read(0)
                got = load_patch(l1r, l2r, args.n_px, rng)
            except Exception:
                continue
            if got is None:
                return None
            x1, x2, y, nv = got
            return k, r, sid, x1, x2, y, nv
        return None

    X1, X2, Y, RID, PID, SID, NV = [], [], [], [], [], [], []
    done = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, k, t, r, sid) for k, (t, r, sid) in enumerate(zip(tids, rois, s2ids))]
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception:
                res = None
            done += 1
            if res is None:
                skipped += 1
                continue
            k, r, sid, x1, x2, y, nv = res
            X1.append(x1); X2.append(x2); Y.append(y); NV.append(nv)
            RID.append(np.full(len(y), r)); PID.append(np.full(len(y), k, np.int32))
            SID.append(np.full(len(y), sid))
            if done % 50 == 0:
                print(f"  [{done}/{len(futs)}] valid={nv}, cls={np.bincount(y, minlength=4)}, skipped={skipped}",
                      flush=True)
    if skipped:
        print(f"  NOTE: skipped {skipped}/{len(futs)} patches after 3 retries (transient reads)", flush=True)

    X1 = np.concatenate(X1); X2 = np.concatenate(X2); Y = np.concatenate(Y)
    RID = np.concatenate(RID); PID = np.concatenate(PID); SID = np.concatenate(SID)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X_l1c=X1, X_l2a=X2, y=Y, roi_id=RID, patch_id=PID, s2_id=SID)
    # ---- built-in verification, printed so it is never silently wrong ----
    # exchangeable units = ROIs unioned when they share a Sentinel-2 product (same as the flagship 195->184)
    parent = {r: r for r in np.unique(RID)}
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    from collections import defaultdict
    s2_rois = defaultdict(set)
    for r, s in zip(RID, SID): s2_rois[s].add(r)
    for rs in s2_rois.values():
        rs = list(rs)
        for r in rs[1:]: parent[find(r)] = find(rs[0])
    n_comp = len({find(r) for r in np.unique(RID)})
    print(f"\nwrote {args.out}: {X1.shape[0]} px, {len(np.unique(PID))} patches, {len(np.unique(RID))} ROIs "
          f"-> {n_comp} scene-components (merged {len(np.unique(RID)) - n_comp})")
    print(f"  median valid px/patch: {int(np.median(NV))}  (must be ~259081 = 509^2 real content)")
    print(f"  L1C reflectance range [{X1.min():.3f}, {X1.max():.3f}] mean {X1.mean():.3f}  (must be ~[0,1], NOT x10000)")
    print(f"  L1C per-band mean: {X1.mean(0).round(3)}")
    print(f"  L2A per-band mean: {X2.mean(0).round(3)}  (B10 col {B10_POS} must be 0.0)")
    print(f"  class balance clear/thick/thin/shadow: {np.bincount(Y, minlength=4)}")
    print(f"  blue B2 L1C {X1[:,1].mean():.3f} -> L2A {X2[:,1].mean():.3f} "
          f"(atmospheric correction lowers blue: delta {X2[:,1].mean()-X1[:,1].mean():+.3f})")


if __name__ == "__main__":
    main()
