# CloudSEN12 validation replication: which label version? (checked 2026-10-14)

## The question

The paper's validation-set replication (main.tex 180; `phase8R5_secondbench.py`, `phase8R5_valdump.py`,
`phase8R5_val_classwise.py`) evaluates on `data/cloudsen12_val_secondbench.npz`, which
`experiments/phase8R5_val_loader.py` drew from the tacofoundation release (`tacofoundation:cloudsen12-l1c`
and `-l2a`). Every other CloudSEN12 result uses CloudSEN12-high (Hugging Face `csaybar/CloudSEN12-high`,
revision `d97728dd61858a21c7d6bc61385687d328433cba`), the original release the paper names (main.tex 73).
The tacofoundation release is CloudSEN12+: its dataset card (Hugging Face `tacofoundation/cloudsen12`)
says of version 1.1.0 that "All labels from the previous version have been curated and refined" and that
the GeoTIFFs are saved "with discard_lsb=2". Are the replication's pixels and labels CloudSEN12-high's?

## What was checked

The loader sampled 400 non-padding pixels from patch k (its `patch_id`) with
`np.random.default_rng(54321 + k)` (its `--seed` default). The script below redoes that draw on
CloudSEN12-high's validation files (`data/cloudsen12/val/`: the same 535 patches, matched one to one by ROI
and Sentinel-2 product) and compares what sits at the same positions. It only reads the data (given a
path, it also saves CloudSEN12-high's labels at the sampled pixels there); run it from the repository root
with the project's venv (`python check.py [labels_high.npy]`).

```python
"""Are the validation replication's pixels and labels CloudSEN12-high's? Read-only; run from the repo root.

phase8R5_val_loader.py drew data/cloudsen12_val_secondbench.npz from the tacofoundation release: patch k (its
patch_id) sampled 400 non-padding pixels with np.random.default_rng(54321 + k). Redo that draw on
CloudSEN12-high's validation files (data/cloudsen12/val/) and compare what sits at the same positions.
With an output path, also save CloudSEN12-high's labels at those pixels, in the sample's order."""
import sys
import numpy as np
import pandas as pd

R = "data/"
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
d = np.load(R + "cloudsen12_val_secondbench.npz", allow_pickle=False)
X, y, pid, sid, rid = d["X_l1c"], d["y"], d["patch_id"], d["s2_id"], d["roi_id"]
meta = pd.read_csv(R + "cloudsen12/val/metadata.csv")
n = len(meta)
lab = np.memmap(R + "cloudsen12/val/LABEL_manual_hq.dat", dtype=np.uint8, mode="r").reshape(n, 512, 512)
l1c = [np.memmap(R + f"cloudsen12/val/L1C_{b}.dat", dtype="<i2", mode="r").reshape(n, 512, 512) for b in BANDS]
row = {}
for i, key in enumerate(zip(meta["roi_id"].astype(str), meta["s2_id"].astype(str))):
    row.setdefault(key, []).append(i)

y_high = np.full(y.shape, -1, np.int16)
dn_hist, differ = {}, []
for k in np.unique(pid):
    m = np.flatnonzero(pid == k)
    (h,) = row[(str(rid[m[0]]), str(sid[m[0]]))]                         # exactly one CloudSEN12-high row
    cube = np.stack([np.asarray(b[h]) for b in l1c]).reshape(13, -1).astype(np.int64)
    idx = np.flatnonzero(cube.sum(0) > 0)                                 # the loader's valid_mask
    pick = np.random.default_rng(54321 + int(k)).choice(idx, len(m), replace=False)
    dd = np.rint(X[m].astype(np.float64) * 10000).astype(np.int64) - cube[:, pick].T
    for v, c in zip(*np.unique(dd, return_counts=True)):
        dn_hist[int(v)] = dn_hist.get(int(v), 0) + int(c)
    y_high[m] = np.asarray(lab[h]).reshape(-1)[pick]
    if (y_high[m] != y[m]).any():
        differ.append((int(k), str(rid[m[0]]), int((y_high[m] != y[m]).sum()),
                       np.bincount(y[m], minlength=4).tolist(), np.bincount(y_high[m], minlength=4).tolist()))

dn = np.rint(X.astype(np.float64) * 10000).astype(np.int64)
print(f"patches {len(np.unique(pid))}, sampled pixels {len(y)}")
print(f"reflectance DN(sample) - DN(CloudSEN12-high) at the same positions: {dict(sorted(dn_hist.items()))}")
print(f"sample DN values that are multiples of 4: {(dn % 4 == 0).mean():.4f}")
print(f"patches whose labels differ at some sampled pixel: {len(differ)}; "
      f"differing pixels {int((y_high != y).sum())} of {len(y)}")
for t in sorted(differ, key=lambda t: -t[2]):
    print("  patch %3d %-9s %3d px differ; class counts sample %s vs CloudSEN12-high %s" % t)
if len(sys.argv) > 1:
    assert (y_high >= 0).all()
    np.save(sys.argv[1], y_high)
    print(f"wrote {sys.argv[1]}")
```

## Result

```
patches 535, sampled pixels 214000
reflectance DN(sample) - DN(CloudSEN12-high) at the same positions: {-1: 693292, 0: 718572, 1: 681042, 2: 689094}
sample DN values that are multiples of 4: 1.0000
patches whose labels differ at some sampled pixel: 23; differing pixels 2815 of 214000
  patch  75 ROI_0178  300 px differ; class counts sample [350, 1, 49, 0] vs CloudSEN12-high [51, 0, 137, 212]
  patch 385 ROI_6785  233 px differ; class counts sample [233, 0, 144, 23] vs CloudSEN12-high [21, 0, 371, 8]
  patch  80 ROI_0225  224 px differ; class counts sample [355, 0, 45, 0] vs CloudSEN12-high [131, 0, 269, 0]
  patch 314 ROI_1354  224 px differ; class counts sample [56, 69, 144, 131] vs CloudSEN12-high [18, 247, 0, 135]
  patch 364 ROI_3846  222 px differ; class counts sample [0, 0, 114, 286] vs CloudSEN12-high [56, 0, 258, 86]
  patch 407 ROI_7047  190 px differ; class counts sample [0, 0, 400, 0] vs CloudSEN12-high [190, 0, 210, 0]
  patch  86 ROI_0257  163 px differ; class counts sample [192, 0, 208, 0] vs CloudSEN12-high [63, 34, 303, 0]
  patch 185 ROI_0720  146 px differ; class counts sample [173, 4, 221, 2] vs CloudSEN12-high [55, 32, 297, 16]
  patch 332 ROI_1631  138 px differ; class counts sample [0, 0, 400, 0] vs CloudSEN12-high [138, 0, 262, 0]
  patch  44 ROI_0129  126 px differ; class counts sample [189, 34, 177, 0] vs CloudSEN12-high [97, 10, 293, 0]
  patch 132 ROI_0501  116 px differ; class counts sample [102, 225, 36, 37] vs CloudSEN12-high [93, 250, 57, 0]
  patch 119 ROI_0487  103 px differ; class counts sample [86, 233, 67, 14] vs CloudSEN12-high [62, 328, 0, 10]
  patch 222 ROI_0928  102 px differ; class counts sample [299, 0, 101, 0] vs CloudSEN12-high [245, 0, 155, 0]
  patch 330 ROI_1631   92 px differ; class counts sample [0, 57, 337, 6] vs CloudSEN12-high [40, 109, 251, 0]
  patch 139 ROI_0505   91 px differ; class counts sample [53, 0, 347, 0] vs CloudSEN12-high [66, 76, 258, 0]
  patch 178 ROI_0636   89 px differ; class counts sample [69, 3, 321, 7] vs CloudSEN12-high [50, 42, 308, 0]
  patch 183 ROI_0714   58 px differ; class counts sample [18, 342, 40, 0] vs CloudSEN12-high [0, 384, 0, 16]
  patch 344 ROI_1953   54 px differ; class counts sample [0, 114, 214, 72] vs CloudSEN12-high [17, 125, 214, 44]
  patch  61 ROI_0160   38 px differ; class counts sample [196, 90, 112, 2] vs CloudSEN12-high [190, 114, 93, 3]
  patch  16 ROI_0050   35 px differ; class counts sample [336, 3, 61, 0] vs CloudSEN12-high [321, 10, 69, 0]
  patch 523 ROI_11651  32 px differ; class counts sample [23, 0, 377, 0] vs CloudSEN12-high [19, 28, 353, 0]
  patch 389 ROI_6785   29 px differ; class counts sample [291, 0, 109, 0] vs CloudSEN12-high [268, 0, 132, 0]
  patch 317 ROI_1404   10 px differ; class counts sample [107, 162, 131, 0] vs CloudSEN12-high [97, 162, 141, 0]
wrote labels_high.npy
```

- Reflectance: every sampled value differs from CloudSEN12-high's at the same position by -1, 0, +1 or
  +2 DN (about a quarter each, over 214,000 x 13 values), and every sampled value is a multiple of 4 DN:
  the tacofoundation release's two discarded low bits. So the draw is reconstructed exactly, and the
  imagery is the same up to that quantisation (at most 0.0002 reflectance). The training and test data,
  from CloudSEN12-high, are not quantised this way.
- Labels: identical at all 400 sampled pixels in 512 of the 535 patches. In 23 patches they differ at
  sampled pixels (so at least 23 patches carry revised labels; only the sampled pixels were compared) --
  2,815 of the 214,000 sampled pixels (1.3 %) -- some grossly: in patch 75 (ROI_0178) 300 of the 400
  pixels differ; CloudSEN12-high has 212 cloud-shadow pixels there, the sample none.

## Does it change the replication's numbers?

The replication was re-run on the 4x P100 box with each trained model put through the evaluation twice:
once with the sample's own labels and once with CloudSEN12-high's at the same pixels (saved by the script
above). Each pass fits its temperatures, calibrates its CRC thresholds and scores with its own labels,
exactly as the replication does, so the comparison answers "what would the replication have reported
with CloudSEN12-high's labels" on the same models, splits and logits. The change to
`phase8R5_secondbench.py` below applies to commit 6c1e392. The run itself used a copy of the working
tree before this round's changes were committed; what it imports differs from 6c1e392 only in comments
and docstrings, `group_center_wavelengths`' input checks, `label_histogram`'s cache directory and
phase8R's new option, none of which touches this computation. It ran as
`python experiments/phase8R5_secondbench.py --seeds <part of 0..9> --y-alt labels_high.npy --out
<part>.csv` in three parts on three GPUs (seeds 0-3, 4-6, 7-9; each seed is trained and evaluated
independently of the others):

```diff
--- a/experiments/phase8R5_secondbench.py
+++ b/experiments/phase8R5_secondbench.py
@@ -101,11 +101,16 @@
     ap.add_argument("--epochs", type=int, default=40)
     ap.add_argument("--val", default="data/cloudsen12_val_secondbench.npz")
     ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
+    ap.add_argument("--y-alt", required=True)
+    ap.add_argument("--out", required=True)
     args = ap.parse_args()
     hw.setup(deterministic=True, prefer=args.device)
 
     d = np.load(args.val, allow_pickle=True)
     Xl1c, Xl2a, y = d["X_l1c"], d["X_l2a"], d["y"].astype(np.int64)
+    y_alt = np.load(args.y_alt).astype(np.int64)
+    assert y_alt.shape == y.shape and set(np.unique(y_alt)) <= {0, 1, 2, 3}
+    print(f"alternative labels: {int((y_alt != y).sum())} of {len(y)} pixels differ", flush=True)
     comp = components(d["roi_id"], d["s2_id"])
     print(f"val second benchmark: {len(y)} px, {len(np.unique(comp))} scene-components, "
           f"class {np.bincount(y, minlength=4)}", flush=True)
@@ -129,23 +134,25 @@
         P2.finetune_proposed(m, Xtr_n, ytr, groups, seed, epochs=args.epochs, bs=bs)
         lc = P8R.logits_at("proposed", m, Xc, groups, [])        # clean L1C
         ll = P8R.logits_at("proposed", m, Xa, groups, [g_b10])   # real L2A, B10 group dropped
-        for ss in range(10):
+        for lab_name, yy in (("tacofoundation", y), ("cloudsen12_high", y_alt)):
+          for ss in range(10):
             mt, mc, me = split3(comp, ss)
-            Tc = fit_temperature(lc[mt], y[mt])                  # clean (source) temperature
-            Tl = fit_temperature(ll[mt], y[mt])                  # L2A (target) temperature
-            thr_n = crc_thr(lc, Tc, y, mc, comp)                 # naive: calibrate on clean
-            thr_m = crc_thr(ll, Tl, y, mc, comp)                 # Mondrian: calibrate on L2A
-            rows.append(dict(seed=seed, split=ss, state="clean", arm="naive",
-                             joint=ce_joint(lc, Tc, y, me, comp, thr_n), cov=cover(lc, Tc, me, thr_n, comp)))
-            rows.append(dict(seed=seed, split=ss, state="L2A", arm="naive",
-                             joint=ce_joint(ll, Tc, y, me, comp, thr_n), cov=cover(ll, Tc, me, thr_n, comp)))
-            rows.append(dict(seed=seed, split=ss, state="L2A", arm="mondrian",
-                             joint=ce_joint(ll, Tl, y, me, comp, thr_m), cov=cover(ll, Tl, me, thr_m, comp)))
+            Tc = fit_temperature(lc[mt], yy[mt])                 # clean (source) temperature
+            Tl = fit_temperature(ll[mt], yy[mt])                 # L2A (target) temperature
+            thr_n = crc_thr(lc, Tc, yy, mc, comp)                # naive: calibrate on clean
+            thr_m = crc_thr(ll, Tl, yy, mc, comp)                # Mondrian: calibrate on L2A
+            rows.append(dict(labels=lab_name, seed=seed, split=ss, state="clean", arm="naive",
+                             joint=ce_joint(lc, Tc, yy, me, comp, thr_n), cov=cover(lc, Tc, me, thr_n, comp)))
+            rows.append(dict(labels=lab_name, seed=seed, split=ss, state="L2A", arm="naive",
+                             joint=ce_joint(ll, Tc, yy, me, comp, thr_n), cov=cover(ll, Tc, me, thr_n, comp)))
+            rows.append(dict(labels=lab_name, seed=seed, split=ss, state="L2A", arm="mondrian",
+                             joint=ce_joint(ll, Tl, yy, me, comp, thr_m), cov=cover(ll, Tl, me, thr_m, comp)))
         print(f"  seed {seed}: clean acc {(lc.argmax(1) == y).mean() * 100:.1f}, "
               f"L2A acc {(ll.argmax(1) == y).mean() * 100:.1f}", flush=True)
 
     df = pd.DataFrame(rows)
-    df.to_csv("paper/results_phase8R5_secondbench_raw.csv", index=False)
+    df.to_csv(args.out, index=False)
+    return
     t = 2.262 if len(args.seeds) >= 10 else (4.303 if len(args.seeds) >= 3 else 12.71)
     print(f"\n=== Second benchmark: CloudSEN12 validation split, {len(np.unique(comp))} within-dataset "
           f"held-out scene-components ===")
```

The patched script stops before its own summary; the table aggregates the 600 rows with the script's own
standard error (`phase8R_perclass_weighting_agg.two_way_se` over seed and split, CGM-ss in
`paper/RESULT_MANIFEST.md`) and a t9 interval. Mean over the 10 seeds x 10 splits; the last column is the
paired difference (CloudSEN12-high minus tacofoundation labels), cell by cell:

| state, arm | tacofoundation labels | CloudSEN12-high labels | paired difference |
|---|---|---|---|
| L2A, naive joint risk | 28.06 ± 0.67 | 28.74 ± 0.64 | +0.68 ± 0.10 [+0.46, +0.91] |
| L2A, Mondrian joint risk | 7.09 ± 0.61 | 7.27 ± 0.61 | +0.18 ± 0.08 [-0.01, +0.37] |
| clean, naive joint risk | 8.44 ± 0.39 | 8.39 ± 0.33 | -0.05 ± 0.11 [-0.30, +0.20] |
| L2A, naive coverage | 92.3 | 92.4 | +0.10 ± 0.06 [-0.04, +0.25] |
| L2A, Mondrian coverage | 39.8 | 39.3 | -0.52 ± 0.21 [-0.99, -0.06] |

(Joint risk and coverage in %. The paper's 2x V100 run, with the tacofoundation labels, reports 28.0 ±
0.7, Mondrian 7.8 and clean 8.1; these P100 models differ from it, as everywhere -- see the H-10 record.)

## What it means

The replication's labels are CloudSEN12+'s revision, not CloudSEN12-high's: 1.3 % of its evaluated pixels
carry a different label than the release the paper names, and its reflectances are quantised to 4 DN.
Calibrated and scored with CloudSEN12-high's labels, the naive breach is about 0.7 points LARGER (the
interval excludes zero): these P100 models give 28.7 instead of 28.1. The Mondrian and clean risks barely
move. The replication's conclusion -- the naive certificate breaches on the validation scenes, and
Mondrian restores it -- holds under either label version. Not re-evaluated: the per-class figures
(`phase8R5_valdump.py`, `phase8R5_val_classwise.py`) and the effect of the 4-DN quantisation of the
inputs, which this comparison holds fixed. Recorded as D14 in `paper/RESULT_MANIFEST.md`.

## Scope

The comparison ran on the 4x P100 box; the paper's numbers come from 2x V100, whose models differ. Its
per-cell rows are committed in
`paper/p100_rerun_2026-10-14/results_phase8R5_secondbench_labelversion_raw.csv`. The CloudSEN12-high
labels at the sampled pixels are not committed (they derive from the CC-BY-NC dataset); the script
regenerates them (the saved array's sha256:
9e6b1eee2bb64688cead6d32ecb95e8dc24c196b38044480d6c9d398cc61d68b).
