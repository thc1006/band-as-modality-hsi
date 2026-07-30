# EMIT cross-sensor reliability — 2nd-dataset external validation (EXPLORATORY, post-v1.1.0)

Independent, cross-**sensor** replication of the silent conformal-certificate failure from the CloudSEN12
reliability paper. Addresses the paper's #1 stated limitation ("a fully independent dataset with its own
labelling protocol and geography").

## Setup
- **Sensor:** EMIT imaging spectrometer (hyperspectral, 244 good bands, 381–2493 nm) — independent of Sentinel-2.
- **Real atmospheric-correction shift:** SOURCE = L1B **radiance** → TARGET = L2A **ISOFIT surface reflectance**
  (analogous to CloudSEN12's L1C→L2A). Same sensor grid, so L1B/L2A pixels are aligned (no reprojection).
- **Geography:** 15 real biomes (arid, vegetated, savanna, forest, cropland …), lat/lon per pixel.
- **Label:** ESA WorldCover 2021 land-cover, remote-sampled from the public S3 COGs at each EMIT pixel
  (product-agnostic; 10 m sampled at ~60 m EMIT pixels → mixed-pixel noise, reported).
- **Certificate:** split conformal risk control at alpha=10%, max-softmax score, temperature scaling — the
  same machinery as the flagship.

## Primary result — pure product shift (pooled pixel split, 5 seeds, ~60k px, K=6 classes)
Same eval pixels; only the product L1B→L2A changes (isolates the stale-normalization contract).

| arm | joint risk (mean ± SE) | coverage |
|---|---|---|
| clean L1B (source) | **9.8 ± 0.3 %** | 93 % |
| **NAIVE L2A (stale L1B-norm)** | **66.2 ± 2.9 %** | 95 % |
| Mondrian (recalibrated on target) | 10.1 ± 0.2 % | 29 % |
| product-aware re-norm (mean/std, eval-disjoint) | 24.2 ± 0.7 % | 85 % |
| per-band quantile transport (R10 mid-CDF, eval-disjoint) | 22.1 ± 0.9 % | 85 % |
| **calibration statistic (source)** | **9.9 %** (≤ alpha) | — |

**Re-normalization is only PARTIALLY effective here, and that is itself a finding.** On CloudSEN12 the shift is
TOA→BOA reflectance (the SAME physical quantity, a rescaling), and a per-band re-norm brings the risk to ~10 %.
On EMIT the shift is radiance→reflectance (DIFFERENT physical quantities: an atmospheric+illumination transform,
not a rescaling), so a per-band MARGINAL transport (mean/std or quantile) only reduces the breach 66→~22 %,
leaving a large residual. The label-free remedy's effectiveness thus depends on whether the product shift is a
rescaling or a fundamental transformation — a nuance the CloudSEN12-only paper could not surface.

**Silent failure reproduced, more dramatically than CloudSEN12 (27.8 %):** the certificate's own statistic
stays at 9.9 % (looks fine) while the deployed confidently-wrong risk is 66 %; accuracy collapses L1B 86 % →
L2A 22–37 %. Mondrian restores control at a heavy abstention cost (coverage 95→29). Product-aware re-norm
substantially reduces the breach (66→24) but leaves a residual (a global per-band mean/std rescaling over 244
bands does not fully realign — a per-band quantile transport is the natural next step).

## 10-seed confirmation (product shift)
The 5-seed table reproduces at 10 seeds with tighter CIs: clean **9.8 ± 0.2**, NAIVE **66.6 ± 1.6**,
Mondrian **10.0 ± 0.1**, re-norm **24.8 ± 0.5**, quantile **22.3 ± 0.6**, calib-stat 9.9 %
(`results_phase8G_emit_shift_10seed.json`).

## Cross-BIOME geographic axis (separate, confounded) — `phase8G_emit_geo.py`
Train + calibrate on the SOURCE product (L1B) of some biomes, deploy on the SOURCE product of UNSEEN
biomes (no product change → pure geographic shift). 10 seeds (`results_phase8G_emit_geo.json`):

| arm | joint (mean ± SE) | coverage |
|---|---|---|
| NAIVE (source-calib, unseen biomes) | **16.4 ± 5.8 %** | 46 % |
| Mondrian (biome-recalib) | 9.9 ± 0.0 % | 40 % |
| calibration statistic | 9.3 % | — |

Geography breaks the certificate **inconsistently and in BOTH directions**: per-seed joint ranges 0.5 → 51.7 %
depending on which biomes are held out — some breach hard (51.7 %@62), others collapse *coverage* instead
(0.5 %@7 %: the model is so OOD it abstains on nearly everything). This is exactly why the paper isolates the
**product** shift as the clean, reproducible mechanism (consistent 66 % every seed) and treats geography as a
confounded axis — the geographic result is a real but high-variance data point, not a headline.

## Architecture independence — `--model band` (band-as-modality)
Re-running the product shift with the paper's actual GroupedCrossBandAttention + SGMAE pretraining (not the
MLP) reproduces the silent failure (`results_phase8G_emit_shift_band.json`, 10 seeds): clean **9.9 ± 0.2** /
NAIVE **57.5 ± 3.6** / Mondrian 9.9 @ 19 % / re-norm 26.7 ± 1.4 / calib-stat 9.9 % — the same catastrophic
silent pattern and partial-re-norm residual as the MLP (9.8 / 66.6 / 10.0 / 24.8). It is a property of the
stale-normalization CONTRACT, not of the classifier family. See MODALITY_GENERALITY_NOTES.md.

**Performance note (band model is LAUNCH-BOUND at the default batch).** The GroupedCrossBandAttention model is
`d_model=64`, so each training step is microseconds of GPU work wrapped in ~15 ms of Python/kernel-launch
overhead (exactly the regime `phase2_degradation.auto_bs` documents). At EMIT's ~30k train px `auto_bs`
floors to **bs=256 → 117 steps/epoch**, which is pure launch overhead and runs at a misleadingly "high"
`nvidia-smi` utilisation while being ~10x slower than it should be (a 100-epoch × 10-seed run stretched to
hours, made worse by concurrent GPU jobs). Profiled: bs=256 **1847 ms/epoch** vs bs=2048 **176 ms/epoch**
(10.5x) vs bs=8192 112 ms/epoch (16.5x). Fix: `phase8G_emit_shift.py --model band --bs 2048` (≈29 s/seed).
bs is a hyperparameter, so `--bs 0` (auto=256) is kept as the default to hold the pushed 40-epoch result
byte-identical; the efficient config `results_phase8G_emit_shift_band_bs2048.json` uses `--bs 2048 --epochs 300`
(~4.5k-update budget, ~15 min for 10 seeds vs hours): clean 9.8 / NAIVE **70.0** / Mondrian 9.8 @ 17 % /
re-norm 30.4 / calib 9.9 (acc L1B 85-86 %).

⚠️ **Honesty correction (adversarial self-review).** An earlier note here claimed "convergence makes the
breach worse (70.0 vs the 40-epoch 57.5)". That was a CONFOUND -- the two runs differ in BOTH epochs (40->300)
AND batch size (256->2048). Isolating each: at fixed bs=2048, NAIVE is 64.3 (40ep) -> 63.7 (100ep) -> 69.0
(300ep), i.e. the epoch trend is small and swamped by huge per-seed variance (e.g. 100ep seeds 49 and 79). The
step from bs=256/40ep (57.5) to bs=2048/40ep (64.3) shows the batch size (a different optimization trajectory)
accounts for most of the gap. So the magnitude is BATCH-SIZE-dependent and NOT cleanly attributable to
convergence. What IS robust across every config (bs 256-2048, 40-300 epochs): the silent failure holds --
NAIVE 57-70 %, calibration statistic 9.9 % <= alpha, always catastrophic and always silent. A separate check
confirmed the classifier is real (clean acc 85 % vs 36 % majority class; per-class recall [88,83,81,64,97,17]),
not a majority-class collapse.

## Honest scope
- A cross-sensor **generalization** of the reliability finding to hyperspectral land-cover under a real
  atmospheric shift — NOT a second cloud dataset (EMIT scenes are arid/low-cloud).
- ⚠️ **This is a CONSTRUCTED cross-product STRESS TEST, more extreme than a realistic pipeline — say so
  plainly.** The source is L1B **radiance** and the target is L2A **reflectance**: the real EMIT product pair,
  but no operational land-cover system trains on radiance (you would train on reflectance). Radiance (~2–7)
  and reflectance (~0.2–0.4) differ ~10× in scale AND are related by a spectrally-varying atmospheric+
  illumination transform, so the stale-normalization failure here is deliberately SEVERE (NAIVE ~66 %). We do
  NOT claim operators train on radiance; we use this real product pair only to show the silent-failure
  MECHANISM generalizes to a hyperspectral spectrometer. It is a demonstration of the mechanism at an extreme,
  not an estimate of a typical deployment breach — the flagship CloudSEN12 L1C→L2A (both reflectance) is the
  realistic-magnitude case.
- Pixel-level CRC (no scene-connected-component exchangeable unit here); WorldCover 10 m labels sampled at
  ~60 m EMIT pixels carry mixed-pixel noise.
- The cross-BIOME (geographic) axis is confounded with the product shift and is measured separately.

## What adversarial self-review CONFIRMED is sound (not artifacts)
- **Source calibration is genuinely HEALTHY**, so the silent failure is real, not a mis-calibration artifact:
  temperature T=1.05 (already well-calibrated, NOT overconfident), ECE 0.9 %, CRC threshold 0.52 (real
  selection at 92 % coverage), held-out source calib-stat 10.0 % ≤ α.
- **The classifier is real, not a majority-class collapse:** clean acc 85 % vs 36 % majority class; per-class
  recall [88,83,81,64,97,17] (5 of 6 classes well-discriminated).
- **K is consistent:** all 10 production seeds keep the SAME 6 classes [tree,shrub,grass,crop,bare,water]
  (the K=5 seen under `--smoke` is a small-subsample artifact, not in the reported runs).

## Reproduce
```
.venv/bin/python experiments/emit_fetch_l1b.py --go          # download L1B radiance (~27 GB, Earthdata netrc)
.venv/bin/python experiments/phase8G_emit_io.py --build-cache # per-biome sample+WorldCover cache (~200 MB)
.venv/bin/python experiments/phase8G_emit_shift.py --seeds 0 1 2 3 4
```
Not committed (rebuildable): the L1B `.nc` granules and the `_shift_cache.npz` files.
