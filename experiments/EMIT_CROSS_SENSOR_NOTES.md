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

## Honest scope
- A cross-sensor **generalization** of the reliability finding to hyperspectral land-cover under a real
  atmospheric shift — NOT a second cloud dataset (EMIT scenes are arid/low-cloud).
- Pixel-level CRC (no scene-connected-component exchangeable unit here); WorldCover mixed-pixel label noise;
  the model trains on radiance (unusual, but that is exactly the stale source the shift then breaks).
- The cross-BIOME (geographic) axis is confounded with the product shift and is measured separately.

## Reproduce
```
.venv/bin/python experiments/emit_fetch_l1b.py --go          # download L1B radiance (~27 GB, Earthdata netrc)
.venv/bin/python experiments/phase8G_emit_io.py --build-cache # per-biome sample+WorldCover cache (~200 MB)
.venv/bin/python experiments/phase8G_emit_shift.py --seeds 0 1 2 3 4
```
Not committed (rebuildable): the L1B `.nc` granules and the `_shift_cache.npz` files.
