# Modality generality of the silent-certificate failure (EXPLORATORY, post-v1.1.0)

The reliability paper's finding — a split conformal-risk-control certificate calibrated on the SOURCE product
silently fails under an operational preprocessing/atmosphere shift (its own calibration statistic stays ≤ α
while the deployed confidently-wrong JOINT risk breaches) — reproduces across **three independent sensor
modalities and three distinct shift types**, and the effectiveness of the label-free remedy follows a clean,
physically-interpretable **gradient**.

## The three modalities

| modality | sensor | bands | shift (source → target) | shift nature |
|---|---|---|---|---|
| multispectral | Sentinel-2 (CloudSEN12) | 13 | L1C TOA → L2A BOA (Sen2Cor) | near per-band rescaling |
| imaging spectrometer | EMIT | 244 | L1B radiance → L2A ISOFIT reflectance | **transformation** |
| hyperspectral | AVIRIS (Indian Pines) | 200 | 6S dry (CWV 0.5) → humid (CWV 4.0) | **pure per-band rescaling** ρ·T |

## The result — silent breach everywhere; re-norm fix follows shift nature

| modality | clean | NAIVE (stale) | Mondrian | product-aware re-norm | re-norm residual (renorm − clean) |
|---|---|---|---|---|---|
| HSI 6S (Indian Pines) | 10.2 ± 0.4 | **86.0 ± 2.8** | 4.9 @ 7 % cov | **11.3 ± 0.5** | **+1.1** (full fix) |
| CloudSEN12 (flagship) | ~10 @ 74 % | **~29** | ~10 (labels + abstention) | ~10 | **~0** (full fix) |
| EMIT (cross-sensor) | 9.8 ± 0.2 | **66.6 ± 1.6** | 10.0 @ 27 % cov | 24.8 ± 0.5 | **+15** (partial) |

α = 10 % throughout; calibration statistic stays ≤ α in every case (the silent part: 9.8 % HSI, 9.9 % EMIT).

## The gradient (the interesting science)

**Product-aware re-normalization fully restores the certificate exactly when the product shift is a per-band
affine rescaling, and only partially when it is a genuine transformation.**

- **HSI 6S is the clean extreme.** The two products are ρ·T(0.5) and ρ·T(4.0): band-wise the target is the
  source times a fixed ratio T(4.0)/T(0.5). A per-band standardization (subtract mean, divide by std) inverts
  a per-band affine map *exactly*, so re-norm lands at 11.3 ≈ 10.2 clean despite an 86 % naive breach. This is
  the mechanism the paper argues for, in its purest possible form.
- **CloudSEN12 sits next to it.** L1C→L2A is TOA→BOA of the *same* reflectance quantity, close to a rescaling
  plus a small additive path term, so re-norm ≈ fully restores (the paper's headline remedy).
- **EMIT is the far end.** L1B radiance → L2A reflectance is an atmospheric+illumination *transformation*
  (spectrally varying, not a single per-band scalar), so a per-band MARGINAL transport (mean/std OR quantile)
  cannot undo it — a +15 residual survives. The remedy's power is bounded by how affine the shift is.

Mondrian (target-label recalibration) restores control in every modality but at a coverage cost that scales
with how badly accuracy collapses: catastrophic on HSI 6S (7 % coverage — the humid atmosphere crushes the
water-vapour bands, dry-trained accuracy falls 82 → ~8 %).

## Architecture independence
On EMIT the breach is the same for a plain MLP and for the paper's band-as-modality GroupedCrossBandAttention
(+ SGMAE pretraining), 10 seeds each:

| model | clean | NAIVE | Mondrian | re-norm | calib-stat |
|---|---|---|---|---|---|
| MLP | 9.8 ± 0.2 | 66.6 ± 1.6 | 10.0 @ 27 % | 24.8 ± 0.5 | 9.9 % |
| band-as-modality | 9.9 ± 0.2 | 57.5 ± 3.6 | 9.9 @ 19 % | 26.7 ± 1.4 | 9.9 % |

Same catastrophic silent breach, same partial-re-norm residual (`phase8G_emit_shift.py --model {mlp,band}`).
The failure is a property of the stale-normalization CONTRACT, not of the classifier family.

## Reproduce
```
.venv/bin/python experiments/phase8H_hsi6s_shift.py       --seeds 0..9            # HSI 6S (Indian Pines)
.venv/bin/python experiments/phase8G_emit_shift.py        --seeds 0..9 --model band   # EMIT band-as-modality
.venv/bin/python experiments/phase8R_scenedump_rich.py    --seeds 0..9            # CloudSEN12 reusable dumps
```
Results: `paper/results_phase8H_hsi6s_shift.json`, `paper/results_phase8G_emit_shift_band.json`.
Indian Pines 6S LUT is the shipped `data/srf_cache/T_6s_grid.npz` (200-band AVIRIS axis; Salinas 204-band and
Pavia ROSIS would each need their own 6S LUT — a CPU Py6S follow-up, not blocking).
