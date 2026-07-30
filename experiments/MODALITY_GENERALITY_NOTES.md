# Modality generality of the silent-certificate failure (EXPLORATORY, post-v1.1.0)

The reliability paper's finding — a split conformal-risk-control certificate calibrated on the SOURCE product
silently fails under an operational preprocessing/atmosphere shift (its own finite-sample calibration statistic
stays ≤ α while the deployed confidently-wrong JOINT risk breaches) — reproduces across **three independent
sensor modalities and three distinct shift types**. Below, the ROBUST shared finding (the silent breach) is
separated carefully from the more delicate claim about *when the label-free re-normalization remedy works*,
because an adversarial self-review showed the naive version of the latter was partly tautological.

## The three modalities

| modality | sensor | bands | shift (source → target) |
|---|---|---|---|
| multispectral | Sentinel-2 (CloudSEN12) | 13 | L1C TOA → L2A BOA (Sen2Cor), real per-pixel |
| imaging spectrometer | EMIT | 244 | L1B radiance → L2A ISOFIT reflectance, real per-pixel |
| hyperspectral | AVIRIS/ROSIS (Indian Pines, Salinas, Pavia) | 200/204/103 | 6S dry→humid (CWV 0.5→4.0) |

## ROBUST finding — the silent breach generalizes (calibration statistic stays ≤ α everywhere)

| case | clean | NAIVE (stale) | Mondrian | calib stat |
|---|---|---|---|---|
| CloudSEN12 (flagship) | ~10 @ 74 % | **~29** | ~10 (labels+abstention) | ≤ α |
| EMIT (product shift, 10 sd) | 9.8 | **66.6 ± 1.6** | 10.0 @ 27 % | 9.9 % |
| HSI 6S Indian Pines (10 sd) | 10.2 | **86.0 ± 2.8** | 4.9 @ 7 % | 9.8 % |
| HSI 6S Salinas (10 sd) | 5.4 | **78.2 ± 3.1** | 4.0 @ 11 % | 5.5 % |
| HSI 6S Pavia (10 sd) | 4.7 | **33.6 ± 1.7** | 9.9 @ 35 % | 4.6 % |

Every case breaches silently. Two honest caveats on the HSI rows:
- **NO clean dose-response.** Within Indian Pines the breach SATURATES: mild 0.5→2.0 gives NAIVE **81.1**,
  severe 0.5→4.0 gives **86.0** (overlapping CIs). Even a mild atmospheric shift already breaks the
  certificate — a *fragility* statement, not a graded curve. The cross-dataset ordering (Pavia 34 < Salinas
  78 < IP 86) is **confounded** with sensor, task, and difficulty, so it is NOT evidence that "breach ∝ shift
  magnitude."
- **Salinas/Pavia clean coverage is 100 %** (the datasets are easy enough that the α=10 % CRC threshold accepts
  everything), so their clean certificate is near-trivial. **Indian Pines (90 % clean coverage) is the primary,
  non-trivial demonstration**; Salinas/Pavia are secondary support.

## DELICATE finding — re-norm's effectiveness depends on the shift's STRUCTURE (two failure modes)

Product-aware re-normalization (re-standardize the target with its OWN per-band statistics) restores the
certificate ONLY when the shift is close to a *global per-band affine* map. There are two distinct ways it
can fall short — one spectral, one spatial:

| shift structure | example | NAIVE → re-norm (10 sd) | why |
|---|---|---|---|
| global per-band multiply | HSI 6S **uniform** (IP) | 86.0 → **11.3** (FULL fix) | per-band standardize is EXACTLY invariant to a per-band multiply (verified \|z_src−z_tgt\|=0) — a POSITIVE CONTROL, not a discovery |
| real atmospheric correction | CloudSEN12 L1C→L2A (Sen2Cor) | ~29 → **~10** (≈ full) | real per-pixel, but dominated by a global per-band component |
| spatially heterogeneous, moderate | HSI 6S **spatial** (IP, `--spatial-cwv`) | 78.4 → **38.9** (PARTIAL) | a global re-norm cannot capture per-pixel variation of the affine map |
| **spectrally** non-affine transform | EMIT radiance→reflectance | 66.6 → **24.8** (PARTIAL) | the map is not per-band affine even globally |
| spatially heterogeneous, **extreme** | HSI 6S **spatial** (Salinas, deep SWIR water bands) | 62.6 → **63.5** (NO benefit) | per-pixel variance in near-opaque bands is so large that global stats help nothing |

**re-norm effectiveness is a SPECTRUM, not a switch: FULL (global per-band) → ≈full (CloudSEN12) → PARTIAL
(moderate spatial / spectral transform) → NONE (extreme spatial heterogeneity).** Two distinct mechanisms
degrade it — *spatial* heterogeneity (per-pixel affine variation, HSI spatial-6S) and *spectral* non-affinity
(EMIT). The key self-review correction: the **uniform** 6S full-fix is guaranteed by the invariance identity
(I *constructed* the shift as `cube·T`), so it is a controlled POSITIVE CONTROL, NOT independent evidence.
The label-free remedy is therefore **not a panacea** — it is the right fix precisely when the deployment shift
is global-per-band-dominated (the CloudSEN12 operational case the paper targets), and it can help little or
nothing when the shift is strongly heterogeneous or non-affine.

## Architecture independence
On EMIT the breach is the same for a plain MLP and the paper's band-as-modality GroupedCrossBandAttention
(+ SGMAE pretraining), 10 seeds each:

| model | clean | NAIVE | Mondrian | re-norm | calib-stat |
|---|---|---|---|---|---|
| MLP | 9.8 ± 0.2 | 66.6 ± 1.6 | 10.0 @ 27 % | 24.8 ± 0.5 | 9.9 % |
| band-as-modality | 9.9 ± 0.2 | 57.5 ± 3.6 | 9.9 @ 19 % | 26.7 ± 1.4 | 9.9 % |

Same catastrophic silent breach and same partial-re-norm residual — the failure is a property of the
stale-normalization CONTRACT, not the classifier family.

## Honest scope / caveats
- The 6S LUTs are computed at each sensor's **nominal reconstructed wavelength axis** (not an
  acquisition-specific calibration), so the shift is physically MOTIVATED but axis-approximate. The re-norm
  conclusions are axis-independent (they concern the shift's affine/spatial structure, not exact wavelengths).
- Single-scene, pixel-level CRC (no exchangeable scene-component unit as in the flagship); block-disjoint
  train/calib/eval splits with a 1 px guard band.
- The spatial-CWV field is an imposed smooth gradient; the specific residual (~34) is illustrative of "re-norm
  is partial under spatial heterogeneity," not a calibrated physical number.

## Reproduce
```
.venv/bin/python experiments/precompute_6s_dataset.py --dataset salinas   # + pavia; IP LUT is shipped
.venv/bin/python experiments/phase8H_hsi6s_shift.py --dataset indian_pines                 # uniform (control)
.venv/bin/python experiments/phase8H_hsi6s_shift.py --dataset indian_pines --spatial-cwv   # spatial (realistic)
.venv/bin/python experiments/phase8G_emit_shift.py  --seeds 0..9 --model band              # EMIT arch-independence
```
Results: `paper/results_phase8H_hsi6s_{shift,salinas,pavia}.json` (uniform),
`paper/results_phase8H_hsi6s_{shift,salinas}_spatial.json` (spatial),
`paper/results_phase8G_emit_shift_band.json`.
