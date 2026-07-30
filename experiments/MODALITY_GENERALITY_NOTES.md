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

## Architecture GENERALITY — the band is NOT "more robust" (AURC/matched-coverage settles it)
The silent failure holds for BOTH a plain MLP and the paper's band-as-modality GroupedCrossBandAttention
(+ SGMAE). The band model's LOWER joint-risk under shift (HSI 6S NAIVE 36.7 vs MLP 86.0) tempted an earlier
"self-protects / graceful degradation" reading — a **coverage-honest AURC / matched-coverage analysis REFUTES
it**. The band's low joint-risk is purely an OPERATING-POINT artifact:

| model | NAIVE joint @ cov | AURC clean→naive | selective-risk @ matched 40 % cov (naive) |
|---|---|---|---|
| MLP | 86.0 @ 97 % | 4.9 → 83.0 | **84** |
| band | 36.7 @ **41 %** | 6.7 → **91.4** | **91** |

(10 seeds.) The band's joint-risk is lower only because it happens to ACCEPT far fewer pixels (41 % vs 97 %).
Hold coverage FIXED and the band is WORSE at every level (selective risk 91 vs 84 at 40 % cov; also 60/80 %).
Under shift the MLP's confidence stays WEAKLY informative (AURC 83 < its full-coverage error 86) while the
band's is UNINFORMATIVE (AURC 91.4 ≈ its error 92 — essentially zero signal about which shifted predictions
are wrong, so no threshold recovers useful coverage). So neither model's confidence is trustworthy under the
stale-normalization shift, and the band's is the WORSE — the apparent robustness was a coverage mirage
(`aurc_matched_coverage.py`).

What IS robust across architectures: BOTH breach the certificate (≫ 10 % target) SILENTLY (calib
9.8–9.9 % ≤ α), and BOTH follow the re-norm spectrum (uniform full-fix MLP 11.3 / band 10.7; spatial partial
MLP 38.9 / band 26.2). The failure is a property of the stale-normalization CONTRACT, not the classifier
family — and no architecture's confidence self-rescues.

## Does the "confidence dies" finding backwash onto the FLAGSHIP? It is UNIT-DEPENDENT (self-review sharpened).
Checked on the CloudSEN12 flagship (band-as-modality) offline from scenedump_rich (AURC is temperature-
invariant, no retrain), 10 seeds. The FIRST pass (`aurc_cloudsen12_core.py`, pixel-level) said "informative";
the scene-component pass (`aurc_scene_component.py`, the paper's EXCHANGEABLE UNIT) shows it does NOT lift to
the unit:

| CloudSEN12 flagship, L1C→L2A naive | AURC | error | gap |
|---|---|---|---|
| pixel-level (accept individual pixels) | 19.6 | 32 | +12 → informative |
| **scene-component-level (accept WHOLE components)** | **30.6** | 32 | **+1 → NOT informative** |

Component-cluster bootstrap 90 % CI on the pixel AURC is [18.9, 22.4] (the pixel result is solid), but the
confidence's ranking power is almost entirely WITHIN-scene: it identifies which PIXELS in a scene are wrong,
NOT which whole SCENES are bad. **IMPLICATION (honest):** the paper's Mondrian remedy accepts individual
confident PIXELS, so it IS supported (pixel AURC 19.6 → recovers ~74 % coverage on CloudSEN12; contrast the
extreme HSI-6S pixel AURC 91.4 ≈ error → Mondrian collapses to 7–13 %). But a whole-SCENE trust decision is
NOT supported by the confidence. So the earlier unqualified "core confidence is informative" was
pixel-level-optimistic; the precise statement is: informative for pixel-level acceptance (Mondrian), NOT for
scene-level trust. The paper's OPERATIONAL claims (pixel-level Mondrian, re-norm, silent CRC failure) stand;
the flagship is safe for what it actually asserts, with this unit caveat recorded.

## Is confidence-informativeness a clean "severity axis"? NO — EMIT refutes it (self-review).
An earlier note claimed informativeness scales inversely with shift SEVERITY. The EMIT AURC (`aurc_emit.py`,
MLP vs band, 5 seeds, pixel-level) is a direct COUNTEREXAMPLE:

| pixel-level naive AURC | AURC | error | gap | verdict |
|---|---|---|---|---|
| CloudSEN12 L1C→L2A (mild) | 19.6 | 32 | +12 | informative |
| **EMIT radiance→reflectance (extreme)** | **~50** | ~71 | **+20** | informative |
| HSI 6S dry→humid, MLP (extreme) | 83 | 86 | +3 | weakly informative |
| HSI 6S dry→humid, band (extreme) | 91 | 92 | +1 | dead |

EMIT is a MORE extreme shift than CloudSEN12 yet its confidence is MORE informative (+20 > +12), so
informativeness is NOT a monotone function of severity — it is shift×model-specific (the HSI-6S *band* going
dead is the outlier, its attention locking onto the corrupted water-vapour bands). Two further honesties:
(1) on EMIT, MLP ≈ band at matched coverage (selective risk @40 % 48.7 vs 50.3; AURC 50.9 vs 48.6, within
seed noise sd~3) — so the band is NOT more robust on EMIT either, confirming the coverage-artifact reading.
(2) "informative" ≠ "usable": EMIT's top-40 %-confident pixels are still ~49 % wrong, which is why EMIT
Mondrian needs ~27 % coverage. So Mondrian's coverage cost depends on BOTH the target error level AND the
ranking (AURC), not on a single severity number.
(`phase8G_emit_shift.py --model {mlp,band}`, `phase8H_hsi6s_shift.py --model band [--spatial-cwv]`.)

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
