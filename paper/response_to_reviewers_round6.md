# Response to Reviewers — Round 6

**Manuscript:** *Aggregate Accuracy Does Not Reveal Reliability: Silent Conformal Certificate Failure under Operational Spectral and Domain Shift in Sentinel-2 Cloud Segmentation*

We thank the reviewer for an exceptionally careful reading. Every specific claim in the report checked out against the manuscript, including two residual numerical inconsistencies we had introduced ourselves. The review prompted the single most useful experiment of the whole project, and we have used it to **strengthen, not merely defend, the paper**. Below we address the three blocking concerns first, then the writing points. New experiments are numbered E1–E6; all are in the released code with their logs.

---

## Overarching change: a sharper mechanism and a cheaper remedy

The reviewer's concern B3 — that our L1C→L2A protocol confounds atmospheric physics with a preprocessing (normalization) mismatch — turned out to be the key that reframes the paper. We ran the disentangling experiment the reviewer asked for (**E4**) and found that the ~29% breach is **dominated by a stale input-normalization contract**: the model normalizes every product with its source (L1C) training statistics, and **re-normalizing the deployed L2A product with its own unlabelled statistics restores the naive joint risk to 10.2% at 74% coverage — without any target labels or abstention** (a disjoint-statistics variant, 10.5%, rules out a transductive artefact).

This has three consequences we have written throughout the revision:

1. **Mechanism.** The breach is a calibration-to-deployment *normalization contract* failure, not atmospheric physics as such (a small residual above clean is the genuine spectral distortion). We reconcile this with the failure of weighted conformal: weighted conformal reweights the calibration while leaving the shifted, miscalibrated scores untouched, whereas re-normalization repairs the model's input *upstream* of the scores.
2. **Remedy.** A **label-free** product-aware re-normalization is the first-line remedy; per-stratum (Mondrian) recalibration, which needs target labels, is demoted to a second, costlier remedy for the residual and for non-normalization shifts.
3. **Novelty positioning.** The label-free fix is a *known* test-time-normalization technique; we now concede this explicitly and cite it (AdaBN; Schneider et al. 2020; Nado et al. 2020; Benz et al. 2021; Tent; DUA; Corley et al. 2024). The contribution we claim is the **silent-failure diagnosis** — that a routine product-normalization mismatch invalidates an operational conformal certificate while the certificate's own statistic still passes — not the fix.

We also retract three over-claims the reviewer's concerns implied, detailed under B1/B3 below.

---

## Blocking concern B1 (§3.1) — "formal" weighted conformal

**Concern.** Our weighted-conformal arm is not a formal instance of Tibshirani et al. (2019): it uses a global threshold from a weighted empirical joint risk with no test-point term, and a pixel-softmax domain classifier against a component-level estimand.

**Agreed; we retract the "formal" framing.** We now describe the arm as *"a likelihood-ratio approximation … an approximation to, not an exact instance of, the exchangeability-restoring weighting, since the true likelihood ratio is unknown."* We do **not** attempt a formal test-dependent weighted CRC (which is semi-open for this component-level estimand); that would over-reach.

We also **soften "no source-only reweighting repairs it"** to *"neither reweighting we tried repairs it,"* and add a diagnosis (E1): the domain classifier separates clean from L2A with AUROC 0.80 but is heavy-tailed, retaining only 61% effective calibration sample (corrected from a stale 54% in the previous version — this was one of the residual bugs). A **synthetic pure-covariate-shift control** now shows the implementation *recovers* control when its assumption holds (naive 6.6%, heuristic 7.1%), so the L2A failure is an assumption-violation, not defective code. Full implementation is disclosed in Appendix (Weighted conformal).

## Blocking concern B2 (§3.2) — missing spatial baseline (per-pixel-specific or task-level?)

**Concern.** The 28.9% flagship is a deliberately per-pixel model; DOFA (spatial) is far more reliable (9.9%). Is the failure per-pixel-specific or task-level?

**We ran the decisive experiment (E6) and isolated the answer in one harness.** On DOFA's exact nine bands, trained *from scratch* with no pretraining, we compare a per-pixel MLP against a spatial U-Net through the *identical* certificate, evaluated on the *same* sampled pixels of the same scene-components (five model seeds × ten splits, BatchNorm- and depth-matched so spatial context is the only structural difference):

| model (9-band, one harness) | naive joint risk (L1C→L2A) | product-aware re-norm |
|---|---|---|
| per-pixel MLP | **24.2 ± 0.6%** (breach) | 8.7% @ 69% cov |
| spatial U-Net (from scratch) | **13.1 ± 0.7%**, *t*₄ CI [11.3, 15.0] (certified breach) | 8.9% @ 83% cov |

So the failure is **not a per-pixel artefact**: a representative spatial model (with *higher* clean accuracy, 82% vs 77%) breaches too. But spatial context attenuates the severity by 11 points *within one harness* (24.2→13.1), and pretraining takes DOFA the rest of the way (→9.9%). The result is a **coherent, honest gradient — per-pixel 24.2%, spatial-from-scratch 13.1%, spatial-and-pretrained 9.9%** — so the crisis is model-*class*-dependent, neither universal nor a per-pixel artefact. Crucially, **the label-free re-normalization restores both arms**, so the diagnosis and its fix generalize beyond the per-pixel model. This is now a dedicated paragraph in §3 (Model-class dependence) and is summarized in the abstract, C2, and the conclusion.

## Blocking concern B3 (§3.3) — preprocessing/normalization confound

**Concern.** Training on L1C and testing on L2A bundles the atmospheric shift with a stale input normalization; the two are not separated.

**This is the reframe above.** We ran the direct disentangling experiment (E4): re-normalizing the L2A input with its own unlabelled statistics — no change to the model, threshold, or temperature — restores the naive joint risk from 28.6% to **10.2% at 74% coverage** (disjoint-stats variant 10.5%). We therefore now state that the breach is *dominated by the normalization contract* rather than the atmospheric physics, retract the claim that *"the atmospheric correction itself is the cause,"* and retract that *"Mondrian is the necessary remedy"* (a label-free alternative exists). §3.3 (band-drop) forwards to this, and the mechanism/remedy are rewritten in §3.flagship, the abstract, C3, the discussion (a remedy hierarchy: label-free re-normalization → Mondrian → adaptive), and the conclusion.

---

## Writing points

- **W1 (weighted rename/soften + disclosure).** Done — see B1. Rename, softening, ESS 54→61 correction, AUROC 0.80, synthetic control, and full appendix disclosure.
- **W2 (surface/geography verdicts).** We keep surface = *suggestive* (its nested scene-component bootstrap [9.6, 14.7] includes the target) and geography = *no clear breach*, and we add a **direct paired difference bootstrap** (E3): the surface-minus-geography naive risk is +1.56% with a 95% interval [+1.11, +2.02] excluding zero, so the two axes differ significantly even though a "one-significant / one-not" comparison alone would not establish it.
- **W3 (ACOLITE "indistinguishable").** Replaced with a **two-one-sided test** (E2): the paired per-run Sen2Cor−ACOLITE difference is −0.08% with interval [−1.33, +1.16] inside a ±2% equivalence margin. We now say "statistically equivalent (a two-one-sided test)" rather than "indistinguishable," in the abstract, §3, and the discussion, and consistently call ACOLITE's output "ACOLITE-surface reflectance."
- **W4 (DOFA/B9 writeup).** The nine-band per-pixel MLP band-set control (48.4→29.2→23.8%) is retained and now feeds directly into the one-harness spatial isolation (E6); "channel-adaptivity is not the differentiator" is kept (both models are channel-adaptive) but the escape is now attributed to a *quantified* gradient (band set + spatial context + pretraining), each a competing contributor.
- **W5 (preprocessing + hyperparameter disclosure).** Added an appendix paragraph (Preprocessing and training): ×10⁻⁴ reflectance scaling; per-band z-scoring with *training* L1C statistics (the stale contract; the product-aware arm swaps only these two per-band statistics); L2A 12-band layout with B10 zero-filled; nodata/saturation screening; Adam (lr 10⁻³, batch 256), unweighted cross-entropy, 40 supervised epochs, 25 pretraining epochs, MLP width 256, DOFA frozen.
- **W6 (metadata gate ≠ Mondrian).** The Implications paragraph now decomposes the remedies into an explicit hierarchy: a metadata gate on product level triggers the **label-free** re-normalization automatically (no labels), and only where a residual remains or the state is not a normalization change does per-stratum labelled recalibration follow, at its stated label-and-abstention cost.
- **W7 (residual bugs).** Fixed: the DOFA seed axis is **ten**, not three (the manuscript reports the 10-seed DOFA run, consistent with its *t*₉ intervals); abstract coverage **94→93** (Table 1); the DOFA band-drop verdict is unified to a **modest, certified breach** (its *t*₉ interval [11.0, 12.4] excludes the target) across §3, the summary table, and the limitations.
- **W8 (minors).** In progress: AUROC described as prevalence-insensitive; the summary-table caption and weighted description softened; EMIT/6S remains a clearly-labelled companion analysis. (Length has grown with the new experiments; we are trimming and will move the EMIT anchor to supplementary material if the editor prefers a 16-page main text.)

---

## New material added in this revision

- **E1** synthetic covariate-shift control + weighted diagnostics (AUROC/ESS).
- **E2** ACOLITE−Sen2Cor paired-difference two-one-sided test.
- **E3** surface−geography difference bootstrap.
- **E4** product-aware normalization control (the reframe; label-free repair to 10.2%).
- **E6** one-harness per-pixel-vs-spatial-U-Net isolation (the decisive B2 experiment).
- **9 references**: AdaBN, Schneider et al. 2020, Nado et al. 2020, Benz et al. 2021, Tent, DUA, Corley et al. 2024 (test-time normalization); Cauchois et al. 2024 (robust conformal); Booksh et al. 2025 (a hyperspectral setting where coverage deviation is an *observable* diagnostic — the opposite of our silent failure, which we cite as a contrast).

All code and result logs are in the public repository. We believe the reframe makes the paper both more honest and more useful, and we are grateful to the reviewer for pushing us to it.
