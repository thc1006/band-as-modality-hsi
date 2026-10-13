# Result manifest — maps each paper number to its script / log / CSV / seed

Release **v1.1.0** (git tag `v1.1.0`). Crossed 10 calibration-split x 10 model-seed design unless noted.
Model seeding: the seed-fixed scripts seed the proposed model immediately **before** its constructor with
`hw.seed_model(seed)` (which seeds with seed+101) or an equivalent seed+101 call (e.g. R10, R14, R17, R18);
the b2 MLP is seeded with `torch.manual_seed(seed)` inside `phase2_degradation.train_mlp`, right before its
constructor (this overrides an earlier `hw.seed_model`, which is why the seed fix left b2 unchanged); the
DOFA heads and `phase8R8_spatial_isolation.py`'s models are also seeded with `torch.manual_seed(seed)`
before construction; `phase8R_perclass_weighting.py` and `phase8R9_acolite_clip_sensitivity.py` construct
their models without a seed call.

Reproduction scope: the committed CSV/JSON/log files below are the outputs of the listed scripts, except
where the section "Checked against the v1.1.0 text" below says a cited file does not hold the number.
`reproduce.sh` holds the commands. The steps that write files write them over the committed ones, so
`git diff paper/` shows what those re-runs changed -- except the flagship, which it writes to
`results_phase8R_{reliability,raw}_10seed_repro.csv`. Many reliability analyses only print their numbers
(their committed record is a `paper/*.log`): for those, `reproduce.sh` keeps the console output in
`paper/logs/reproduce/<step>.log` (gitignored), to be compared with the committed log -- `git diff` shows
nothing for them. The 4x P100 re-run of the flagship trained different models than the committed 2x V100
run: its headline is 28.71 ± 0.82 against 27.85 ± 0.79, and clean accuracy differs in all 100 cells
(`experiments/H10_TRAIN_ONCE_VERIFICATION.md`; the P100 outputs are in `paper/p100_rerun_2026-10-13/`); expect
a re-run on another machine to be a new draw of these numbers, not a digit-for-digit copy.
`experiments/verify_round7.py` is a **function/regression** suite (18 checks over the round-7 code paths
and their guards) — it validates the building blocks, not a full re-execution of every 10x10 campaign; the
campaign numbers are reproduced by running the scripts themselves.

Standard errors: where a script reports a two-way (split seed x model seed) standard error, it is the
Cameron–Gelbach–Miller cluster-robust estimator in one of two variants. `phase8R_reliability.py`,
`phase8R2_landcover_reliability.py` and the `phase8E2_*` DOFA scripts compute it without small-sample
factors (**CGM** below); the other phase8R scripts that report one import
`phase8R_perclass_weighting_agg.two_way_se`, which multiplies each one-way term by G/(G−1) and the iid term
by N/(N−1) (**CGM-ss** below). On the per-arm means of the committed 10 x 10 outputs CGM-ss is about 6 %
larger; on some paired contrasts, and on R15's 5 x 10 design, much more (R15's L2A k1_w5−k5: 0.115 against
0.074). Bootstrap-only analyses report no such SE, and R11's synthetic controls use iid standard errors.

| Paper item | Value | Script | Output |
|---|---|---|---|
| Flagship naive L2A joint (Tab. flagship, abstract) | 27.85±0.79 | `experiments/phase8R_reliability.py` | `paper/results_phase8R_reliability_10seed.csv` |
| Band-drop table | 8.53 / 8.22 / 8.56 / 27.85 | `phase8R_reliability.py` | same CSV (states clean/dropB10/dropB1B9B10/L2A_real) |
| Temp×threshold factorial | 28.3/8.5/29.7/8.6 | `phase8R_sceneboot_2x2.py` | `paper/results_sceneboot2x2_seedfix.log` |
| Pixel-pooled sanity | 28.3 | `phase8R_sceneboot_2x2.py` | none committed: the log and script compute no pixel-pooled value (D3 below) |
| Alpha 5–20% sweep | 24.0/28.3/31.1/32.5 | `phase8R6_alpha_sensitivity.py` | `paper/results_phase8R6_alpha_seedfix.log` |
| Weighted heuristic | 32.2, 32.6 | `phase8R7_weighted_conformal.py`, `phase8R9_formal_weighted_crc.py` | `paper/results_phase8R7_seedfix.log`, `paper/results_R9formal_seedfix.log` (`results_phase8R9_formal_weighted_crc.log`, 32.67, is the pre-fix run) |
| Formal weighted CRC (test-point term) | real L2A 1.87@7% DEPLOYMENT mean, **feasible only 12%** (P0-3 FIXED: `weighted_crc_perunit` returns a feasibility flag — an infeasible unit, target<0, is an abstain-FALLBACK, NOT a formal 0-risk; conditional-on-feasible 15.4@53 reported separately); est-w PC 7.1 tracks true 6.2 @ AUROC 0.88 overlap | `phase8R11_weighted_crc_formal.py` | `paper/results_phase8R11_estweight.log` |
| Representation AUROC (raw/product) | 1.00 / 0.56 | `phase8R13_weighted_representation.py` | `paper/results_phase8R13_weighted_representation_groupcv.log` |
| Radiometric audit (magnitude self-consistency) | L1C p1≈15 ⇒ under a +1000 offset the true refl there would be ≈−0.10, far beyond dark-target noise (~−0.001) ⇒ stored on no-offset scale; all versions pre-N04.00; and the product-aware fix is offset-invariant anyway, so the headline does not rest on this audit (physics wording corrected in review: TOA refl CAN be slightly negative — a magnitude test, not a sign impossibility) | `phase8R12_radiometric_audit.py` | `paper/results_phase8R12_radiometric_audit.log` (its verdict line predates that wording correction) |
| Normalization control (disjoint / pooled) | 10.8±0.3 / 10.5±0.3 (from 28.4) | `phase8R10_normalization_control.py` | `paper/results_phase8R10_normalization_control_{summary.json,percell.csv,10seed.log}` |
| P0-2 quantile-tie convention — FIXED IN CODE | `phase8R10.quantile_match` now uses the tie-aware mid-distribution CDF on unique knots (`_mid_cdf_knots`) + a self-test (identity 9.5e-08, monotone, tie F(0)=0.375) + per-band diagnostics (knots/clamp/endpoint); R17/R18 import it (R19 has its own tie-aware CDF). The committed R10/R17/R18 outputs predate it: their full-calibration quantile-transport arm (9.546 ± 0.238, printed as 9.6) used the earlier CDF, which gives a tied value the rank of its last duplicate. On the same trained models (4x P100, 10 × 10, both CDFs in one run) the change moves that arm's joint risk by −0.039 (CGM-ss SE 0.018; [−0.079, +0.002], p = 0.06; with plain CGM p = 0.047) and its coverage by +0.07 points; of the 0.1–0.2 recorded earlier, R10's part is the mis-rounding of 9.546 and R17's 0.2 is not explained by it, and the attribution of that shift to GPU non-determinism is contradicted by R10, R17 and R18 reproducing this arm bit for bit on the V100 box. The product-aware (mean/std) arms, the headline 10.8, do not use this function | `phase8R10_normalization_control.py`; the check: `experiments/PHASE8R10_QUANTILE_CONVENTION_CHECK.md` | `paper/p100_rerun_2026-10-14/results_phase8R10_quantile_convention_{percell.csv,summary.json}` |
| Normalization decomposition | product +17.9 / composition +0.0 / TTA +1.0 | same | same |
| Quantile-transport generalization (3.2) | full-calib 9.55 (anchor≈headline 9.6); sample-size n8/16/24/40 = 13.0/11.2/10.4/9.8; random halves 10.1/9.9 (agree); bright/dark calib 12.5/14.4 (surface-dependent); cov ~70% all arms (df=9) | `phase8R17_quantile_generalization.py` (10 model × 10 split seed) | `paper/results_phase8R17_quantile_generalization{_summary.json,_percell.csv,.log}` |
| Quantile transfer: acquisition-month + Sen2Cor product-version (LOOSE split) | full 9.55; Mar–Aug/Sep–Feb 10.4/9.8; new/old Sen2Cor product-version 9.9/10.8; both ≪ surface (df=9). HONEST (adversarial review): (a) month split = calendar bin under NH naming, NOT hemisphere-aware season (CloudSEN12 spans both hemispheres); (b) `s2_sen2cor_version` = Sen2Cor product-version field, NOT the ESA Processing Baseline; (c) MIXING (measured): 158/184 components span both month-halves, 183/184 span >1 version → these arms are a LOOSE stratification that RECONFIRMS R17 random-subset robustness, NOT a clean cross-season/version transfer; surface (R17) is the cleanly-separable axis | `phase8R18_quantile_season_baseline.py` (10 model × 10 split seed) | `paper/results_phase8R18_quantile_season_baseline{_summary.json,_percell.csv,.log}` |
| Weighted CRC on re-normalized rep (R19, full 10×10, factorial) | src\_uniform 28.1@94 (reproduces flagship), src\_weighted 1.5@7 **feasible only 12%** (infeasible→deployment abstain-fallback, not a chosen operating point), ren\_uniform 9.4@71, ren\_weighted 8.9@69; **re-norm alone −18.7 [−19.7,−17.8]** restores the operating point, **weight\|renorm −0.5 [−1.0,+0.0] incl 0** (weighting adds no measurable value on the re-normalized rep); held-out AUROC 0.98→0.57, calib-ESS 10→75% (eval-ESS 39→67 — the earlier "41" mislabel was eval-ESS). Rewritten after adversarial review (mid-CDF ECDF, feasibility flag, calib-ESS via wc, factorial contrasts) | `phase8R19_weighted_crc_on_renorm.py` (10 model × 10 split seed) | `paper/results_phase8R19_weighted_crc_on_renorm{_summary.json,_percell.csv,.log}` |
| ACOLITE paired diff | 29.6 vs 29.34, +0.26 [−1.20,+1.71] (CGM-ss, t9; the paper prints the seed-cluster bootstrap [−1.02,+1.42]) TOST pass | `phase8R3_acolite.py`+`phase8R10_acolite_paired_diff.py` | `paper/results_phase8R3_acolite10.csv` (per run), `paper/results_phase8R10_acolite_paired_diff.log` (both intervals) |
| Validation replication | 28.0±0.7 | `phase8R5_secondbench.py` | `paper/results_phase8R5_secondbench_seedfix.log` |
| Validation per-class | IoU 54.4→30.0 | `phase8R5_valdump.py`+`phase8R5_val_classwise.py` | `paper/results_val_classwise_seedfix.log` |
| Class-wise decomposition (Tab. classwise) | full 10×10: clean 8.43 / L2A 28.25 joint | `phase8R16_classwise_10x10.py` (10 split × 10 model seed) | `paper/results_phase8R16_classwise_10x10{_summary.json,_percell.csv,.log}` |
| DOFA (atmos / band-drop) | 9.95 / 11.77 | `phase8E2_dofa_crc.py` (10 split × 10 model seed) | `paper/results_phase8E2_dofa_crc_seedfix.log`; no committed CSV holds this run -- `results_phase8E2_dofa_crc.csv` is an earlier 3-model-seed run (D5 below) |
| Band-set MLP 13/−B9/9 | 48.4/29.2/24.1 | `phase8R9_bandset_control.py` | `paper/results_phase8R9_bandset_seedfix.log` |
| Flagship band-set 13/−B9/9 | 28.3/22.6/21.1 | `phase8R14_flagship_bandset.py` | `paper/results_phase8R14_flagship_bandset.log` |
| Spatial U-Net vs per-pixel | 14.4 / 23.6 (restore 8.9 / 8.8) | `phase8R10_unet_spatial.py` | `paper/results_phase8R10_unet_seedfix.log` |
| Receptive-field isolation (capacity-matched, 7 arm) | breach k1 20.4→k7 17.4; capacity-matched k1\_w5−k5 +2.11 [1.79,2.42], k1\_w7−k7 +3.42 [1.34,5.50] exclude 0 (RF 17/25px); k1\_w3−k3 +0.74 incl 0; capacity-only all incl 0 (CGM-ss, df=4) | `phase8R15_receptive_field.py` (5 model × 10 split seed) | `paper/results_phase8R15_receptive_field{_summary.json,_percell.csv,.log}` |
| Surface / geography / diff | 11.34 / 9.71 / +1.62 | `phase8R2_landcover_reliability.py`+`phase8R10_surface_geo_diff.py` | `paper/results_phase8R2_{landcover,geography}_raw_10seed.csv`; `paper/results_surface_geo_diff_seedfix.log` (the SEs as printed, from CGM-ss, and the bootstrap interval) |
| ACOLITE retained-subset bias | retained 181 comps / 590 patches; effect sizes negligible–small (conditional on successful processing) | `phase8R9_acolite_retained_bias.py` | `paper/results_phase8R9_acolite_retained_bias.log` |
| Surface nested bootstrap (seed-averaged estimand) | θ̂ 11.47, mean 11.7, 95% CI [9.5,14.2], feasible 100%, includes 10 → inconclusive | `phase8R9_surface_nested_boot.py --B 600` (B recorded in the JSON) | `paper/results_phase8R9_surface_nested_boot_summary.json` |
| Figures fig2/fig3 | regenerated seed-fixed | `make_paper_figs.py` | `paper/figs/fig2_flagship_reliability.pdf`, `fig3_domain_gaps.pdf` |

## Checked against the v1.1.0 text (2026-10-13/14)

The Value column gives each number as its committed file holds it (rounded) or as the paper prints it.
Every row was compared with `paper/latex/main.tex` (unchanged since v1.1.0) and with the committed files.
Each matches its cited file except the pixel-pooled row, which has no committed source (D3), the
normalization-control and quantile-generalization rows, which give the paper's mis-rounded values (D11),
and the acquisition-month row, whose mixing counts have no committed source (D15). Where a printed value
differs from the committed one, a D item says so (D4, D5, D11); some values (e.g. R11's 15.4@53, the
nested bootstrap's θ̂, R19's 28.1@94) are committed results the paper does not print. The P0-2 row
records a code change made after the paper's runs.

The statements below, in the paper's text, are not supported by the committed outputs, or the code and the
data contradict them. By origin:
- pre-fix numbers left in the text, which the pre-fix release's (eae9962) files reproduce: D6 (except its
  scene-component bootstrap), D5's SEs and its "9.9", D2's coverage cost, D12's ACOLITE clip sensitivity;
- text that entered with the seed-fixed re-run (c59100a) or later: D1, D2's 28.0 with the product drop, D3,
  D5's means, D6's surface "2.39 ± 0.21", D7's [9.0, 14.4], D8's 9.71 ± 0.26, D11;
- text that no committed file in any version supports (present since eae9962): D4, D2's "AURC 6.2 → 21.6",
  D6's flagship scene-component bootstrap [25.7, 33.1], D7's [9.1, 12.4] and [9.3, 13.6], and D12's B9
  shift and abstain-all remark;
- descriptions of the methods or the data that the code or the data contradict: D9, D10, D12's
  dark/bright unit counts, D13, D14 (D15's counts are right but have no committed source).
None of them reverses a conclusion; each would need a correction in a revised version. CGM and CGM-ss are the
two SE variants defined at the top of this file.

| # | Where (main.tex) | What the committed outputs show |
|---|---|---|
| D1 | Tab. banddrop, "Δ vs clean" (160–161); text (149) | The −B10 and −B1/B9/B10 entries (−0.15 [−0.4,0.1], +0.13 [−0.3,0.5]) are naive minus that state's own Mondrian. Naive minus clean, from `results_phase8R_raw_10seed.csv` (CGM, t9): −0.32 [−0.57,−0.06] and +0.03 [−0.46,+0.52]; the L2A entry (+19.3 [17.5,21.1]) is naive minus clean. So "both within noise of zero" does not hold for −B10, which LOWERS the joint risk slightly; that band removal does not break the certificate is unaffected. |
| D2 | Validation replication (180; also 35, 261, 285) | The text says the two train-shared products were dropped (0.4 % of pixels). The reported 28.0 ± 0.7 [26.3,29.6], Mondrian 7.8, clean 8.1 and the per-class figures come from runs on all 214,000 pixels (`results_phase8R5_secondbench_seedfix.log:1`, `results_valdump_seedfix.log:1`); `phase8R5_secondbench.py` and `phase8R5_valdump.py` apply no such filter. The only run with the drop (`results_phase8R5_val_leakfix.log`) predates the seed fix: 29.14 ± 0.84, from 29.44 without it. Same paragraph: "coverage cost (93 → 42 %)" is the pre-fix run's (seed-fix log: 92 → 44 %); "AURC 6.2 → 21.6" has no committed output. |
| D3 | Pixel-pooled check (205) | No committed file holds 28.3 vs 28.3 % (coverage 92.7 vs 92.7, selective 30.0 vs 30.0, ±0.8). The only committed pixel-pooled flagship aggregate, `results_phase8R_weighting_agg.csv`, is pre-fix (29.08 component-equal / 29.04 pixel-pooled). |
| D4 | Alpha sweep (103) | "25–33 %", "15–20 points", Mondrian "3.6": the seed-fix log gives 24.0–32.5 %, excess 12.5–19.0 points, Mondrian 3.5. |
| D5 | DOFA (245, 268–269) | The text's means are the seed-fix log's (9.95, 8.79, 11.77, 8.63); the SEs ±0.21, ±0.19, ±0.36 and coverage 76.4 are those of `results_phase8E2_dofa_crc_r4_10seed.csv`, an earlier 10 × 10 run (committed in eae9962) on all 975 test patches -- before the train-overlap guard the seed-fix run applies (971); the DOFA heads were seeded the same way in both; "±0.42" and "80.0" match no committed output, and ±0.42 does not fit the printed t9 interval [10.9,12.6] (SE ≈ 0.37). Tab. summary (269) prints the band-drop risk as 11.7 ± 0.4, the earlier run's 11.685; the seed-fix 11.77 rounds to 11.8. The "9.9" printed for the L2A risk (35, 54, 245, 247 twice, 268) is unchanged pre-fix text: it matches the committed 10 × 3 run (`results_phase8E2_dofa_crc.csv`, 9.865), not the earlier 10 × 10 (9.953, which rounds to 10.0); the seed-fix log's 9.95 keeps only two decimals. |
| D6 | Pre-fix values left in the text | Tab. flagship selective-risk and accuracy SEs (124, 126) and Tab. banddrop selective SEs (159, 162); the sensitivity intervals at 103 -- seed-cluster bootstrap [28.0,29.9], seed-only t9 [27.8,30.1], surface [10.9,11.8] (current data ≈ [26.3,29.4], [25.9,29.8], ≈ [10.7,12.0]; all still exclude 10 %) -- and the flagship scene-component bootstrap [25.7,33.1] (seed-fix log [25.3,31.7]); the AURC paragraph (147): L2A AURC 20.9, AUGRC 12.3 (clean 4.9), AUROC 68.6 ± 1.1, state-temperature AURC 20.7 (current 19.1, 11.6 (4.8), 70.3 ± 0.7, 18.9); the surface paragraph: "2.39 ± 0.21" (211; the SE is the pre-fix one, Tab. deploytriple prints ±0.25), b2 "11.9 → 2.5" (211, 287; a 3-seed run, the 10 × 10 gives 12.1 → 2.5) and "41 %" coverage (217, 228; current 39.85). |
| D7 | Intervals without a computed source | "[9.0,14.4]" (211) is the superseded nested bootstrap (B = 1500 log) where 103/274/289 use [9.5,14.2] (the B = 600 JSON); "[9.3,13.6]" (103, 289) is a constant the nested-bootstrap script used to print as a reminder (the print was removed in 0e6a77f; the two committed nested logs still show it), not a result any committed file computes; "[9.1,12.4]" (213) appears in no committed output. |
| D8 | SE estimators | Both variants are two-way cluster-robust, as the paper says, but it uses both without saying which, sometimes in one sentence: 213 gives the proposed model 9.71 ± 0.26 (CGM-ss, `phase8R10_surface_geo_diff.py`) and b2 10.12 ± 0.23 (CGM, the phase8R2 CSV); in Tab. flagship the joint-risk and coverage SEs are CGM while the selective-risk and accuracy SEs are pre-fix CGM-ss, as are the AURC paragraph's AUGRC and AUROC SEs (147; D6). |
| D9 | AURC / AUGRC definitions (95, 147, 314) | The text calls AURC component-equal, but `phase8R_reliability.py` computes it over pooled pixels (`bandsim.reliability.aurc` takes no weights); AUGRC is described for the clean-temperature detector, but its only column uses each state's own temperature. |
| D10 | Band-set MLP control (245) | `phase8R9_bandset_control.py` loads the test split without the train-overlap guard the paper describes (101): 390,000 test pixels against 388,400 everywhere else (`results_phase8R9_bandset_seedfix.log:1`). |
| D11 | Rounding | Printed values that are not the 1-dp rounding of the committed value: 10.8 (10.748), 9.6 (9.546), 10.4 (10.348), 28.2 (28.255), 9.3 (9.350), 39.9 (39.849, 235); SE 0.5 for 0.448 (264). |
| D12 | Minor | 312: "152 dark / 33 bright" counts the one mixed component twice (184 components; the nested JSON: 151 / 32 / 1 excluded), and 211's "bootstrap over the 33 bright units" resamples 32 (both nested-bootstrap versions keep only purely-bright components). 316: the ACOLITE clip sensitivity is a pre-fix 3-seed run (its log compares against the pre-fix 28.56). 245: "B9 shifts by ~+0.2–0.3" has no committed output. 88: "[the abstain-all threshold] is reached only near the conformal feasibility floor, e.g. the surface Mondrian arm" -- no committed 10 × 10 output selects it (no infinite threshold in the flagship, surface or geography raw CSVs; the lowest surface Mondrian coverage is 20.7 % for b2, 21.5 % for the proposed model). |
| D13 | Reproducibility details (308) | "spectral-group masked-autoencoder pretraining for 25 epochs", "batch size 256", "plain-MLP baselines (one hidden layer of width 256)": in the CloudSEN12 scripts the pretraining runs `max(1, epochs // 2)` = 20 epochs (phase8R; phase8R2, whose v1.1.0 code had its own copy of the same recipe and which trains through phase8R since 6c1e392; R10 and R16–R19; 25 and 256 are the Indian Pines defaults of `phase2_degradation.py`), and the batch size is `auto_bs` of the training-set size -- 4096 for the flagship (720,000 training pixels; `train_bs` in its sidecars) and for R10/R16–R19 (900,000), 2048 for the surface and geography axes (552,000 and 533,520). The b2 baseline (`bandsim.model.MLPBaseline`) and the band-set control's MLP have two hidden layers of width 256; the U-Net comparison's per-pixel MLP has two of width 128 with batch normalisation (60 epochs, batch 4096), and the band-set control trains 30 epochs at batch 8192. Learning rate 10⁻³ and 40 supervised epochs are right for the flagship-family scripts. |
| D14 | Validation set (35, 180, 252, 261, 285; framed as the original CloudSEN12 release, 73) | The validation sample (`data/cloudsen12_val_secondbench.npz`) was drawn from the tacofoundation release (`phase8R5_val_loader.py`), not from CloudSEN12-high. Reconstructing its sampled pixels exactly and reading the same positions in CloudSEN12-high's validation files: the reflectances agree to within 2 DN (the tacofoundation release stores multiples of 4 DN), but the labels differ at the sampled pixels of 23 of the 535 patches (only those pixels were compared) -- 2,815 of the 214,000 (1.3 %) -- so the replication's labels are CloudSEN12+'s later revision. Calibrated and scored with CloudSEN12-high's labels on the same models (4x P100, 10 × 10), the naive L2A joint risk is 0.68 points higher, [+0.46, +0.91] (28.7 against 28.1); Mondrian and clean barely move, and the breach replicates either way (`experiments/CLOUDSEN12_VAL_LABEL_VERSION_CHECK.md`). |
| D15 | R18 mixing counts (176; the acquisition-month row) | "158 of 184 components contain both calendar-month halves and 183 span more than one Sen2Cor version" has no committed source: the committed R18 log and summary predate the mixing line (added in a81f29e). Recomputed from the test metadata and the flagship's component map, both numbers are right; but the line the current R18 code prints counts something else -- components spanning both of its new/old product-version groups (160 of 184). |

## Provenance caveats (recorded 2026-10-13/14)

Commit `c59100a` (the seed-fixed 2x V100 re-run) replaced the `proposed` rows of the three 10x10 aggregates
below and the `proposed` half of their `_raw` companions -- the numbers this manifest quotes. Their `b2` rows
are unchanged from the pre-fix version; for the flagship the seed-fix log prints the same b2 values, so the
fix did not change the b2 model there (the surface and geography logs print only the proposed model). The
commit did not regenerate the `.provenance.json` sidecars, which still describe the pre-fix runs
(development-repository commits that are not in this repository). It rewrote only the `cwd` path in four
of these six -- as in 71 of the 82 sidecars it changed (10 others also had home-directory paths rewritten,
and `results_phase8F_multi_perband.csv`'s sidecar was replaced by another run's):

| CSV (and its `_raw` companion) | its sidecar still records |
|---|---|
| `paper/results_phase8R_reliability_10seed.csv` | commit `6d8c343` (working tree dirty), 2026-07-22 |
| `paper/results_phase8R2_landcover_10seed.csv` | commit `5e1111c`, 2026-07-23 |
| `paper/results_phase8R2_geography_10seed.csv` | commit `98363bb`, 2026-07-23 |

The CSVs' proposed-model means agree with the seed-fix run logs
(`paper/results_phase8R_reliability_seedfix.log`: 27.85, 2x V100;
`paper/results_landcover_surface_seedfix.log`: 11.34 and `paper/results_landcover_geo_seedfix.log`: 9.71,
each 1x V100). `c59100a` also replaced the per-run ACOLITE files `results_phase8R3_acolite10.csv` and
`results_phase8R3_acolite.csv` (byte-identical to each other) and added
`results_phase8R_classwise_10seed.csv`, `supp_scene_components.csv` and
`results_phase8R10_normalization_control_percell.csv` (extended in 0e6a77f with the robust and
quantile-transport arms; its earlier rows are unchanged).

Of the seed-fix logs, only the flagship, surface, geography and ACOLITE ones have a committed CSV with the
same numbers; for DOFA, the validation set and the single class-wise run the committed CSVs hold pre-fix
numbers (`results_phase8E2_dofa_crc.csv`, `results_phase8E2_dofa_crc_r4_10seed.csv`,
`results_phase8R5_secondbench_raw.csv`, `results_phase8R_classwise.csv`).

The seed-fix logit dumps were written to `paper/scenedump_flagship_seedfix/`
(`results_phase8R_scenedump_seedfix.log`), but the scripts that read the flagship dumps
(`phase8R6_alpha_sensitivity.py`, `phase8R7_weighted_conformal.py`, `phase8R9_formal_weighted_crc.py`,
`phase8R10_weighted_synthetic.py`, `phase8R11_weighted_crc_formal.py`, `phase8R_sceneboot_2x2.py`) read
`paper/scenedump_flagship/`. Their seed-fix logs differ from the pre-fix
ones (R9's naive L2A joint risk 29.16 against 29.79), so they read different dumps -- presumably the
seed-fix set under the default name; the rename is not recorded. The dumps themselves are not committed
(`paper/scenedump_*/` is ignored).
Nineteen committed result CSVs in `paper/` (outside the `p100_rerun_*` folders) have no sidecar:
`results_phase8E2_dofa_crc_raw.csv`, `results_phase8E2_dofa_crc_raw_r4_10seed.csv`,
`results_phase8R10_normalization_control_percell.csv`, the R15–R19 `*_percell.csv` (five files),
`results_phase8R2_landcover_raw.csv`, `results_phase8R3_acolite.csv`, `results_phase8R3_acolite10.csv`,
`results_phase8R5_secondbench_raw.csv`, `results_phase8R_classwise.csv`,
`results_phase8R_classwise_10seed.csv`, `results_phase8R_perclass_agg.csv`,
`results_phase8R_perclass_weighting_gpu0.csv`, `results_phase8R_perclass_weighting_gpu1.csv`,
`results_phase8R_weighting_agg.csv` and `supp_scene_components.csv`.
