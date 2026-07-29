# Result manifest — maps each paper number to its script / log / CSV / seed

Release **v1.1.0** (git tag `v1.1.0`). All experiments seed the model **before** the constructor
(`hw.seed_model(seed)`, seed+101); crossed 10 calibration-split x 10 model-seed design unless noted.

Reproduction scope: the committed CSV/JSON files below **are** the end-to-end numeric outputs of the
listed scripts (re-run a script to regenerate its file). `experiments/verify_round7.py` is a
**function/regression** suite (18 checks over the round-7 code paths and their guards) — it validates the
building blocks, not a full re-execution of every 10x10 campaign; the campaign numbers are reproduced by
running the scripts themselves.

| Paper item | Value | Script | Output |
|---|---|---|---|
| Flagship naive L2A joint (Tab. flagship, abstract) | 27.85±0.79 | `experiments/phase8R_reliability.py` | `paper/results_phase8R_reliability_10seed.csv` |
| Band-drop table | 8.53 / 8.22 / 8.56 / 27.85 | `phase8R_reliability.py` | same CSV (states clean/dropB10/dropB1B9B10/L2A_real) |
| Temp×threshold factorial | 28.3/8.5/29.7/8.6 | `phase8R_sceneboot_2x2.py` | `paper/results_sceneboot2x2_seedfix.log` |
| Pixel-pooled sanity | 28.3 | `phase8R_sceneboot_2x2.py` | same log |
| Alpha 5–20% sweep | 24.0/28.3/31.1/32.5 | `phase8R6_alpha_sensitivity.py` | `paper/results_phase8R6_alpha_seedfix.log` |
| Weighted heuristic | 32.2, 32.6 | `phase8R7_weighted_conformal.py`, `phase8R9_formal_weighted_crc.py` | `paper/results_phase8R7_seedfix.log`, `..._R9formal_seedfix.log` |
| Formal weighted CRC (test-point term) | real L2A 1.87@7% (breach unfixed); est-w PC 7.1 tracks true 6.2 @ AUROC 0.88 overlap | `phase8R11_weighted_crc_formal.py` | `paper/results_phase8R11_estweight.log` |
| Representation AUROC (raw/product) | 1.00 / 0.56 | `phase8R13_weighted_representation.py` | `paper/results_phase8R13_weighted_representation_groupcv.log` |
| Radiometric audit | L1C p1<1000 | `phase8R12_radiometric_audit.py` | (stdout) |
| Normalization control (disjoint / pooled) | 10.8±0.3 / 10.5±0.3 (from 28.4) | `phase8R10_normalization_control.py` | `paper/results_phase8R10_normalization_control_{summary.json,percell.csv,10seed.log}` |
| Normalization decomposition | product +17.9 / composition +0.0 / TTA +1.0 | same | same |
| Quantile-transport generalization (3.2) | full-calib 9.55 (anchor≈headline 9.6); sample-size n8/16/24/40 = 13.0/11.2/10.4/9.8; random halves 10.1/9.9 (agree); bright/dark calib 12.5/14.4 (surface-dependent); cov ~70% all arms (df=9) | `phase8R17_quantile_generalization.py` (10 model × 10 split seed) | `paper/results_phase8R17_quantile_generalization_{summary.json,percell.csv,log}` |
| Quantile transfer: season + Sen2Cor baseline | full 9.55; warm/cold calib 10.4/9.8 (cold matches full, warm +0.9); new/old Sen2Cor baseline 9.9/10.8 (older +1.2 above full); season+baseline effects ≪ surface (df=9) | `phase8R18_quantile_season_baseline.py` (10 model × 10 split seed) | `paper/results_phase8R18_quantile_season_baseline_{summary.json,percell.csv,log}` |
| Weighted CRC on re-normalized rep (overlap diag) | domain AUROC 0.988→0.606, calib ESS 41→75% over 10×10 (re-norm restores the overlap source-only reweighting lacked). NOTE: the formal weighted-CRC coverage arms abstain (0@0%) on the self-trained model — the flagship dumps that phase8R11 uses lack the input reflectance needed to re-normalize, so a matched useful-coverage demo is future work; only the AUROC/ESS diagnostic is cited in the paper | `phase8R19_weighted_crc_on_renorm.py` (10 model × 10 split seed) | `paper/results_phase8R19_weighted_crc_on_renorm_{summary.json,percell.csv,log}` |
| ACOLITE paired diff | 29.6 vs 29.34, +0.26 [−1.2,+1.7] TOST pass | `phase8R3_acolite.py`+`phase8R10_acolite_paired_diff.py` | `paper/results_phase8R3_acolite10.csv` |
| Validation replication | 28.0±0.7 | `phase8R5_secondbench.py` | `paper/results_phase8R5_secondbench_seedfix.log` |
| Validation per-class | IoU 54.4→30.0 | `phase8R5_valdump.py`+`phase8R5_val_classwise.py` | `paper/results_val_classwise_seedfix.log` |
| Class-wise decomposition (Tab. classwise) | full 10×10: clean 8.43 / L2A 28.25 joint | `phase8R16_classwise_10x10.py` (10 split × 10 model seed) | `paper/results_phase8R16_classwise_10x10_{summary.json,percell.csv,log}` |
| DOFA (atmos / band-drop) | 9.95 / 11.77 | `phase8E2_dofa_crc.py` | `paper/results_phase8E2_dofa_crc.csv` |
| Band-set MLP 13/−B9/9 | 48.4/29.2/24.1 | `phase8R9_bandset_control.py` | `paper/results_phase8R9_bandset_seedfix.log` |
| Flagship band-set 13/−B9/9 | 28.3/22.6/21.1 | `phase8R14_flagship_bandset.py` | (stdout) |
| Spatial U-Net vs per-pixel | 14.4 / 23.6 (restore 8.9 / 8.8) | `phase8R10_unet_spatial.py` | `paper/results_phase8R10_unet_seedfix.log` |
| Receptive-field isolation (capacity-matched, 7 arm) | breach k1 20.4→k7 17.4; capacity-matched k1\_w5−k5 +2.11 [1.79,2.42], k1\_w7−k7 +3.42 [1.34,5.50] exclude 0 (RF 17/25px); k1\_w3−k3 +0.74 incl 0; capacity-only all incl 0 (two-way SE df=4) | `phase8R15_receptive_field.py` (5 model × 10 split seed) | `paper/results_phase8R15_receptive_field_{summary.json,percell.csv,log}` |
| Surface / geography / diff | 11.34 / 9.71 / +1.62 | `phase8R2_landcover_reliability.py`+`phase8R10_surface_geo_diff.py` | `paper/results_phase8R2_{landcover,geography}_raw_10seed.csv` |
| ACOLITE retained-subset bias | retained 181 comps / 590 patches; effect sizes negligible–small (conditional on successful processing) | `phase8R9_acolite_retained_bias.py` | `paper/results_phase8R9_acolite_retained_bias.log` |
| Surface nested bootstrap (seed-averaged estimand) | θ̂ 11.47, mean 11.7, 95% CI [9.5,14.2], feasible 100%, includes 10 → inconclusive | `phase8R9_surface_nested_boot.py` | `paper/results_phase8R9_surface_nested_boot_summary.json` |
| Figures fig2/fig3 | regenerated seed-fixed | `make_paper_figs.py` | `paper/figs/fig2_flagship_reliability.pdf`, `fig3_domain_gaps.pdf` |
