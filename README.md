# Aggregate Accuracy Does Not Reveal Reliability

Code and manuscript for:

> **Aggregate Accuracy Does Not Reveal Reliability: Silent Conformal Certificate Failure under Operational Spectral and Domain Shift in Sentinel-2 Cloud Segmentation**
> Hsiu-Chi Tsai, Chia-Tung Chung.

This repository measures how a conformal *certificate* — a split conformal risk-control (CRC) operating point — fails **silently** under the real Sentinel-2 L1C→L2A atmospheric-correction shift on CloudSEN12 cloud segmentation: the calibration statistic the procedure minimises stays below target while the deployed confidently-wrong risk runs far above it. It characterises *when* the failure occurs (conditional on the kind of shift), shows it is model-class-dependent, and quantifies the honestly-costed remedy (per-stratum / Mondrian recalibration).

It also ships `bandsim`, a small reusable library for physically-grounded missing-band robustness and conformal-reliability evaluation.

## Repository layout
- `bandsim/` — reusable library: conformal risk control, temperature scaling, two-way cluster-robust standard error, physical band grouping, and the band-as-modality model.
- `experiments/` — experiment scripts. The reliability study is `phase8R*`; the round-5 review controls are `phase8R9_*` (formal covariate-shift weighted conformal, nested surface bootstrap, band-set / B9 control, ACOLITE retained-subset bias, ACOLITE clip sensitivity) and `phase8R5_val_leakfix.py` (validation product-leak audit).
- `paper/latex/` — manuscript source (`main.tex`), bibliography, and figures.
- `paper/results_*.csv`, `paper/results_*.log`, `paper/*.provenance.json` — committed result tables, experiment logs, and provenance for every phase.
- `configs/`, `scripts/`, `tests/` — configuration, data-staging scripts, and tests.

## Key results (proposed small per-pixel model; CloudSEN12; target risk α = 10 %)
- Under the operational L1C→L2A shift, a clean-calibrated conformal threshold's confidently-wrong (joint) risk rises **8.5 % → 28.9 %** while its own calibration statistic stays ≤ α — a *silent* breach; pixel accuracy falls only 12 points and hides a per-class collapse.
- The breach is not an artefact of one processor, scene set, or operating point: it reproduces under a second, algorithmically different atmospheric processor (ACOLITE, 28.6 %), on the official held-out validation scenes (29.1 %), and across target risks 5–20 %; a covariate-shift weighted-conformal threshold does **not** repair it (32.7 %).
- Per-stratum (Mondrian) recalibration restores the target (8.6 %) but by abstaining (coverage 93 % → 45 %) — a trade we report rather than hide.
- The crisis is conditional and model-dependent: a channel-adaptive foundation model (DOFA) run through the identical protocol is far more reliable under the same shift (naive joint risk 9.9 %).

## Reproducing
- Environment: see `ENVIRONMENT_SETUP.md` and `requirements-lock.txt` (uv-managed; runs on 2×V100).
- Data: CloudSEN12 is public (see the manuscript for the exact tier and splits). The ~145 GB of raw Sentinel-2 / CloudSEN12 / ACOLITE data and the intermediate per-scene logit dumps are **not** shipped here; the result tables, logs, provenance, and the code that produces them are. The round-5 controls (`phase8R9_*`) run offline from the per-scene dumps once regenerated.
- Data-staging scripts read a repo/data root from the `BANDSIM_REPO` / `BANDSIM_DATA` environment variables (defaulting to the checkout).

## License
See `LICENSE`, `LICENSES/`, and `REUSE.toml` (code under MIT; manuscript text and figures under CC-BY-4.0 unless a file states otherwise).

## Citation
```bibtex
@article{tsai_chung_reliability,
  title   = {Aggregate Accuracy Does Not Reveal Reliability: Silent Conformal
             Certificate Failure under Operational Spectral and Domain Shift in
             Sentinel-2 Cloud Segmentation},
  author  = {Tsai, Hsiu-Chi and Chung, Chia-Tung},
  year    = {2026}
}
```
