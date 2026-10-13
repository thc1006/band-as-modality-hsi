#!/usr/bin/env bash
# One-command reproduction, hardware-ADAPTIVE.
#
# The phase scripts auto-detect the machine (GPUs + usable CPU cores) and fan the independent
# per-seed jobs across all GPUs + CPU cores via bandsim.parallel — no flags needed. Tune with:
#   BANDSIM_DEVICE=cpu|cuda|auto   force device      (default: auto -> cuda if present)
#   BANDSIM_WORKERS=N              force concurrent seed workers (1 = serial reference)
#   BANDSIM_GPU_OVERSUB=K          seed workers per GPU when on CUDA (default: size-aware, from the
#                                  smallest GPU's memory and a per-worker VRAM budget; 2 on a 16 GiB P100
#                                  or a 32 GB V100 at the default 8 GB budget)
#   BANDSIM_PER_WORKER_GB=G        that per-worker budget in decimal GB (default 8; some scripts pass a
#                                  larger one for heavy stages, e.g. phase8R2's evaluation uses 11). When
#                                  set it REPLACES those too, so a small value can run them out of memory
#   BANDSIM_THREADS=T              torch intra-op threads per worker (default cores/workers)
# e.g.  BANDSIM_WORKERS=1 bash reproduce.sh     # serial reference run (phase8R/phase8R2 were observed
#                                                bitwise repeatable on CUDA on the 4x P100 box --
#                                                experiments/H10_TRAIN_ONCE_VERIFICATION.md; on the CPU
#                                                only at BANDSIM_THREADS=1, on small synthetic campaigns
#                                                -- bandsim/parallel.py, "Determinism")
#
# Usage: bash reproduce.sh [TIER]      (from repo root; bootstraps .venv via uv if missing)
#   core         HSI experiments on Indian Pines / Pavia / Salinas                     (hours)
#   physics      6S atmosphere + the A+B flagship, on the shipped 6S table             (hours)
#   flagship     real Sentinel-2 CloudSEN12: degradation + reliability                 (many hours)
#   reliability  the paper's other CloudSEN12 analyses: the two deployment axes, controls and
#                decompositions, the logit dumps and what is computed from them offline, DOFA,
#                ACOLITE, the validation scene set                                      (many hours)
#   emit         real EMIT hyperspectral + its reliability study                       (hours)
#   all          the five tiers above, in that order                                    (default)
#   exploratory  post-release studies that are NOT in the v1.1.0 paper (EMIT L1B->L2A shift,
#                HSI 6S shift, AURC); runs only when named
#
# A step whose input files are absent is skipped with a message (the files: ENVIRONMENT_SETUP.md §4).
# Other things can still stop a run under `set -e`: network access on first use (pyspectral's RSR
# store, the DOFA weights, WorldCover tiles for the EMIT studies), phase7 refusing to time on a busy
# machine, and the self-checks noted at their steps. Steps that write result files write them where the
# committed ones are, so `git diff paper/` shows what those re-runs changed -- except the phase8R
# flagship, which keeps its own tag (see that step). Many reliability analyses only print their numbers
# (their committed record is a paper/*.log): the reliability and exploratory tiers keep each step's
# console output in paper/logs/reproduce/<step>.log (gitignored) -- compare those with the committed
# logs; git diff shows nothing for them. Results depend on the machine: the 4x P100 box trained
# different models than the paper's 2x V100 (see the flagship step). The paper's two figures are drawn
# by experiments/make_paper_figs.py from the committed 10-seed CSVs; this script does not redraw them.
#
# The core, physics, flagship and emit tiers took roughly the better part of a day on 2x V100 (an
# estimate, not a recorded timing); reliability adds a few dozen CloudSEN12 runs, most of them over 10
# seeds, and was not timed as a whole. Pick a tier when you only need part of it.
set -e
cd "$(dirname "$0")"
TIER="${1:-all}"
case "$TIER" in core|physics|flagship|reliability|emit|all|exploratory) ;; *)
  echo "unknown tier '$TIER' (want: core|physics|flagship|reliability|emit|all|exploratory)"; exit 2 ;; esac
want(){ { [ "$TIER" = "all" ] && [ "$1" != "exploratory" ]; } || [ "$TIER" = "$1" ]; }
have_all(){ local f; for f in "$@"; do [ -f "$f" ] || return 1; done; }
have_ip(){ have_all data/indian_pines/Indian_pines_corrected.mat data/indian_pines/Indian_pines_gt.mat; }
have_pavia(){ have_all data/pavia/PaviaU.mat data/pavia/PaviaU_gt.mat; }
have_salinas(){ have_all data/salinas/Salinas_corrected.mat data/salinas/Salinas_gt.mat; }
have_s2(){ local s; for s in train test; do
             have_all data/cloudsen12/$s/metadata.csv data/cloudsen12/$s/LABEL_manual_hq.dat data/cloudsen12/$s/L1C_B2.dat || return 1
           done; have_all data/cloudsen12/test/L2A_B2.dat; }
# logged NAME CMD... : run CMD with its console output also kept in paper/logs/reproduce/NAME.log.
# Returns CMD's status, or tee's if CMD succeeded and tee failed; either stops the run under set -e.
logged(){ local name=$1 st; shift; mkdir -p paper/logs/reproduce
          set +e; "$@" 2>&1 | tee "paper/logs/reproduce/$name.log"; st=("${PIPESTATUS[@]}"); set -e
          [ "${st[0]}" -ne 0 ] && return "${st[0]}"; return "${st[1]}"; }

# Bootstrap the LOCKED env with uv if .venv is absent (uv = fast + reproducible; the project
# is uv-managed — see pyproject.toml [tool.uv] and requirements-lock.txt).
if [ ! -x .venv/bin/python ]; then
  echo "[env] .venv not found — bootstrapping with uv from requirements-lock.txt ..."
  command -v uv >/dev/null || { echo "  install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
  uv venv --python 3.12 .venv
  uv pip sync --torch-backend cu124 --python .venv/bin/python requirements-lock.txt \
      --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
fi
source .venv/bin/activate 2>/dev/null || true
# Ensure the host CUDA driver library is used (bypass any broken forward-compat libcuda).
[ -f scripts/gpu_env.sh ] && source scripts/gpu_env.sh

echo "======================================================================"
echo " Band-as-Modality — adaptive reproduce   (tier: $TIER)"
python - <<'PY'
from bandsim import hw, parallel
print(" HW:", hw.info())
print(" fan-out (5 seeds):", parallel.plan_jobs(5).describe())
PY
echo " override via BANDSIM_DEVICE / BANDSIM_WORKERS / BANDSIM_GPU_OVERSUB / BANDSIM_PER_WORKER_GB / BANDSIM_THREADS"
echo "======================================================================"
_t0=$(date +%s)

echo "== unit tests =="
python -m pytest tests/ -q

echo "== integrity harness (every phase runs at smoke size, fresh output, no NaN) =="
python experiments/integrity_check.py

# Commands below reproduce the configurations of the committed results: from their .provenance.json
# sidecars where those describe the committed numbers (the "command" and "args" fields), otherwise from
# the run logs committed beside them, the seeds/splits those logs print and the scripts' own usage
# lines. Execution-only flags (--jobs, --device) are left to the adaptive defaults.
# paper/RESULT_MANIFEST.md maps each paper number to its file and lists the known gaps.

# ---------------------------------------------------------------------------------------------
if want core; then
echo "== Controlled synthetic PoC (multi-seed) + synthetic code-generality check =="
# experiment_synthetic_multiseed.py is the CONTROLLED synthetic mechanism check: a world where the
# ground truth is known by construction, used to show the band-as-modality mechanism does what is
# claimed before moving to real sensors. It had been archived during the repo reorg while this
# script kept calling it, so `set -e` aborted the entire reproduction here, before a single paper
# number was produced. Restored and provenance-stamped 2026-07-20.
#
# phase6 --dataset synthetic is a DIFFERENT thing and not a substitute: its own help calls the data
# FABRICATED and "a code-generality check only (never quote it as a second dataset)". Both are run
# because they answer different questions.
python experiments/experiment_synthetic_multiseed.py --seeds 0 1 2 3 4
python experiments/phase6_second_dataset.py --dataset synthetic --seeds 0 1 2 3 4

if have_ip; then
echo "== Phase 1 (Indian Pines Table 1) =="
# phase1 has no --seeds option (its axes are --split-offsets / --model-seeds); the line here used to
# pass one, argparse rejected it, and set -e stopped the core tier. The committed tables come from
# the defaults.
python experiments/phase1_indian_pines.py

echo "== Phase 2 (degradation curve, B1-B6 + Proposed) [CORE] =="
python experiments/phase2_degradation.py --seeds 0 1 2 3 4 --epochs 60 --trials 12

echo "== Phase 2 cross-sensor (real pyspectral RSR) =="
python experiments/phase2_cross_sensor.py --seeds 0 1 2 3 4 --srf pyspectral

echo "== Phase 2 group-size ablation =="
# Defaults = the committed run (5 seeds, groups 5/10/20); this line used to ask for 2 seeds and so
# replaced the 5-seed deliverable with a different, smaller run.
python experiments/phase2_group_ablation.py

echo "== Phase 2X (PE x dropout 2x2 attribution) =="
# The canonical Indian-Pines/checkerboard run, which reproduces results_phase2X_pe_dropout.csv and
# self-verifies against phase2's raw (exits non-zero if the seeding contract drifts). Then the
# committed Indian Pines replications (spatial-blocks split, sampling baselines); the Pavia and Salinas
# ones and the sensor-adaptive wl_scale probe run with Phase 6 below.
python experiments/phase2X_pe_dropout_2x2.py --seeds 0 1 2 3 4
python experiments/phase2X_pe_dropout_2x2.py --dataset indian_pines --split blocks
python experiments/phase2X_pe_dropout_2x2.py --sampling-baselines --dataset indian_pines --seeds 0 1 2 3 4
python experiments/phase2X_pe_dropout_2x2.py --sampling-baselines --dataset indian_pines --split blocks --seeds 0 1 2 3 4

echo "== Phase 4 (C/D ablation) =="
python experiments/phase4_ablation.py --seeds 0 1 2 3 4

echo "== Phase 9 (E2: does it matter WHICH bands go missing?) [HOOK] =="
if [ -f data/srf_cache/T_6s_grid.npz ]; then
  python experiments/phase9_physics_vs_random.py --seeds 0 1 2 3 4
else
  echo "  (skip: phase9 selects drops by 6S transmittance and needs the table)"
fi

echo "== Phase 4R (reliability / abstention on HSI) [ACE] =="
# Defaults = the committed run (32 trials); this line used to ask for 8.
python experiments/phase4R_reliability.py

# NOT distillation. This phase has no teacher, no student and no KD loss -- B4/B6 are architecture
# ablations. The banner advertised it because the frozen submission listed a distilled student as a
# deployment GOAL (its own caption says "not claimed as achieved results"); nothing here measures it,
# and a reproduction script that names a measurement it does not take is how a goal becomes a result.
echo "== Phase 7 (efficiency: params/MACs/memory/latency, partial dynamic INT8) =="
python experiments/phase7_efficiency.py --seed 0
else
echo "== Phases 1-2X, 4, 4R, 7, 9: skipped (needs data/indian_pines/Indian_pines_{corrected,gt}.mat) =="
fi

echo "== Phase 6 (second/third HSI datasets) + their Phase 2X replications =="
if have_pavia; then
  python experiments/phase6_second_dataset.py --dataset pavia   --seeds 0 1 2 3 4
  python experiments/phase2X_pe_dropout_2x2.py --dataset pavia
  python experiments/phase2X_pe_dropout_2x2.py --dataset pavia --wl-scale auto
else
  echo "  (skip Pavia: needs data/pavia/PaviaU.mat + PaviaU_gt.mat)"
fi
if have_salinas; then
  python experiments/phase6_second_dataset.py --dataset salinas --seeds 0 1 2 3 4
  python experiments/phase2X_pe_dropout_2x2.py --dataset salinas
else
  echo "  (skip Salinas: needs data/salinas/Salinas_corrected.mat + Salinas_gt.mat)"
fi
fi

# ---------------------------------------------------------------------------------------------
if want physics; then
echo "== Phase 3 (atmosphere) + Phase 5 (A+B flagship) =="
if [ ! -f data/srf_cache/T_6s_grid.npz ]; then
  # 6S is built from source with gfortran (ENVIRONMENT_SETUP.md §3), not taken from a conda environment;
  # an earlier version of this script pointed at $HOME/miniforge3/envs/sixs, which does not exist.
  echo "  (skip: the shipped 6S table data/srf_cache/T_6s_grid.npz is missing. Restore it from git, or"
  echo "         build 6SV1.1 from source (ENVIRONMENT_SETUP.md §3) and run"
  echo "         .venv/bin/python experiments/precompute_6s_table.py )"
elif ! have_ip; then
  echo "  (skip: needs data/indian_pines/Indian_pines_{corrected,gt}.mat)"
else
  # Two committed runs, in the order they were made (their sidecars): the tau sweep + random control
  # first (results_phase3_tau_sweep.csv, results_phase3_random_control.csv), then the plain run, which
  # wrote the main table and per-seed rows last. Both runs write the main table, so the order decides
  # which one it ends up holding.
  # --tau-sweep values are pinned, NOT bare: phase3's default grid is [0.3..0.7], but the committed
  # results_phase3_tau_sweep.csv was built on [0.2..0.8] (see scripts/campaign_addenda_20260720.sh).
  # A bare --tau-sweep here would regenerate a DIFFERENT grid that agrees with the deliverable only
  # at the shared tau=0.5, so the exact grid travels with the reproduction command.
  python experiments/phase3_atmosphere.py --seeds 0 1 2 3 4 \
    --tau-sweep 0.2 0.35 0.5 0.65 0.8 --random-control
  python experiments/phase3_atmosphere.py --seeds 0 1 2 3 4
  python experiments/phase5_ab_flagship.py --seeds 0 1 2 3 4
fi
fi

# ---------------------------------------------------------------------------------------------
if want flagship; then
if have_s2; then
echo "== Phase 8 (real Sentinel-2 CloudSEN12 degradation) [FLAGSHIP] =="
# The committed run: 5 seeds over ALL 8490 train patches (--patches-train unset). This line used to ask
# for 7 seeds on 3000 patches, a different configuration written over the committed one.
python experiments/phase8_cloudsen12.py --seeds 0 1 2 3 4

echo "== Phase 8R (conformal risk control on CloudSEN12, ROI-disjoint) [FLAGSHIP] =="
# The paper's flagship is the crossed 10 split-seed x 10 model-seed campaign (paper/RESULT_MANIFEST.md:
# results_phase8R_reliability_10seed.csv -- the seed-fixed 2x V100 re-run committed in c59100a; its
# .provenance.json was not regenerated then and still describes the earlier, pre-fix run). It is
# reproduced under its OWN tag rather than over the committed file, because the numbers depend on the
# machine: on 4x P100 this code gives a headline naive L2A joint risk of 28.707 +/- 0.815 against the
# paper's 27.849 +/- 0.786, every trained model coming out different (clean accuracy differs in all
# 100 cells), while the pre-H-10 per-cell code gives byte-identical CSVs on the
# same P100s -- so the gap is not H-10's: the models are re-drawn on the other machine (same data and
# numeric packages; different GPU, CPU instruction set, driver patch, Python patch release, OS and
# container -- not isolated further; experiments/H10_TRAIN_ONCE_VERIFICATION.md).
# Compare the two files rather than overwrite the paper's.
# This script has no --seeds option -- the line here used to pass one, argparse rejected it, and set -e
# stopped reproduce.sh there.
python experiments/phase8R_reliability.py --split-seeds $(seq 0 9) --model-seeds $(seq 0 9) --out-tag _10seed_repro

echo "== Phase 8D (difficulty stratification) =="
python experiments/phase8D_difficulty.py --seeds 0 1 2

echo "== Phase 8E (DOFA foundation-model baseline) =="
python experiments/phase8E_dofa.py --seeds 0 1 2
else
echo "== flagship: skipped (needs CloudSEN12 under data/cloudsen12/, see ENVIRONMENT_SETUP.md §4) =="
fi
fi

# ---------------------------------------------------------------------------------------------
if want reliability; then
if have_s2; then
echo "== Reliability analyses on CloudSEN12 (beyond the flagship table) =="
# Several of these analyses report their numbers on stdout only; where a record of those was committed
# it is the log next to the results (paper/results_*.log), and the re-run's is
# paper/logs/reproduce/<step>.log. Two committed results have no producer in this repository:
# results_phase8R_classwise_10seed.csv (with paper/results_classwise10.log) and supp_scene_components.csv.

echo "-- the two deployment axes: surface (dark->bright) and geography (north->south), 10 x 10 --"
logged phase8R2_surface   python experiments/phase8R2_landcover_reliability.py --split-seeds $(seq 0 9) --model-seeds $(seq 0 9) --axis surface --out-tag _10seed
logged phase8R2_geography python experiments/phase8R2_landcover_reliability.py --split-seeds $(seq 0 9) --model-seeds $(seq 0 9) --axis geography --out-tag _10seed
logged phase8R10_surface_geo_diff python experiments/phase8R10_surface_geo_diff.py

echo "-- channel-sampling comparison (band-group dropout vs ChannelViT-HCS vs DiChaViT-DCS): two 5-seed blocks, merged --"
# phase8_sampling checks its dropout arm against results_phase8_cloudsen12_curve.csv and exits 1 on a
# mismatch, so on a machine other than the one that made that file, run the flagship tier first.
logged phase8_sampling        python experiments/phase8_sampling.py --seeds 0 1 2 3 4
logged phase8_sampling_s5-9   python experiments/phase8_sampling.py --seeds 5 6 7 8 9 --out-tag _s5-9
logged phase8_sampling_10seed python experiments/phase8_sampling_10seed.py --raw results_phase8_sampling_raw.csv --raw results_phase8_sampling_raw_s5-9.csv

echo "-- flagship logit dumps (10 seeds, paper/scenedump_flagship/) and the analyses run on them --"
logged phase8R_scenedump python experiments/phase8R_scenedump.py --seeds $(seq 0 9) --out scenedump_flagship
logged phase8R_sceneboot_2x2 python experiments/phase8R_sceneboot_2x2.py
logged phase8R6_alpha_sensitivity python experiments/phase8R6_alpha_sensitivity.py
logged phase8R7_weighted_conformal python experiments/phase8R7_weighted_conformal.py
logged phase8R9_formal_weighted_crc python experiments/phase8R9_formal_weighted_crc.py
logged phase8R10_weighted_synthetic python experiments/phase8R10_weighted_synthetic.py
logged phase8R11_weighted_crc_formal python experiments/phase8R11_weighted_crc_formal.py

echo "-- surface-axis logit dumps (10 seeds, paper/scenedump_surface/) and the bootstraps run on them --"
logged phase8R2_surfacedump python experiments/phase8R2_surfacedump.py --seeds $(seq 0 9) --axis surface
logged phase8R2_surface_sceneboot python experiments/phase8R2_surface_sceneboot.py
# B = 600: the committed results_phase8R9_surface_nested_boot_summary.json records it (the default is 1000).
logged phase8R9_surface_nested_boot python experiments/phase8R9_surface_nested_boot.py --B 600

echo "-- controls and decompositions, each its own training campaign --"
logged phase8R10_normalization_control python experiments/phase8R10_normalization_control.py
logged phase8R14_flagship_bandset python experiments/phase8R14_flagship_bandset.py
logged phase8R9_bandset_control python experiments/phase8R9_bandset_control.py
logged phase8R10_unet_spatial python experiments/phase8R10_unet_spatial.py --model-seeds 0 1 2 3 4
logged phase8R15_receptive_field python experiments/phase8R15_receptive_field.py
logged phase8R16_classwise_10x10 python experiments/phase8R16_classwise_10x10.py
logged phase8R17_quantile_generalization python experiments/phase8R17_quantile_generalization.py --seeds $(seq 0 9)
logged phase8R18_quantile_season_baseline python experiments/phase8R18_quantile_season_baseline.py
logged phase8R19_weighted_crc_on_renorm python experiments/phase8R19_weighted_crc_on_renorm.py
logged phase8R12_radiometric_audit python experiments/phase8R12_radiometric_audit.py
logged phase8R13_weighted_representation python experiments/phase8R13_weighted_representation.py
# DOFA factor isolation (per-pixel MLP vs a small CNN -- three 3x3 convolutions, a 7x7 receptive field -- on
# DOFA's nine bands); stdout only, no committed log.
logged phase8R8_spatial_isolation python experiments/phase8R8_spatial_isolation.py

echo "-- class-wise decomposition and per-class weighting --"
logged phase8R_classwise python experiments/phase8R_classwise.py
logged phase8R_perclass_weighting_gpu0 python experiments/phase8R_perclass_weighting.py --seeds 0 1 2 3 4 --out-tag gpu0
logged phase8R_perclass_weighting_gpu1 python experiments/phase8R_perclass_weighting.py --seeds 5 6 7 8 9 --out-tag gpu1
logged phase8R_perclass_weighting_agg python experiments/phase8R_perclass_weighting_agg.py

echo "-- the frozen DOFA foundation model under the flagship protocol, 10 x 10 --"
# 10 x 10 as in the paper and its seed-fix log; the committed results_phase8E2_dofa_crc.csv (and its
# sidecar) is an earlier 10 x 3 run, which this replaces. The DOFA weights are fetched on first use.
logged phase8E2_dofa_crc python experiments/phase8E2_dofa_crc.py --split-seeds $(seq 0 9) --model-seeds $(seq 0 9)
logged phase8E2_dofa_r4  python experiments/phase8E2_dofa_r4.py --split-seeds $(seq 0 9) --model-seeds $(seq 0 9)

echo "-- ACOLITE as a second atmospheric processor --"
aco=(); for b in B1 B2 B3 B4 B5 B6 B7 B8 B8A B11 B12; do aco+=("data/cloudsen12/test/ACOLITE_$b.dat"); done
if have_all "${aco[@]}"; then
  logged phase8R3_acolite python experiments/phase8R3_acolite.py --seeds $(seq 0 9) --out-tag acolite10
  logged phase8R10_acolite_paired_diff python experiments/phase8R10_acolite_paired_diff.py
  logged phase8R9_acolite_clip_sensitivity python experiments/phase8R9_acolite_clip_sensitivity.py --seeds 0 1 2
  logged phase8R9_acolite_retained_bias python experiments/phase8R9_acolite_retained_bias.py
else
  echo "  (skip: needs data/cloudsen12/test/ACOLITE_*.dat -- build them with experiments/acolite_batch.py"
  echo "   (needs ACOLITE + GDAL, see its header), then experiments/acolite_crops_to_dat.py)"
fi

echo "-- the official CloudSEN12 validation split as a second scene set --"
if [ -f data/cloudsen12_val_secondbench.npz ]; then
  logged phase8R5_secondbench python experiments/phase8R5_secondbench.py
  logged phase8R5_valdump python experiments/phase8R5_valdump.py
  logged phase8R5_val_leakfix python experiments/phase8R5_val_leakfix.py
  logged phase8R5_val_classwise python experiments/phase8R5_val_classwise.py
  logged phase8R5_val_analysis python experiments/phase8R5_val_analysis.py
else
  echo "  (skip: needs data/cloudsen12_val_secondbench.npz. experiments/phase8R5_val_loader.py builds it from"
  echo "   the tacofoundation CloudSEN12 release -- a later revision than the train/test files; its header"
  echo "   says it needs tacoreader<1.0 + rasterio, i.e. a separate venv, not this one)"
fi
else
echo "== reliability: skipped (needs CloudSEN12 under data/cloudsen12/, see ENVIRONMENT_SETUP.md §4) =="
fi
fi

# ---------------------------------------------------------------------------------------------
if want emit; then
echo "== Phase 8F (real EMIT hyperspectral, multi-region) =="
if compgen -G "data/emit*/*RFL*.nc" >/dev/null; then
  python scripts/emit/preflight_emit.py
  # --require-region drops a granule that has no declared region box. The committed CSVs were made
  # without it; every one of the 15 regions has a box (checked 2026-10-13), so it drops nothing there and
  # only guards a granule added later. The committed run analysed 14 granules (the 'sumatra' slot then
  # held a granule with no pixel in its box); a re-run with today's granule set may analyse 15.
  python experiments/phase8F_multi.py --seeds 0 1 2 --require-region
  # phase8F_multi draws no figure; this separate script does (paper/figs/fig_emit_ndvi_trend.{png,pdf},
  # not committed).
  python experiments/plot_phase8F_multi.py
else
  echo "  (skip: no EMIT granules under data/emit*/ — see scripts/emit/dl_emit_ndvi.py)"
fi
echo "== Phase 8G (EMIT per-band uncertainty as a reliability proxy) =="
if [ -f data/emit/emit_sample.npz ]; then
  python experiments/phase8G_emit_reliability.py --seeds 0 1 2
else
  echo "  (skip: needs data/emit/emit_sample.npz, a pixel extract of one EMIT granule that no script in this"
  echo "   repository builds -- ENVIRONMENT_SETUP.md §4)"
fi
fi

# ---------------------------------------------------------------------------------------------
# Post-release studies (after the v1.1.0 release), not part of the paper: experiments/
# MODALITY_GENERALITY_NOTES.md and EMIT_CROSS_SENSOR_NOTES.md. Their JSON results carry no provenance
# stamp, so these commands are reconstructed from the notes, from the output names (each variant flag
# adds a suffix) and from the number of seeds each committed file holds.
if want exploratory; then
echo "== Exploratory: HSI 6S dry->humid shift (uniform control, spatial field) + AURC =="
if have_ip; then
  logged phase8H_indian_pines_mlp python experiments/phase8H_hsi6s_shift.py --dataset indian_pines --model mlp
  logged phase8H_indian_pines_band python experiments/phase8H_hsi6s_shift.py --dataset indian_pines --model band
  logged phase8H_indian_pines_spatial_mlp python experiments/phase8H_hsi6s_shift.py --dataset indian_pines --spatial-cwv --model mlp
  logged phase8H_indian_pines_spatial_band python experiments/phase8H_hsi6s_shift.py --dataset indian_pines --spatial-cwv --model band
  logged phase8H_cwv05_20 python experiments/phase8H_hsi6s_shift.py --dataset indian_pines --src-cwv cwv0.5 --tgt-cwv cwv2.0 --out paper/results_phase8H_hsi6s_cwv05_20
  logged phase8H_cwv20_40 python experiments/phase8H_hsi6s_shift.py --dataset indian_pines --src-cwv cwv2.0 --tgt-cwv cwv4.0 --out paper/results_phase8H_hsi6s_cwv20_40
  # The committed results_aurc_matched_coverage.json predates a later change to the script (ec6a83c), so a
  # re-run differs from it even on the same machine.
  logged aurc_matched_coverage python experiments/aurc_matched_coverage.py
else
  echo "  (skip Indian Pines: needs data/indian_pines/)"
fi
if have_salinas; then
  logged phase8H_salinas_mlp python experiments/phase8H_hsi6s_shift.py --dataset salinas --model mlp
  logged phase8H_salinas_band python experiments/phase8H_hsi6s_shift.py --dataset salinas --model band
  logged phase8H_salinas_spatial_mlp python experiments/phase8H_hsi6s_shift.py --dataset salinas --spatial-cwv --model mlp
  logged phase8H_salinas_spatial_band python experiments/phase8H_hsi6s_shift.py --dataset salinas --spatial-cwv --model band
  logged aurc_salinas python experiments/aurc_matched_coverage.py --dataset salinas --seeds 0 1 2 3 4
  logged aurc_salinas_spatial python experiments/aurc_matched_coverage.py --dataset salinas --spatial-cwv --seeds 0 1 2 3 4
else
  echo "  (skip Salinas: needs data/salinas/)"
fi
if have_pavia; then
  logged phase8H_pavia_mlp python experiments/phase8H_hsi6s_shift.py --dataset pavia --model mlp
  logged phase8H_pavia_band python experiments/phase8H_hsi6s_shift.py --dataset pavia --model band
  logged phase8H_pavia_spatial_band python experiments/phase8H_hsi6s_shift.py --dataset pavia --spatial-cwv --model band
  logged aurc_pavia python experiments/aurc_matched_coverage.py --dataset pavia --seeds 0 1 2 3 4
  logged aurc_pavia_spatial python experiments/aurc_matched_coverage.py --dataset pavia --spatial-cwv --seeds 0 1 2 3 4
else
  echo "  (skip Pavia: needs data/pavia/)"
fi

echo "== Exploratory: EMIT L1B radiance -> L2A reflectance (a real processing shift on a second sensor) =="
if compgen -G "data/emit*/*L1B_RAD*.nc" >/dev/null; then
  logged phase8G_emit_io python experiments/phase8G_emit_io.py --build-cache
  logged phase8G_emit_shift python experiments/phase8G_emit_shift.py --seeds 0 1 2 3 4
  logged phase8G_emit_shift_10seed python experiments/phase8G_emit_shift.py --seeds $(seq 0 9) --out paper/results_phase8G_emit_shift_10seed
  logged phase8G_emit_shift_band python experiments/phase8G_emit_shift.py --seeds $(seq 0 9) --model band --out paper/results_phase8G_emit_shift_band
  logged phase8G_emit_shift_band_bs2048 python experiments/phase8G_emit_shift.py --seeds $(seq 0 9) --model band --bs 2048 --epochs 300 --out paper/results_phase8G_emit_shift_band_bs2048
  logged phase8G_emit_geo python experiments/phase8G_emit_geo.py --seeds $(seq 0 9)
  logged aurc_emit python experiments/aurc_emit.py
else
  echo "  (skip: needs the EMIT L1B radiance granules -- .venv/bin/python experiments/emit_fetch_l1b.py --go"
  echo "   downloads them with their OBS files (~30 GB, an Earthdata login in ~/.netrc))"
fi

echo "== Exploratory: AURC on CloudSEN12 from the rich logit bank =="
if have_s2; then
  bank=paper/scenedump_rich
  bank_files=("$bank/inputs_test.npz" "$bank/norm_stats.npz")
  for s in $(seq 0 9); do bank_files+=("$bank/logits_seed$s.npz" "$bank/model_seed$s.pt"); done
  if [ ! -e "$bank" ]; then
    logged phase8R_scenedump_rich python experiments/phase8R_scenedump_rich.py --seeds $(seq 0 9) --out scenedump_rich
  fi
  if have_all "${bank_files[@]}"; then
    # An existing bank is never rebuilt: the one in a development checkout is the 2x V100 bank the
    # restored CloudSEN12 files were verified against.
    logged aurc_cloudsen12_core python experiments/aurc_cloudsen12_core.py
    logged aurc_scene_component python experiments/aurc_scene_component.py
  else
    echo "  (skip: $bank/ exists but is incomplete -- not rebuilt over, not analysed; move it aside to rebuild)"
  fi
else
  echo "  (skip: needs CloudSEN12 under data/cloudsen12/)"
fi
fi

# make_paper_tables.py builds the baselines table from paper/results_phase2_curve.csv, which only
# the `core` tier produces. Running it after `flagship` or `emit` would regenerate that table from
# whatever phase2 CSV happened to be on disk, stamping a fresh timestamp on stale numbers.
if want core && have_ip; then
echo "== paper tables (the earlier phase-2 baselines table; not part of the v1.1.0 paper) =="
python experiments/make_paper_tables.py
else
echo "== paper tables: skipped (needs the core tier's results_phase2_curve.csv) =="
fi

_t1=$(date +%s)
echo "== DONE ($TIER) in $((_t1-_t0))s. Results in paper/, tables in paper/tables/ =="
echo "   Most scripts write a *.provenance.json next to each result (git commit, dirty state, args,"
echo "   library versions); paper/RESULT_MANIFEST.md lists the committed result CSVs that have none."
