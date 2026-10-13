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
#                                                bitwise repeatable on CUDA on the 4x P100 box; on the
#                                                CPU only at BANDSIM_THREADS=1, on small synthetic
#                                                campaigns -- bandsim/parallel.py, "Determinism")
#
# Usage: bash reproduce.sh [TIER]      (from repo root; bootstraps .venv via uv if missing)
#   core      HSI experiments on Indian Pines / Pavia / Salinas          (hours)
#   physics   6S atmosphere + the A+B flagship, needs the 6S table       (hours)
#   flagship  real Sentinel-2 CloudSEN12: degradation + reliability      (many hours)
#   emit      real EMIT hyperspectral + its reliability study            (hours)
#   all       every tier, in the order above                             (default)
#
# Running EVERY tier serially takes the better part of a day on 2x V100. That is a property of the
# work, not of this script: the flagship alone trains ~6 models per seed over ~2.5M pixels. Pick a
# tier when you only need part of it.
set -e
cd "$(dirname "$0")"
TIER="${1:-all}"
case "$TIER" in core|physics|flagship|emit|all) ;; *)
  echo "unknown tier '$TIER' (want: core|physics|flagship|emit|all)"; exit 2 ;; esac
want(){ [ "$TIER" = "all" ] || [ "$TIER" = "$1" ]; }

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

echo "== Phase 1 (Indian Pines Table 1) =="
python experiments/phase1_indian_pines.py --seeds 0 1 2 3 4

echo "== Phase 2 (degradation curve, B1-B6 + Proposed) [CORE] =="
python experiments/phase2_degradation.py --seeds 0 1 2 3 4 --epochs 60 --trials 12

echo "== Phase 2 cross-sensor (real pyspectral RSR) =="
python experiments/phase2_cross_sensor.py --seeds 0 1 2 3 4 --srf pyspectral

echo "== Phase 2 group-size ablation =="
python experiments/phase2_group_ablation.py --seeds 0 1 --groups 5 10 20

echo "== Phase 2X (PE x dropout 2x2 attribution) =="
# The canonical Indian-Pines/checkerboard run, which reproduces results_phase2X_pe_dropout.csv and
# self-verifies against phase2's raw (exits non-zero if the seeding contract drifts). The Pavia and
# Salinas replications and the sensor-adaptive-wl_scale probe are separate --dataset/--wl-scale
# invocations documented in the script header; the default run is the one the paper's decomposition
# table is built from.
python experiments/phase2X_pe_dropout_2x2.py --seeds 0 1 2 3 4

echo "== Phase 4 (C/D ablation) =="
python experiments/phase4_ablation.py --seeds 0 1 2 3 4

echo "== Phase 9 (E2: does it matter WHICH bands go missing?) [HOOK] =="
if [ -f data/srf_cache/T_6s_grid.npz ]; then
  python experiments/phase9_physics_vs_random.py --seeds 0 1 2 3 4
else
  echo "  (skip: phase9 selects drops by 6S transmittance and needs the table)"
fi

echo "== Phase 4R (reliability / abstention on HSI) [ACE] =="
python experiments/phase4R_reliability.py --seeds 0 1 2 3 4 --epochs 60 --trials 8

echo "== Phase 6 (second/third HSI datasets) =="
python experiments/phase6_second_dataset.py --dataset pavia   --seeds 0 1 2 3 4
python experiments/phase6_second_dataset.py --dataset salinas --seeds 0 1 2 3 4

# NOT distillation. This phase has no teacher, no student and no KD loss -- B4/B6 are architecture
# ablations. The banner advertised it because the frozen submission listed a distilled student as a
# deployment GOAL (its own caption says "not claimed as achieved results"); nothing here measures it,
# and a reproduction script that names a measurement it does not take is how a goal becomes a result.
echo "== Phase 7 (efficiency: params/MACs/memory/latency, partial dynamic INT8) =="
python experiments/phase7_efficiency.py --seed 0
fi

# ---------------------------------------------------------------------------------------------
if want physics; then
echo "== Phase 3 (atmosphere) + Phase 5 (A+B flagship) =="
if [ -f data/srf_cache/T_6s_grid.npz ]; then
  # --tau-sweep values are pinned, NOT bare: phase3's default grid is [0.3..0.7], but the committed
  # results_phase3_tau_sweep.csv was built on [0.2..0.8] (see scripts/campaign_addenda_20260720.sh).
  # A bare --tau-sweep here would regenerate a DIFFERENT grid that agrees with the deliverable only
  # at the shared tau=0.5, so the exact grid travels with the reproduction command.
  python experiments/phase3_atmosphere.py --seeds 0 1 2 3 4 \
    --tau-sweep 0.2 0.35 0.5 0.65 0.8 --random-control
  python experiments/phase5_ab_flagship.py --seeds 0 1 2 3 4
else
  # 6S is built from SOURCE here with gfortran — there is no conda environment. The previous
  # version of this script pointed at $HOME/miniforge3/envs/sixs, which does not exist.
  echo "  (skip: no 6S table. Build 6SV1.1 from source with gfortran, then:"
  echo "         .venv/bin/python experiments/precompute_6s_table.py )"
fi
fi

# ---------------------------------------------------------------------------------------------
if want flagship; then
echo "== Phase 8 (real Sentinel-2 CloudSEN12 degradation) [FLAGSHIP] =="
python experiments/phase8_cloudsen12.py --seeds 0 1 2 3 4 5 6 --patches-train 3000 --epochs 40

echo "== Phase 8R (conformal risk control on CloudSEN12, ROI-disjoint) [FLAGSHIP] =="
# The paper's flagship is the crossed 10 split-seed x 10 model-seed campaign (paper/RESULT_MANIFEST.md:
# results_phase8R_reliability_10seed.csv -- the seed-fixed 2x V100 re-run committed in c59100a; its
# .provenance.json was not regenerated then and still describes the earlier, pre-fix run). It is
# reproduced under its OWN tag rather than over the committed file, because the numbers depend on the
# machine: on 4x P100 this code gives a headline naive L2A joint risk of 28.707 +/- 0.815 against the
# paper's 27.849 +/- 0.786, every one of the 100 cells' trained models coming out different (clean
# accuracy differs in all of them), while the pre-H-10 per-cell code gives byte-identical CSVs on the
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
fi

# ---------------------------------------------------------------------------------------------
if want emit; then
echo "== Phase 8F (real EMIT hyperspectral, multi-region) =="
if compgen -G "data/emit*/*RFL*.nc" >/dev/null; then
  python scripts/emit/preflight_emit.py
  python experiments/phase8F_multi.py --seeds 0 1 2 --require-region
  # phase8F_multi has ZERO savefig calls -- its plotting is deliberately a separate script, which
  # was never wired in here, so paper/figs/ has no EMIT figure at all despite the CSVs existing.
  python experiments/plot_phase8F_multi.py
  echo "== Phase 8G (EMIT per-band uncertainty as reliability ground truth) =="
  python experiments/phase8G_emit_reliability.py --seeds 0 1 2
else
  echo "  (skip: no EMIT granules under data/emit*/ — see scripts/emit/dl_emit_ndvi.py)"
fi
fi

# make_paper_tables.py builds the baselines table from paper/results_phase2_curve.csv, which only
# the `core` tier produces. Running it after `flagship` or `emit` would regenerate that table from
# whatever phase2 CSV happened to be on disk, stamping a fresh timestamp on stale numbers.
if want core; then
echo "== paper tables =="
python experiments/make_paper_tables.py
else
echo "== paper tables: skipped (needs the core tier's results_phase2_curve.csv) =="
fi

_t1=$(date +%s)
echo "== DONE ($TIER) in $((_t1-_t0))s. Figures in paper/figs/, tables in paper/tables/ =="
echo "   Each results CSV is written with a *.provenance.json recording the git commit, dirty"
echo "   state, args and library versions that produced it. A result without one is not citable."
