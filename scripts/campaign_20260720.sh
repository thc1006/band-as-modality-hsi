#!/usr/bin/env bash
# Full re-run campaign on the merged 2026-07-20 baseline (main >= 5ad762d, corrected axis).
# 2x V100-32GB + 8 cgroup cores. Lanes A/B run concurrently on separate GPUs; C uses both;
# D (phase7 timing) runs EXCLUSIVE at the end. SIGTERM only -- never KILL a GPU process.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STAMP=paper/campaign_20260720
mkdir -p "$STAMP"
echo "campaign start $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD)" | tee "$STAMP/state.log"

run() {  # run <lane> <name> <gpu-list> <workers> <cmd...>
  local lane="$1" name="$2" gpus="$3" workers="$4"; shift 4
  local log="$STAMP/${lane}_${name}.log"
  if [ -f "$STAMP/${lane}_${name}.ok" ]; then
    echo "[skip] $lane/$name (already ok)" | tee -a "$STAMP/state.log"; return 0
  fi
  echo "[start] $lane/$name gpus=$gpus $(date -u +%T)" | tee -a "$STAMP/state.log"
  CUDA_VISIBLE_DEVICES="$gpus" BANDSIM_WORKERS="$workers" \
    "$PY" "$@" >"$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$STAMP/${lane}_${name}.ok"
    echo "[ok]    $lane/$name $(date -u +%T)" | tee -a "$STAMP/state.log"
  else
    echo "[FAIL]  $lane/$name rc=$rc $(date -u +%T) (log: $log)" | tee -a "$STAMP/state.log"
  fi
  return 0   # independent phases continue; dependencies are encoded by lane order
}

lane_A() {  # GPU0
  run A phase1      0 4 experiments/phase1_indian_pines.py
  run A phase2      0 4 experiments/phase2_degradation.py --seeds 0 1 2 3 4
  run A phase4      0 4 experiments/phase4_ablation.py --seeds 0 1 2 3 4
  run A phase9      0 4 experiments/phase9_physics_vs_random.py
  run A phase8G     0 2 experiments/phase8G_emit_reliability.py --seeds 0 1 2 --device cuda
}

lane_B() {  # GPU1
  run B phase6_syn  1 4 experiments/phase6_second_dataset.py --dataset synthetic
  run B phase6_pav  1 4 experiments/phase6_second_dataset.py --dataset pavia
  run B phase6_sal  1 4 experiments/phase6_second_dataset.py --dataset salinas
  run B phase2cs    1 4 experiments/phase2_cross_sensor.py
  run B phase2gabl  1 4 experiments/phase2_group_ablation.py
  run B synth_multi 1 4 experiments/experiment_synthetic_multiseed.py
  run B phase3      1 4 experiments/phase3_atmosphere.py --seeds 0 1 2 3 4
  run B phase5      1 4 experiments/phase5_ab_flagship.py --seeds 0 1 2 3 4
}

lane_A & PID_A=$!
lane_B & PID_B=$!
trap 'kill -TERM $PID_A $PID_B 2>/dev/null; wait; exit 143' TERM INT
wait $PID_A $PID_B
echo "[lanes A+B done] $(date -u +%T)" | tee -a "$STAMP/state.log"

# LANE C -- CloudSEN12 + EMIT + 4R. Both GPUs; measured: --jobs 2 beats --jobs 7 (latency-bound).
run C phase8      0,1 2 experiments/phase8_cloudsen12.py --seeds 0 1 2 3 4 --jobs 2
run C phase8D     0,1 2 experiments/phase8D_difficulty.py
run C phase8R     0,1 2 experiments/phase8R_reliability.py --jobs 2
run C phase8E     0,1 2 experiments/phase8E_dofa.py
run C phase8Fm    0,1 2 experiments/phase8F_multi.py
run C phase4R     0,1 2 experiments/phase4R_reliability.py

# LANE D -- efficiency timing, EXCLUSIVE (nothing else may run)
run D phase7      0,1 8 experiments/phase7_efficiency.py

# FINISH -- tables + doctor + summary
run F tables      "" 1 experiments/make_paper_tables.py
"$PY" scripts/doctor.py > "$STAMP/doctor_final.log" 2>&1
doctor_rc=$?
echo "doctor exit=$doctor_rc (see $STAMP/doctor_final.log)" | tee -a "$STAMP/state.log"
OK=$(ls "$STAMP"/*.ok 2>/dev/null | wc -l); FAILS=$(grep -c '^\[FAIL\]' "$STAMP/state.log")
echo "campaign end $(date -u +%FT%TZ): $OK phases ok, $FAILS failed" | tee -a "$STAMP/state.log"

# Propagate failure. run() intentionally returns 0 so independent phases continue, but the CAMPAIGN
# as a whole must NOT exit 0 when a phase failed or doctor is unhappy -- otherwise a CI or an
# operator reads "campaign finished" over missing/stale outputs. Any [FAIL] or a non-zero doctor is
# a non-zero campaign.
if [ "$FAILS" -gt 0 ] || [ "$doctor_rc" -ne 0 ]; then
  echo "campaign FAILED: $FAILS phase failure(s), doctor_rc=$doctor_rc" | tee -a "$STAMP/state.log"
  exit 1
fi
echo "campaign OK: all phases succeeded and doctor is clean" | tee -a "$STAMP/state.log"
