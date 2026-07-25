#!/usr/bin/env bash
# Run ONE campaign phase with the same bookkeeping as campaign_20260720.sh's run().
#
# Why this exists: the driver serialises lane C, but 8D/8R/8E/8F/4R read none of phase8's
# artefacts -- they each train their own models -- so they can run alongside it on the idle
# cores. Launching them by hand is unsafe though: the driver reaches `run C phase8E` later and,
# finding no .ok, starts a SECOND phase8E writing the same CSVs concurrently. Touching the .ok
# by hand afterwards is exactly the step a human (or an agent that loses context) forgets, and
# the failure is silent file corruption. So the .ok is written HERE, by the same rule the driver
# uses: only on rc=0.
#
# usage: scripts/campaign_run_one.sh <lane> <name> <gpus> <workers> <cmd...>
set -u
cd "$(dirname "$0")/.."
lane="$1"; name="$2"; gpus="$3"; workers="$4"; shift 4
STAMP=paper/campaign_20260720
log="$STAMP/${lane}_${name}.log"
if [ -f "$STAMP/${lane}_${name}.ok" ]; then
  echo "[skip] $lane/$name (already ok)" | tee -a "$STAMP/state.log"; exit 0
fi
echo "[start] $lane/$name gpus=$gpus $(date -u +%T) (parallel, out-of-band)" | tee -a "$STAMP/state.log"
CUDA_VISIBLE_DEVICES="$gpus" BANDSIM_WORKERS="$workers" PYTHONUNBUFFERED=1 \
  .venv/bin/python "$@" >"$log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  touch "$STAMP/${lane}_${name}.ok"
  echo "[ok]    $lane/$name $(date -u +%T) (parallel)" | tee -a "$STAMP/state.log"
else
  echo "[FAIL]  $lane/$name rc=$rc $(date -u +%T) (log: $log)" | tee -a "$STAMP/state.log"
fi
exit $rc
