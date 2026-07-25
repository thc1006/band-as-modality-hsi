#!/usr/bin/env bash
# Post-campaign addenda: the two never-run modes shared_layer flagged. Run AFTER the main driver.
set -u; cd "$(dirname "$0")/.."
S=paper/campaign_20260720
# Pick the LESS LOADED GPU at launch instead of hardcoding 0. This host is shared: a foreign
# tenant (PIDs outside our namespace) intermittently saturates a card, and pinning to it both
# starves this job and steals from whatever critical-path worker is already there.
GPU=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "addenda: least-loaded GPU is $GPU" >> "$S/state.log"
CUDA_VISIBLE_DEVICES=$GPU BANDSIM_WORKERS=4 .venv/bin/python experiments/phase3_atmosphere.py \
  --seeds 0 1 2 3 4 --tau-sweep 0.2 0.35 0.5 0.65 0.8 --random-control \
  > "$S/X_phase3_tausweep.log" 2>&1 && touch "$S/X_phase3_tausweep.ok" \
  || echo "[FAIL] X_phase3_tausweep" >> "$S/state.log"
echo "[addenda done] $(date -u +%T)" >> "$S/state.log"
