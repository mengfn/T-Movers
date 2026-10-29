#!/usr/bin/env bash
# Unattended GPU pipeline for a Delta sandbox (run from the bundle root):
# waits for setup.sh to finish, then validation -> benchmark -> MAPPO training
# -> evaluation against the paper's rule. Everything lands in runs/.
#   bash scripts/delta/pipeline.sh <training hours>
set -uo pipefail
cd "$(dirname "$0")/../.."
HOURS=${1:-3}
mkdir -p runs
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a runs/pipeline.log; }

until grep -q "EXIT=" /root/setup.log 2>/dev/null; do sleep 20; done
grep -q "EXIT=0" /root/setup.log || { log "setup failed"; tail -20 /root/setup.log >> runs/pipeline.log; exit 1; }
PY=.venv/bin/python
log "setup ok: $($PY -c 'import jax; print(jax.devices())' 2>&1 | tail -1)"

log "validation"
$PY scripts/validate_interface.py --puzzles 64 --minutes 15 > runs/validation.txt 2>&1
log "validation exit $?"

log "benchmark"
$PY scripts/benchmark.py --batch 256 1024 2048 > runs/benchmark.txt 2>&1
log "benchmark exit $?"

log "training for ${HOURS} h"
$PY scripts/train.py --envs 1024 --rollout 64 --updates 100000 --hours "$HOURS" \
  --warm-start --random-start 0.5 --train-minutes 5 \
  --eval-puzzles 64 --eval-minutes 30 --out runs/mappo > runs/train.txt 2>&1
log "training exit $?"
log "done"
