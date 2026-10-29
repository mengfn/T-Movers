#!/usr/bin/env bash
# One-shot small-arena evaluation for a Delta sandbox (run from the bundle
# root): setup -> interface validation -> 64-attempt evaluation of the paper
# model, the paper rule and the run B actors. Writes runs/small_*; the last
# line of runs/small.log is "done" when everything finished.
#   bash scripts/delta/small_eval.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p runs
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a runs/small.log; }

log "setup"
# Mirrors behind some providers drop connections; finished downloads stay in
# uv's cache, so retrying resumes rather than restarts.
export UV_HTTP_TIMEOUT=120
for attempt in 1 2 3 4; do
  bash scripts/delta/setup.sh > runs/setup.txt 2>&1 && break
  log "setup attempt $attempt failed"
  [ "$attempt" = 4 ] && { log "setup failed"; exit 1; }
done
PY=.venv/bin/python

log "validation"
$PY scripts/validate_interface.py --arena small --puzzles 64 --minutes 15 \
  > runs/small_validation.txt 2>&1
log "validation exit $?"

log "evaluation"
$PY scripts/evaluate_arena.py --arena small --puzzles 64 --minutes 30 \
  --actor MAPPO=runs/delta/run_B/actor.json \
  --actor BC=runs/delta/run_B/actor_bc.json \
  --out runs/small_eval.json > runs/small_eval.txt 2>&1
log "evaluation exit $?"
log "done"
