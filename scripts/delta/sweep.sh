#!/usr/bin/env bash
# Group-size sweep on the large arena (run from the bundle root, after setup):
# for each N_max, a 64 x 30 min evaluation of the paper rule and the run B
# actors (zero-shot) plus carrier counts from the paper's model (16 x 6 min).
# A 64-attempt evaluation leaves most of the GPU idle, so up to $PARALLEL
# values run at once.
#   PARALLEL=3 bash scripts/delta/sweep.sh 1 60 150 185 370
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p runs
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a runs/sweep.log; }
PY=.venv/bin/python
PARALLEL=${PARALLEL:-1}

one() {
  local n=$1
  log "N_max $n evaluation"
  $PY scripts/evaluate_arena.py --arena large --n-sites "$n" --puzzles 64 \
    --minutes 30 --no-model \
    --actor MAPPO=runs/delta/run_B/actor.json \
    --actor BC=runs/delta/run_B/actor_bc.json \
    --out "runs/sweep_N$n.json" > "runs/sweep_N$n.txt" 2>&1
  log "N_max $n evaluation exit $?"
  $PY scripts/validate_interface.py --arena large --n-sites "$n" \
    --puzzles 16 --minutes 6 > "runs/sweep_N${n}_validation.txt" 2>&1
  log "N_max $n validation exit $?"
}

for n in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  one "$n" &
done
wait
log "done"
