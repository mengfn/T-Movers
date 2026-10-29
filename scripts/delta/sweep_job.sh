#!/usr/bin/env bash
# One-shot sweep job for a fresh Delta sandbox: setup (with retries), then the
# group-size sweep. Progress in runs/sweep.log; last line "done".
#   TMOVERS_INDEX=https://mirrors.aliyun.com/pypi/simple \
#     bash scripts/delta/sweep_job.sh 1 60 150 185 370
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p runs
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a runs/sweep.log; }
export UV_HTTP_TIMEOUT=120
log "setup"
for attempt in 1 2 3 4; do
  bash scripts/delta/setup.sh > runs/setup.txt 2>&1 && break
  log "setup attempt $attempt failed"
  [ "$attempt" = 4 ] && { log "setup failed"; exit 1; }
done
log "setup ok"
PARALLEL=${PARALLEL:-3} bash scripts/delta/sweep.sh "$@"
